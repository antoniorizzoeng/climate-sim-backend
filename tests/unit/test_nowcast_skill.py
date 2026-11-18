import json
import sys
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from scripts import nowcast_skill


@pytest.fixture
def skill_case():
    start = datetime(2026, 1, 1, 23, tzinfo=UTC)
    payload = {
        "hourly_units": {
            "time": "unixtime",
            "cloud_cover_low": "%",
            "cloud_cover_mid": "%",
            "cloud_cover_high": "%",
            "cloud_cover": "%",
        },
        "hourly": {
            "time": [int(start.timestamp()) + 3600 * i for i in range(3)],
            "cloud_cover_low": [10, 40, None],
            "cloud_cover_mid": [20, 20, 0],
            "cloud_cover_high": [30, 0, 0],
            "cloud_cover": [60, 30, 0],
        },
    }
    rows = [
        dict(
            time=(start + timedelta(minutes=minutes)).isoformat(),
            nowcast=dict(low=0, mid=0, high=0, total=0, clear=1),
            model=dict(low=0.2, mid=0.2, high=0.2, total=0.5, clear=0.5),
            blend=dict(low=0.15, mid=0.2, high=0.25, total=0.55, clear=0.45),
        )
        for minutes in (0, 10, 60, 70)
    ]
    return rows, payload


def test_mae_aligns_times_and_uses_common_samples(skill_case):
    rows, payload = skill_case
    matched, errors = nowcast_skill.evaluate(
        rows, nowcast_skill.reference_hours(payload)
    )
    assert matched == 3
    assert errors["nowcast"] == pytest.approx(
        dict(low=65 / 3, mid=20, high=55 / 3, total=145 / 3, clear=145 / 3)
    )
    assert errors["model"] == pytest.approx(
        dict(low=35 / 3, mid=0, high=35 / 3, total=35 / 3, clear=35 / 3)
    )
    assert errors["blend"] == pytest.approx(
        dict(low=10, mid=0, high=10, total=10, clear=10)
    )
    with pytest.raises(ValueError, match="No matching valid times"):
        nowcast_skill.evaluate(rows[-1:], nowcast_skill.reference_hours(payload))


def test_past_query_around_midnight(skill_case):
    rows, _ = skill_case
    now = datetime(2026, 1, 2, 2, 1, tzinfo=UTC)
    assert nowcast_skill.past_range(rows, now) == dict(past_hours=3, forecast_hours=0)
    assert nowcast_skill.past_range(rows, now + timedelta(days=1)) == dict(
        past_days=2, forecast_days=0
    )
    assert nowcast_skill.past_range(
        rows, now + timedelta(hours=22) - timedelta(minutes=2)
    ) == dict(past_hours=24, forecast_hours=0)
    started_earlier = [dict(rows[0], time="2026-01-01T22:00:00+00:00"), *rows[1:]]
    assert nowcast_skill.past_range(started_earlier, now + timedelta(hours=21)) == dict(
        past_days=1, forecast_days=1
    )
    with pytest.raises(ValueError, match="reference hours"):
        nowcast_skill.past_range(rows, now - timedelta(minutes=2))
    with pytest.raises(ValueError, match="92 past days"):
        nowcast_skill.past_range(rows, now + timedelta(days=92))


def test_reference_uses_saved_observer_weights(skill_case):
    rows, payload = skill_case
    _, errors = nowcast_skill.evaluate(
        rows[:1], nowcast_skill.reference_hours(payload), dict(low=0, mid=0.5, high=1)
    )
    # 20% mid at half weight and 30% high overlap to 37% above the site.
    assert errors["nowcast"]["total"] == pytest.approx(37)
    assert errors["nowcast"]["clear"] == pytest.approx(37)
    assert errors["nowcast"]["low"] == 10


@pytest.mark.parametrize(
    ("section", "value", "message"),
    [("hourly_units", "fraction", "units"), ("hourly", [-1, 0, 10], "percentage")],
)
def test_rejects_invalid_open_meteo_data(skill_case, section, value, message):
    _, payload = skill_case
    payload[section]["cloud_cover_low"] = value
    with pytest.raises(ValueError, match=message):
        nowcast_skill.reference_hours(payload)


@pytest.mark.parametrize("legacy", [False, True])
def test_reads_saved_nowcast_with_mocked_request(
    skill_case, tmp_path, monkeypatch, capsys, legacy
):
    rows, payload = skill_case
    if legacy:
        for row in rows:
            del row["blend"]
    record = tmp_path / "nowcast.json"
    record.write_text(
        json.dumps(dict(status="done", params=dict(lat=45, lon=10), site=rows))
    )
    monkeypatch.setattr(sys, "argv", ["nowcast_skill", str(record)])

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 2, 2, 1, tzinfo=tz)

    def get(url, *, params, timeout):
        assert url == nowcast_skill.URL
        assert params == dict(
            latitude=45,
            longitude=10,
            hourly="cloud_cover_low,cloud_cover_mid,cloud_cover_high,cloud_cover",
            timezone="GMT",
            timeformat="unixtime",
            past_hours=3,
            forecast_hours=0,
        )
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(nowcast_skill, "datetime", Clock)
    monkeypatch.setattr(httpx, "get", get)
    nowcast_skill.main()
    output = capsys.readouterr().out.splitlines()
    assert output[0].startswith("Matched 3/4 site times")
    assert output[3].split()[0] == "nowcast" and output[4].split()[0] == "model"
    assert output[5].split() == (
        ["blend", "18.51", "16.84", "20.42", "40.44", "40.44"]
        if legacy
        else ["blend", "10.00", "0.00", "10.00", "10.00", "10.00"]
    )


def test_cli_rejects_unfinished_record(tmp_path, monkeypatch, capsys):
    record = tmp_path / "nowcast.json"
    record.write_text(json.dumps(dict(status="running", site=None)))
    monkeypatch.setattr(sys, "argv", ["nowcast_skill", str(record)])
    with pytest.raises(SystemExit) as error:
        nowcast_skill.main()
    assert error.value.code == 2
    assert "A completed nowcast" in capsys.readouterr().err


def test_cli_from_repository_without_pythonpath(tmp_path):
    import os
    import subprocess
    from pathlib import Path

    record = tmp_path / "metadata.json"
    record.write_text(json.dumps({"status": "running", "site": None}))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "scripts/nowcast_skill.py", str(record)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "A completed nowcast with site data is required" in result.stderr
