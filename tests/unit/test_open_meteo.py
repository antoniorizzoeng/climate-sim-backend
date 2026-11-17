import asyncio
import copy
import json
import sqlite3
from contextlib import closing
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from app.models import NowcastParams
from app.open_meteo import BATCH_SIZE, UNITS, ForecastClient, parse


@pytest.fixture
def recorded():
    return json.loads(
        (Path(__file__).parents[1] / "fixtures/open_meteo.json").read_text()
    )


def test_recorded_response(recorded):
    times = recorded[0]["hourly"]["time"]
    parsed = parse(recorded, 2, times)
    assert set(parsed) == set(UNITS)
    assert parsed["cloud_cover_mid"][1].tolist() == [0.09, 0.88]
    assert parsed["wind_speed_250hPa"][0].tolist() == [66.87, 67.59]
    assert parse(recorded[0], 1, times)["cloud_cover"][0].tolist() == [0.86, 0.99]
    for key, value in [("utc_offset_seconds", 3600), ("location_id", 1)]:
        bad = dict(recorded[0], **{key: value})
        with pytest.raises(ValueError, match="location or UTC"):
            parse(bad, 1, times)
    with pytest.raises(ValueError, match="number of locations"):
        parse(recorded, 1, times)
    with pytest.raises(ValueError, match="UTC"):
        parse(recorded, 2, times[::-1])
    recorded[0]["hourly_units"]["wind_speed_250hPa"] = "km/h"
    with pytest.raises(ValueError, match="units"):
        parse(recorded, 2, times)


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("cloud_cover", [None, 20]),
        ("cloud_cover_low", [101, 0]),
        ("wind_direction_500hPa", [361, 3]),
        ("wind_speed_250hPa", [-1, 0]),
        ("wind_speed_300hPa", []),
    ],
)
def test_reject_bad_records(recorded, name, values):
    recorded[0]["hourly"][name] = values
    with pytest.raises(ValueError, match=name):
        parse(recorded, 2, recorded[0]["hourly"]["time"])


def test_batches_cache_refresh_and_expiry(tmp_path, clock, recorded, monkeypatch):
    calls = []

    async def get(client, url, params):
        assert url == "https://api.open-meteo.com/v1/forecast"
        assert client.headers["User-Agent"] == "sky-nowcast/1.1"
        assert list(UNITS) == params["hourly"].split(",")
        assert params["wind_speed_unit"] == "ms"
        size = len(params["latitude"].split(","))
        assert size == len(params["longitude"].split(","))
        calls.append(size)
        data = []
        for i in range(size):
            point = copy.deepcopy(recorded[i % 2])
            point["location_id"] = i
            point["hourly"]["time"] = [params["start_hour"], params["end_hour"]]
            data.append(point)
        return httpx.Response(200, json=data)

    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    p = NowcastParams(lat=45, lon=10, hours=1, radius_km=100)
    start = clock()

    def fetch(fresh):
        api = ForecastClient(tmp_path, clock)
        api.start()
        return asyncio.run(api.fetch(p, start, fresh))

    assert fetch(False)["cloud_cover_low"].shape == (81, 2)
    assert calls == [BATCH_SIZE, 17]
    fetch(False)
    assert len(calls) == 2
    fetch(True)
    assert len(calls) == 4
    clock.time += timedelta(seconds=3599)
    fetch(False)
    assert len(calls) == 4
    clock.time += timedelta(seconds=1)
    fetch(False)
    assert len(calls) == 6


def test_quota_and_429(tmp_path, clock, recorded):
    api = ForecastClient(tmp_path, clock)
    api.start()
    now = clock().timestamp()
    with closing(sqlite3.connect(api.db)) as db, db:
        db.execute("INSERT INTO usage VALUES (?, ?)", (now - 1, 590))
        assert api.reserve(db, 64, now) == 59
        db.execute("INSERT INTO usage VALUES (?, ?)", (now - 200, 4900))
        with pytest.raises(ValueError, match="quota"):
            api.reserve(db, 64, now)
        db.execute("DELETE FROM usage")

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(429, headers={"Retry-After": "120"})
            )
        ) as c:
            for message in ("HTTP 429", "Retry in 120"):
                with pytest.raises(ValueError, match=message):
                    await api.batch(c, [(0, 0)], recorded[0]["hourly"]["time"], False)

    asyncio.run(check())
