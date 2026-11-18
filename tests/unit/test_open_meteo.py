import asyncio
import copy
import json
import sqlite3
from contextlib import closing
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from app import open_meteo
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
    assert set(parsed) == {*UNITS, "elevation"}
    assert parsed["elevation"] == [40.0, 38.0]
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

    data = fetch(False)
    assert data["cloud_cover_low"].shape == (81, 2)
    assert data["site_elevation"] == dict(
        elevation_m=40, source="open-meteo", layer_weights=dict(low=1, mid=1, high=1)
    )
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


@pytest.mark.parametrize("override", [None, 0, 4205])
def test_observer_uses_centre_and_initial_hour(
    tmp_path, clock, recorded, monkeypatch, override
):
    profile = json.loads(
        (Path(__file__).parents[1] / "fixtures/mauna_kea_profile.json").read_text()
    )
    params = NowcastParams(
        lat=19.8206, lon=-155.4681, radius_km=200, hours=1, elevation_m=override
    )
    site = (f"{params.lat:.6f}", f"{params.lon:.6f}")
    profile_calls = []

    async def get(client, url, params):
        points = list(
            zip(params["latitude"].split(","), params["longitude"].split(","))
        )
        if params["hourly"].split(",") == list(open_meteo.PROFILE_UNITS):
            assert params["start_hour"] == params["end_hour"]
            assert points == [site]
            profile_calls.append(points)
            profile["hourly"]["time"] = [params["start_hour"]]
            return httpx.Response(200, json=profile)
        rows = []
        for i, point in enumerate(points):
            row = copy.deepcopy(recorded[0])
            row.update(location_id=i, elevation=4196 if point == site else 20)
            row["hourly"]["time"] = [params["start_hour"], params["end_hour"]]
            rows.append(row)
        return httpx.Response(200, json=rows)

    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    api = ForecastClient(tmp_path, clock)
    api.start()
    result = asyncio.run(api.fetch(params, clock(), False))["site_elevation"]
    elevation = 4196 if override is None else override
    assert result["elevation_m"] == elevation
    assert result["source"] == ("open-meteo" if override is None else "user")
    assert len(profile_calls) == int(elevation > 0)
    # In this profile the middle band is clear, with limits at 3178 and 8576.22 m.
    mid = min(1, (8576.22 - (elevation - 250)) / (8576.22 - 3178))
    assert result["layer_weights"] == pytest.approx(
        dict(low=int(elevation == 0), mid=mid, high=1)
    )
