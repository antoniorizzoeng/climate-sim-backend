from datetime import UTC, datetime, timedelta

import pytest
from astronomy import Body, Observer

from app.astronomy import altitude, ephemeris_time, night_at
from app.sky import best_window


# Independent JPL DE421 ephemeris, evaluated with Skyfield 1.55's built-in
# timescale. WGS84 sea level, UT event times rounded to the nearest second.
# dark_twilight_day/find_discrete, Earth + site -> apparent Moon.altaz()
# (refraction disabled), and almanac.fraction_illuminated at 0, 1 and 2 h.
# https://rhodesmill.org/skyfield/almanac.html#the-phases-of-twilight
@pytest.mark.parametrize(
    "lat,lon,start,dusk,dawn,altitudes,fractions",
    [
        (
            28.7606,
            -17.8816,
            "2026-03-21T00:00",
            "2026-03-20T20:41:38",
            "2026-03-21T05:55:36",
            [-32.104176, -40.452028, -45.951721],
            [0.050566, 0.052723, 0.054923],
        ),
        (
            -24.6272,
            -70.4042,
            "2026-07-29T04:00",
            "2026-07-28T23:35:00",
            "2026-07-29T10:01:06",
            [82.621891, 83.035048, 69.787951],
            [0.997656, 0.998005, 0.998321],
        ),
        (
            69.6492,
            18.9553,
            "2026-01-15T00:00",
            "2026-01-14T16:31:48",
            "2026-01-15T05:13:53",
            [-35.328607, -30.450850, -25.545308],
            [0.134867, 0.132187, 0.129528],
        ),
    ],
    ids=["La Palma", "Paranal", "Tromso"],
)
def test_night_against_de421(lat, lon, start, dusk, dawn, altitudes, fractions):
    def utc(value):
        return datetime.fromisoformat(value).replace(tzinfo=UTC)

    start = utc(start)
    night = night_at(start, lat, lon, 24)
    for key, reference in (("dusk", dusk), ("dawn", dawn)):
        assert abs(utc(night[key]) - utc(reference)) <= timedelta(minutes=2)
    for i, row in enumerate(night["moon"]):
        assert utc(row["time"]) == start + timedelta(minutes=10 * i)
    assert (
        utc(night["moon"][-1]["time"])
        <= utc(night["dawn"])
        < utc(night["moon"][-1]["time"]) + timedelta(minutes=10)
    )
    for i in range(3):
        assert night["moon"][6 * i]["altitude_deg"] == pytest.approx(
            altitudes[i], abs=1
        )
        assert night["moon"][6 * i]["illumination"] == pytest.approx(
            fractions[i], abs=0.002
        )


def test_polar_summer_and_upcoming_dusk(client, clock):
    # DE421 has no twilight transitions at Longyearbyen on the June solstice.
    clock.time = datetime(2026, 6, 21, tzinfo=UTC)
    params = dict(lat=78.2232, lon=15.6469, window="tonight")
    # Skyfield DE421 meridian_transits: the lower transit is still above the horizon.
    midnight_sun = datetime(2026, 6, 21, 22, 59, 20, tzinfo=UTC)
    sun = altitude(
        Body.Sun,
        Observer(params["lat"], params["lon"], 0),
        ephemeris_time(midnight_sun),
    )
    assert sun == pytest.approx(11.657494, abs=0.02)
    assert client.post("/nowcasts", json=params).status_code == 422
    assert client.get("/nowcasts").json()["items"] == []
    assert (
        client.post("/nowcasts", json=dict(params, window="now")).json()["night"]
        is None
    )
    clock.time = datetime(2026, 3, 20, 12, tzinfo=UTC)
    record = client.post(
        "/nowcasts", json=dict(lat=28.7606, lon=-17.8816, window="tonight")
    ).json()
    assert record["night"]["dusk"] == record["night"]["moon"][0]["time"]
    assert datetime.fromisoformat(record["night"]["dusk"]) > clock()
    rows = [
        dict(
            time=(clock() + timedelta(hours=i)).isoformat(),
            seeing="good",
            blend={"clear": 1},
        )
        for i in range(25)
    ]
    window = best_window(rows, record["night"])
    assert window["start"] == rows[9]["time"] and window["end"] == rows[17]["time"]


def test_polar_night_horizon_is_capped(client, clock):
    clock.time = datetime(2026, 12, 21, tzinfo=UTC)
    params = dict(lat=86, lon=0, radius_km=50, window="tonight")
    response = client.post("/nowcasts", json=params)
    assert response.status_code == 202
    record = response.json()
    assert datetime.fromisoformat(record["night"]["dawn"]) > clock() + timedelta(days=1)
    stored = client.app.state.manager.nowcasts.records.records[record["id"]]
    assert stored["_horizon_seconds"] == 86400
    clock.time = datetime(2027, 6, 21, tzinfo=UTC)
    assert client.post("/nowcasts/" + record["id"] + "/refresh").status_code == 422
