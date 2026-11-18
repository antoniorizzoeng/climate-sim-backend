import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from app import netcdf, sky
from app.models import NowcastParams


@pytest.fixture
def column():
    heights = {
        1000: 100,
        925: 700,
        850: 1500,
        800: 2000,
        700: 3000,
        600: 4200,
        500: 5600,
        400: 7200,
        350: 8000,
        300: 9200,
        250: 10400,
        200: 11800,
        150: 13500,
        100: 16000,
    }
    return {level: (height, 1.0) for level, height in heights.items()}


@pytest.mark.parametrize(
    ("elevation", "expected"),
    [
        (0, (1, 1, 1)),
        (1750, (0.5, 1, 1)),
        (4205, (0, 0.809, 1)),
        (9000, (0, 0, 0.90625)),
    ],
)
def test_observer_in_forecast_bands(column, elevation, expected):
    assert sky.layer_weights(elevation, column) == pytest.approx(
        dict(zip(sky.LEVELS, expected))
    )


def test_cloud_at_site_and_marine_deck(column):
    # Only the 850 hPa sample is cloudy. Its slab spans 1100–1750 m.
    profile = {p: (z, float(p == 850)) for p, (z, _) in column.items()}
    assert sky.layer_weights(1500, profile)["low"] == pytest.approx(500 / 650)
    assert sky.layer_weights(2100, profile)["low"] == 0
    profile[850] = (2000, 1)
    assert sky.layer_weights(2100, profile)["low"] > 0


def test_missing_cloud_samples_use_local_band_thickness(column):
    profile = {p: (z, np.nan) for p, (z, _) in column.items()}
    profile[700] = (3300, np.nan)
    assert sky.layer_weights(1750, profile)["low"] == pytest.approx(1800 / 3300)
    profile[700] = (np.nan, np.nan)
    top = np.interp(math.log(700), np.log([600, 800]), [4200, 2000])
    assert sky.layer_weights(1750, profile)["low"] == pytest.approx((top - 1500) / top)
    # No measurable ceilings needed for the band entirely above the observer.
    profile[100] = (np.nan, np.nan)
    assert sky.layer_weights(1750, profile)["high"] == 1
    with pytest.raises(ValueError, match="band heights"):
        sky.layer_weights(9000, profile)


def test_weighted_timeline_and_legacy_queued_record(monkeypatch, tmp_path):
    weights = dict(low=0, mid=0.5, high=1)
    raw = dict(low=1, mid=0.4, high=0)
    forecast = dict(low=1, mid=0, high=0.2, total=1, jet_ms=10)
    record = dict(
        site_elevation=dict(elevation_m=5750, source="user", layer_weights=weights),
        data_time="2026-01-01T00:00:00+00:00",
        params=dict(hours=1, blend_tau_hours=2),
        layers=[dict(layer=k, run_id=k) for k in sky.LEVELS],
        _model={k: [v, v] for k, v in forecast.items()},
    )
    monkeypatch.setattr(sky, "site_samples", lambda path: [raw[path.parent.name]] * 7)
    rows = sky.timeline(record, tmp_path)
    transport = math.exp(-0.5)
    mid, high = 0.4 * transport, 0.2 * (1 - transport)
    clear = (1 - mid * 0.5) * (1 - high)
    assert rows[-1]["blend"] == pytest.approx(
        dict(low=1, mid=mid, high=high, total=1 - clear, clear=clear)
    )
    assert rows[-1]["nowcast"] == pytest.approx(dict(**raw, total=0.2, clear=0.8))
    assert rows[-1]["model"] == pytest.approx(
        dict(low=1, mid=0, high=0.2, total=0.2, clear=0.8)
    )
    assert sky.best_window(rows)["end"] == rows[-1]["time"]
    del record["site_elevation"]
    legacy = sky.timeline(record, tmp_path)
    assert all(
        row[series]["total"] == 1
        for row in legacy
        for series in ("nowcast", "model", "blend")
    )
    assert sky.best_window(legacy) is None


def test_geometry_and_sampling():
    p = NowcastParams(lat=60, lon=179.99, resolution=16)
    g = sky.grid(p)
    assert g["dx_m"] == g["dy_m"] == 600_000 / 16
    lat_span = math.degrees(7.5 * g["dy_m"] / 6371000)
    assert g["lat0"] == pytest.approx(60 - lat_span)
    assert g["lon0"] == pytest.approx(179.99 - 2 * lat_span)
    points = sky.weather_points(p)
    assert points[0] == pytest.approx((g["lat0"], g["lon0"]))
    assert points[144] == pytest.approx((p.lat, p.lon))
    assert points[-1][0] == pytest.approx(60 + lat_span)
    assert points[-1][1] < -170
    with pytest.raises(ValidationError, match="poles"):
        NowcastParams(lat=-89.9, lon=0)


def test_resampling_and_wind():
    np.testing.assert_allclose(
        sky.resample(np.array([[2, 4], [6, 8]]), 3), [[2, 3, 4], [4, 5, 6], [6, 7, 8]]
    )
    u, v = sky.wind_components(np.ones(8), np.arange(0, 360, 45))
    q = math.sqrt(0.5)
    np.testing.assert_allclose(u, [0, -q, -1, -q, 0, q, 1, q], atol=1e-12)
    np.testing.assert_allclose(v, [-1, -q, 0, q, 1, q, 0, -q], atol=1e-12)


def test_interpolate_components_before_angles():
    data = dict(
        cloud_cover=np.zeros((4, 2)),
        cloud_cover_low=np.ones((4, 2)),
        wind_speed_850hPa=np.ones((4, 2)),
        wind_direction_850hPa=np.tile([[359, 90], [1, 270]], (2, 1)),
    )
    plane, u, v = sky.layer_inputs(data, "low", 3)
    assert np.all(plane == 1)
    np.testing.assert_allclose(u[:, 1, 1], [0, 0], atol=1e-12)
    np.testing.assert_allclose(v[:, 1, 1], [-math.cos(math.pi / 180), 0], atol=1e-12)


def test_overlap_seeing_and_stability():
    assert sky.cover(0, 0, 0)["clear"] == 1
    assert sky.cover(0.2, 0.3, 0.4)["total"] == pytest.approx(1 - 0.8 * 0.7 * 0.6)
    assert sky.cover(0.1, 1, 0.5)["clear"] == 0
    assert [sky.seeing(v) for v in (19.9, 20, 35, 35.01)] == [
        "good",
        "fair",
        "fair",
        "poor",
    ]
    p = NowcastParams(lat=0, lon=0, radius_km=50, resolution=256, hours=12)
    req, limit = sky.run_request(p, "field", np.array([0, -100]), np.array([75, 0]))
    dx = 100000 / 256
    assert limit == pytest.approx(1 / (175 / dx + 4 * sky.DIFFUSION / dx**2))
    assert req.dt <= 0.9 * limit and req.dt * req.out_every == pytest.approx(600)
    assert req.dt * req.steps == pytest.approx(12 * 3600)


@pytest.mark.parametrize("n", [4, 5])
def test_bilinear_centre(tmp_path, n):
    path = tmp_path / "output.nc"
    plane = np.zeros((n, n))
    plane[(n - 1) // 2, (n - 1) // 2] = 1
    with netcdf.dataset(path, "w", format="NETCDF3_64BIT_DATA") as ds:
        for dim, size in (("time", 2), ("y", n), ("x", n)):
            ds.createDimension(dim, size)
        ds.createVariable("u", "f8", ("time", "y", "x"))[:] = [plane, plane * 0]
    np.testing.assert_allclose(sky.site_samples(path), [1 if n % 2 else 0.25, 0])


def test_best_window_requires_contiguous_clear_samples_and_prefers_earliest_tie():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        dict(
            time=(start + timedelta(minutes=10 * i)).isoformat(),
            blend=dict(clear=clear),
            seeing=seeing,
        )
        for i, (clear, seeing) in enumerate(
            [
                (1, "good"),
                (1, "poor"),
                (0.70, "good"),
                (0.8, "fair"),
                (0.69, "good"),
                (0.9, "good"),
                (1, "good"),
            ]
        )
    ]
    assert sky.best_window(rows) == dict(
        start=rows[2]["time"], end=rows[3]["time"], mean_clear=0.75, seeing="fair"
    )
    assert sky.best_window(rows[1:2] + rows[4:5]) is None
