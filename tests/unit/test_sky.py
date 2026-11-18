import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from app import netcdf, sky
from app.models import NowcastParams


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
