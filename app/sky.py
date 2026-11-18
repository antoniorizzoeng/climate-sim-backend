"""Physical grid, cloud transport inputs and site sampling (SI units, UTC)."""

import math
from datetime import datetime, timedelta
from itertools import groupby
from statistics import mean

import numpy as np

from . import netcdf
from .models import RunRequest, stability_limit

LEVELS = {"low": 850, "mid": 500, "high": 300}
DIFFUSION = 10.0  # m²/s
FRAME_SECONDS = 600


def grid(params):
    n = params.resolution
    spacing = 2000 * params.radius_km / n
    half_span = math.degrees(spacing * (n - 1) / (2 * 6371000))
    west = params.lon - half_span / math.cos(math.radians(params.lat))
    return dict(
        nx=n,
        ny=n,
        dx_m=spacing,
        dy_m=spacing,
        lat0=params.lat - half_span,
        lon0=(west + 180) % 360 - 180,
    )


def weather_points(params):
    # Odd side includes the observing site, even on an even-sized engine grid.
    side = min(17, 2 * math.ceil(params.radius_km / 25) + 1)
    g = grid(params)
    offsets = np.linspace(g["lat0"] - params.lat, params.lat - g["lat0"], side)
    return [
        (
            params.lat + y,
            (params.lon + x / math.cos(math.radians(params.lat)) + 180) % 360 - 180,
        )
        for y in offsets
        for x in offsets
    ]


def wind_components(speed, direction):
    radians = np.deg2rad(direction)
    return -speed * np.sin(radians), -speed * np.cos(radians)


def resample(values, n):
    """Bilinear interpolation along the last two axes of a square source grid."""
    x = np.linspace(0, values.shape[-1] - 1, n)
    left = x.astype(int)
    right = np.minimum(left + 1, values.shape[-1] - 1)
    w = x - left
    rows = values[..., left] * (1 - w) + values[..., right] * w
    return rows[..., left, :] * (1 - w[:, None]) + rows[..., right, :] * w[:, None]


def layer_inputs(data, layer, n):
    # Source arrays are [location, hour], locations ordered south-west first.
    side = math.isqrt(len(data["cloud_cover"]))

    def cube(key):
        return data[key].T.reshape(-1, side, side)

    level = LEVELS[layer]
    u, v = wind_components(
        cube(f"wind_speed_{level}hPa"), cube(f"wind_direction_{level}hPa")
    )
    return (
        resample(cube(f"cloud_cover_{layer}")[0], n),
        resample(u, n),
        resample(v, n),
    )


def run_request(params, field_id, u, v):
    dx = grid(params)["dx_m"]
    limit = stability_limit(
        dx, dx, DIFFUSION, float(np.abs(u).max()), float(np.abs(v).max())
    )
    every = max(1, math.ceil(FRAME_SECONDS / (0.9 * limit)))
    req = RunRequest(
        source={"kind": "field", "field_id": field_id},
        dx=dx,
        dy=dx,
        D=DIFFUSION,
        dt=FRAME_SECONDS / every,
        steps=params.hours * 6 * every,
        out_every=every,
        bc=dict.fromkeys(("left", "right", "bottom", "top"), "neumann"),
    )
    return req, limit


def wind_config(path):
    return dict(mode="file", path=str(path), vx_var="vx", vy_var="vy", interval=3600)


def write_wind(path, u, v):
    with netcdf.dataset(path, "w", format="NETCDF3_64BIT_DATA") as ds:
        dims = ("time", "y", "x")
        for name, size in zip(dims, u.shape):
            ds.createDimension(name, size)
        for name, value in (("vx", u), ("vy", v)):
            var = ds.createVariable(name, "f8", dims, fill_value=False)
            var.units = "m/s"
            var[:] = value


def site_samples(path):
    # The centre has weight 1 on an odd grid, or 1/4 on its four neighbours.
    with netcdf.dataset(path) as ds:
        var = ds["u"]
        _, ny, nx = var.shape
        centre = var[:, (ny - 1) // 2 : ny // 2 + 1, (nx - 1) // 2 : nx // 2 + 1]
        return np.clip(centre.mean(axis=(1, 2)), 0, 1)


def cover(low, mid, high, total=None):
    if total is None:
        total = 1 - (1 - low) * (1 - mid) * (1 - high)
    return dict(low=low, mid=mid, high=high, total=total, clear=1 - total)


def seeing(speed):
    if speed < 20:
        return "good"
    return "poor" if speed > 35 else "fair"


def blend_site(rows, tau_hours):
    if not rows:
        return []
    start = datetime.fromisoformat(rows[0]["time"])
    blended = []
    for row in rows:
        lead_hours = (
            datetime.fromisoformat(row["time"]) - start
        ).total_seconds() / 3600
        weight = math.exp(-lead_hours / tau_hours)
        cloud = {
            key: weight * row["nowcast"][key] + (1 - weight) * row["model"][key]
            for key in (*LEVELS, "total")
        }
        blended.append(dict(row, blend=cover(**cloud)))
    return blended


def best_window(rows):
    def qualifies(row):
        return row["blend"]["clear"] >= 0.70 and row["seeing"] != "poor"

    stretches = (list(group) for good, group in groupby(rows, qualifies) if good)
    best = max(stretches, key=len, default=None)
    if best is None:
        return None
    return dict(
        start=best[0]["time"],
        end=best[-1]["time"],
        mean_clear=mean(row["blend"]["clear"] for row in best),
        seeing="fair" if any(row["seeing"] == "fair" for row in best) else "good",
    )


def timeline(record, runs_dir):
    model = record["_model"]
    samples = {
        entry["layer"]: site_samples(runs_dir / entry["run_id"] / "result.nc")
        for entry in record["layers"]
    }
    start = datetime.fromisoformat(record["data_time"])
    hours = range(record["params"]["hours"] + 1)
    rows = []
    for i in range(6 * (len(hours) - 1) + 1):
        at_site = {k: s[i] for k, s in samples.items()}
        forecast = {
            k: float(np.interp(i / 6, hours, model[k]))
            for k in ("low", "mid", "high", "total")
        }
        jet = model["jet_ms"][i // 6]
        rows.append(
            dict(
                time=(start + timedelta(seconds=i * FRAME_SECONDS)).isoformat(),
                nowcast=cover(**at_site),
                model=cover(**forecast),
                jet_ms=jet,
                seeing=seeing(jet),
            )
        )
    return blend_site(rows, record["params"]["blend_tau_hours"])
