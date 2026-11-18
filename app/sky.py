"""Physical grid, cloud transport inputs and site sampling (SI units, UTC)."""

import math
from datetime import datetime, timedelta
from itertools import groupby
from statistics import mean

import numpy as np

from . import netcdf
from .models import RunRequest, stability_limit

LEVELS = {"low": 850, "mid": 500, "high": 300}
# Pressure surfaces bracketing the nominal 3 and 8 km layer boundaries.
BAND_LEVELS = {
    "low": (1000, 925, 850, 800, 700),
    "mid": (700, 600, 500, 400, 350),
    "high": (350, 300, 250, 200, 150, 100),
}
FOG_MARGIN_M = 250
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


def run_request(params, field_id, u, v, seconds=None):
    dx = grid(params)["dx_m"]
    limit = stability_limit(
        dx, dx, DIFFUSION, float(np.abs(u).max()), float(np.abs(v).max())
    )
    seconds = seconds or params.hours * 3600
    interval = math.gcd(seconds, FRAME_SECONDS)
    subdivisions = max(1, math.ceil(interval / (0.9 * limit)))
    every = (FRAME_SECONDS // interval) * subdivisions
    req = RunRequest(
        source={"kind": "field", "field_id": field_id},
        dx=dx,
        dy=dx,
        D=DIFFUSION,
        dt=FRAME_SECONDS / every,
        steps=(seconds // interval) * subdivisions,
        units={"length": "m", "time": "s"},
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


def layer_weights(elevation, profile):
    """Estimate each band's above-observer share at the forecast's initial UTC hour.

    Open-Meteo describes low/mid/high as roughly <3/3–8/>8 km. We use
    the local 700 and 350 hPa heights as boundaries, from sea level up to
    100 hPa. Geopotential heights are relative to sea level. Each pressure
    sample represents the slab halfway to its neighbours within the band.
    Integrate its cloud fraction times slab thickness above elevation minus
    250 m (includes fog at site height), divided by the full-band integral.
    Interpolate missing boundary heights in log pressure between available
    forecast surfaces. A clear or unavailable cloud profile gets the geometric
    thickness share. The original horizontal layer covers are preserved.
    See https://open-meteo.com/en/docs
    """
    cutoff = elevation - FOG_MARGIN_M
    if cutoff <= 0:
        return dict.fromkeys(LEVELS, 1.0)
    # Interpolation in log pressure approximates the hypsometric relation.
    # Require bracketing data rather than extrapolating beyond the column.
    measured = sorted((p, z) for p, (z, _) in profile.items() if np.isfinite(z))
    if not measured:
        raise ValueError("Missing Open-Meteo pressure-level heights")
    pressure, height = np.array(measured).T
    if (np.diff(height) > 0).any():
        raise ValueError("Unordered Open-Meteo pressure-level heights")
    log_pressure = np.log(pressure)
    weights = {}
    for layer, levels in BAND_LEVELS.items():
        heights, cloud = np.array([profile[p] for p in levels]).T
        bottom, top = np.interp(
            np.log([levels[0], levels[-1]]),
            log_pressure,
            height,
            left=np.nan,
            right=np.nan,
        )
        if layer == "low":
            bottom = 0.0
        if cutoff <= bottom:
            weights[layer] = 1.0
            continue
        if cutoff >= top:
            weights[layer] = 0.0
            continue
        if not (np.isfinite(bottom) and np.isfinite(top) and bottom < top):
            raise ValueError(f"Missing or invalid {layer} cloud band heights")
        valid = np.isfinite(heights) & np.isfinite(cloud)
        heights = np.clip(heights[valid], bottom, top)
        edges = np.r_[bottom, (heights[:-1] + heights[1:]) / 2, top]
        density = cloud[valid]
        share = (top - cutoff) / (top - bottom)
        if density.size:
            amount = np.dot(density, np.diff(edges))
            if amount > 0:
                above = np.diff(np.maximum(edges, cutoff))
                share = np.dot(density, above) / amount
        weights[layer] = float(np.clip(share, 0, 1))
    return weights


def cover(low, mid, high, total=None, *, weights=None):
    layers = dict(low=low, mid=mid, high=high)
    if total is None or (weights is not None and min(weights.values()) < 1):
        weights = weights or dict.fromkeys(LEVELS, 1.0)
        total = 1 - math.prod(1 - layers[k] * weights[k] for k in LEVELS)
    return dict(**layers, total=total, clear=1 - total)


def seeing(speed):
    if speed < 20:
        return "good"
    return "poor" if speed > 35 else "fair"


def blend_site(rows, tau_hours, weights=None):
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
            for key in ((*LEVELS, "total") if weights is None else LEVELS)
        }
        blended.append(dict(row, blend=cover(**cloud, weights=weights)))
    return blended


def best_window(rows, night=None):
    if night and rows:
        dusk, dawn = (datetime.fromisoformat(night[key]) for key in ("dusk", "dawn"))
        start = datetime.fromisoformat(rows[0]["time"])

        def in_night(row):
            time = datetime.fromisoformat(row["time"])
            return (
                dusk <= time <= dawn
                and (time - start).total_seconds() % FRAME_SECONDS == 0
            )

        rows = filter(in_night, rows)

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
    elevation = record.get("site_elevation")
    weights = elevation["layer_weights"] if elevation else None
    samples = {
        entry["layer"]: site_samples(runs_dir / entry["run_id"] / "result.nc")
        for entry in record["layers"]
    }
    start = datetime.fromisoformat(record["data_time"])
    seconds = record.get("_horizon_seconds", record["params"]["hours"] * 3600)
    hours = range(math.ceil(seconds / 3600) + 1)
    rows = []
    for i in range(math.ceil(seconds / FRAME_SECONDS) + 1):
        elapsed = min(i * FRAME_SECONDS, seconds)
        at_site = {k: s[i] for k, s in samples.items()}
        forecast = {
            k: float(np.interp(elapsed / 3600, hours, model[k]))
            for k in ("low", "mid", "high", "total")
        }
        jet = model["jet_ms"][elapsed // 3600]
        rows.append(
            dict(
                time=(start + timedelta(seconds=elapsed)).isoformat(),
                nowcast=cover(**at_site, weights=weights),
                model=cover(**forecast, weights=weights),
                jet_ms=jet,
                seeing=seeing(jet),
            )
        )
    return blend_site(rows, record["params"]["blend_tau_hours"], weights)
