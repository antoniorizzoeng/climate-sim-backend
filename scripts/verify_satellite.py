"""Verify site cloud forecasts with GOES ABI full-disk mask and cloud-top height.

Score both cloud above site elevation plus a margin and any cloud in the column.
See docs/satellite-verification.md for alignment, quality criteria and limits.
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

import fsspec
import h5py
import httpx
import numpy as np
from pyproj import CRS, Geod, Proj, Transformer

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import DEFAULT_BLEND_TAU_HOURS
from app.sky import blend_site

SERIES = ("nowcast", "model", "blend", "persistence")
TOLERANCE = timedelta(minutes=10)
ACM = "ABI-L2-ACMF"
ACHA = "ABI-L2-ACHAF"
OBSERVATORIES = (
    ("Mauna Kea", 19.82, -155.47, 4205),
    ("Paranal", -24.63, -70.40, 2635),
    ("Roque de los Muchachos", 28.76, -17.89, 2396),
)
SLOTS = {"G18": -137.0, "G19": -75.2}
GEOD = Geod(ellps="GRS80")
ECEF = Transformer.from_crs(4979, 4978, always_xy=True)


def zenith(lat, lon, slot):
    """Angle between the local geodetic normal and the site-to-platform ray."""
    x, y, z = ECEF.transform(lon, lat, np.zeros_like(lat))
    sx, sy, sz = ECEF.transform(slot, 0, 35786023)
    dx, dy, dz = sx - x, sy - y, sz - z
    phi, lam = np.radians(lat), np.radians(lon)
    up = np.cos(phi) * (dx * np.cos(lam) + dy * np.sin(lam)) + dz * np.sin(phi)
    return np.degrees(
        np.arccos(np.clip(up / np.sqrt(dx * dx + dy * dy + dz * dz), -1, 1))
    )


def view(lat, lon, radius):
    sat = min(SLOTS, key=lambda key: zenith(lat, lon, SLOTS[key]))
    az = np.arange(360)
    lons, lats, _ = GEOD.fwd(
        np.full(360, lon), np.full(360, lat), az, np.full(360, radius * 1000)
    )
    center = float(zenith(lat, lon, SLOTS[sat]))
    return sat, center, max(center, float(zenith(lats, lons, SLOTS[sat]).max()))


@dataclass(frozen=True)
class Scan:
    midpoint: datetime
    key: str


def hour(at):
    return at.replace(minute=0, second=0, microsecond=0)


def scan_time(stamp):
    return datetime.strptime(stamp, "%Y%j%H%M%S%f").replace(tzinfo=UTC)


class Archive:
    def __init__(self, client, cache):
        self.client = client
        self.cache = cache

    @staticmethod
    def endpoint(sat):
        return f"https://noaa-goes{sat[1:]}.s3.amazonaws.com"

    @lru_cache(maxsize=256)
    def listing(self, sat, start, product=ACM):
        params = {"list-type": 2, "prefix": f"{product}/{start:%Y/%j/%H}/"}
        scans = []
        while True:
            response = self.client.get(self.endpoint(sat), params=params)
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
            for key in root.findall("{*}Contents/{*}Key"):
                match = re.search(
                    rf"_{sat}_s(\d{{14}})_e(\d{{14}})_c\d+\.nc$", key.text
                )
                if match:
                    begin, end = map(scan_time, match.groups())
                    scans.append(Scan(begin + (end - begin) / 2, key.text))
            token = root.findtext("{*}NextContinuationToken")
            if not token:
                return scans
            params["continuation-token"] = token

    def nearest(self, sat, at, product=ACM):
        # S3 partitions by scan start. Include the preceding hour even when
        # the closest midpoint belongs to a scan that straddled midnight.
        first = hour(at - TOLERANCE - timedelta(minutes=15))
        last = hour(at + TOLERANCE)
        scans = list(self.listing(sat, first, product))
        if last != first:
            scans.extend(self.listing(sat, last, product))
        eligible = [s for s in scans if abs(s.midpoint - at) <= TOLERANCE]
        return min(
            eligible,
            key=lambda s: (abs(s.midpoint - at), s.midpoint, s.key),
            default=None,
        )

    def remote(self, sat, scan):
        return fsspec.open(
            f"{self.endpoint(sat)}/{scan.key}",
            block_size=65536,
            cache_type="blockcache",
        )

    def window(self, sat, scan, lat, lon, radius, height_scan):
        path = (
            self.cache
            / "windows-v2"
            / Path(scan.key).stem
            / (Path(height_scan.key).stem if height_scan else "no-height-scan")
            / f"{lat}_{lon}_{radius}.npz"
        )
        keys = ("distance_km", "bcm", "dqf", "height_m")
        if path.exists():
            with np.load(path) as data:
                return tuple(data[k] for k in keys)
        with self.remote(sat, scan) as remote, h5py.File(remote) as ds:
            distance, bcm, dqf, lons, lats = read_window(ds, lat, lon, radius)
        if height_scan:
            with self.remote(sat, height_scan) as remote, h5py.File(remote) as ds:
                heights = mapped_heights(ds, lons, lats)
        else:
            heights = np.full(distance.shape, np.nan)
        window = distance, bcm, dqf, heights
        path.parent.mkdir(exist_ok=True, parents=True)
        with TemporaryDirectory(dir=path.parent) as tmp:
            staged = Path(tmp) / "window.npz"
            np.savez_compressed(staged, **dict(zip(keys, window, strict=True)))
            staged.replace(path)
        return window


def scalar(value):
    value = np.asarray(value).item()
    return value.decode() if isinstance(value, bytes) else value


def grid(ds):
    attrs = ds["goes_imager_projection"].attrs
    proj = Proj(
        CRS.from_cf(
            {
                key: scalar(attrs[key])
                for key in (
                    "grid_mapping_name",
                    "longitude_of_projection_origin",
                    "latitude_of_projection_origin",
                    "perspective_point_height",
                    "semi_major_axis",
                    "semi_minor_axis",
                    "sweep_angle_axis",
                )
            }
        )
    )
    axes = []
    for axis in ("x", "y"):
        v = ds[axis]
        radians = v[:].astype(float) * scalar(v.attrs.get("scale_factor", 1)) + scalar(
            v.attrs.get("add_offset", 0)
        )
        axes.append(radians * scalar(attrs["perspective_point_height"]))
    return proj, *axes


def read_window(ds, lat, lon, radius):
    proj, x, y = grid(ds)
    east, north = proj(lon, lat)

    # A deliberately oversized projected square encloses the surface disk.
    # At geostationary distance the scan-angle displacement times instrument
    # height is at most ~1.02 times the surface displacement. Test membership
    # exactly on the ellipsoid after inverse projecting each candidate centre.
    def span(axis, center):
        idx = np.flatnonzero(abs(axis - center) <= 2000 * radius)
        return slice(idx[0], idx[-1] + 1) if idx.size else slice(0, 0)

    rows, cols = span(y, north), span(x, east)
    xx, yy = np.meshgrid(x[cols], y[rows])
    lons, lats = proj(xx, yy, inverse=True)
    geod = proj.crs.get_geod()
    _, _, meters = geod.inv(np.full(xx.shape, lon), np.full(xx.shape, lat), lons, lats)
    return meters / 1000, ds["BCM"][rows, cols], ds["DQF"][rows, cols], lons, lats


def mapped_heights(ds, lons, lats):
    """Sample the ACHA fixed-grid cell containing each ACM center, without interpolation."""
    proj, x, y = grid(ds)
    east, north = proj(lons, lats)
    # ABI fixed-grid axes are equally spaced. Keep out-of-footprint cells missing.
    col = np.floor((east - x[0]) / (x[1] - x[0]) + 0.5)
    row = np.floor((north - y[0]) / (y[1] - y[0]) + 0.5)
    inside = (col >= 0) & (col < len(x)) & (row >= 0) & (row < len(y))
    heights = np.full(lons.shape, np.nan)
    if not inside.any():
        return heights
    r, c = row[inside].astype(int), col[inside].astype(int)
    tile = np.s_[r.min() : r.max() + 1, c.min() : c.max() + 1]
    indices = r - r.min(), c - c.min()
    ht = ds["HT"]
    raw = ht[tile][indices]
    lo, hi = ht.attrs["valid_range"]
    valid = (ds["DQF"][tile][indices] == 0) & (raw != scalar(ht.attrs["_FillValue"]))
    valid &= (lo <= raw) & (raw <= hi)
    if scalar(ht.attrs["units"]) != "m":
        raise ValueError("Expected ACHA HT in metres")
    meters = raw.astype(float) * scalar(ht.attrs["scale_factor"]) + scalar(
        ht.attrs["add_offset"]
    )
    heights[inside] = np.where(valid, meters, np.nan)
    return heights


def observe(window, radii, elevation_m, margins):
    distance, bcm, dqf, heights = window
    result = {}
    for r in radii:
        inside = distance <= r
        valid = inside & (dqf == 0) & ((bcm == 0) | (bcm == 1))
        n, good = int(inside.sum()), int(valid.sum())
        cloudy = valid & (bcm == 1)
        cloud_tops = heights[cloudy]
        missing = int((~np.isfinite(cloud_tops)).sum())
        usable = good > 0 and 2 * good >= n
        above = {}
        for margin in margins:
            count = int((cloud_tops > elevation_m + margin).sum())
            above[margin] = dict(
                cloudy=count if not missing else None,
                fraction=count / good if usable and not missing else None,
            )
        result[r] = dict(
            valid=good,
            bad_or_missing=n - good,
            cloudy=cloud_tops.size,
            column_fraction=cloud_tops.size / good if usable else None,
            missing_height_pixels=missing,
            above_observer=above,
        )
    return result


def site_elevation(lat, lon, override, client):
    if override is not None:
        return override, "--elevation-m"
    for name, site_lat, site_lon, height in OBSERVATORIES:
        if GEOD.inv(lon, lat, site_lon, site_lat)[2] <= 1000:
            return height, f"{name} published elevation"
    response = client.get(
        "https://api.open-meteo.com/v1/elevation",
        params={"latitude": lat, "longitude": lon},
    )
    response.raise_for_status()
    (height,) = response.json()["elevation"]
    if height is None or not math.isfinite(height):
        raise ValueError("Open-Meteo returned no finite site elevation")
    return height, "Open-Meteo Elevation API (Copernicus DEM)"


def utc(text):
    at = datetime.fromisoformat(text)
    if at.tzinfo is None:
        raise ValueError("Nowcast timestamps must include a UTC offset")
    return at.astimezone(UTC)


def bucket(hours):
    if hours < 1:
        return "0-1"
    if hours < 2:
        return "1-2"
    if hours < 4:
        return "2-4"
    return "4-6" if hours <= 6 else ">6"


def verify(record, archive, radii, elevation_m=None, margins=(300, 0, 1000)):
    rows = record["site"]
    if record["status"] != "done" or not rows:
        raise ValueError("A completed nowcast with site data is required")
    params = record["params"]
    lat, lon = float(params["lat"]), float(params["lon"])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Invalid site coordinates")
    start = utc(record["data_time"])
    sat, angle, edge = view(lat, lon, max(radii))
    elevation, source = site_elevation(lat, lon, elevation_m, archive.client)
    result = dict(
        elevation_m=elevation,
        elevation_source=source,
        id=record.get("id"),
        lat=lat,
        lon=lon,
        data_time=start.isoformat(),
        satellite=sat,
        zenith_deg=angle,
        max_footprint_zenith_deg=edge,
        requested=len(rows),
        status="unreliable view" if edge > 70 else "ok",
        samples=[],
    )
    if edge > 70:
        return result
    if "blend" not in rows[0]:
        rows = blend_site(rows, params.get("blend_tau_hours", DEFAULT_BLEND_TAU_HOURS))

    @lru_cache(maxsize=32)
    def observation(scan):
        height_scan = archive.nearest(sat, scan.midpoint, ACHA)
        window = archive.window(sat, scan, lat, lon, max(radii), height_scan)
        return dict(
            height_scan=height_scan.key if height_scan else None,
            height_midpoint=height_scan.midpoint.isoformat() if height_scan else None,
            height_offset_from_acm_seconds=(
                (height_scan.midpoint - scan.midpoint).total_seconds()
                if height_scan
                else None
            ),
            observation=observe(window, radii, elevation, margins),
        )

    def match(at):
        scan = archive.nearest(sat, at)
        if scan is None:
            return dict(scan=None, height_scan=None, observation={})
        return dict(
            scan=scan.key,
            midpoint=scan.midpoint.isoformat(),
            offset_seconds=(scan.midpoint - at).total_seconds(),
            **observation(scan),
        )

    result["initial"] = match(start)
    for row in rows:
        at = utc(row["time"])
        lead = (at - start).total_seconds() / 3600
        if lead < 0:
            raise ValueError("Site time precedes data_time")
        forecasts = {s: row[s]["total"] for s in SERIES[:-1]}
        if not all(0 <= v <= 1 for v in forecasts.values()):
            raise ValueError("Forecast total must be a fraction in [0,1]")
        result["samples"].append(
            dict(
                time=at.isoformat(),
                lead_hours=lead,
                bucket=bucket(lead),
                forecast=forecasts,
                **match(at),
            )
        )
    if not any(
        obs["column_fraction"] is not None
        for sample in result["samples"]
        for obs in sample["observation"].values()
    ):
        result["status"] = "no usable observations"
    elif not any(
        obs["above_observer"][margins[0]]["fraction"] is not None
        for sample in result["samples"]
        for obs in sample["observation"].values()
    ):
        result["status"] = "column only"
    return result


def metrics(pairs):
    hits = misses = false_alarms = correct_negatives = 0
    error_sum = absolute_sum = 0.0
    for forecast, obs in pairs:
        delta = 100 * (forecast - obs)
        error_sum += delta
        absolute_sum += abs(delta)
        if obs <= 0.30:
            hits += forecast <= 0.30
            misses += forecast > 0.30
        else:
            false_alarms += forecast <= 0.30
            correct_negatives += forecast > 0.30
    n = len(pairs)
    return dict(
        n=n,
        mae_pp=absolute_sum / n if n else None,
        bias_pp=error_sum / n if n else None,
        hits=hits,
        misses=misses,
        false_alarms=false_alarms,
        correct_negatives=correct_negatives,
        hit_rate=hits / (hits + misses) if hits + misses else None,
        false_alarm_rate=(
            false_alarms / (false_alarms + correct_negatives)
            if false_alarms + correct_negatives
            else None
        ),
    )


def fraction(obs, margin):
    if margin is None:
        return obs["column_fraction"]
    return obs["above_observer"][margin]["fraction"]


def summarize(records, radii, margin=None):
    groups = defaultdict(list)
    for rec in records:
        for row in rec["samples"]:
            groups[row["bucket"]].append((row, rec["initial"]["observation"]))
    tables = []
    for r in radii:
        for lead in ("0-1", "1-2", "2-4", "4-6", ">6"):
            if lead not in groups:
                continue
            pairs = {s: [] for s in SERIES}
            bad = good = missing_scans = low_valid = 0
            missing_height_samples = missing_height_pixels = 0
            observed = []
            for row, initial in groups[lead]:
                obs = row["observation"].get(r)
                if obs is None:
                    missing_scans += 1
                    continue
                good += obs["valid"]
                bad += obs["bad_or_missing"]
                missing_heights = (
                    obs["missing_height_pixels"] if margin is not None else 0
                )
                missing_height_pixels += missing_heights
                missing_height_samples += bool(missing_heights)
                if obs["column_fraction"] is None:
                    low_valid += 1
                actual = fraction(obs, margin)
                if actual is None:
                    continue
                observed.append(actual)
                persistence = fraction(initial[r], margin) if r in initial else None
                for s, v in dict(row["forecast"], persistence=persistence).items():
                    if v is not None:
                        pairs[s].append((v, actual))
            tables.append(
                dict(
                    reference=(
                        "any cloud in the column"
                        if margin is None
                        else "above observer"
                    ),
                    margin_m=margin,
                    mean_observed_fraction=(
                        sum(observed) / len(observed) if observed else None
                    ),
                    missing_height_samples=missing_height_samples,
                    missing_height_pixels=missing_height_pixels,
                    radius_km=r,
                    lead=lead,
                    requested=len(groups[lead]),
                    missing_scans=missing_scans,
                    low_valid=low_valid,
                    valid_pixels=good,
                    bad_or_missing_pixels=bad,
                    scores={s: metrics(p) for s, p in pairs.items()},
                )
            )
    return tables


def print_table(tables):
    if not tables:
        return
    print(
        "R km  Lead h   n/p    Valid/Bad  NoHT(t/px)  Obs %     "
        + "  ".join(f"{s:>23}" for s in SERIES)
    )
    for t in tables:
        scores = t["scores"]
        counts = f"{scores['nowcast']['n']}/{scores['persistence']['n']}"
        pixels = f"{t['valid_pixels']}/{t['bad_or_missing_pixels']}"
        no_ht = (
            f"{t['missing_height_samples']}/{t['missing_height_pixels']}"
            if t["margin_m"] is not None
            else "–"
        )
        actual = t["mean_observed_fraction"]
        obs_pct = "–" if actual is None else f"{100 * actual:.2f}"
        cells = []
        for m in scores.values():
            values = (m["mae_pp"], m["bias_pp"], m["hit_rate"], m["false_alarm_rate"])
            cells.append("/".join("–" if v is None else f"{v:.2f}" for v in values))
        print(
            f"{t['radius_km']:4g}  {t['lead']:>6}  {counts:>5}  {pixels:>11}  {no_ht:>10}  {obs_pct:>5}     "
            + "  ".join(f"{c:>23}" for c in cells)
        )


def elevation_overrides(values, sites):
    """A single site's scalar or repeated LAT,LON=METRES assignments."""
    overrides = {}
    for value in values:
        location, sep, meters = value.rpartition("=")
        if sep:
            lat, lon = map(float, location.split(","))
            site = lat, lon
            if site not in sites:
                raise ValueError(
                    f"Elevation coordinates {site} do not match any input site"
                )
        elif len(sites) == 1:
            site = next(iter(sites))
        else:
            raise ValueError("Use --elevation-m=LAT,LON=METRES with multiple sites")
        height = float(meters)
        if not math.isfinite(height) or site in overrides:
            raise ValueError(f"Expected one finite elevation for {site}")
        overrides[site] = height
    return overrides


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument(
        "--radius-km",
        type=float,
        default=10,
        help="Main radius in km (also evaluates 5 and 25)",
    )
    parser.add_argument(
        "--elevation-m",
        action="append",
        default=[],
        metavar="[LAT,LON=]METRES",
        help="Site elevation override. Use coordinate assignments for multiple sites.",
    )
    parser.add_argument(
        "--margin-m",
        type=float,
        default=300,
        help="Primary height margin in metres (also evaluates 0 and 1000)",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "sat-cache")
    parser.add_argument("--json", type=Path, default=Path("satellite-summary.json"))
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Pool matched samples by site across records",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.radius_km) or args.radius_km <= 0:
        parser.error("Radius must be positive and finite")
    radii = sorted({5, args.radius_km, 25})
    if not math.isfinite(args.margin_m) or args.margin_m < 0:
        parser.error("Margin must be nonnegative and finite")
    margins = list(dict.fromkeys([args.margin_m, 0, 1000]))
    records = []
    try:
        inputs = {
            p: json.loads(p.read_text())
            for p in dict.fromkeys(p.resolve() for p in args.records)
        }
        sites = {
            p: (float(rec["params"]["lat"]), float(rec["params"]["lon"]))
            for p, rec in inputs.items()
        }
        overrides = elevation_overrides(args.elevation_m, set(sites.values()))
        with httpx.Client(timeout=30) as client:
            archive = Archive(client, args.cache_dir)
            for path, saved in inputs.items():
                rec = verify(saved, archive, radii, overrides.get(sites[path]), margins)
                records.append(dict(rec, source=str(path)))
        print(
            "Cells: MAE pp / bias pp / clear hit rate / clear false-alarm rate. – = undefined."
        )
        print(
            "n/p = matched forecasts / persistence. Valid/Bad = ACM pixels. "
            "NoHT(t/px) = samples skipped for missing heights / cloudy pixels missing HT."
        )
        groups = defaultdict(list)
        for i, rec in enumerate(records):
            print(
                f"{rec['lat']:.2f}, {rec['lon']:.2f}  {rec['data_time']}  {rec['satellite']}  "
                f"zenith {rec['zenith_deg']:.2f}° (footprint max {rec['max_footprint_zenith_deg']:.2f}°)  "
                f"{rec['elevation_m']:g} m ({rec['elevation_source']})  {rec['requested']} times  {rec['status']}"
            )
            groups[rec["lat"], rec["lon"], None if args.aggregate else i].append(rec)
        summaries = []
        for (lat, lon, _), group in groups.items():
            tables = []
            summaries.append(
                dict(
                    lat=lat,
                    lon=lon,
                    records=[rec["source"] for rec in group],
                    table=tables,
                )
            )
            if any(rec["samples"] for rec in group):
                print(
                    f"\n{lat:.2f}, {lon:.2f}  ({len(group)} record(s), first {group[0]['data_time']})"
                )
                for margin in [*margins, None]:
                    print(
                        "Any cloud in the column"
                        if margin is None
                        else f"Above observer, margin {margin:g} m"
                    )
                    rows = summarize(group, radii, margin)
                    tables.extend(rows)
                    print_table(rows)
        args.json.write_text(
            json.dumps(
                dict(
                    radii_km=radii,
                    margins_m=margins,
                    records=records,
                    summaries=summaries,
                ),
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        print(f"\nJSON: {args.json}")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        httpx.HTTPError,
        ElementTree.ParseError,
    ) as exc:
        parser.exit(2, f"Verification failed: {exc}\n")


if __name__ == "__main__":
    main()
