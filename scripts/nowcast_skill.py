"""Compare a saved sky nowcast with Open-Meteo past model estimates."""

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import httpx

from app.models import DEFAULT_BLEND_TAU_HOURS
from app.open_meteo import URL
from app.sky import LEVELS, blend_site

HOUR = 3600
CLOUD_FIELDS = {**{key: f"cloud_cover_{key}" for key in LEVELS}, "total": "cloud_cover"}
METRICS = (*CLOUD_FIELDS, "clear")
SERIES = ("nowcast", "model", "blend")


def timestamp(iso_time):
    return datetime.fromisoformat(iso_time).timestamp()


def past_range(rows, now):
    start = math.floor(timestamp(rows[0]["time"]) / HOUR) * HOUR
    end = math.ceil(timestamp(rows[-1]["time"]) / HOUR) * HOUR
    current_hour = math.floor(now.timestamp() / HOUR) * HOUR
    if end >= current_hour:
        raise ValueError(
            "Wait until all valid times and their reference hours are past"
        )
    hours = (current_hour - start) // HOUR
    if hours <= 24:
        return dict(past_hours=hours, forecast_hours=0)
    days = (now.date() - datetime.fromtimestamp(start, UTC).date()).days
    if days > 92:
        raise ValueError("The Forecast API supports up to 92 past days")
    return dict(
        past_days=days,
        forecast_days=int(datetime.fromtimestamp(end, UTC).date() == now.date()),
    )


def reference_hours(payload):
    units = payload["hourly_units"]
    if units["time"] != "unixtime" or any(
        units[key] != "%" for key in CLOUD_FIELDS.values()
    ):
        raise ValueError("Unexpected Open-Meteo units")
    hourly = payload["hourly"]
    reference = {}
    for hour, *cloud in zip(
        hourly["time"], *(hourly[key] for key in CLOUD_FIELDS.values()), strict=True
    ):
        if None in cloud:
            continue
        if any(not 0 <= value <= 100 for value in cloud):
            raise ValueError("Invalid Open-Meteo cloud percentage")
        reference[hour] = cloud
    return reference


def evaluate(rows, reference):
    errors = {series: dict.fromkeys(METRICS, 0.0) for series in SERIES}
    matched = 0
    for row in rows:
        at = timestamp(row["time"])
        hour, offset = divmod(at, HOUR)
        before = reference.get(hour * HOUR)
        after = reference.get(hour * HOUR + (HOUR if offset else 0))
        if before is None or after is None:
            continue
        weight = offset / HOUR
        values = [a + weight * (b - a) for a, b in zip(before, after)]
        values.append(100 - values[-1])
        for series in SERIES:
            for key, value in zip(METRICS, values):
                errors[series][key] += abs(100 * row[series][key] - value)
        matched += 1
    if not matched:
        raise ValueError("No matching valid times with complete Open-Meteo cloud data")
    return matched, {
        series: {key: error / matched for key, error in metrics.items()}
        for series, metrics in errors.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "record", type=Path, help="nowcast metadata.json or saved API response"
    )
    args = parser.parse_args()
    try:
        record = json.loads(args.record.read_text())
        rows = record["site"]
        if record["status"] != "done" or not rows:
            raise ValueError("A completed nowcast with site data is required")
        params = record["params"]
        if "blend" not in rows[0]:
            rows = blend_site(
                rows, params.get("blend_tau_hours", DEFAULT_BLEND_TAU_HOURS)
            )
        response = httpx.get(
            URL,
            params=dict(
                latitude=params["lat"],
                longitude=params["lon"],
                hourly=",".join(CLOUD_FIELDS.values()),
                timezone="GMT",
                timeformat="unixtime",
                **past_range(rows, datetime.now(UTC)),
            ),
            timeout=30,
        )
        response.raise_for_status()
        matched, errors = evaluate(rows, reference_hours(response.json()))
    except (OSError, ValueError, TypeError, KeyError, httpx.HTTPError) as exc:
        parser.error(str(exc))

    print(
        f"Matched {matched}/{len(rows)} site times against Open-Meteo past model estimates."
    )
    print("Mean absolute error (percentage points)")
    print(f"{'Series':8}" + "".join(f"{key:>8}" for key in METRICS))
    for series, metrics in errors.items():
        print(f"{series:8}" + "".join(f"{error:8.2f}" for error in metrics.values()))


if __name__ == "__main__":
    main()
