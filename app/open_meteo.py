"""Open-Meteo forecast client. Keep a common UTC valid hour across all batches."""

import asyncio
import hashlib
import json
import math
import sqlite3
from contextlib import closing
from datetime import timedelta
from email.utils import parsedate_to_datetime

import httpx
import numpy as np

from .sky import LEVELS, weather_points

URL = "https://api.open-meteo.com/v1/forecast"
BATCH_SIZE = 64
UNITS = {
    **{f"cloud_cover_{layer}": "%" for layer in LEVELS},
    "cloud_cover": "%",
    **{
        f"wind_{kind}_{h}hPa": unit
        for h in LEVELS.values()
        for kind, unit in (("speed", "m/s"), ("direction", "°"))
    },
    "wind_speed_250hPa": "m/s",
}
HOUR_FORMAT = "%Y-%m-%dT%H:%M"


def parse(payload, size, times):
    points = payload if isinstance(payload, list) else [payload]
    if len(points) != size:
        raise ValueError("Open-Meteo returned the wrong number of locations")
    result = {key: [] for key in UNITS}
    for index, point in enumerate(points):
        if (
            point.get("location_id", 0) != index
            or point.get("utc_offset_seconds") != 0
            or point["hourly"]["time"] != times
        ):
            raise ValueError("Unexpected Open-Meteo location or UTC hours")
        for key, unit in UNITS.items():
            values = np.asarray(point["hourly"][key], dtype=float)
            if point["hourly_units"][key] != unit:
                raise ValueError(f"Unexpected Open-Meteo units for {key}")
            if (
                values.shape != (len(times),)
                or not np.isfinite(values).all()
                or (values < 0).any()
                or (unit == "%" and (values > 100).any())
                or (unit == "°" and (values > 360).any())
            ):
                raise ValueError(f"Incomplete or invalid Open-Meteo values for {key}")
            result[key].append(values / 100 if unit == "%" else values)
    return result


class ForecastClient:
    def __init__(self, directory, clock):
        self.db = directory / "open-meteo.sqlite3"
        self.clock = clock
        self.lock = asyncio.Lock()

    def start(self):
        with closing(sqlite3.connect(self.db)) as db, db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, stamp REAL, body BLOB);
                CREATE TABLE IF NOT EXISTS usage (stamp REAL, cost INTEGER);
                CREATE TABLE IF NOT EXISTS cooldown (until REAL);
            """
            )

    def cleanup(self):
        now = self.clock().timestamp()
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute("DELETE FROM cache WHERE stamp <= ?", (now - 3600,))
            db.execute("DELETE FROM usage WHERE stamp <= ?", (now - 31 * 86400,))
            db.execute("DELETE FROM cooldown WHERE until <= ?", (now,))

    def reserve(self, db, size, now):
        cooldown = db.execute("SELECT MAX(until) FROM cooldown").fetchone()[0]
        if cooldown and cooldown > now:
            raise ValueError(
                f"Open-Meteo rate limit. Retry in {math.ceil(cooldown-now)} seconds"
            )
        cost = math.ceil(size * len(UNITS) / 10)
        for seconds, limit in (
            (31 * 86400, 300000),
            (86400, 10000),
            (3600, 5000),
            (60, 600),
        ):
            used, oldest = db.execute(
                "SELECT SUM(cost), MIN(stamp) FROM usage WHERE stamp > ?",
                (now - seconds,),
            ).fetchone()
            if used and used + cost > limit:
                if seconds == 60:
                    return max(0.1, oldest + seconds - now)
                raise ValueError(
                    "Local Open-Meteo forecast quota exhausted. Refresh later"
                )
        db.execute("INSERT INTO usage VALUES (?, ?)", (now, cost))
        return 0

    async def batch(self, client, points, times, fresh):
        params = dict(
            latitude=",".join(f"{p[0]:.6f}" for p in points),
            longitude=",".join(f"{p[1]:.6f}" for p in points),
            hourly=",".join(UNITS),
            wind_speed_unit="ms",
            start_hour=times[0],
            end_hour=times[-1],
        )
        key = hashlib.sha256(json.dumps(params).encode()).hexdigest()
        async with self.lock:
            while True:
                now = self.clock().timestamp()
                with closing(sqlite3.connect(self.db)) as db, db:
                    cached = db.execute(
                        "SELECT body FROM cache WHERE key=? AND stamp>?",
                        (key, now - 3600),
                    ).fetchone()
                    if cached and not fresh:
                        return parse(json.loads(cached[0]), len(points), times)
                    delay = self.reserve(db, len(points), now)
                if not delay:
                    break
                await asyncio.sleep(delay)
            response = await client.get(URL, params=params)
            now = self.clock().timestamp()
            with closing(sqlite3.connect(self.db)) as db, db:
                if response.status_code == 429:
                    retry = response.headers.get("Retry-After", "")
                    try:
                        until = (
                            now + int(retry)
                            if retry.isdecimal()
                            else parsedate_to_datetime(retry).timestamp()
                        )
                    except (ValueError, TypeError):
                        until = now + 60
                    db.execute("INSERT INTO cooldown VALUES (?)", (until,))
                elif response.is_success:
                    result = parse(response.json(), len(points), times)
                    db.execute(
                        "INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
                        (key, now, response.content),
                    )
                    return result
            raise ValueError(
                f"Open-Meteo returned HTTP {response.status_code}. Refresh later"
            )

    async def fetch(self, params, start, fresh):
        times = [
            (start + timedelta(hours=i)).strftime(HOUR_FORMAT)
            for i in range(params.hours + 1)
        ]
        points = weather_points(params)
        data = {key: [] for key in UNITS}
        async with httpx.AsyncClient(
            headers={"User-Agent": "sky-nowcast/1.1"}, timeout=30
        ) as client:
            for i in range(0, len(points), BATCH_SIZE):
                batch = await self.batch(
                    client, points[i : i + BATCH_SIZE], times, fresh
                )
                for key, rows in batch.items():
                    data[key].extend(rows)
        return {key: np.array(rows) for key, rows in data.items()}
