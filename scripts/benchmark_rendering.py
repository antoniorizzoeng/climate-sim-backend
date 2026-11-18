"""Run manually with python -m scripts.benchmark_rendering."""

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import netCDF4
import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings


def seed(data, side):
    directory = data / "runs" / "benchmark"
    directory.mkdir(parents=True)
    now = datetime.now(UTC).isoformat()
    record = dict(
        id="benchmark",
        _order=1,
        status="done",
        finished_at=now,
        frames=1,
        resolved=dict(expected_frames=1, nx=side, ny=side),
        display_range=[-2.0, 2.0],
    )
    (directory / "metadata.json").write_text(json.dumps(record))
    with netCDF4.Dataset(
        directory / "result.nc", "w", format="NETCDF3_64BIT_DATA"
    ) as ds:
        for name, size in [("time", None), ("y", side), ("x", side)]:
            ds.createDimension(name, size)
        frame = ds.createVariable("u", "f8", ("time", "y", "x"))
        x = np.linspace(0, 64, side)
        for start in range(0, side, 128):
            frame[0, start : start + 128] = np.sin(
                x[start : start + 128, None]
            ) + np.cos(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    width = min(1024, args.side)
    offset = (args.side - width) // 2
    endpoints = {
        "PNG native": "/0.png",
        "PNG max_side=512": "/0.png?max_side=512",
        "Values native": "/0/values",
        "Values window / stride": (
            f"/0/values?i0={offset}&j0={offset}&w={width}&h={width}&stride=4"
        ),
    }
    with TemporaryDirectory(dir=Path.cwd(), prefix=".render-bench-") as tmp:
        data = Path(tmp)
        seed(data, args.side)
        settings = replace(Settings.from_env(), data=data, engine="/engine-unavailable")
        with TestClient(create_app(settings)) as client:
            for label, suffix in endpoints.items():
                url = "/runs/benchmark/frames" + suffix
                assert client.get(url).status_code == 200  # warm filesystem and code
                timings = []
                for _ in range(args.samples):
                    start = perf_counter()
                    response = client.get(url)
                    timings.append((perf_counter() - start) * 1000)
                    response.raise_for_status()
                print(
                    f"{label:25} {median(timings):8.2f} ms  {len(response.content):>10,} bytes"
                )


if __name__ == "__main__":
    main()
