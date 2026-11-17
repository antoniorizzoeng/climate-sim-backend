import os
import shutil
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings


class Clock:
    def __init__(self):
        self.time = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self):
        return self.time

    def advance(self, days):
        self.time += timedelta(days=days)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def settings(tmp_path):
    return replace(
        Settings.from_env(),
        data=tmp_path / "data",
        cpu_count=4,
        engine="/engine-unavailable",
        mpirun="missing-mpirun",
        periodic=False,
    )


@pytest.fixture
def engine_settings(settings):
    engine = os.getenv(
        "ENGINE_BIN",
        str(Path.home() / "climate-sim/climate-sim-mpi-cpp/build/src/climate_sim"),
    )
    path = shutil.which(os.path.expanduser(engine))
    if not path:
        pytest.skip(
            f"Real MPI integration requires executable ENGINE_BIN (looked for {engine})"
        )
    return replace(
        settings,
        engine=str(Path(path).resolve()),
        mpirun=os.getenv("MPIRUN", "mpirun"),
        mpi_args=("--oversubscribe",),
        cpu_count=2,
    )


@pytest.fixture
def client(settings, clock):
    with TestClient(create_app(settings, clock)) as c:
        yield c


@pytest.fixture
def live(engine_settings, clock):
    with TestClient(create_app(engine_settings, clock)) as c:
        yield c


def preset(**changes):
    return dict(
        source=dict(kind="preset", preset="gaussian_hotspot", nx=7, ny=5),
        **(dict(dt=0.1, steps=5, processes=2) | changes),
    )


def poll(client, url, predicate, timeout=25):
    start = time.monotonic()
    while True:
        response = client.get(url)
        if predicate(response):
            return response
        assert time.monotonic() - start < timeout, (
            url,
            response.status_code,
            response.text[:500],
        )
        time.sleep(0.025)


def wait_run(client, identifier, state="done"):
    def ready(r):
        assert r.status_code == 200
        run = r.json()
        assert run["status"] != "error" or state == "error", run
        return run["status"] == state

    return poll(client, f"/runs/{identifier}", ready).json()


def upload_data(path, fmt="NETCDF4"):
    with netCDF4.Dataset(path, "w", format=fmt) as ds:
        for dim, size in (("time", 2), ("y", 5), ("x", 7)):
            ds.createDimension(dim, size)
            coord = ds.createVariable(dim, "i4", (dim,))
            coord[:] = np.arange(size) if dim == "time" else np.arange(size)[::-1]
        ds["time"].units = "days since 2000-01-01"
        ds.createVariable("labels", "S1", ("y", "x"))
        v = ds.createVariable("heat", "i2", ("time", "y", "x"), fill_value=-999)
        v.scale_factor, v.add_offset, v.missing_value = 0.5, 10.0, np.int16(-998)
        v.units, v.long_name = "K", "Temperature"
        v.set_auto_maskandscale(False)
        raw = np.arange(70, dtype=np.int16).reshape(2, 5, 7)
        raw[1, 0, :2] = [-999, -998]
        v[:] = raw
    return path.read_bytes()


def draw(client, plane=None, **extra):
    if plane is None:
        plane = np.arange(20, dtype="<f4").reshape(4, 5)
    return client.post(
        "/fields/drawn",
        files={"data": ("grid.bin", plane.tobytes())},
        data={"nx": plane.shape[1], "ny": plane.shape[0], **extra},
    )


def eastward_bank(params):
    """A triangular cloud bank centred one source-grid interval west of the site."""
    from app import sky
    from app.open_meteo import UNITS

    points = sky.weather_points(params)
    side = int(len(points) ** 0.5)
    shape = (len(points), params.hours + 1)
    data = {key: np.zeros(shape) for key in UNITS}
    west_distance = (params.resolution - 1) * sky.grid(params)["dx_m"] / (side - 1)
    for layer, level in sky.LEVELS.items():
        data[f"wind_speed_{level}hPa"][:] = west_distance / 3600
        data[f"wind_direction_{level}hPa"][:] = 270
        data[f"cloud_cover_{layer}"][side // 2 - 1 :: side] = 1
    data["cloud_cover"][:] = np.linspace(0.2, 0.8, shape[1])
    data["wind_speed_250hPa"][:] = np.linspace(20, 40, shape[1])
    return data
