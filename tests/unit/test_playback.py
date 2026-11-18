import sys
from dataclasses import replace

import netCDF4
import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings
from tests.conftest import poll, preset, wait_run


def test_progress_is_authoritative_and_export_survives_restart(
    settings, tmp_path, clock, monkeypatch
):
    # Stage a complete snapshot file before the launcher publishes its second
    # frame. This ordering is difficult to stop at reliably with the real engine.
    fixture = tmp_path / "snapshots.nc"
    with netCDF4.Dataset(fixture, "w", format="NETCDF3_64BIT_DATA") as ds:
        for name, size in [("time", None), ("y", 5), ("x", 7)]:
            ds.createDimension(name, size)
        ds.createVariable("u", "f8", ("time", "y", "x"))[:] = np.arange(70).reshape(
            2, 5, 7
        )

    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import json, shutil, sys, time\n"
        "from pathlib import Path\n"
        "shutil.copyfile(sys.argv[1], 'result.nc')\n"
        "def publish(n, status):\n"
        "    Path('pending').write_text(json.dumps(dict(step=n-1, time=n-1, frames=n, status=status)))\n"
        "    Path('pending').replace('progress.json')\n"
        "publish(1, 'running')\n"
        "while not Path('finish').exists():\n"
        "    time.sleep(.01)\n"
        "publish(2, 'done')\n"
    )
    settings = replace(
        settings,
        engine=sys.executable,
        mpirun=sys.executable,
        mpi_args=(str(launcher), str(fixture)),
    )
    with TestClient(create_app(settings, clock)) as c:
        run = c.post("/runs", json=preset(dt=1, steps=1)).json()
        url = "/runs/" + run["id"]
        directory = settings.data / "runs" / run["id"]
        poll(c, url, lambda r: r.json()["frames"] == 1)
        assert c.get(url + "/frames/0.png").status_code == 200
        assert c.get(url + "/frames/1/values").status_code == 409
        assert c.get(url + "/result.nc").status_code == 409
        assert c.get(url + "/export.gif").status_code == 409
        assert c.get("/runs").json()["items"][0]["frames"] == 1
        queued = c.post("/runs", json=preset()).json()
        assert c.delete("/runs/" + queued["id"]).status_code == 204
        (directory / "finish").touch()
        wait_run(c, run["id"])
        assert (
            c.get(url + "/frames/1/values").content
            == np.arange(35, 70, dtype="<f4").tobytes()
        )
        download = tmp_path / "download.nc"
        download.write_bytes(c.get(url + "/result.nc").content)
        with netCDF4.Dataset(download) as annotated, netCDF4.Dataset(
            fixture
        ) as original:
            np.testing.assert_array_equal(annotated["u"][:], original["u"][:])
            assert all(annotated[v].units == "1" for v in ("time", "y", "x", "u"))
        contents = {}
        for fmt in ("gif", "mp4"):
            ep = url + "/export." + fmt
            assert c.get(ep).status_code == 202
            ready = poll(c, ep, lambda r: r.status_code != 202)
            assert ready.status_code == 200
            contents[fmt] = ready.content

        # Shutdown leaves a durable running record even when result.nc creation
        # was interrupted. Queued work still has its own complete config.
        interrupted = c.post("/runs", json=preset()).json()["id"]
        poll(c, "/runs/" + interrupted, lambda r: r.json()["frames"] == 1)
        pending = c.post("/runs", json=preset()).json()["id"]
    (settings.data / "runs" / interrupted / "result.nc").unlink()
    with TestClient(create_app(settings, clock)) as c:
        assert (
            c.get("/runs/" + interrupted).json()["message"] == "interrupted by restart"
        )
        poll(c, "/runs/" + pending, lambda r: r.json()["frames"] == 1)
        assert c.delete("/runs/" + pending).status_code == 204
        monkeypatch.setenv("PATH", str(tmp_path))  # a cached export needs no encoder
        for fmt, cached in contents.items():
            assert c.get(url + "/export." + fmt).content == cached
        ep = url + "/export.mp4?fps=24"
        assert c.get(ep).status_code == 202
        assert poll(c, ep, lambda r: r.status_code != 202).status_code == 500
        assert not list(directory.glob("*.partial.*"))


def test_settings_parse_paths_origins_and_engine_identity(tmp_path, monkeypatch):
    binary = tmp_path / "bin" / "climate_sim"
    binary.parent.mkdir()
    binary.write_bytes(b"engine")
    binary.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", "state")
    monkeypatch.setenv("ENGINE_BIN", "./bin/climate_sim")
    monkeypatch.setenv("MPIRUN", "./bin/launcher")
    monkeypatch.setenv(
        "MPIRUN_ARGS", '--oversubscribe --prefix "/mpi installations/openmpi"'
    )
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5173, http://localhost:3000,")
    monkeypatch.setenv("ENGINE_PERIODIC", "1")
    conf = Settings.from_env()
    assert conf.data == tmp_path / "state" and conf.engine == str(binary)
    assert conf.mpirun == str(tmp_path / "bin/launcher") and conf.periodic
    assert conf.mpi_args == (
        "--oversubscribe",
        "--prefix",
        "/mpi installations/openmpi",
    )
    assert conf.origins == ("http://localhost:5173", "http://localhost:3000")
    assert "sha256" in conf.engine_description()
    (tmp_path / "climate-sim-mpi-cppConfigVersion.cmake").write_text(
        'set(PACKAGE_VERSION "0.1.0")'
    )
    assert "0.1.0" in conf.engine_description()
