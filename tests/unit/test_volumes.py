import json
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import netcdf
from app.main import create_app
from app.models import RunRequest, fits, resolve
from tests.conftest import draw, preset, upload_data


def test_vertical_stability_defaults_and_limits(client):
    req = preset(dx=2, dy=4, dz=0.5, D=0.25, vx=-2, vy=3, vz=-0.5)
    req["source"]["nz"] = 3
    for Dz in (0.25, 0, 0.125):
        limit = 1 / (1 + 0.75 + 1 + 2 * (0.25 / 4 + 0.25 / 16 + Dz / 0.25))
        if Dz != 0.25:
            req["Dz"] = Dz
        req["dt"] = limit
        r = client.post("/runs/validate", json=req).json()["resolved"]
        assert r["dt_max"] == limit
        assert r["estimated_output_bytes"] == 8192 + 8 * (
            (7 * 5 * 3 + 1) * r["expected_frames"] + 7 + 5 + 3
        )
        req["dt"] = np.nextafter(limit, np.inf)
        assert not client.post("/runs/validate", json=req).json()["ok"]
    req.update(D=0, vx=0, vy=0, vz=0, Dz=1e-200, dz=1e-180, dt=1e-162)
    assert client.post("/runs/validate", json=req).json()["resolved"][
        "dt_max"
    ] == pytest.approx(5e-161, abs=0)


def test_volume_process_grid(settings):
    assert fits(16, 2, 2, 4) and fits(7, 1, 1, 7)
    assert not fits(7, 4, 4, 4)
    settings = replace(settings, cpu_count=17)
    req = RunRequest.model_validate(
        preset()
        | {"source": dict(kind="preset", preset="constant_zero", nx=4, ny=4, nz=2)}
    )
    assert not resolve(
        req.model_copy(update={"processes": 17}), 4, 4, settings, 100000, 2
    )["ok"]
    assert (
        resolve(req.model_copy(update={"processes": None}), 4, 4, settings, 100000, 2)[
            "resolved"
        ]["processes"]
        == 1
    )


def test_legacy_field_and_run_metadata(settings, clock, tmp_path):
    path = tmp_path / "legacy.nc"
    with netcdf.dataset(path, "w") as ds:
        for name in ("time", "depth", "y", "x"):
            ds.createDimension(name, 2)
        ds.createVariable("u", "f8", ("time", "depth", "y", "x"))[:] = 3
    with TestClient(create_app(settings, clock)) as client:
        field = client.post(
            "/fields", files={"file": ("legacy.nc", path.read_bytes())}
        ).json()
        run = client.post("/runs", json=preset()).json()
    for kind, record in [("fields", field), ("runs", run)]:
        path = settings.data / kind / record["id"] / "metadata.json"
        saved = json.loads(path.read_text())
        if kind == "runs":
            for key in ("nz", "dz"):
                del saved["resolved"][key]
        else:
            for key in ("nz", "z_dim"):
                del saved["variables"][0][key]
        path.write_text(json.dumps(saved))
    with TestClient(create_app(settings, clock)) as client:
        grid = client.get("/runs/" + run["id"]).json()["resolved"]
        assert grid["nz"] == grid["dz"] == 1
        (var,) = client.get("/fields/" + field["id"]).json()["variables"]
        assert var["z_dim"] == "depth" and var["nz"] == 2
        req = dict(
            source=dict(kind="field", field_id=field["id"], volume=True), dt=1, steps=1
        )
        assert client.post("/runs/validate", json=req).json()["resolved"]["nz"] == 2


def test_vertical_validation_and_size_guard(client, tmp_path, monkeypatch):
    bad = [
        {"dz": 0},
        {"Dz": -1},
        {"bc": {"front": "open"}},
        {"bc": {"back": 0}},
        {"vz": None},
    ]
    body = preset()
    for changes in bad:
        assert client.post("/runs/validate", json=body | changes).status_code == 422
    body["source"]["nz"] = 0
    assert client.post("/runs", json=body).status_code == 422
    # A leading time dimension alone is still a plane selection with volume=true.
    field = client.post(
        "/fields", files={"file": ("time.nc", upload_data(tmp_path / "time.nc"))}
    ).json()
    source = dict(
        kind="field", field_id=field["id"], volume=True, index={"time": 1}, fill_value=0
    )
    body = dict(source=source, steps=1, dt=1)
    assert client.post("/runs/validate", json=body).json()["resolved"]["nz"] == 1
    source.update(replicate_z=0)
    assert client.post("/runs", json=body).status_code == 422
    field = draw(client).json()
    source.update(field_id=field["id"], index={}, replicate_z=12)
    disk = client.app.state.manager.disk
    size = 4096 + 8 * 20 + 4096 + 8 * 12 * 20 * 2
    monkeypatch.setattr(type(disk), "free", property(lambda _: disk.margin + size - 1))
    assert client.post("/runs/validate", json=body).json()["warnings"]
    assert client.post("/runs", json=body).status_code == 422


def test_volume_reads_are_spatial_hyperslabs(monkeypatch):
    shape = (10, 70, 80, 100)
    requests = []

    class Output:
        ndim = 4

        def __getitem__(self, key):
            assert len(key) == 4 and key[0] == 5
            assert sum(isinstance(i, int) for i in key) == 2
            requests.append(key)
            return np.ones(
                tuple(
                    len(range(*i.indices(n)))
                    for i, n in zip(key, shape)
                    if isinstance(i, slice)
                )
            )

    output = Output()
    output.shape = shape

    @contextmanager
    def dataset(_):
        yield {"u": output}

    monkeypatch.setattr(netcdf, "dataset", dataset)
    volume = netcdf.read_volume(None, 5, 64)
    assert volume.shape == (35, 40, 50) and volume.dtype == "<f4"
    assert all(key[-1].step == key[-2].step == 2 for key in requests)
    requests.clear()
    values = netcdf.read_frame(
        None, 5, (2, 1, 5, 3, 2), netcdf.FrameView("x", None, "mean")
    )
    assert len(requests) == 100 and values.shape == (2, 3)
    assert requests[-1] == (5, slice(1, 4, 2), slice(2, 7, 2), 99)
