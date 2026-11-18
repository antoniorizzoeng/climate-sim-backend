import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import memory
from app.main import create_app
from app.settings import Settings
from tests.conftest import draw, preset, wait_run

MIB = 1024**2


@pytest.mark.parametrize("nz, components", [(1, 0), (64, 0), (1, 2), (64, 3)])
def test_engine_estimate(nz, components):
    assert memory.estimate_engine_memory(1024, 1024, nz, 12, components) == (
        (28 + 8 * components) * MIB * nz + (64 + 16 * 12) * MIB
    )


def test_engine_capacity_from_total_cgroup_and_env(monkeypatch, tmp_path):
    root = tmp_path / "cgroup"
    parent = root / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    membership = tmp_path / "self.cgroup"
    membership.write_text("0::/parent/child\n")
    (root / "memory.max").write_text("max")
    (parent / "memory.max").write_text(str(3_000 * MIB))
    (child / "memory.max").write_text(str(4_000 * MIB))
    for directory in (parent, child):
        (directory / "memory.current").write_text(str(2_999 * MIB))

    monkeypatch.setattr(memory, "CGROUP_ROOT", root)
    monkeypatch.setattr(memory, "SELF_CGROUP", membership)
    monkeypatch.setattr(
        memory.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=5_000 * MIB, available=100 * MIB),
    )
    monkeypatch.delenv("ENGINE_MEMORY_BYTES", raising=False)
    monkeypatch.delenv("ENGINE_MEMORY_RESERVE_BYTES", raising=False)
    assert Settings.from_env().engine_memory_capacity() == (3_000 - 512) * MIB

    (parent / "memory.max").write_text(str(7_000 * MIB))
    (child / "memory.max").unlink()
    monkeypatch.setenv("ENGINE_MEMORY_RESERVE_BYTES", str(256 * MIB))
    assert Settings.from_env().engine_memory_capacity() == (5_000 - 256) * MIB
    membership.unlink()
    assert Settings.from_env().engine_memory_capacity() == (5_000 - 256) * MIB
    monkeypatch.setenv("ENGINE_MEMORY_RESERVE_BYTES", str(6_000 * MIB))
    assert Settings.from_env().engine_memory_capacity() == 0
    monkeypatch.setenv("ENGINE_MEMORY_BYTES", "6800000000")
    assert Settings.from_env().engine_memory_capacity() == 6_800_000_000


@pytest.mark.parametrize(
    "variable", ["ENGINE_MEMORY_BYTES", "ENGINE_MEMORY_RESERVE_BYTES"]
)
def test_engine_capacity_rejects_negative_settings(monkeypatch, variable):
    monkeypatch.setenv(variable, "-1")
    with pytest.raises(ValueError, match=variable):
        Settings.from_env()


def test_extruded_field_memory_validation_and_submission(settings):
    settings = replace(settings, cpu_count=12, engine_memory_bytes=6_800_000_000)
    with TestClient(create_app(settings)) as client:
        field = draw(client, np.zeros((1024, 1024), dtype="<f4"))
        assert field.status_code == 201
        body = dict(
            source=dict(kind="field", field_id=field.json()["id"], replicate_z=1024),
            steps=1,
            dt=0.1,
            processes=12,
        )
        error = dict(
            loc=["body", "source"],
            msg=(
                "Needs about 30.3 GB of memory; this server can give the engine "
                "6.8 GB. Use fewer layers or a smaller grid."
            ),
            type="value_error",
        )
        validation = client.post("/runs/validate", json=body)
        assert validation.status_code == 200
        report = validation.json()
        assert report["ok"] is False and report["errors"] == [error]
        resolved = report["resolved"]
        assert resolved["nz"] == 1024 and resolved["processes"] == 12
        assert resolved["estimated_memory_bytes"] == 30_333_206_528
        assert resolved["memory_capacity_bytes"] == 6_800_000_000
        rejected = client.post("/runs", json=body)
        assert rejected.status_code == 422 and rejected.json() == {"detail": [error]}
        assert client.get("/runs").json()["items"] == []

        body["source"]["replicate_z"] = 64
        report = client.post("/runs/validate", json=body).json()
        assert report["ok"] and report["resolved"]["estimated_memory_bytes"] == 2**31
        accepted = client.post("/runs", json=body)
        assert accepted.status_code == 202
        assert accepted.json()["resolved"] == report["resolved"]


@pytest.mark.parametrize("stored_processes", [True, False])
def test_old_run_record_memory_fields_are_backfilled(settings, stored_processes):
    with TestClient(create_app(settings)) as client:
        identifier = client.post("/runs", json=preset()).json()["id"]
        wait_run(client, identifier, "error")
    metadata = settings.data / "runs" / identifier / "metadata.json"
    record = json.loads(metadata.read_text())
    for key in ("nz", "dz", "estimated_memory_bytes", "memory_capacity_bytes"):
        del record["resolved"][key]
    if not stored_processes:
        del record["resolved"]["processes"]
    metadata.write_text(json.dumps(record))

    with TestClient(
        create_app(replace(settings, engine_memory_bytes=1_000_000_000))
    ) as client:
        run = client.get(f"/runs/{identifier}").json()
        assert run["status"] == "error"
        assert run["resolved"]["nz"] == 1
        assert run["resolved"]["processes"] == 2
        assert run["resolved"]["estimated_memory_bytes"] == 7 * 5 * 28 + 96 * MIB
        assert run["resolved"]["memory_capacity_bytes"] == 1_000_000_000


def test_engine_nice_setting(monkeypatch):
    from app.settings import Settings

    monkeypatch.setenv("ENGINE_NICE", "7")
    assert Settings.from_env().engine_nice == 7
    monkeypatch.setenv("ENGINE_NICE", "25")
    try:
        Settings.from_env()
    except ValueError as exc:
        assert "ENGINE_NICE" in str(exc)
    else:
        raise AssertionError("ENGINE_NICE=25 accepted")
