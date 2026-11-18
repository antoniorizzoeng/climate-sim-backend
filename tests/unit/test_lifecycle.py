from datetime import datetime

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from app.main import create_app
from tests.conftest import draw, preset, wait_run


def clean(client):
    m = client.app.state.manager

    async def cleanup():
        async with m.lock:
            await m.cleanup()

    client.portal.call(cleanup)


def test_contract(client):
    import yaml
    from openapi_spec_validator import validate

    from app.main import CONTRACT

    spec = yaml.safe_load(CONTRACT.read_text())
    validate(spec)
    assert (
        spec["info"]["version"]
        == client.get("/health").json()["api_version"]
        == "1.4.0"
    )
    expected = {
        (p, method.upper()): set(map(int, op["responses"]))
        for p, path in spec["paths"].items()
        for method, op in path.items()
        if method != "parameters"
    }
    registered = {
        (route.path, method): {route.status_code or 200, *map(int, route.responses)}
        for route in client.app.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    }
    assert registered == expected and spec == client.get("/openapi.json").json()
    for path in ("frames/{n}.png", "frames/{n}/values", "export.{format}"):
        params = spec["paths"]["/runs/{runId}/" + path]["get"]["parameters"]
        refs = {p.get("$ref") for p in params}
        for name in ("Axis", "Slice", "Projection"):
            assert "#/components/parameters/" + name in refs
    request = preset(dz=2, Dz=0.5, vz=1, bc={"front": "neumann", "back": "neumann"})
    request["source"]["nz"] = 4
    created = client.post("/runs", json=request)
    field = draw(client).json()
    rename = {"name": "Baseline"}
    for schema, data in [
        ("RenameRequest", rename),
        ("Units", created.json()["request"]["units"]),
        ("StorageUsage", client.get("/storage").json()),
        ("ServerConfig", client.get("/config").json()),
        ("Run", created.json()),
        ("Nowcast", client.post("/nowcasts", json={"lat": 45, "lon": 10}).json()),
        ("Run", wait_run(client, created.json()["id"], "error")),
        ("Field", field),
        ("Field", client.patch("/fields/" + field["id"], json=rename).json()),
        ("Run", client.patch("/runs/" + created.json()["id"], json=rename).json()),
        ("RunValidation", client.post("/runs/validate", json=request).json()),
        ("ValidationError", client.post("/runs", json={}).json()),
        ("MessageError", client.get("/fields/missing").json()),
    ]:
        Draft202012Validator(
            {"components": spec["components"], "$ref": f"#/components/schemas/{schema}"}
        ).validate(data)
    for path in ("/runs", "/fields", "/health"):
        schema = spec["paths"][path]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        Draft202012Validator({"components": spec["components"], **schema}).validate(
            client.get(path).json()
        )
    assert client.get("/health").json()["status"] == "ok"
    assert (
        "periodic" not in client.get("/config").json()["supported_boundary_conditions"]
    )


def test_retention_boundaries_and_reactivation(settings, clock):
    with TestClient(create_app(settings, clock)) as c:
        field = draw(c).json()
        f = "/fields/" + field["id"]
        req = {
            "source": {"kind": "field", "field_id": field["id"]},
            "dt": 1.0,
            "steps": 3,
        }
        created = c.post("/runs", json=req).json()
        wait_run(c, created["id"], "error")
        url = "/runs/" + created["id"]
        run_dir = settings.data / "runs" / created["id"]
        for name in ("result.nc", "export-example.mp4"):
            (run_dir / name).write_bytes(b"bytes")
        clock.advance(6)
        clean(c)
        assert (
            c.get(f).json()["state"] == "active"
            and c.get(url).json()["status"] == "error"
        )
        clock.advance(1)
    with TestClient(create_app(settings, clock)) as c:
        assert c.get("/fields").json()["items"][0]["state"] == "archived"
        assert c.get(f + "/original").status_code == 200
        for path in ["/preview.png", "/stats", "/values"]:
            assert c.get(f + path).status_code == 409
        expired = c.get(url).json()
        assert expired["status"] == "expired" and expired["request"] == req | {
            "units": {"length": "cell", "time": "t"}
        }
        assert expired["expires_at"] == c.get(f).json()["expires_at"]
        assert {p.name for p in run_dir.iterdir()} == {"metadata.json"}
        for path in ["/result.nc", "/export.mp4", "/frames/0.png"]:
            assert c.get(url + path).status_code == 404
        new = c.post(url + "/rerun")
        assert new.status_code == 202 and c.get(f).json()["state"] == "active"
        assert datetime.fromisoformat(c.get(f).json()["last_used_at"]) == clock()
        wait_run(c, new.json()["id"], "error")
        clock.advance(30)
        clean(c)
        assert c.get(url).status_code == 404
        assert c.get(f).json()["state"] == "archived"
        clock.advance(7)
        clean(c)
        assert c.get(f).status_code == 404 and c.get("/runs").json()["items"] == []


def test_rerun_source_deleted(client):
    item = draw(client).json()
    body = {"source": {"kind": "field", "field_id": item["id"]}, "steps": 1, "dt": 0.2}
    r = client.post("/runs", json=body).json()
    wait_run(client, r["id"], "error")
    assert client.delete("/fields/" + item["id"]).status_code == 204
    assert client.post("/runs/" + r["id"] + "/rerun").status_code == 404


def test_owner_lock_and_staging_recovery(settings):
    directory = settings.data / "runs" / ".new-unfinished"
    directory.mkdir(parents=True)
    with TestClient(create_app(settings)):
        assert not directory.exists()
        with pytest.raises(BlockingIOError), TestClient(create_app(settings)):
            pass


def test_launcher_failure_includes_diagnostic(settings, tmp_path):
    import sys
    from dataclasses import replace

    script = tmp_path / "launcher.py"
    script.write_text(
        "import sys\nsys.stderr.write('launcher diagnostic')\nsys.exit(17)"
    )
    with TestClient(
        create_app(replace(settings, mpirun=sys.executable, mpi_args=(str(script),)))
    ) as c:
        run = c.post("/runs", json=preset()).json()
        error = wait_run(c, run["id"], "error")
        assert "17" in error["message"] and "launcher diagnostic" in error["message"]


def test_rerun_expired_preset(settings, clock):
    with TestClient(create_app(settings, clock)) as c:
        old = c.post("/runs", json=preset()).json()
        wait_run(c, old["id"], "error")
        clock.advance(36)
        clean(c)
        new = c.post(f"/runs/{old['id']}/rerun")
        assert new.status_code == 202 and new.json()["request"] == old["request"]


@pytest.mark.parametrize("path", ["/fields", "/runs"])
def test_list_contract_declares_pagination_errors(client, path):
    result = client.get(path, params={"limit": 0})
    assert result.status_code == 422
    responses = client.get("/openapi.json").json()["paths"][path]["get"]["responses"]
    assert responses["422"] == {"$ref": "#/components/responses/ValidationError"}


def test_preexisting_upload_stats_repaired_on_startup(settings, clock):
    import json

    from app import netcdf

    directory = settings.data / "fields" / "legacy"
    directory.mkdir(parents=True)
    path = directory / "original.nc"
    with netcdf.dataset(path, "w") as ds:
        ds.createDimension("y", 2)
        ds.createDimension("x", 2)
        ds.createVariable("u", "f8", ("y", "x"))[0] = 1
    meta = netcdf.inspect_field(path)
    meta["default_stats"].update(missing_count=0, max=9.969209968386869e36)
    record = dict(
        **meta,
        id="legacy",
        source="upload",
        last_used_at=clock().isoformat(),
        state="active",
    )
    metadata = directory / "metadata.json"
    metadata.write_text(json.dumps(record))
    with TestClient(create_app(settings, clock)) as client:
        stats = client.get("/fields/legacy").json()["default_stats"]
        assert stats["missing_count"] == 2 and stats["max"] == 1
    assert json.loads(metadata.read_text())["default_stats"] == stats


def test_health_reports_stopped_worker(settings, clock, monkeypatch):
    from app.manager import RunManager

    async def stopped(manager):
        return

    monkeypatch.setattr(RunManager, "work", stopped)
    with TestClient(create_app(settings, clock)) as client:
        assert client.get("/health").status_code == 503
