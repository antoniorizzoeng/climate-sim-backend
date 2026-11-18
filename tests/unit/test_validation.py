from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import nowcasts
from app.main import create_app
from app.models import RunRequest, fits, resolve
from app.settings import Settings
from tests.conftest import preset


def test_combined_stability(client):
    body = preset(D=0.25, vx=-2.0, vy=3.0, dx=2.0, dy=4.0)
    limit = 1 / (2 / 2 + 3 / 4 + 0.5 * (0.25 + 0.0625))
    body["dt"] = limit
    assert (
        client.post("/runs/validate", json=body).json()["resolved"]["dt_max"] == limit
    )
    body["dt"] += 1e-9
    report = client.post("/runs/validate", json=body).json()
    assert not report["ok"] and report["errors"][0]["loc"] == ["body", "dt"]
    assert client.post("/runs", json=body).status_code == 422
    assert client.get("/runs").json()["items"] == []


@pytest.mark.parametrize(
    ("options", "every", "count"),
    [
        ({}, 3, 69),
        ({"frames": 2}, 202, 2),
        ({"frames": 500}, 1, 203),
        ({"out_every": 20}, 20, 12),
        ({"out_every": 202}, 202, 2),
        ({"out_every": 300}, 300, 2),
    ],
)
def test_timeline_and_no_limit(client, options, every, count):
    report = client.post("/runs/validate", json=preset(steps=202, **options)).json()
    assert report["ok"] and report["warnings"] == []
    resolved = report["resolved"]
    assert (resolved["out_every"], resolved["expected_frames"], resolved["dt_max"]) == (
        every,
        count,
        None,
    )
    assert resolved["estimated_output_bytes"] >= 8 * 7 * 5 * count


@pytest.mark.parametrize(
    "changes",
    [
        {"D": -1},
        {"steps": 0},
        {"steps": 2147483648},
        {"frames": 1},
        {"dx": 0},
        {"dt": "0.1"},
        {"other": 0},
        {"frames": None},
        {"bc": {"east": "neumann"}},
        {"source": {"kind": "unknown"}},
        {"source": {"kind": "field", "field_id": "a", "index": {"time": -1}}},
        {"source": {"kind": "field", "field_id": "a", "typo": True}},
    ],
)
def test_malformed_requests(client, changes):
    r = client.post("/runs/validate", json=preset() | changes)
    assert r.status_code == 422
    assert all(set(e) == {"loc", "msg", "type"} for e in r.json()["detail"])


@pytest.mark.parametrize(
    "changes",
    [
        {"processes": 5},
        {"frames": 3, "out_every": 2},
        {"bc": {"left": "periodic", "right": "periodic"}},
        {"dt": 1e308},
        {"source": {"kind": "field", "field_id": "absent"}},
    ],
)
def test_semantic_errors(client, changes):
    body = preset() | changes
    r = client.post("/runs/validate", json=body)
    assert r.status_code == 200
    assert r.json()["errors"] and r.json()["resolved"] is None
    assert client.post("/runs", json=body).status_code == (
        404 if "source" in changes else 422
    )


def test_periodic_enabled(settings, monkeypatch):
    frame = np.zeros((5, 5))
    frame[-1, 0] = frame[0, -1] = 0.25

    def probe_frame(settings, plane, bc, **physics):
        if physics.get("wind") is not None:
            raise OSError("file wind unsupported")
        assert set(bc.values()) == {"periodic"}
        assert physics == dict(vx=0.25, vy=0.25)
        return frame

    monkeypatch.setattr(nowcasts, "probe_frame", probe_frame)
    with TestClient(create_app(replace(settings, periodic=True))) as client:
        assert (
            "periodic" in client.get("/config").json()["supported_boundary_conditions"]
        )
        for axis in [("left", "right"), ("top", "bottom"), ("front", "back")]:
            bc = {edge: "periodic" for edge in axis}
            req = preset(bc=bc)
            req["source"]["nz"] = 3 if axis[0] == "front" else 1
            assert client.post("/runs/validate", json=req).json()["ok"]
            del bc[axis[1]]
            assert not client.post("/runs/validate", json=req).json()["ok"]


@pytest.mark.parametrize("silently_ignored", [True, False])
def test_periodic_default_falls_back_for_old_engine(
    settings, monkeypatch, silently_ignored
):
    monkeypatch.delenv("ENGINE_PERIODIC", raising=False)
    assert Settings.from_env().periodic

    def probe_frame(settings, plane, bc, **physics):
        if silently_ignored:
            return plane
        raise OSError("periodic unsupported")

    monkeypatch.setattr(nowcasts, "probe_frame", probe_frame)
    with TestClient(create_app(replace(settings, periodic=True))) as client:
        assert client.get("/config").json()["supported_boundary_conditions"] == [
            "dirichlet",
            "neumann",
        ]
        req = preset(bc=dict(left="periodic", right="periodic"))
        assert not client.post("/runs/validate", json=req).json()["ok"]


def test_mpi_dims_and_disk_warning(settings):
    assert fits(72, 12, 6) and not fits(72, 9, 8)
    assert fits(12, 4, 3) and not fits(12, 3, 4)
    req = RunRequest.model_validate(
        {k: v for k, v in preset().items() if k != "processes"}
    )
    report = resolve(req, 4, 20, replace(settings, cpu_count=19), 0)
    assert report["warnings"] and report["resolved"]["processes"] == 16
    assert not resolve(
        req.model_copy(update={"processes": 7}),
        4,
        20,
        replace(settings, cpu_count=8),
        1000,
    )["ok"]


def test_frontend_clamp_and_extreme_scales(client):
    # Reordering the diffusion term shifts this boundary by a float64 ULP.
    dx, dy, D, vx, vy = 0.1, 0.3, 0.13, 0.4, -0.5
    dt = 1 / (abs(vx) / dx + abs(vy) / dy + 2 * D * (1 / (dx * dx) + 1 / (dy * dy)))
    report = client.post(
        "/runs/validate", json=preset(dx=dx, dy=dy, D=D, vx=vx, vy=vy, dt=dt)
    ).json()
    assert report["ok"] and report["resolved"]["dt_max"] == dt
    extreme = client.post(
        "/runs/validate", json=preset(dx=1e-180, D=1e-200, dt=1e-170)
    ).json()
    assert extreme["ok"] and extreme["resolved"]["dt_max"] == pytest.approx(
        5e-161, abs=0
    )
    wide = client.post("/runs/validate", json=preset(out_every=2**40)).json()
    assert wide["warnings"] and wide["resolved"]["expected_frames"] == 2
