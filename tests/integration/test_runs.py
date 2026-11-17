import io
import json
import subprocess

import netCDF4
import numpy as np
import psutil
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.manager import process_tree
from tests.conftest import draw, poll, preset, upload_data, wait_run

pytestmark = pytest.mark.integration


def test_upload_run_frames_result_and_exports(live, tmp_path):
    original = upload_data(tmp_path / "packed.nc")
    item = live.post("/fields", files={"file": ("packed.nc", original)}).json()
    source = dict(
        kind="field",
        field_id=item["id"],
        var="heat",
        index={"time": 1},
        fill_value=-2.0,
    )
    req = dict(source=source, dt=0.1, D=0.1, steps=5, out_every=2, processes=2)
    created = live.post("/runs", json=req)
    assert created.status_code == 202, created.text
    run = wait_run(live, created.json()["id"])
    assert run["resolved"]["expected_frames"] == run["frames"] == 4
    assert (
        run["step"] == 5 and run["time"] == 0.5 and run["display_range"] == [-2.0, 44.5]
    )
    url = "/runs/" + run["id"]
    initial = np.arange(35, 70, dtype="f8").reshape(5, 7)[::-1, ::-1] * 0.5 + 10.0
    initial[-1, -2:] = -2.0
    input_path = live.app.state.manager.runs.root / run["id"] / "input.nc"
    with netCDF4.Dataset(input_path) as ds:
        assert ds.file_format == "NETCDF3_64BIT_DATA" and ds["u"].dtype == "float64"
        assert np.array_equal(ds["u"][:], initial)
    png = live.get(url + "/frames/0.png")
    assert png.status_code == 200 and "immutable" in png.headers["cache-control"]
    assert Image.open(io.BytesIO(png.content)).size == (7, 5)
    values = live.get(url + "/frames/0/values")
    assert np.array_equal(np.frombuffer(values.content, "<f4").reshape(5, 7), initial)
    result = live.get(url + "/result.nc")
    assert (
        result.content.startswith(b"CDF\x05")
        and "attachment" in result.headers["content-disposition"]
    )
    result_file = tmp_path / "download.nc"
    result_file.write_bytes(result.content)
    with netCDF4.Dataset(result_file) as ds:
        assert ds["u"].dimensions == ("time", "y", "x") and ds["u"].shape == (4, 5, 7)
        assert np.array_equal(ds["u"][0], initial)
        assert (
            ds["u"][-1].astype("<f4").tobytes()
            == live.get(url + "/frames/3/values").content
        )
    assert live.get(url + "/frames/4.png").status_code == 404

    for fmt in ("mp4", "gif"):
        endpoint = url + f"/export.{fmt}?fps=3&colormap=gray"
        response = live.get(endpoint)
        assert response.status_code == 202 and int(response.headers["retry-after"]) >= 1
        assert 0 <= response.json()["progress"] <= 1
        ready = poll(live, endpoint, lambda r: r.status_code != 202)
        assert ready.status_code == 200, (
            ready.text[:200] if ready.status_code != 200 else ""
        )
        assert "attachment" in ready.headers["content-disposition"]
        assert live.get(endpoint).content == ready.content
        path = tmp_path / f"download.{fmt}"
        path.write_bytes(ready.content)
        probe = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)]
        )
        (stream,) = json.loads(probe)["streams"]
        assert int(stream["nb_frames"]) == run["frames"]
        if fmt == "mp4":
            assert (
                stream["width"],
                stream["height"],
                stream["pix_fmt"],
                stream["codec_name"],
            ) == (8, 6, "yuv420p", "h264")
        else:
            assert (stream["width"], stream["height"]) == (7, 5)


def test_live_frames_queue_and_delete_all_ranks(live):
    field = draw(live).json()
    src = {"kind": "field", "field_id": field["id"]}
    request = dict(source=src, dt=0.1, steps=100000000, out_every=1000, processes=2)
    run = live.post("/runs", json=request).json()
    url = "/runs/" + run["id"]
    running = poll(live, url, lambda r: r.json()["frames"] >= 2).json()
    assert running["status"] == "running"
    directory = live.app.state.manager.runs.root / run["id"]
    processes = process_tree(directory)
    assert sum(p.name() == "climate_sim" for p in processes) == 2
    assert live.get(url + "/frames/1.png").status_code == 200
    assert (
        live.get(url + "/frames/1/values").content
        == np.arange(20, dtype="<f4").tobytes()
    )
    assert live.get(url).json()["status"] == "running"
    assert (
        live.get(
            url + f"/frames/{run['resolved']['expected_frames'] - 1}.png"
        ).status_code
        == 409
    )
    assert (
        live.get(url + "/result.nc").status_code
        == live.get(url + "/export.gif").status_code
        == 409
    )

    queued = [live.post("/runs", json=preset()).json() for _ in range(3)]
    assert [r["queue_position"] for r in queued] == [1, 2, 3]
    assert len(live.get("/runs").json()["items"]) == 4
    assert live.delete("/fields/" + field["id"]).status_code == 409
    assert live.delete("/runs/" + queued[1]["id"]).status_code == 204
    assert live.get("/runs/" + queued[2]["id"]).json()["queue_position"] == 2
    assert live.delete(url).status_code == 204
    assert not directory.exists() and live.get(url).status_code == 404
    assert all(
        not p.is_running() or p.status() == psutil.STATUS_ZOMBIE for p in processes
    )
    assert live.delete("/fields/" + field["id"]).status_code == 204
    for item in [queued[0], queued[2]]:
        wait_run(live, item["id"])


def test_restart_recovery(engine_settings, clock):
    app = create_app(engine_settings, clock)
    with TestClient(app) as first:
        interrupted = first.post(
            "/runs", json=preset(steps=100000000, frames=2)
        ).json()["id"]
        url = "/runs/" + interrupted
        poll(first, url, lambda r: r.json()["frames"] == 1)
        queued = [first.post("/runs", json=preset()).json()["id"] for _ in range(2)]
    assert (
        json.loads(
            (engine_settings.data / "runs" / interrupted / "metadata.json").read_text()
        )["status"]
        == "running"
    )
    with TestClient(create_app(engine_settings, clock)) as second:
        run = second.get(url).json()
        assert run["status"] == "error" and run["message"] == "interrupted by restart"
        assert run["expires_at"] and run["frames"] == 1
        for identifier in queued:
            wait_run(second, identifier)
        new = second.post(url + "/rerun")
        assert new.status_code == 202 and new.json()["id"] != interrupted
        assert second.delete("/runs/" + new.json()["id"]).status_code == 204


def test_recovery_after_server_sigkill(engine_settings, tmp_path):
    """Exercise OS ownership, surviving MPI ranks and durable FIFO state."""
    import os
    import socket
    import sys
    import time

    import httpx

    settings = engine_settings
    with socket.socket() as sock, (tmp_path / "server.log").open("wb") as log:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        endpoint = f"http://127.0.0.1:{sock.getsockname()[1]}"
        env = dict(
            os.environ,
            DATA_DIR=str(settings.data),
            ENGINE_BIN=settings.engine,
            MPIRUN=settings.mpirun,
            MPIRUN_ARGS="--oversubscribe",
        )
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--fd",
                str(sock.fileno()),
            ],
            env=env,
            pass_fds=(sock.fileno(),),
            stdout=log,
            stderr=log,
        )
        try:
            with httpx.Client(base_url=endpoint, trust_env=False) as client:
                deadline = time.monotonic() + 15
                while True:
                    try:
                        if client.get("/health", timeout=0.25).is_success:
                            break
                    except httpx.TransportError:
                        pass
                    assert server.poll() is None and time.monotonic() < deadline
                long_req = preset(steps=100000000, frames=2)
                interrupted = client.post("/runs", json=long_req).json()["id"]
                url = "/runs/" + interrupted
                poll(client, url, lambda r: r.json()["frames"] == 1)
                ranks = process_tree(settings.data / "runs" / interrupted)
                assert sum(p.name() == "climate_sim" for p in ranks) == 2
                queued = [
                    client.post("/runs", json=req).json()["id"]
                    for req in [long_req, preset()]
                ]
        finally:
            server.kill()
            server.wait(timeout=10)
    with TestClient(create_app(settings)) as restarted:
        recovered = restarted.get(url).json()
        assert (
            recovered["status"] == "error"
            and recovered["message"] == "interrupted by restart"
        )
        assert all(
            not p.is_running() or p.status() == psutil.STATUS_ZOMBIE for p in ranks
        )
        wait_run(restarted, queued[0], "running")
        assert restarted.get("/runs/" + queued[1]).json()["queue_position"] == 1
        assert restarted.delete("/runs/" + queued[0]).status_code == 204
        wait_run(restarted, queued[1])


def test_export_coalesces_while_encoder_is_busy_and_recovers(live, monkeypatch):
    import threading

    from app import manager

    run = live.post("/runs", json=preset()).json()
    wait_run(live, run["id"])
    url = f"/runs/{run['id']}/export.mp4"
    entered, release = threading.Event(), threading.Event()
    calls = []

    def failed_encode(*args):
        calls.append(args[1])
        entered.set()
        assert release.wait(5)
        raise OSError("encoder unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(manager, "encode", failed_encode)
        try:
            assert live.get(url).status_code == 202 and entered.wait(3)
            assert live.get(url).json() == {"progress": 0}
            assert live.get("/health").status_code == 200
            assert len(calls) == 1
        finally:
            release.set()
        failed = poll(live, url, lambda r: r.status_code != 202)
        assert (
            failed.status_code == 500
            and "encoder unavailable" in failed.json()["detail"]
        )

    assert live.get(url).status_code == 202
    assert poll(live, url, lambda r: r.status_code != 202).status_code == 200
