import io
import json
import re
import shutil
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
        stream = probe_video(path)
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
        sock.bind(("127.0.0.1", int(os.environ.get("TEST_SERVER_PORT", 0))))
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


def probe_video(path):
    """Stream facts via ffprobe, or via plain ffmpeg when ffprobe is absent (CI)."""
    if shutil.which("ffprobe"):
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)]
        )
        (stream,) = json.loads(out)["streams"]
        return stream
    info = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path)], capture_output=True, text=True
    ).stderr
    match = re.search(r"Video: (\w+).*?, (\w+)(?:\([^)]*\))?, (\d+)x(\d+)", info)
    codec, pix_fmt, width, height = match.groups()
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ]
    )
    return {
        "codec_name": codec,
        "pix_fmt": pix_fmt,
        "width": int(width),
        "height": int(height),
        "nb_frames": len(raw) // (int(width) * int(height)),
    }


def test_implicit_fill_is_missing_in_stats_validation_and_frame(live, tmp_path):
    path = tmp_path / "unwritten.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("y", 4)
        ds.createDimension("x", 4)
        ds.createVariable("u", "f8", ("y", "x"))[:3] = 1
    item = live.post("/fields", files={"file": (path.name, path.read_bytes())}).json()
    assert item["default_stats"]["missing_count"] == 4
    source = dict(kind="field", field_id=item["id"])
    body = dict(source=source, steps=1, dt=1, processes=1)
    assert not live.post("/runs/validate", json=body).json()["ok"]
    source["fill_value"] = 7
    run = wait_run(live, live.post("/runs", json=body).json()["id"])
    values = live.get(f"/runs/{run['id']}/frames/0/values")
    np.testing.assert_array_equal(
        np.frombuffer(values.content, "<f4").reshape(4, 4), [[1] * 4] * 3 + [[7] * 4]
    )


@pytest.mark.parametrize("transition", ["running", "done"])
def test_worker_retries_metadata_write_in_fifo_order(live, monkeypatch, transition):
    import errno
    from threading import Event

    from app import storage

    failed, restored = Event(), Event()
    save = storage.save
    starts = []
    finishes = []

    def flaky_save(directory, record):
        status = record.get("status")
        if status == transition and not restored.is_set():
            failed.set()
            raise OSError(errno.ENOSPC, "injected metadata failure")
        save(directory, record)
        if status == "running":
            starts.append(record["id"])
        if status == "done":
            finishes.append(record["id"])

    monkeypatch.setattr(storage, "save", flaky_save)
    first = live.post("/runs", json=preset()).json()["id"]
    try:
        assert failed.wait(5)
        assert live.get("/health").status_code == 503
        second = live.post("/runs", json=preset()).json()["id"]
        state = live.get("/runs/" + first).json()
        if transition == "running":
            assert (state["status"], state["queue_position"]) == ("queued", 1)
        else:
            assert state["status"] == "running"
        assert live.get("/runs/" + second).json()["status"] == "queued"
    finally:
        restored.set()
    for identifier in (first, second):
        wait_run(live, identifier)
    assert starts == finishes == [first, second]
    assert live.get("/health").json()["status"] == "ok"


def test_cf_download_sizes_and_retention(live, clock, tmp_path, monkeypatch):
    from app import netcdf
    from tests.unit.test_catalog import disk_bytes
    from tests.unit.test_lifecycle import clean

    m = live.app.state.manager
    annotate = netcdf.annotate_result
    layout = {}

    def inspect_engine_output(path, run, units):
        with netcdf.dataset(path) as ds:
            assert ds.dimensions["time"].isunlimited()
            frame_bytes = ds["u"].size * 8
        with path.open("rb") as stream:
            layout["offset"] = stream.seek(-frame_bytes, 2)
            layout["payload"] = stream.read()
        annotate(path, run, units)

    monkeypatch.setattr(netcdf, "annotate_result", inspect_engine_output)
    path = tmp_path / "temp.nc"
    with netCDF4.Dataset(path, "w") as ds:
        for axis in ("z", "y", "x"):
            ds.createDimension(axis, 4)
        u = ds.createVariable("temperature", "f8", ("z", "y", "x"))
        u.units = "K"
        u[:] = np.arange(64).reshape(4, 4, 4)
    field = live.post("/fields", files={"file": ("temp.nc", path.read_bytes())}).json()
    assert field["size_bytes"] == disk_bytes(m.fields.root)
    body = dict(
        source=dict(kind="field", field_id=field["id"], volume=True),
        dt=0.2,
        steps=5,
        out_every=2,
        units={"length": "km", "time": "h"},
        dx=0.25,
        dy=0.75,
        dz=1.5,
        name="  All levels  ",
    )
    run = wait_run(live, live.post("/runs", json=body).json()["id"])
    url = "/runs/" + run["id"]
    assert run["name"] == run["request"]["name"] == "All levels"
    assert run["request"]["units"] == body["units"]
    assert run["size_bytes"] == disk_bytes(m.runs.root)
    path.write_bytes(live.get(url + "/result.nc").content)
    with path.open("rb") as stream:
        stream.seek(layout["offset"])
        assert stream.read(len(layout["payload"])) == layout["payload"]
    with netCDF4.Dataset(path) as ds:
        assert not ds.dimensions["time"].isunlimited()
        assert [ds[k].units for k in ("time", "z", "y", "x", "u")] == [
            "h",
            "km",
            "km",
            "km",
            "K",
        ]
        np.testing.assert_allclose(ds["time"][:], [0, 0.4, 0.8, 1])
        for axis in ("z", "y", "x"):
            np.testing.assert_allclose(ds[axis][:], np.arange(4) * body["d" + axis])
        np.testing.assert_array_equal(ds["u"][0], np.arange(64).reshape(4, 4, 4))
    ep = url + "/export.gif"
    poll(live, ep, lambda r: r.status_code == 200)
    assert (
        live.get(url).json()["size_bytes"]
        == disk_bytes(m.runs.root)
        > run["size_bytes"]
    )
    assert live.get("/runs").json()["total_bytes"] == disk_bytes(m.runs.root)
    storage = live.get("/storage").json()
    assert storage["used_bytes"] == disk_bytes(m.runs.root) + disk_bytes(m.fields.root)
    assert 0 < storage["free_bytes"] <= storage["total_bytes"]
    clock.advance(7)
    clean(live)
    assert live.get(url).json()["size_bytes"] == 0
    storage = live.get("/storage").json()
    assert storage["runs_bytes"] == disk_bytes(m.runs.root)
    assert storage["fields_bytes"] == disk_bytes(m.fields.root)
    assert live.delete(url).status_code == 204
    assert live.get("/storage").json()["runs_bytes"] == 0
