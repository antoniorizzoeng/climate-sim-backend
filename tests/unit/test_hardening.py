import asyncio
import io
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import netCDF4
import numpy as np
import psutil
import pytest
from fastapi.testclient import TestClient
from matplotlib import colormaps
from PIL import Image

from app import events, netcdf, render
from app.disk import CHUNK_BYTES, DiskSpaceError
from app.main import create_app
from app.routes import Download
from tests.conftest import poll, preset, upload_data, wait_run
from tests.unit.test_lifecycle import clean


@pytest.fixture
def saved_run(settings, clock):
    identifier = "finished"
    directory = settings.data / "runs" / identifier
    directory.mkdir(parents=True)
    record = dict(
        id=identifier,
        created_at=clock().isoformat(),
        request=preset(),
        _order=1,
        status="done",
        finished_at=clock().isoformat(),
        frames=2,
        resolved=dict(expected_frames=2, nx=256, ny=256),
        display_range=[0, 1],
    )
    (directory / "metadata.json").write_text(json.dumps(record))
    with netcdf.dataset(
        directory / "result.nc", "w", format="NETCDF3_64BIT_DATA"
    ) as ds:
        for key, n in [("t", 2), ("y", 256), ("x", 256)]:
            ds.createDimension(key, n)
        var = ds.createVariable("u", "f8", ("t", "y", "x"))
        var[:] = np.linspace(0, 1, 256)
    return identifier, directory


def test_many_gets_during_partial_frame_write(settings, clock, tmp_path, monkeypatch):
    launcher = tmp_path / "writer.py"
    launcher.write_text(
        "import json, time\n"
        "from pathlib import Path\n"
        "import netCDF4, numpy as np\n"
        "def publish(n, status):\n"
        "    Path('pending').write_text(json.dumps(dict(step=n-1, time=n-1, frames=n, status=status)))\n"
        "    Path('pending').replace('progress.json')\n"
        "with netCDF4.Dataset('result.nc', 'w', format='NETCDF3_64BIT_DATA') as ds:\n"
        "    for key, n in [('t', None), ('y', 5), ('x', 7)]: ds.createDimension(key, n)\n"
        "    var = ds.createVariable('u', 'f8', ('t', 'y', 'x'))\n"
        "    var[0] = np.arange(35).reshape(5, 7)\n"
        "    ds.sync()\n"
        "    publish(1, 'running')\n"
        "    var[1, :2] = np.arange(35, 49).reshape(2, 7)\n"
        "    ds.sync()\n"
        "    Path('writing').touch()\n"
        "    while not Path('finish').exists(): time.sleep(.01)\n"
        "    var[1, 2:] = np.arange(49, 70).reshape(3, 7)\n"
        "    ds.sync()\n"
        "    publish(2, 'running')\n"
        "publish(2, 'done')\n"
    )
    settings = replace(settings, mpirun=sys.executable, mpi_args=(str(launcher),))
    with TestClient(create_app(settings, clock)) as c:
        run = c.post("/runs", json=preset(steps=1)).json()["id"]
        url = f"/runs/{run}"
        directory = settings.data / "runs" / run
        poll(
            c,
            url,
            lambda r: (directory / "writing").exists() and r.json()["frames"] == 1,
        )
        queued = c.post("/runs", json=preset(steps=1)).json()["id"]
        health = c.get("/health")
        assert health.headers["x-active-run-id"] == run
        assert health.headers["x-queue-length"] == "1"
        assert int(health.headers["x-disk-free-bytes"]) > 0
        assert c.delete(f"/runs/{queued}").status_code == 204
        window = "?i0=1&j0=1&w=5&h=4&stride=2"
        expected = np.arange(35, dtype="<f4").reshape(5, 7)[1:5:2, 1:6:2]

        def check_get(n):
            if n % 3 == 0:
                response = c.get(url + "/frames/0/values" + window)
                assert (
                    response.status_code == 200
                    and response.content == expected.tobytes()
                )
            elif n % 3 == 1:
                response = c.get(url + "/frames/1/values")
                assert response.status_code == 409
            else:
                response = c.get(url + "/frames/0.png")
                assert response.status_code == 200
                assert Image.open(io.BytesIO(response.content)).size == (7, 5)

        with ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(check_get, range(48)))

        with monkeypatch.context() as patch:

            def unavailable(*args):
                raise RuntimeError("concurrent header update")

            patch.setattr(netcdf, "read_frame", unavailable)
            assert c.get(url + "/frames/0/values").status_code == 409
        (directory / "finish").touch()
        wait_run(c, run)
        assert (
            c.get(url + "/frames/1/values").content
            == np.arange(35, 70, dtype="<f4").tobytes()
        )


@pytest.fixture
def blocked_encoder(monkeypatch, tmp_path):
    encoder = tmp_path / "encoder.py"
    encoder.write_text(
        "import os, time\n"
        "from pathlib import Path\n"
        "Path('encoder.pid').write_text(str(os.getpid()))\n"
        "os.write(1, b'incomplete export')\n"
        "time.sleep(30)\n"
    )
    monkeypatch.setattr(
        render, "encoder_command", lambda *_: [sys.executable, str(encoder)]
    )


def start_export(c, saved_run):
    identifier, directory = saved_run
    url = f"/runs/{identifier}/export.mp4"
    assert c.get(url).status_code == 202
    partial = directory / render.export_name("mp4", 12, render.Style())
    partial = partial.with_suffix(".partial.mp4")
    poll(c, url, lambda r: partial.exists() and partial.stat().st_size > 0)
    return url, partial, int((directory / "encoder.pid").read_text())


def test_delete_while_ffmpeg_is_blocked(saved_run, client, blocked_encoder):
    identifier, directory = saved_run
    url, _, pid = start_export(client, saved_run)
    assert psutil.pid_exists(pid)
    assert client.delete(f"/runs/{identifier}").status_code == 204
    assert not directory.exists() and not psutil.pid_exists(pid)
    assert client.get(url).status_code == 404


def test_restart_during_export(
    settings, clock, saved_run, blocked_encoder, monkeypatch
):
    with TestClient(create_app(settings, clock)) as c:
        url, partial, pid = start_export(c, saved_run)
        assert psutil.pid_exists(pid)
    assert not partial.exists() and not psutil.pid_exists(pid)

    # A process killed before releasing its owner lock can leave a disk artifact.
    partial.write_bytes(b"stale partial")
    monkeypatch.undo()
    with TestClient(create_app(settings, clock)) as restarted:
        assert not partial.exists()
        assert restarted.get(url).status_code == 202
        response = poll(restarted, url, lambda r: r.status_code != 202)
        assert response.status_code == 200
        assert response.content[4:8] == b"ftyp"


def test_retention_during_active_download(saved_run, client, clock, monkeypatch):
    identifier, directory = saved_run
    original = directory / "result.nc"
    expected = client.get(f"/runs/{identifier}/result.nc").content
    entered, release = threading.Event(), threading.Event()
    chunks = Download.chunks
    opened = []

    async def paused(self):
        opened.append(self.stream)
        async for block in chunks(self):
            yield block
            if not entered.is_set():
                entered.set()
                assert await asyncio.to_thread(release.wait, 5)

    monkeypatch.setattr(Download, "chunks", paused)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.get, f"/runs/{identifier}/result.nc")
        try:
            assert entered.wait(5)
            clock.advance(7)
            clean(client)
            assert not original.exists()
            assert client.get(f"/runs/{identifier}/result.nc").status_code == 404
        finally:
            release.set()
        response = future.result(timeout=5)
    assert response.status_code == 200
    assert int(response.headers["content-length"]) == len(expected) > CHUNK_BYTES
    assert response.content == expected and opened[0].closed


def test_upload_streams_to_owned_staging_and_cleans_up_if_disk_tight(
    client, monkeypatch
):
    disk = client.app.state.manager.disk
    writes = []
    write = disk.write

    def observe(stream, data, description):
        writes.append((stream.name, len(data)))
        return write(stream, data, description)

    monkeypatch.setattr(disk, "write", observe)
    root = client.app.state.manager.fields.root
    budget = disk.margin + 2 * CHUNK_BYTES
    monkeypatch.setattr(
        type(disk), "free", property(lambda _: budget - sum(n for _, n in writes[:-1]))
    )
    payload = b"a" * (3 * CHUNK_BYTES)
    response = client.post("/fields", files={"file": ("x.nc", payload)})
    assert response.status_code == 422 and "safety margin" in response.text
    assert 2 <= len(writes) and all(0 < n <= CHUNK_BYTES for _, n in writes)
    assert not list(root.iterdir())


def test_declared_netcdf_size_is_checked_before_any_decode(
    client, monkeypatch, tmp_path
):
    path = tmp_path / "sparse.nc"
    with netCDF4.Dataset(path, "w") as ds:
        for name, size in [("t", 128), ("y", 1024), ("x", 1024)]:
            ds.createDimension(name, size)
        v = ds.createVariable(
            "heat", "f8", ("t", "y", "x"), zlib=True, chunksizes=(1, 32, 32)
        )
        v[0, :32, :32] = 1
    assert path.stat().st_size < 128 * 1024

    def unexpected_decode(*args):
        pytest.fail("An oversized declaration must be rejected before any decode")

    disk = client.app.state.manager.disk
    monkeypatch.setattr(
        type(disk), "free", property(lambda _: disk.margin + 500 * 1024**2)
    )
    monkeypatch.setattr(netcdf, "unpack", unexpected_decode)
    response = client.post("/fields", files={"file": ("sparse.nc", path.read_bytes())})
    assert (
        response.status_code == 422 and "declared decoded NetCDF data" in response.text
    )
    assert not list(client.app.state.manager.fields.root.iterdir())


def test_run_disk_warning_is_nonblocking_until_submission(client, monkeypatch):
    disk = client.app.state.manager.disk
    monkeypatch.setattr(type(disk), "free", property(lambda _: disk.margin + 4000))
    report = client.post("/runs/validate", json=preset()).json()
    assert report["ok"] and "safety margin" in report["warnings"][0]
    create = client.post("/runs", json=preset())
    assert create.status_code == 422 and "safety margin" in create.text
    assert not client.get("/runs").json()["items"]


def test_field_values_only_unpack_the_requested_descending_window(
    client, monkeypatch, tmp_path
):
    payload = upload_data(tmp_path / "packed.nc")
    response = client.post("/fields", files={"file": ("packed.nc", payload)})
    assert response.status_code == 201
    url = f"/fields/{response.json()['id']}/values"
    read = netcdf.unpack
    slices = []

    def observe(variable, selection):
        if variable.name == "heat":
            slices.append(selection)
        return read(variable, selection)

    monkeypatch.setattr(netcdf, "unpack", observe)
    response = client.get(url + "?index[time]=1&i0=1&j0=1&w=4&h=3&stride=2")
    assert response.status_code == 200
    assert slices == [(1, slice(1, 4, 2), slice(3, 6, 2))]
    expected = np.arange(35, 70).reshape(5, 7)[::-1, ::-1][1:4:2, 1:5:2] * 0.5 + 10
    assert response.content == expected.astype("<f4").tobytes()
    assert client.get(url + "?stride=" + "9" * 40).headers["x-nx"] == "1"
    assert client.get(url + "?w=" + "9" * 40).status_code == 422
    assert client.get(url + "?index[time]=" + "9" * 40).status_code == 422


def test_upload_content_types_filenames_and_strict_cors(settings, clock, tmp_path):
    origin = "http://localhost:5173"
    conf = replace(settings, origins=(origin, "*"))
    original = upload_data(tmp_path / "original.nc")
    with TestClient(create_app(conf, clock)) as c:
        assert (
            c.post(
                "/fields",
                content=original,
                headers={"content-type": "application/x-netcdf"},
            ).status_code
            == 422
        )
        assert (
            c.post(
                "/fields", files={"file": ("wrong.nc", original, "text/html")}
            ).status_code
            == 422
        )
        assert (
            c.post("/fields", files={"unexpected": ("wrong.nc", original)}).status_code
            == 422
        )
        assert (
            c.post(
                "/fields",
                content=b"--bad--",
                headers={"content-type": "multipart/form-data"},
            ).status_code
            == 422
        )
        name = "../../outside.nc"
        response = c.post(
            "/fields", files={"file": (name, original, "application/octet-stream")}
        )
        assert response.status_code == 201
        assert response.json()["name"] == name
        item = response.json()["id"]
        result = c.get(f"/fields/{item}/original", headers={"origin": origin})
        assert result.headers["access-control-allow-origin"] == origin
        assert result.headers["content-disposition"].endswith("..%2F..%2Foutside.nc")
        assert (
            result.content == original and not (settings.data / "outside.nc").exists()
        )
        assert (
            "access-control-allow-origin"
            not in c.get("/health", headers={"origin": origin + ".bad"}).headers
        )
        headers = {"origin": "null", "access-control-request-method": "POST"}
        assert c.options("/fields", headers=headers).status_code == 400


def test_downscale_matches_pillow_box_with_alpha():
    shape = (33, 72)
    rng = np.random.default_rng(123)
    values = rng.uniform(-2, 2, size=shape)
    values[::2, ::6] = np.nan
    style = render.Style("coolwarm", -1, 1, 20)
    rgba = colormaps["coolwarm"](np.ma.masked_invalid((values + 1) / 2), bytes=True)
    expected = Image.fromarray(rgba[::-1]).resize((20, 9), Image.Resampling.BOX)
    assert render.render(values, style, None).tobytes() == expected.tobytes()


def test_request_and_run_state_logs_are_json(client, caplog):
    response = client.post("/runs", json=preset())
    identifier = response.json()["id"]
    wait_run(client, identifier, "error")
    assert client.delete(f"/runs/{identifier}").status_code == 204
    records = [
        json.loads(r.message) for r in caplog.records if r.name == events.logger.name
    ]
    states = [record for record in records if record.get("run_id") == identifier]
    assert [r["status"] for r in states] == ["queued", "running", "error", "deleted"]
    requests = [record for record in records if record.get("path") == "/runs"]
    assert len(requests) == 1 and requests[0]["status"] == 202
    assert requests[0]["method"] == "POST" and requests[0]["duration_ms"] >= 0


def test_ffmpeg_output_checks_disk_before_writing(
    saved_run, client, blocked_encoder, monkeypatch
):
    manager = client.app.state.manager
    disk = manager.disk
    monkeypatch.setattr(type(disk), "free", property(lambda _: disk.margin))
    identifier, directory = saved_run
    url = f"/runs/{identifier}/export.mp4"
    assert client.get(url).status_code == 202
    result = poll(client, url, lambda r: r.status_code != 202)
    assert result.status_code == 500 and "safety margin" in result.json()["detail"]
    assert not list(directory.glob("*.partial.*"))


def test_disk_reservations_and_short_writes(client, monkeypatch):
    disk = client.app.state.manager.disk
    monkeypatch.setattr(type(disk), "free", property(lambda _: disk.margin + 100))
    with disk.reserve(75, "queued write"):
        assert disk.available == 25
        with pytest.raises(DiskSpaceError, match="safety margin"):
            disk.write(io.BytesIO(), b"x" * 26, "other write")
    assert disk.available == 100

    class ShortWrite:
        def write(self, data):
            return len(data) - 1

    monkeypatch.setattr(
        type(disk), "free", property(lambda _: disk.margin + CHUNK_BYTES)
    )
    with pytest.raises(DiskSpaceError, match="Short write"):
        disk.write(ShortWrite(), b"x", "short write")


def test_legacy_field_disk_guard_is_visible_in_validation(
    client, tmp_path, monkeypatch
):
    payload = upload_data(tmp_path / "packed.nc")
    response = client.post("/fields", files={"file": ("packed.nc", payload)})
    field = response.json()["id"]
    disk = client.app.state.manager.disk
    monkeypatch.setattr(type(disk), "free", property(lambda _: disk.margin))
    request = {
        "source": {"kind": "field", "field_id": field, "fill_value": 0},
        "dt": 1,
        "steps": 1,
    }
    report = client.post("/runs/validate", json=request).json()
    assert not report["ok"] and "declared decoded NetCDF data" in report["warnings"][0]
    values = client.get(f"/fields/{field}/values")
    assert values.status_code == 422 and "safety margin" in values.text


def test_internal_error_body_omits_exception(settings):
    app = create_app(settings)

    @app.get("/error-test")
    async def error():
        raise RuntimeError("internal path /local/secret")

    with TestClient(app, raise_server_exceptions=False) as c:
        response = c.get("/error-test")
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
