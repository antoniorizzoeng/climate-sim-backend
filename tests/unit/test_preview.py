import io
import json
import os
import resource
import subprocess
import sys
from dataclasses import replace

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app import image_memory, netcdf
from app.main import create_app
from app.render import Style, render
from app.settings import Settings
from tests.conftest import draw


def pixels(response):
    assert response.status_code == 200, response.text
    return np.asarray(Image.open(io.BytesIO(response.content)).convert("RGBA"))


def test_preview_matches_full_plane_with_packing_and_missing(settings, clock, tmp_path):
    path = tmp_path / "packed.nc"
    ny, nx = 33, 72
    raw = np.random.default_rng(42).integers(-300, 300, (ny, nx), dtype=np.int16)
    raw[::2, ::3] = -998
    raw[-5:, :8] = -999
    with netcdf.dataset(path, "w") as ds:
        for name, length in (("time", 2), ("y", ny), ("x", nx)):
            ds.createDimension(name, length)
        for name, length in (("y", ny), ("x", nx)):
            coordinate = ds.createVariable(name, "i4", (name,), chunksizes=(8,))
            coordinate[:] = np.arange(length)[::-1]
        var = ds.createVariable(
            "u", "i2", ("time", "y", "x"), fill_value=-999, chunksizes=(1, 2, nx)
        )
        var._Unsigned = "true"
        var.missing_value = np.int16(-998)
        var.scale_factor = 0.5
        var.add_offset = 15.0
        var.set_auto_maskandscale(False)
        var[1] = raw

    settings = replace(settings, preview_memory_bytes=14 * 1024)
    decoded = raw.view(np.uint16).astype(float) * 0.5 + 15
    decoded[np.isin(raw, [-998, -999])] = np.nan
    decoded = decoded[::-1, ::-1]
    with TestClient(create_app(settings, clock)) as client:
        uploaded = client.post(
            "/fields", files={"file": ("packed.nc", path.read_bytes())}
        )
        assert uploaded.status_code == 201, uploaded.text
        url = f"/fields/{uploaded.json()['id']}/preview.png"
        for explicit in ({}, {"vmin": 0, "vmax": 33000}):
            style = Style("coolwarm", max_side=20, **explicit)
            actual = pixels(
                client.get(
                    url,
                    params=dict(
                        max_side=20,
                        colormap="coolwarm",
                        **{"index[time]": 1},
                        **explicit,
                    ),
                )
            )
            expected = render(decoded, style, (np.nanmin(decoded), np.nanmax(decoded)))
            np.testing.assert_array_equal(actual, np.asarray(expected))
            assert actual[-1, -1, 3] == 0
            assert np.any((actual[:, :, 3] > 0) & (actual[:, :, 3] < 255))
        assert (pixels(client.get(url, params={"max_side": 20}))[:, :, 3] == 0).all()


def test_tall_preview_preserves_pillow_axis_order(client):
    plane = np.random.default_rng(1).normal(size=(407, 4)).astype("<f4")
    uploaded = draw(client, plane)
    assert uploaded.status_code == 201, uploaded.text
    item = uploaded.json()
    response = client.get(
        f"/fields/{item['id']}/preview.png?max_side=16&colormap=coolwarm&vmin=-1&vmax=1"
    )
    expected = render(plane, Style("coolwarm", -1, 1, 16), None)
    np.testing.assert_array_equal(pixels(response), np.asarray(expected))


def test_projected_frame_and_native_memory_errors(settings, clock, monkeypatch):
    directory = settings.data / "runs" / "finished"
    directory.mkdir(parents=True)
    metadata = dict(
        id="finished",
        _order=1,
        status="done",
        finished_at=clock().isoformat(),
        frames=2,
        resolved=dict(expected_frames=2, nx=70, ny=45, nz=3),
        display_range=[-2, 2],
    )
    (directory / "metadata.json").write_text(json.dumps(metadata))
    volume = np.random.default_rng(22).normal(size=(3, 45, 70))
    volume[:, :5, :8] = np.nan
    with netcdf.dataset(directory / "result.nc", "w") as ds:
        for dim, size in zip("tzyx", (2, *volume.shape)):
            ds.createDimension(dim, size)
        var = ds.createVariable("u", "f8", tuple("tzyx"), chunksizes=(1, 1, 3, 70))
        var[0] = volume

    settings = replace(settings, preview_memory_bytes=50 * 1024)
    with TestClient(create_app(settings, clock)) as client:
        item = draw(client, np.ones((45, 70), dtype="<f4")).json()
        field = f"/fields/{item['id']}/preview.png"
        frame = "/runs/finished/frames/0.png"
        monkeypatch.setattr(image_memory, "available_ram", lambda: 500_000)
        for endpoint in (frame, field):
            response = client.get(endpoint)
            assert response.status_code == 413
            assert "smaller max_side" in response.json()["detail"]
            assert client.get(endpoint, params={"max_side": 80}).status_code == 413
        assert client.get(field, params={"max_side": 16}).is_success
        for axis, name in enumerate("zyx"):
            response = client.get(
                frame, params=dict(axis=name, projection="mean", max_side=16)
            )
            plane = volume.mean(axis=axis, dtype=np.longdouble).astype(float)
            expected = render(
                plane, Style(max_side=16), metadata["display_range"], whole_field=False
            )
            np.testing.assert_array_equal(pixels(response), np.asarray(expected))
        response = client.get(field, params={"max_side": 69})
        assert response.status_code == 413
        assert "preview memory budget" in response.json()["detail"]
        response = client.get(frame, params={"max_side": 67})
        assert response.status_code == 413
        assert "NetCDF chunk" in response.json()["detail"]


def test_huge_netcdf4_preview_peak_rss(settings, clock):
    n = 8192
    directory = settings.data / "fields" / "large"
    directory.mkdir(parents=True)
    with netcdf.dataset(directory / "original.nc", "w") as ds:
        for dim in "yx":
            ds.createDimension(dim, n)
        var = ds.createVariable(
            "u", "f4", tuple("yx"), chunksizes=(64, n), fill_value=False
        )
        row = np.arange(n, dtype=np.float32)
        for start in range(0, n, 64):
            var[start : start + 64] = row + start
    metadata = dict(
        id="large",
        state="active",
        source="upload",
        last_used_at=clock().isoformat(),
        default_var="u",
        variables=[dict(name="u", nx=n, ny=n, nz=1, extra_dims=[])],
        default_stats=dict(min=0, max=16319, missing_count=0),
    )
    (directory / "metadata.json").write_text(json.dumps(metadata))
    worker = subprocess.run(
        [sys.executable, "-m", "tests.unit.test_preview", clock().isoformat()],
        env=dict(os.environ, DATA_DIR=str(settings.data), ENGINE_PERIODIC="0"),
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    baseline, peak = map(int, worker.stdout.split())
    overhead = 16 * 1024**2
    assert (
        peak - baseline
    ) * 1024 <= settings.preview_memory_bytes + overhead, (
        f"preview added {(peak - baseline) / 1024:.1f} MiB of RSS"
    )


if __name__ == "__main__":
    from datetime import datetime

    with TestClient(
        create_app(Settings.from_env(), lambda: datetime.fromisoformat(sys.argv[1]))
    ) as client:
        baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        image = pixels(client.get("/fields/large/preview.png?max_side=512"))
        assert image.shape == (512, 512, 4)
        assert image[0, 0, 3] == 255
        print(baseline, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
