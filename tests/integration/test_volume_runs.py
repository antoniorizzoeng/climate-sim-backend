import io

import numpy as np
import pytest
from matplotlib import colormaps
from PIL import Image

from app import netcdf
from tests.conftest import draw, poll, preset, wait_run

pytestmark = pytest.mark.integration


def complete(client, request):
    created = client.post("/runs", json=request)
    assert created.status_code == 202, created.text
    run = wait_run(client, created.json()["id"])
    return run, "/runs/" + run["id"]


def raw(response, dims):
    assert response.status_code == 200, response.text[:120] if response.is_error else ""
    shape = tuple(int(response.headers["X-N" + dim]) for dim in dims)
    assert len(response.content) == 4 * np.prod(shape)
    return np.frombuffer(response.content, "<f4").reshape(shape)


def test_preset_on_two_ranks(live):
    request = preset(
        vz=-0.5,
        D=0.1,
        Dz=0.2,
        dz=0.75,
        out_every=2,
        bc={"front": "neumann", "back": "dirichlet"},
    )
    request["source"].update(nx=24, ny=20, nz=12)
    run, url = complete(live, request)
    assert run["resolved"]["processes"] == 2
    assert run["resolved"]["nz"] == 12 and run["resolved"]["dz"] == 0.75
    assert run["resolved"]["expected_frames"] == run["frames"] == 4
    path = live.app.state.manager.runs.root / run["id"] / "result.nc"
    with netcdf.dataset(path) as ds:
        assert float(ds.Dz) == 0.2 and float(ds.dz) == 0.75
        assert "front=neumann back=dirichlet" in ds.boundary_conditions.lower()
        assert float(ds.velocity.strip("()").split(",")[2]) == -0.5
        var = ds["u"]
        assert var.dimensions == ("time", "z", "y", "x") and var.shape == (
            4,
            12,
            20,
            24,
        )
        middle = var[0, 6]
        final = var[3, 6]
        assert not np.array_equal(middle, final)
        expected = var[3, ::3, ::3, ::3]
    np.testing.assert_array_equal(
        raw(live.get(url + "/frames/0/values"), "yx"), middle.astype("<f4")
    )
    response = live.get(
        url + "/frames/3/volume?max_side=8", headers={"Origin": "http://localhost:3000"}
    )
    assert "X-Nz" in response.headers["access-control-expose-headers"]
    np.testing.assert_array_equal(raw(response, "zyx"), expected.astype("<f4"))
    assert live.get(url + "/frames/4/volume").status_code == 404


def test_uploaded_volume_and_views(live, tmp_path):
    path = tmp_path / "layers.nc"
    data = np.arange(2 * 4 * 5 * 7, dtype="i2").reshape(2, 4, 5, 7)
    data[1, -1, 1, 2] = -999
    with netcdf.dataset(path, "w") as ds:
        for dim, size in zip(("member", "sigma", "y", "x"), data.shape):
            ds.createDimension(dim, size)
        for dim in ("sigma", "y", "x"):
            coord = ds.createVariable(dim, "i4", (dim,))
            coord[:] = np.arange(len(ds.dimensions[dim]))[::-1]
        ds["sigma"].axis = "Z"
        v = ds.createVariable(
            "heat", "i2", ("member", "sigma", "y", "x"), fill_value=-999
        )
        v[:] = data
        v.scale_factor = 0.5
    field = live.post(
        "/fields", files={"file": ("layers.nc", path.read_bytes())}
    ).json()
    (var,) = field["variables"]
    assert (var["z_dim"], var["nz"]) == ("sigma", 4)
    src = dict(
        kind="field", field_id=field["id"], index={"member": 1, "sigma": 1}, volume=True
    )
    req = dict(source=src, steps=1, dt=0.1)
    assert not live.post("/runs/validate", json=req).json()["ok"]
    assert live.post("/runs", json=req).status_code == 422
    src["fill_value"] = 17
    run, url = complete(live, req)
    expected = data[1, ::-1, ::-1, ::-1].astype("<f4") * 0.5
    expected[expected == -499.5] = 17
    assert run["display_range"] == [0, expected.max()]
    endpoint = url + "/frames/0"
    np.testing.assert_array_equal(raw(live.get(endpoint + "/volume"), "zyx"), expected)
    for axis, name in enumerate("zyx"):
        for projection in ("none", "max", "mean"):
            query = dict(axis=name, projection=projection, slice=1)
            plane = (
                np.take(expected, 1, axis)
                if projection == "none"
                else getattr(expected, projection)(axis=axis)
            )
            np.testing.assert_allclose(
                raw(live.get(endpoint + "/values", params=query), "yx"), plane
            )
            query.update(i0=1, j0=1, w=3, h=3, stride=2)
            np.testing.assert_allclose(
                raw(live.get(endpoint + "/values", params=query), "yx"),
                plane[1:4:2, 1:4:2],
            )
        query = dict(axis=name, slice=1, colormap="gray", vmin=0, vmax=256)
        png = live.get(endpoint + ".png", params=query)
        pixels = np.asarray(Image.open(io.BytesIO(png.content)).convert("RGBA"))[
            :, :, 0
        ]
        expected_colors = colormaps["gray"](
            np.take(expected, 1, axis)[::-1] / 256, bytes=True
        )
        np.testing.assert_array_equal(pixels, expected_colors[:, :, 0])
    for suffix in ("/frames/0.png", "/frames/0/values", "/export.gif"):
        assert live.get(url + suffix + "?axis=x&slice=7").status_code == 422
    for axis, size in [("x", (5, 4)), ("y", (7, 4))]:
        export_url = url + "/export.gif?axis=" + axis + "&projection=max"
        assert live.get(export_url).status_code == 202
        gif = poll(live, export_url, lambda r: r.status_code != 202)
        assert gif.status_code == 200
        assert Image.open(io.BytesIO(gif.content)).size == size
    assert (
        len(list((live.app.state.manager.runs.root / run["id"]).glob("export-*.gif")))
        == 2
    )
    src.update(replicate_z=2)
    assert not live.post("/runs/validate", json=req).json()["ok"]
    src.update(replicate_z=1, volume=False)
    plane_run, plane_url = complete(live, req)
    assert plane_run["resolved"]["nz"] == 1
    np.testing.assert_array_equal(
        raw(live.get(plane_url + "/frames/0/values"), "yx"), expected[2]
    )


def test_extrude_drawing_and_single_layer(live):
    plane = np.arange(20, dtype="<f4").reshape(4, 5)
    field = draw(live, plane).json()
    assert field["variables"][0]["nz"] == 1 and "z_dim" not in field["variables"][0]
    source = dict(kind="field", field_id=field["id"], volume=True, replicate_z=3)
    request = dict(source=source, dt=0.25, steps=1)
    _, url = complete(live, request)
    np.testing.assert_array_equal(
        raw(live.get(url + "/frames/0/volume"), "zyx"), np.stack([plane] * 3)
    )
    source.pop("replicate_z")
    request.update(Dz=1000, vz=1000)
    run, url = complete(live, request)
    assert run["resolved"]["dt_max"] is None
    query = "?slice=999&axis=x&projection=max"
    for suffix in (".png", "/values"):
        base = url + "/frames/1" + suffix
        assert live.get(base + query).content == live.get(base).content
    np.testing.assert_array_equal(
        raw(live.get(url + "/frames/1/volume"), "zyx"), plane[None]
    )
    completed = poll(live, url + "/export.gif", lambda r: r.status_code != 202)
    assert completed.status_code == 200
    assert live.get(url + "/export.gif" + query).content == completed.content


@pytest.mark.parametrize("bottom,top", [(0, 1), (1, 1)])
def test_uniform_volume_views_share_field_scale(live, tmp_path, bottom, top):
    path = tmp_path / "layers.nc"
    with netcdf.dataset(path, "w") as ds:
        for name, size in [("z", 2), ("y", 4), ("x", 4)]:
            ds.createDimension(name, size)
        u = ds.createVariable("u", "f8", ("z", "y", "x"))
        u[0], u[1] = bottom, top
    field = live.post("/fields", files={"file": (path.name, path.read_bytes())}).json()
    source = dict(kind="field", field_id=field["id"], volume=True)
    run, url = complete(live, dict(source=source, dt=1, steps=1, processes=1))
    assert run["display_range"] == [0, 1]
    for view, expected in [("slice=0", 0), ("slice=1", 255), ("projection=max", 255)]:
        if bottom == top:
            expected = 128
        query = "?colormap=gray&axis=z&" + view
        for endpoint in ("/frames/0.png", "/export.gif"):
            response = poll(
                live, url + endpoint + query, lambda r: r.status_code != 202
            )
            assert response.status_code == 200
            image = Image.open(io.BytesIO(response.content)).convert("RGBA")
            assert image.getpixel((0, 0)) == (expected,) * 3 + (255,)
