import io

import netCDF4
import numpy as np
import pytest
from PIL import Image

from app import netcdf
from app.render import Style, render
from tests.conftest import draw, upload_data


@pytest.mark.parametrize(
    "fmt", ["NETCDF3_CLASSIC", "NETCDF3_64BIT_OFFSET", "NETCDF3_64BIT_DATA", "NETCDF4"]
)
def test_upload_packed_descending(client, tmp_path, fmt):
    original = upload_data(tmp_path / "input.nc", fmt)
    uploaded = client.post("/fields", files={"file": ("example.nc", original)})
    assert uploaded.status_code == 201, uploaded.text
    item = uploaded.json()
    (var,) = item["variables"]
    assert (var["name"], var["nx"], var["ny"], var["units"]) == ("heat", 7, 5, "K")
    assert var["extra_dims"] == [
        {
            "name": "time",
            "size": 2,
            "values": ["2000-01-01T00:00:00", "2000-01-02T00:00:00"],
        }
    ]
    assert item["default_stats"]["mean"] == pytest.approx(18.5)
    url = "/fields/" + item["id"]
    assert client.get(url + "/original").content == original
    assert client.get(url + "/stats?index[time]=1").json()["missing_count"] == 2
    values = client.get(url + "/values?index[time]=1")
    grid = np.frombuffer(values.content, dtype="<f4").reshape(5, 7)
    assert grid[0, 0] == 10 + 69 * 0.5
    assert np.isnan(grid[-1, -2:]).all()
    src = {"kind": "field", "field_id": item["id"], "index": {"time": 1}}
    req = {"source": src, "dt": 1, "steps": 1}
    assert not client.post("/runs/validate", json=req).json()["ok"]
    assert client.post("/runs", json=req).status_code == 422
    src["fill_value"] = 0
    assert client.post("/runs/validate", json=req).json()["ok"]
    src["index"] = {"time": 2}
    assert not client.post("/runs/validate", json=req).json()["ok"]


def test_drawn_windows_and_preview(client):
    plane = np.arange(20, dtype="<f4").reshape(4, 5)
    item = draw(client, plane, name="paint").json()
    url = "/fields/" + item["id"]
    assert client.get(url).json() == item
    assert client.get(url + "/original").content.startswith(b"CDF\x05")
    r = client.get(
        url + "/values?i0=1&j0=0&w=3&h=3&stride=2",
        headers={"Origin": "http://localhost:5173"},
    )
    assert [r.headers[h] for h in ("x-nx", "x-ny", "x-i0", "x-j0", "x-stride")] == [
        "2",
        "2",
        "1",
        "0",
        "2",
    ]
    assert "X-Stride" in r.headers["access-control-expose-headers"]
    assert r.content == plane[:3:2, 1:4:2].tobytes()
    image = Image.open(
        io.BytesIO(client.get(url + "/preview.png?colormap=gray").content)
    )
    assert image.getpixel((0, 3)) == (0, 0, 0, 255)
    assert image.getpixel((4, 0)) == (255, 255, 255, 255)
    for query in [
        "i0=5",
        "h=5",
        "var=wrong",
        "index[wrong]=0",
        "index=1",
        "index[t]=-1",
        "index[t]=0&index[t]=1",
    ]:
        assert client.get(url + "/values?" + query).status_code == 422
    for query in [
        "vmin=1",
        "vmax=2",
        "vmin=2&vmax=2",
        "vmin=nan&vmax=2",
        "colormap=anything",
        "max_side=2",
    ]:
        assert client.get(url + "/preview.png?" + query).status_code == 422
    assert client.delete(url).status_code == 204
    assert client.get(url).status_code == 404


def test_pagination_and_upload_errors(client, tmp_path):
    ids = {draw(client).json()["id"] for _ in range(2)}
    page = client.get("/fields?limit=1").json()
    other = client.get("/fields", params={"cursor": page["next_cursor"]}).json()
    assert {page["items"][0]["id"], other["items"][0]["id"]} == ids and other[
        "next_cursor"
    ] is None
    assert client.get("/fields?cursor=bad").status_code == 422
    assert (
        client.post("/fields", files={"file": ("bad", b"not a dataset")}).status_code
        == 422
    )
    assert draw(client, unknown="x").status_code == 422
    assert draw(client, np.full((4, 4), np.nan, dtype="<f4")).status_code == 422
    assert draw(client, nx=100).status_code == 422
    assert client.post("/fields").status_code == 422
    with netCDF4.Dataset(tmp_path / "empty", "w"):
        pass
    assert (
        client.post(
            "/fields", files={"file": ("empty", (tmp_path / "empty").read_bytes())}
        ).status_code
        == 422
    )


def test_stats_all_missing_and_rendering():
    values = np.array([[np.nan, 4.0], [4.0, 4.0]])
    im = render(values, Style(colormap="gray"), (4, 4))
    assert im.getpixel((0, 1))[3] == 0
    assert im.getpixel((1, 1)) == (128, 128, 128, 255)
    values[:] = np.nan
    info = netcdf.stats(values, "u", {})
    assert (
        info["missing_count"] == 4
        and info["min"] is info["max"] is info["mean"] is None
    )
    assert not np.asarray(render(values, Style(), (None, None)))[:, :, 3].any()
    checker = np.tile([[0.0, 1.0], [1.0, 0.0]], (16, 32))
    im = render(checker, Style("gray", 0, 1, 16), (0, 0))
    assert im.size == (16, 8) and im.getpixel((0, 0))[0] == 128


def test_grouped_variables_unsigned_values_and_coordinates(tmp_path):
    path = tmp_path / "groups.nc"
    with netCDF4.Dataset(path, "w") as ds:
        group = ds.createGroup("forecast")
        for name in ("member", "level", "channel", "y", "x"):
            group.createDimension(name, 2)
        member = group.createVariable("member", str, ("member",))
        member[:] = np.array(["early", "late"], dtype=object)
        level = group.createVariable("level", "f8", ("level",))
        level.units = "days since nonsense"
        level[:] = [1, 2]
        heat = group.createVariable(
            "heat", "i2", ("member", "level", "channel", "y", "x")
        )
        heat.set_auto_maskandscale(False)
        heat._Unsigned = "true"
        heat[:] = -2
        group.createDimension("empty", 0)
        group.createVariable("empty_field", "i4", ("y", "empty"))
    field = netcdf.inspect_field(path)
    assert field["default_var"] == "forecast/heat"
    dims = field["variables"][0]["extra_dims"]
    assert [d.get("values") for d in dims] == [["early", "late"], [1, 2], None]
    values, index = netcdf.read_plane(path, field, index={"member": 1})
    assert index == {"member": 1, "level": 0, "channel": 0}
    assert np.all(values == 65534)
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        netcdf.read_plane(path, field, name="forecast/empty_field")


def test_constant_dirichlet_render_and_override():
    plane = np.ones((4, 5)) * 6
    assert render(plane, Style("gray"), [0, 6]).getpixel((0, 0))[0] == 128
    assert render(plane, Style("gray", 0, 6), [0, 6]).getpixel((0, 0))[0] == 255


def test_child_group_coordinates_for_inherited_dimensions(tmp_path):
    path = tmp_path / "inherited.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("y", 2)
        ds.createDimension("x", 3)
        group = ds.createGroup("experiment")
        group.createVariable("y", "i4", ("y",))[:] = [2, 1]
        # The end points alone are insufficient to classify this axis.
        ds.createVariable("x", "i4", ("x",))[:] = [3, 1, 2]
        group.createVariable("u", "f8", ("y", "x"))[:] = np.arange(6).reshape(2, 3)
    field = netcdf.inspect_field(path)
    plane, _ = netcdf.read_plane(path, field)
    assert plane.tolist() == [[3, 4, 5], [0, 1, 2]]


def test_implicit_fill_in_non_native_packed_data(tmp_path):
    import sys

    endian = "big" if sys.byteorder == "little" else "little"
    dtype = ">i2" if endian == "big" else "<i2"
    path = tmp_path / "packed.nc"
    with netcdf.dataset(path, "w") as ds:
        ds.createDimension("y", 2)
        ds.createDimension("x", 2)
        var = ds.createVariable("packed", dtype, ("y", "x"), endian=endian)
        var[0] = [-2, 384]
        var._Unsigned = "true"
        var.scale_factor = 0.5
    field = netcdf.inspect_field(path)
    plane, _ = netcdf.read_plane(path, field)
    np.testing.assert_equal(plane, [[32767, 192], [np.nan, np.nan]])


@pytest.mark.parametrize("legacy", [False, True])
def test_drawn_default_fill_number_remains_a_finite_input(client, legacy):
    value = netCDF4.default_fillvals["f8"]
    field = draw(client, np.full((4, 4), value, dtype="<f4")).json()
    assert field["default_stats"]["missing_count"] == 0
    if legacy:
        path = client.app.state.manager.field_path(field)
        with netcdf.dataset(path, "w", format="NETCDF3_64BIT_DATA") as ds:
            for name in ("y", "x"):
                ds.createDimension(name, 4)
            ds.createVariable("u", "f8", ("y", "x"), fill_value=False)[:] = value
    url = "/fields/" + field["id"]
    assert client.get(url + "/stats").json()["min"] == value
    np.testing.assert_equal(
        np.frombuffer(client.get(url + "/values").content, "<f4"), value
    )
    body = dict(source=dict(kind="field", field_id=field["id"]), steps=1, dt=1)
    assert client.post("/runs/validate", json=body).json()["ok"]
