import numpy as np
import pytest
from jsonschema import Draft202012Validator

from app import netcdf
from app.models import automatic_processes
from tests.conftest import preset


@pytest.mark.parametrize(
    "x,y,z,xs,ys,expected",
    [
        (
            "kilometres",
            "meters",
            "m",
            [4, 2, 0],
            [0, 2000, 4000],
            (2, 2, 0.05, "km", False),
        ),
        ("m", "km", "Pa", [0, 1, 2.1], [2, 3, 4], (1.05, 1000, None, "m", True)),
        (
            "degrees_E",
            "degreeNorth",
            "km",
            [179, -180, -179],
            [59, 60, 61],
            (55.59746, 111.19493, 50, "km", True),
        ),
        ("unknown", "m", "m", [1, 2, 3], [0, 1, 2], None),
        ("m", "m", "m", [0, 1, 0], [0, 1, 2], None),
        ("m", "m", "m", [0, np.nan, 2], [0, 1, 2], None),
    ],
)
def test_cf_spacing(client, tmp_path, x, y, z, xs, ys, expected):
    path = tmp_path / "group.nc"
    with netcdf.dataset(path, "w") as ds:
        for dim, values, unit in (("z", [0, 50], z), ("y", ys, y), ("x", xs, x)):
            ds.createDimension(dim, len(values))
            c = ds.createVariable(dim, "f8", (dim,))
            c[:] = values
            c.units = unit
        ds.createGroup("weather").createVariable("heat", "f8", ("z", "y", "x"))[:] = 1
    field = client.post(
        "/fields", files={"file": ("grid.nc", path.read_bytes())}
    ).json()
    spacing = field["variables"][0]["spacing"]
    if expected is None:
        assert spacing is None
    else:
        dx, dy, dz, unit, approximate = expected
        assert spacing["dx"] == pytest.approx(dx)
        assert spacing["dy"] == pytest.approx(dy)
        assert spacing["dz"] == dz
        assert (spacing["length_unit"], spacing["approximate"]) == (unit, approximate)
        spec = client.get("/openapi.json").json()
        Draft202012Validator(spec["components"]["schemas"]["GridSpacing"]).validate(
            spacing
        )


def test_auto_ranks_and_explicit_requests(client):
    assert [automatic_processes(n, n, 1, 12) for n in (64, 256, 768)] == [1, 1, 4]
    assert [automatic_processes(n, n, n, 12) for n in (16, 48, 96)] == [1, 1, 6]
    assert automatic_processes(1024, 1024, 2, 2) == 2
    assert automatic_processes(4, 700000, 1, 19) == 16
    for invalid in ({"length": "m"}, {"length": "m", "time": "days"}, None):
        assert client.post("/runs", json=preset(units=invalid)).status_code == 422
    for name in ("    ", "s" * 121):
        assert client.post("/runs", json=preset(name=name)).status_code == 422
