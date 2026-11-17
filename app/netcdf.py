"""Copy data out before releasing the lock. libnetcdf/HDF5 calls cannot overlap."""

from contextlib import contextmanager
from threading import Lock

import netCDF4
import numpy as np

from .models import invalid

_nc_lock = Lock()


@contextmanager
def dataset(path, *args, **kwargs):
    with _nc_lock, netCDF4.Dataset(path, *args, **kwargs) as ds:
        ds.set_auto_maskandscale(False)
        yield ds


def unpack(variable, selection):
    raw = np.asarray(variable[selection])
    missing = np.zeros(raw.shape, dtype=bool)
    for attr in ("_FillValue", "missing_value"):
        if attr in variable.ncattrs():
            missing |= np.isin(raw, variable.getncattr(attr))
    if getattr(variable, "_Unsigned", "") == "true" and raw.dtype.kind == "i":
        raw = raw.view(np.dtype(f"u{raw.itemsize}").newbyteorder(raw.dtype.byteorder))
    values = raw.astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        values *= getattr(variable, "scale_factor", 1.0)
        values += getattr(variable, "add_offset", 0.0)
    values[missing | ~np.isfinite(values)] = np.nan
    return values


def coordinate(group, dim):
    while group is not None:
        var = group.variables.get(dim.name)
        if var is not None and var.dimensions == (dim.name,):
            return var
        if dim.name in group.dimensions:
            break
        group = group.parent
    return None


def extra_dim(group, dim):
    meta = dict(name=dim.name, size=len(dim))
    coord = coordinate(group, dim)
    if coord is None:
        return meta
    if np.dtype(coord.dtype).kind in "iuf":
        values = unpack(coord, slice(None))
        if not np.isfinite(values).all():
            return meta
        units = getattr(coord, "units", "")
        if isinstance(units, str) and " since " in units:
            try:
                dates = netCDF4.num2date(
                    values, units, calendar=getattr(coord, "calendar", "standard")
                )
                meta["values"] = [d.isoformat() for d in dates]
                return meta
            except (ValueError, OverflowError):
                pass  # Unrecognised calendar or units: preserve the coordinate labels.
        meta["values"] = values.tolist()
    elif np.dtype(coord.dtype).kind in "SU":
        meta["values"] = np.asarray(coord[:]).astype(str).tolist()
    return meta


def variables(group, prefix=""):
    for name, var in group.variables.items():
        if (
            var.ndim < 2
            or isinstance(var.datatype, netCDF4.VLType)
            or np.dtype(var.dtype).kind not in "iuf"
        ):
            continue
        yield dict(
            name=prefix + name,
            nx=var.shape[-1],
            ny=var.shape[-2],
            extra_dims=[extra_dim(group, d) for d in var.get_dims()[:-2]],
            **{
                attr: value
                for attr in ("units", "long_name")
                if isinstance(value := getattr(var, attr, None), str)
            },
        )
    for name, subgroup in group.groups.items():
        yield from variables(subgroup, f"{prefix}{name}/")


def read_plane(path, field, name=None, index=None):
    name = field["default_var"] if name is None else name
    var = next((v for v in field["variables"] if v["name"] == name), None)
    if var is None:
        raise invalid("source.var", f"Unknown variable {name!r}")
    if not var["nx"] or not var["ny"]:
        raise invalid("source.var", "Selected spatial dimensions are empty")
    indices = {d["name"]: 0 for d in var["extra_dims"]}
    if (index or {}).keys() - indices.keys():
        raise invalid("source.index", "Index contains an unknown dimension")
    indices.update(index or {})
    for dim in var["extra_dims"]:
        if not 0 <= indices[dim["name"]] < dim["size"]:
            raise invalid("source.index", f"Index outside dimension {dim['name']}")
    with dataset(path) as ds:
        variable = ds[name]
        selection = tuple(indices[d["name"]] for d in var["extra_dims"]) + (
            slice(None),
            slice(None),
        )
        values = unpack(variable, selection)
        for axis, dim in enumerate(variable.get_dims()[-2:]):
            coord = coordinate(variable.group(), dim)
            if (
                coord is not None
                and len(dim) > 1
                and np.dtype(coord.dtype).kind in "iuf"
            ):
                positions = unpack(coord, slice(None))
                if positions[0] > positions[-1] and np.all(
                    positions[1:] <= positions[:-1]
                ):
                    values = np.flip(values, axis)
    return values, indices


def stats(values, name, indices):
    finite = values[np.isfinite(values)]
    return dict(
        var=name,
        index=indices,
        nx=values.shape[1],
        ny=values.shape[0],
        min=float(finite.min()) if finite.size else None,
        max=float(finite.max()) if finite.size else None,
        mean=float(np.mean(finite, dtype=np.longdouble)) if finite.size else None,
        missing_count=values.size - finite.size,
    )


def inspect_field(path):
    try:
        with dataset(path) as ds:
            available = list(variables(ds))
        default = next(
            (
                v
                for v in available
                if v["ny"] and v["nx"] and all(d["size"] for d in v["extra_dims"])
            ),
            None,
        )
        if default is None:
            raise ValueError("No numeric variable with a selectable (y, x) plane")
        field = dict(variables=available, default_var=default["name"])
        plane, indices = read_plane(path, field)
        return dict(**field, default_stats=stats(plane, default["name"], indices))
    except (OSError, RuntimeError, ValueError, TypeError, IndexError) as exc:
        raise invalid("file", f"Invalid NetCDF field: {exc}") from exc


def write_plane(path, values):
    with dataset(path, "w", format="NETCDF3_64BIT_DATA") as ds:
        ds.createDimension("y", values.shape[0])
        ds.createDimension("x", values.shape[1])
        ds.createVariable("u", "f8", ("y", "x"), fill_value=False)[:] = values


def read_frame(path, n):
    with dataset(path) as ds:
        return np.asarray(ds["u"][n], dtype=np.float64)
