"""Copy data out before releasing the lock. libnetcdf/HDF5 calls cannot overlap."""

from contextlib import contextmanager
from math import prod
from mmap import ACCESS_READ, mmap
from threading import Lock

import netCDF4
import numpy as np

from .disk import CHUNK_BYTES, DiskSpaceError
from .models import invalid

_nc_lock = Lock()


@contextmanager
def dataset(path, *args, **kwargs):
    with _nc_lock, netCDF4.Dataset(path, *args, **kwargs) as ds:
        ds.set_auto_maskandscale(False)
        yield ds


def unpack(variable, selection):
    raw = np.asarray(variable[selection])
    attrs = variable.ncattrs()
    missing = [
        variable.getncattr(attr)
        for attr in ("_FillValue", "missing_value")
        if attr in attrs
    ]
    if getattr(variable, "_Unsigned", "") == "true" and raw.dtype.kind == "i":
        packed = raw.view(
            np.dtype(f"u{raw.itemsize}").newbyteorder(raw.dtype.byteorder)
        )
    else:
        packed = raw
    values = packed.astype(np.float64, copy=False)
    for sentinel in missing:
        values[np.isin(raw, sentinel)] = np.nan
    with np.errstate(invalid="ignore", over="ignore"):
        if "scale_factor" in attrs:
            values *= variable.scale_factor
        if "add_offset" in attrs:
            values += variable.add_offset
    values[~np.isfinite(values)] = np.nan
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


def window_slices(nx, ny, window=None):
    if window is None:
        return slice(0, ny, 1), slice(0, nx, 1)
    i0, j0, w, h, stride = window
    end_i, end_j = nx if w is None else i0 + w, ny if h is None else j0 + h
    if i0 >= nx or j0 >= ny or end_i > nx or end_j > ny:
        raise invalid("window", "Window lies outside the plane", "query")
    # Bound the effective stride by the axis length before passing it to C.
    return slice(j0, end_j, min(stride, ny)), slice(i0, end_i, min(stride, nx))


def selected_indices(field, name, index):
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
    return var, indices


def descending(variable, dim):
    coord = coordinate(variable.group(), dim)
    if coord is None or len(dim) < 2 or np.dtype(coord.dtype).kind not in "iuf":
        return False
    first, last = unpack(coord, [0, len(dim) - 1])
    if not first > last:
        return False
    cells = CHUNK_BYTES // 8
    for start in range(0, len(dim) - 1, cells):
        positions = unpack(coord, slice(start, start + cells + 1))
        if not np.all(positions[1:] <= positions[:-1]):
            return False
    return True


def reverse_slice(selection, length):
    last = (
        selection.start
        + (selection.stop - 1 - selection.start) // selection.step * selection.step
    )
    return slice(length - 1 - last, length - selection.start, selection.step)


def read_spatial(variable, leading, spatial, reversed_axes):
    selection = tuple(
        reverse_slice(part, length) if reverse else part
        for part, length, reverse in zip(spatial, variable.shape[-2:], reversed_axes)
    )
    values = unpack(variable, leading + selection)
    return np.flip(
        values, axis=tuple(i for i, reverse in enumerate(reversed_axes) if reverse)
    )


def declared_bytes(group):
    total = 0
    for var in group.variables.values():
        cells = prod(var.shape)
        itemsize = np.dtype(var.dtype).itemsize or 8
        chunks = var.chunking()
        # A single HDF5 chunk may also declare padding beyond the logical array.
        total += (
            max(
                cells,
                prod(chunks) if cells and isinstance(chunks, (list, tuple)) else 0,
            )
            * itemsize
        )
    return total + sum(declared_bytes(g) for g in group.groups.values())


def check_decoded_size(ds, disk):
    if disk is not None and ds.data_model.startswith("NETCDF4"):
        disk.check(declared_bytes(ds), "declared decoded NetCDF data")


def read_plane(path, field, name=None, index=None, window=None, disk=None):
    var, indices = selected_indices(field, name, index)
    spatial = window_slices(var["nx"], var["ny"], window)
    with dataset(path) as ds:
        check_decoded_size(ds, disk)
        variable = ds[var["name"]]
        leading = tuple(indices.values())
        reversed_axes = tuple(descending(variable, d) for d in variable.get_dims()[-2:])
        values = read_spatial(variable, leading, spatial, reversed_axes)
    return values, indices


def rectangles(ny, nx, cells=CHUNK_BYTES // 8):
    # Keep scratch arrays bounded even when a single row is very wide.
    columns = min(cells, nx)
    rows = cells // columns
    for j in range(0, ny, rows):
        for i in range(0, nx, columns):
            yield slice(j, min(j + rows, ny)), slice(i, min(i + columns, nx))


def summarize(blocks, ny, nx, name, indices):
    count = 0
    low, high = np.inf, -np.inf
    total = np.longdouble(0)
    for block in blocks:
        finite = np.isfinite(block)
        count += int(np.count_nonzero(finite))
        low = min(low, np.min(block, where=finite, initial=np.inf))
        high = max(high, np.max(block, where=finite, initial=-np.inf))
        total += np.sum(block, where=finite, dtype=np.longdouble)
    return dict(
        var=name,
        index=indices,
        nx=nx,
        ny=ny,
        min=float(low) if count else None,
        max=float(high) if count else None,
        mean=float(total / count) if count else None,
        missing_count=ny * nx - count,
    )


def stats(values, name, indices):
    ny, nx = values.shape
    return summarize(
        (values[rect] for rect in rectangles(ny, nx)), ny, nx, name, indices
    )


def variable_stats(ds, var, indices):
    variable = ds[var["name"]]
    ny, nx = var["ny"], var["nx"]
    leading = tuple(indices.values())
    blocks = (unpack(variable, leading + rect) for rect in rectangles(ny, nx))
    return summarize(blocks, ny, nx, var["name"], indices)


def field_stats(path, field, name, index, disk):
    var, indices = selected_indices(field, name, index)
    with dataset(path) as ds:
        check_decoded_size(ds, disk)
        return variable_stats(ds, var, indices)


def inspect_field(path, disk=None):
    try:
        with dataset(path) as ds:
            check_decoded_size(ds, disk)
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
            indices = {d["name"]: 0 for d in default["extra_dims"]}
            return dict(
                variables=available,
                default_var=default["name"],
                default_stats=variable_stats(ds, default, indices),
            )
    except DiskSpaceError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError, IndexError) as exc:
        raise invalid("file", f"Invalid NetCDF field: {exc}") from exc


def write_plane(path, values, *, drawn=False):
    ny, nx = values.shape
    with dataset(path, "w", format="NETCDF3_64BIT_DATA") as ds:
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        variable = ds.createVariable("u", "f8", ("y", "x"), fill_value=False)
        for rect in rectangles(ny, nx):
            block = values[rect]
            if drawn and not np.isfinite(block).all():
                raise invalid("data", "Drawn values must all be finite")
            variable[rect] = block


def write_drawing(path, raw, shape):
    # Map the file for bounded, rectangular conversion via write_plane.
    with raw.open("rb") as stream, mmap(
        stream.fileno(), 0, access=ACCESS_READ
    ) as mapped:
        values = np.ndarray(shape, dtype="<f4", buffer=mapped)
        write_plane(path, values, drawn=True)


def frame_count(path):
    with dataset(path) as ds:
        return ds["u"].shape[0]


def read_frame(path, n, window=None):
    with dataset(path) as ds:
        var = ds["u"]
        if window is None or window == (0, 0, None, None, 1):
            return np.asarray(var[n])
        ny, nx = var.shape[-2:]
        return np.asarray(var[(n,) + window_slices(nx, ny, window)])
