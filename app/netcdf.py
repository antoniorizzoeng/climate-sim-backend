"""Copy data out before releasing the lock. libnetcdf/HDF5 calls cannot overlap."""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from math import prod
from mmap import ACCESS_READ, mmap
from operator import getitem
from threading import Lock

import netCDF4
import numpy as np

from .disk import CHUNK_BYTES, DiskSpaceError
from .models import initial_bytes, invalid

_nc_lock = Lock()


@contextmanager
def dataset(path, *args, **kwargs):
    with _nc_lock, netCDF4.Dataset(path, *args, **kwargs) as ds:
        ds.set_auto_maskandscale(False)
        yield ds


def unpack(variable, selection):
    raw = np.asarray(variable[selection])
    attrs = variable.ncattrs()
    missing = []
    if "_FillValue" in attrs:
        missing.append(variable._FillValue)
    elif variable.get_fill_value() is not None:
        # libnetcdf returns the implicit value in native byte order, even for
        # a non-native variable. Use the declared type's NetCDF constant.
        missing.append(netCDF4.default_fillvals[raw.dtype.str[1:]])
    if "missing_value" in attrs:
        missing.append(variable.missing_value)
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


def vertical_metadata(variable):
    if variable.ndim >= 3:
        dim = variable.get_dims()[-3]
        coord = coordinate(variable.group(), dim)
        if (
            dim.name.lower()
            in {
                "z",
                "nz",
                "depth",
                "height",
                "altitude",
                "level",
                "lev",
                "layer",
                "plev",
                "pressure",
                "isobaricinhpa",
                "isobaricinpa",
            }
            or str(getattr(coord, "axis", "")).upper() == "Z"
            or str(getattr(coord, "positive", "")).lower() in {"up", "down"}
            or str(getattr(coord, "standard_name", ""))
            in ("depth", "height", "altitude", "air_pressure")
        ):
            return dict(nz=len(dim), z_dim=dim.name)
    return dict(nz=1)


def upgrade_field(path, field):
    with dataset(path) as ds:
        for var in field["variables"]:
            var.update(vertical_metadata(ds[var["name"]]))


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
            **vertical_metadata(var),
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


def selected_indices(field, name, index, *, volume=False):
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
    if volume and "z_dim" in var:
        indices[var["z_dim"]] = 0
    for dim in var["extra_dims"]:
        if not 0 <= indices[dim["name"]] < dim["size"]:
            raise invalid("source.index", f"Index outside dimension {dim['name']}")
    return var, indices


def descending(variable, dim, cells=CHUNK_BYTES // 8, configure=lambda _: None):
    coord = coordinate(variable.group(), dim)
    if coord is None or len(dim) < 2 or np.dtype(coord.dtype).kind not in "iuf":
        return False
    configure(coord)
    first, last = unpack(coord, [0, len(dim) - 1])
    if not first > last:
        return False
    for start in range(0, len(dim) - 1, cells):
        positions = unpack(coord, slice(start, start + cells + 1))
        if not np.all(positions[1:] <= positions[:-1]):
            return False
    return True


def reverse_slice(selection, length):
    step = selection.step or 1
    last = selection.start + (selection.stop - 1 - selection.start) // step * step
    return slice(length - 1 - last, length - selection.start, selection.step)


def field_reader(field):
    # Drawings contain only validated, unpacked finite doubles. Older CDF-5
    # drawings have no explicit fill attribute to distinguish valid fill numbers.
    return getitem if field.get("source") == "drawn" else unpack


def read_spatial(variable, leading, spatial, reversed_axes, read):
    selection = tuple(
        reverse_slice(part, length) if reverse else part
        for part, length, reverse in zip(spatial, variable.shape[-2:], reversed_axes)
    )
    values = read(variable, leading + selection)
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
        values = field_regions(variable, field, indices)(spatial)
    return values, indices


def field_regions(variable, field, indices, **coordinate_options):
    leading = tuple(indices.values())
    reversed_axes = tuple(
        descending(variable, dim, **coordinate_options)
        for dim in variable.get_dims()[-2:]
    )
    read = field_reader(field)

    def region(spatial):
        return read_spatial(variable, leading, spatial, reversed_axes, read)

    return region


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


def variable_stats(ds, var, indices, read):
    variable = ds[var["name"]]
    ny, nx = var["ny"], var["nx"]
    leading = tuple(indices.values())
    blocks = (read(variable, leading + rect) for rect in rectangles(ny, nx))
    return summarize(blocks, ny, nx, var["name"], indices)


def field_stats(path, field, name, index, disk):
    var, indices = selected_indices(field, name, index)
    with dataset(path) as ds:
        check_decoded_size(ds, disk)
        return variable_stats(ds, var, indices, field_reader(field))


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
                default_stats=variable_stats(ds, default, indices, unpack),
            )
    except DiskSpaceError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError, IndexError) as exc:
        raise invalid("file", f"Invalid NetCDF field: {exc}") from exc


def prepare_input(path, field, source, disk, target=None):
    """Validate or stage one IC in bounded tiles, holding the NetCDF lock."""
    var, indices = selected_indices(
        field, source.var, source.index, volume=source.volume
    )
    layers = var["nz"] if source.volume else 1
    volume = layers > 1
    ny, nx = var["ny"], var["nx"]
    with dataset(path) as ds, ExitStack() as stack:
        check_decoded_size(ds, disk)
        variable = ds[var["name"]]
        leading = tuple(indices.values())
        dest = None
        if target is not None:
            size = initial_bytes(source, nx, ny, layers)
            stack.enter_context(disk.reserve(size, "staged input", target))
            out = stack.enter_context(
                netCDF4.Dataset(target, "w", format="NETCDF3_64BIT_DATA")
            )
            if volume:
                out.createDimension("z", layers)
            out.createDimension("y", ny)
            out.createDimension("x", nx)
            dest = out.createVariable(
                "u", "f8", tuple(out.dimensions), fill_value=False
            )
        dims = variable.get_dims()
        flip = tuple(descending(variable, d) for d in dims[-2:])
        reverse_z = volume and descending(variable, dims[-3])
        read = field_reader(field)
        for k in range(layers):
            if volume:
                input_z = layers - 1 - k if reverse_z else k
                leading = (*leading[:-1], input_z)
            output_layer = (k,) if volume else ()
            for rect in rectangles(ny, nx):
                block = read_spatial(variable, leading, rect, flip, read)
                missing = ~np.isfinite(block)
                if missing.any():
                    if source.fill_value is None:
                        raise invalid(
                            "source.fill_value", "Selected field contains missing cells"
                        )
                    block[missing] = source.fill_value
                if dest is not None:
                    dest[output_layer + rect] = block


def write_plane(path, values, *, drawn=False):
    ny, nx = values.shape
    with dataset(path, "w", format="NETCDF3_64BIT_DATA") as ds:
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        # CDF-5 does not persist no-fill mode. An explicit NaN sentinel keeps
        # every finite drawing value valid when another NetCDF reader opens it.
        variable = ds.createVariable(
            "u", "f8", ("y", "x"), fill_value=np.nan if drawn else False
        )
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


@dataclass(frozen=True)
class FrameView:
    axis: str = "z"
    index: int | None = None
    projection: str = "none"

    def resolve(self, shape):
        if shape[0] == 1:
            return FrameView()
        extent = shape["zyx".index(self.axis)]
        if self.projection != "none":
            return FrameView(self.axis, None, self.projection)
        index = extent // 2 if self.index is None else self.index
        if not 0 <= index < extent:
            raise invalid("slice", f"Slice outside the {self.axis} axis", "query")
        return FrameView(self.axis, index)

    def plane_shape(self, shape):
        axis = "zyx".index(self.axis)
        return shape[:axis] + shape[axis + 1 :]


def read_frame(path, n, window=None, view=FrameView(), cancelled=None):
    with dataset(path) as ds:
        var = ds["u"]
        return frame_plane(var, n, window, view, cancelled)


def read_image_plane(path, n, view, auto_range, cancelled=None):
    with dataset(path) as ds:
        var = ds["u"]
        plane = frame_plane(var, n, None, view, cancelled)
        if var.ndim == 3:
            return plane, True
        # A uniform projection or slice can hide variation along the viewing axis.
        value = plane.flat[0]
        uniform = auto_range and np.isfinite(value) and np.all(plane == value)
        return plane, uniform and volume_matches(var, n, value, cancelled=cancelled)


def volume_matches(var, n, value, cells=CHUNK_BYTES // 8, cancelled=None):
    for k in range(var.shape[1]):
        if cancelled is not None and cancelled.is_set():
            raise InterruptedError("Export cancelled")
        for rect in rectangles(*var.shape[-2:], cells=cells):
            if not np.all(var[(n, k) + rect] == value):
                return False
    return True


def frame_regions(var, n, view, cancelled=None):
    shape = var.shape[1:]
    if var.ndim == 3:
        return shape, lambda spatial: np.asarray(var[(n,) + spatial])
    view = view.resolve(shape)
    axis = "zyx".index(view.axis)

    def region(spatial):
        def plane(k):
            return np.asarray(var[(n,) + spatial[:axis] + (k,) + spatial[axis:]])

        if view.projection == "none":
            return plane(view.index or 0)
        reduced = plane(0)
        if view.projection == "mean":
            reduced = reduced.astype(np.longdouble)
        for k in range(1, shape[axis]):
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("Export cancelled")
            if view.projection == "max":
                np.maximum(reduced, plane(k), out=reduced)
            else:
                reduced += plane(k)
        if view.projection == "mean":
            reduced /= shape[axis]
            return reduced.astype(np.float64)
        return reduced

    return view.plane_shape(shape), region


def frame_plane(var, n, window, view, cancelled):
    (ny, nx), region = frame_regions(var, n, view, cancelled)
    return region(window_slices(nx, ny, window))


def read_volume(path, n, max_side):
    with dataset(path) as ds:
        var = ds["u"]
        nz, ny, nx = (1, *var.shape[-2:]) if var.ndim == 3 else var.shape[1:]
        stride = (max(nx, ny, nz) + max_side - 1) // max_side
        layers = range(0, nz, stride)
        result = np.empty(
            (len(layers), len(range(0, ny, stride)), len(range(0, nx, stride))),
            dtype="<f4",
        )
        for i, k in enumerate(layers):
            leading = (n,) if var.ndim == 3 else (n, k)
            result[i] = var[leading + (slice(None, None, stride),) * 2]
        return result


def initial_range(path, boundaries):
    low, high = np.inf, -np.inf
    with dataset(path) as ds:
        var = ds["u"]
        volume = var.ndim == 4
        for k in range(var.shape[1] if volume else 1):
            leading = (0, k) if volume else (0,)
            for rect in rectangles(*var.shape[-2:]):
                block = np.asarray(var[leading + rect])
                finite = np.isfinite(block)
                low = min(low, block.min(where=finite, initial=np.inf))
                high = max(high, block.max(where=finite, initial=-np.inf))
    if low > high:
        return None
    edges = ["left", "right", "bottom", "top"]
    if volume:
        edges.extend(("front", "back"))
    if any(boundaries.get(edge, "dirichlet") == "dirichlet" for edge in edges):
        low, high = min(low, 0), max(high, 0)
    return [float(low), float(high)]
