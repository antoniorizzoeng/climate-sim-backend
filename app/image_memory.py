from math import prod

import psutil

from .memory import cgroup_directories

# Allow for NetCDF reads and unpacking, projection accumulators, and PNG
# conversion buffers. A masked, packed field can need several arrays per cell.
SOURCE_BYTES_PER_CELL = 64
OUTPUT_BYTES_PER_PIXEL = 16


class ImageTooLarge(Exception):
    pass


def available_ram():
    available = psutil.virtual_memory().available
    # A container can have less headroom than the host reports. On cgroup v2,
    # a parent also limits the memory available to its descendants.
    for directory in cgroup_directories():
        try:
            limit = (directory / "memory.max").read_text().strip()
            if limit != "max":
                used = int((directory / "memory.current").read_text())
                available = min(available, max(0, int(limit) - used))
        except OSError:
            continue
    return available


def native_image_budget(nx, ny, ram_fraction):
    required = nx * ny * SOURCE_BYTES_PER_CELL
    limit = int(available_ram() * ram_fraction)
    if required >= limit:
        raise ImageTooLarge(
            f"Native PNG requires an estimated {required} bytes for the decoded "
            f"plane and rendering, exceeding the {limit}-byte RAM limit. "
            "Request a smaller max_side."
        )
    return limit - required


def bound_chunk_cache(variable, budget):
    chunks = variable.chunking()
    if not isinstance(chunks, (list, tuple)):
        return 0
    chunk_bytes = prod(chunks) * variable.dtype.itemsize
    # Allow for the current chunk, cached chunks, and temporary HDF5 filter
    # buffers. A hyperslab cannot avoid decompressing a whole filtered chunk.
    cache = min(max(chunk_bytes, budget // 16), 8 * 1024**2)
    required = cache + 3 * chunk_bytes
    if required >= budget:
        raise ImageTooLarge(
            f"A NetCDF chunk requires an estimated {required} bytes to decode, "
            f"exceeding the {budget} bytes available for NetCDF reads. "
            "Request a smaller max_side or use a file with smaller NetCDF chunks."
        )
    variable.set_var_chunk_cache(size=cache)
    return required
