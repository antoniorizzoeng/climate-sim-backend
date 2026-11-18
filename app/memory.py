from collections.abc import Iterator
from pathlib import Path

import psutil

ENGINE_BYTES_PER_CELL = 28
ENGINE_FIXED_BYTES = 64 * 1024**2
ENGINE_BYTES_PER_RANK = 16 * 1024**2
WIND_BYTES_PER_CELL_COMPONENT = 8
DEFAULT_ENGINE_MEMORY_RESERVE_BYTES = 512 * 1024**2
CGROUP_ROOT = Path("/sys/fs/cgroup")
SELF_CGROUP = Path("/proc/self/cgroup")


def cgroup_directories() -> Iterator[Path]:
    try:
        groups = SELF_CGROUP.read_text().splitlines()
        relative = next(line[3:] for line in groups if line.startswith("0::"))
    except (OSError, StopIteration):
        return

    group = CGROUP_ROOT / relative.lstrip("/")
    for directory in (group, *group.parents):
        if directory != CGROUP_ROOT and CGROUP_ROOT not in directory.parents:
            break
        yield directory


def estimate_engine_memory(
    nx: int, ny: int, nz: int, ranks: int, wind_components: int = 0
) -> int:
    bytes_per_cell = (
        ENGINE_BYTES_PER_CELL + WIND_BYTES_PER_CELL_COMPONENT * wind_components
    )
    return (
        nx * ny * nz * bytes_per_cell
        + ENGINE_FIXED_BYTES
        + ENGINE_BYTES_PER_RANK * ranks
    )


def engine_memory_capacity(reserve: int, override: int | None) -> int:
    if override is not None:
        return override

    capacity = psutil.virtual_memory().total
    for directory in cgroup_directories():
        try:
            limit = (directory / "memory.max").read_text().strip()
            if limit != "max":
                capacity = min(capacity, int(limit))
        except OSError:
            continue
    return max(0, capacity - reserve)
