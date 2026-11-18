import math
import sys
from decimal import Decimal
from typing import Annotated, Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import StringConstraints

from .memory import estimate_engine_memory

Boundary = Literal["dirichlet", "neumann", "periodic"]
Preset = Literal["gaussian_hotspot", "constant_zero"]
Colormap = Literal[
    "viridis", "inferno", "magma", "plasma", "cividis", "turbo", "coolwarm", "gray"
]
Index = Annotated[int, Field(ge=0)]
ENGINE_INT_MAX = 2**31 - 1
DEFAULT_BLEND_TAU_HOURS = 2.0


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def reject_null(cls, data):
        # Optional properties may be omitted. Explicit null needs contract support.
        if isinstance(data, dict):
            for name, value in data.items():
                if value is None and name not in ("fill_value", "elevation_m"):
                    raise ValueError(f"{name} cannot be null")
        return data


class FieldSource(RequestModel):
    kind: Literal["field"]
    field_id: str
    var: str | None = None
    index: dict[str, Index] = Field(default_factory=dict)
    fill_value: float | None = None
    volume: bool = False
    replicate_z: int = Field(1, ge=1)


class PresetSource(RequestModel):
    kind: Literal["preset"]
    preset: Preset
    nx: int = Field(ge=4)
    ny: int = Field(ge=4)
    nz: int = Field(1, ge=1)


class BoundaryConfig(RequestModel):
    left: Boundary = "dirichlet"
    right: Boundary = "dirichlet"
    bottom: Boundary = "dirichlet"
    top: Boundary = "dirichlet"
    front: Boundary = "dirichlet"
    back: Boundary = "dirichlet"


DisplayName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)
]


class RenameRequest(RequestModel):
    name: DisplayName


class Units(RequestModel):
    length: Literal["cell", "m", "km"]
    time: Literal["t", "s", "h"]

    @classmethod
    def grid(cls):
        return cls(length="cell", time="t")


class RunRequest(RequestModel):
    units: Units = Field(default_factory=Units.grid)
    name: DisplayName | None = None
    source: Annotated[FieldSource | PresetSource, Field(discriminator="kind")]
    dt: float = Field(gt=0)
    steps: int = Field(ge=1, le=ENGINE_INT_MAX)
    dx: float = Field(1.0, gt=0)
    dy: float = Field(1.0, gt=0)
    dz: float = Field(1.0, gt=0)
    D: float = Field(0.0, ge=0)
    Dz: float = Field(default_factory=lambda data: data["D"], ge=0)
    vz: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    out_every: int | None = Field(None, ge=1)
    frames: int | None = Field(None, ge=2)
    bc: BoundaryConfig = Field(default_factory=BoundaryConfig)
    processes: int | None = Field(None, ge=1)


def issue(name, message, where="body"):
    return {"loc": [where, *name.split(".")], "msg": message, "type": "value_error"}


def invalid(name, message, where="body"):
    return HTTPException(422, [issue(name, message, where)])


def fits(ranks, nx, ny, nz=1):
    if nz > 1:
        # The volume engine tries alternative factorizations for thin grids.
        return any(
            ranks % x == 0 and ranks // x % y == 0 and ranks // (x * y) <= nz
            for x in range(1, min(nx, ranks) + 1)
            for y in range(1, min(ny, ranks // x) + 1)
        )
    # Open MPI distributes prime factors largest first to the shorter axis.
    factors = []
    divisor = 2
    remaining = ranks
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            factors.append(divisor)
            remaining //= divisor
        divisor += 1
    if remaining > 1:
        factors.append(remaining)
    short = long = 1
    for prime in reversed(factors):
        short, long = sorted((short * prime, long))
    return long <= nx and short <= ny


def stability_limit(dx, dy, diffusion, vx, vy):
    if not (vx or vy or diffusion):
        return None

    def inverse_rate(dx, dy, diffusion, vx, vy):
        rate = abs(vx) / dx + abs(vy) / dy
        if diffusion:
            rate += 2 * diffusion * (1 / (dx * dx) + 1 / (dy * dy))
        return 1 / rate

    args = (dx, dy, diffusion, vx, vy)
    try:
        limit = inverse_rate(*args)
        if 0 < limit < math.inf:
            return limit
    except ZeroDivisionError:
        pass
    # Intermediate overflow or underflow can occur for finite inputs. Compute
    # the same expression with a wider exponent range, then return a JSON number.
    return min(float(inverse_rate(*map(Decimal, args))), sys.float_info.max)


def volume_stability(req):
    # Evaluate the engine's combined rate with an extended exponent range.
    dx, dy, dz = map(Decimal, (req.dx, req.dy, req.dz))
    D, Dz = map(Decimal, (req.D, req.Dz))
    rate = (
        abs(Decimal(req.vx)) / dx
        + abs(Decimal(req.vy)) / dy
        + abs(Decimal(req.vz)) / dz
        + 2 * (D / dx**2 + D / dy**2 + Dz / dz**2)
    )
    return min(float(1 / rate), sys.float_info.max) if rate else None


def initial_bytes(source, nx, ny, nz):
    if not isinstance(source, FieldSource):
        return 0
    layers = 1 if source.replicate_z > 1 else nz
    return 4096 + 8 * nx * ny * layers


def automatic_processes(nx, ny, nz, cpu_count):
    """Allow one rank per 131,072 cells (at least one), capped by cpu_count.
    Reduce until the engine can factor the ranks across the spatial axes.
    """
    ranks = min(cpu_count, max(1, nx * ny * nz // 131072))
    while not fits(ranks, nx, ny, nz):
        ranks -= 1
    return ranks


def header_extent(source_units):
    """PnetCDF alignment for the engine header and the added CF descriptors."""
    return 4096 * (1 + (len(source_units.encode()) + 4095) // 4096)


def resolve(
    req, nx, ny, settings, available_bytes, nz=1, source_units="1", *, wind_components=0
):
    errors = []
    if req.frames is not None and req.out_every is not None:
        errors.append(issue("frames", "frames and out_every are mutually exclusive"))
    for a, b in [
        (req.bc.left, req.bc.right),
        (req.bc.bottom, req.bc.top),
        (req.bc.front, req.bc.back),
    ]:
        if (a == "periodic") != (b == "periodic"):
            errors.append(
                issue("bc", "Periodic boundaries must be paired on opposite edges")
            )
    if not settings.periodic and "periodic" in req.bc.model_dump().values():
        errors.append(
            issue("bc", "Installed engine does not support periodic boundaries")
        )

    dt_max = (
        volume_stability(req)
        if nz > 1
        else stability_limit(req.dx, req.dy, req.D, req.vx, req.vy)
    )
    if dt_max is not None and req.dt > dt_max:
        errors.append(issue("dt", f"dt exceeds the stability limit {dt_max}"))
    if not math.isfinite(req.dt * req.steps):
        errors.append(issue("dt", "Total simulation time exceeds float64 range"))
    if max(nx, ny, nz) > ENGINE_INT_MAX:
        errors.append(
            issue("source", "Grid dimensions exceed the engine's signed integer range")
        )
    ranks = req.processes
    if ranks is None:
        ranks = automatic_processes(nx, ny, nz, settings.cpu_count)
    elif ranks > settings.cpu_count or not fits(ranks, nx, ny, nz):
        errors.append(
            issue("processes", "MPI process grid must fit the grid and cpu_count")
        )

    intervals = (req.frames or 101) - 1
    every = req.out_every or ((req.steps + intervals - 1) // intervals)
    count = 1 + (req.steps + every - 1) // every
    coordinates = nx + ny + (nz if nz > 1 else 0) + count
    size = header_extent(source_units) + 8 * (nx * ny * nz * count + coordinates)
    memory = estimate_engine_memory(nx, ny, nz, ranks, wind_components)
    capacity = settings.engine_memory_capacity()
    resolved = dict(
        nx=nx,
        ny=ny,
        nz=nz,
        dz=req.dz,
        dt=req.dt,
        steps=req.steps,
        out_every=every,
        expected_frames=count,
        processes=ranks,
        dt_max=dt_max,
        estimated_output_bytes=size,
        estimated_memory_bytes=memory,
        memory_capacity_bytes=capacity,
    )
    if memory > capacity:
        errors.append(
            issue(
                "source",
                f"Needs about {Decimal(memory) / 10**9:.1f} GB of memory; "
                f"this server can give the engine {Decimal(capacity) / 10**9:.1f} GB. "
                "Use fewer layers or a smaller grid.",
            )
        )
    elif errors:
        resolved = None
    warnings = (
        [
            "Estimated output and staged input exceed available disk space "
            f"above the {settings.disk_margin_bytes}-byte safety margin"
        ]
        if size + initial_bytes(req.source, nx, ny, nz) > available_bytes
        else []
    )
    if every > ENGINE_INT_MAX:
        warnings.append(
            "The engine stores out_every as INT_MAX; both intervals write only the initial and final frames"
        )
    return dict(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        resolved=resolved,
    )


class NowcastParams(RequestModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius_km: float = Field(300, ge=50, le=1000)
    window: Literal["now", "tonight"] = "now"
    hours: int = Field(6, ge=1, le=12)
    resolution: int = Field(96, ge=16, le=256)
    blend_tau_hours: float = Field(DEFAULT_BLEND_TAU_HOURS, gt=0)
    elevation_m: float | None = Field(None, ge=-500, le=9000)

    @model_validator(mode="after")
    def check_projection(self):
        if abs(self.lat) + math.degrees(self.radius_km / 6371) >= 90:
            raise ValueError("Local equirectangular grid must stay short of the poles")
        return self
