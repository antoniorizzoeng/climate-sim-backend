import math
import sys
from decimal import Decimal
from typing import Annotated, Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
        # Optional properties may be omitted. Only fill_value is nullable.
        if isinstance(data, dict):
            for name, value in data.items():
                if value is None and name != "fill_value":
                    raise ValueError(f"{name} cannot be null")
        return data


class FieldSource(RequestModel):
    kind: Literal["field"]
    field_id: str
    var: str | None = None
    index: dict[str, Index] = Field(default_factory=dict)
    fill_value: float | None = None


class PresetSource(RequestModel):
    kind: Literal["preset"]
    preset: Preset
    nx: int = Field(ge=4)
    ny: int = Field(ge=4)


class BoundaryConfig(RequestModel):
    left: Boundary = "dirichlet"
    right: Boundary = "dirichlet"
    bottom: Boundary = "dirichlet"
    top: Boundary = "dirichlet"


class RunRequest(RequestModel):
    source: Annotated[FieldSource | PresetSource, Field(discriminator="kind")]
    dt: float = Field(gt=0)
    steps: int = Field(ge=1, le=ENGINE_INT_MAX)
    dx: float = Field(1.0, gt=0)
    dy: float = Field(1.0, gt=0)
    D: float = Field(0.0, ge=0)
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


def fits(ranks, nx, ny):
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


def initial_bytes(req, nx, ny):
    return 4096 + 8 * nx * ny if isinstance(req.source, FieldSource) else 0


def resolve(req, nx, ny, settings, available_bytes):
    errors = []
    if req.frames is not None and req.out_every is not None:
        errors.append(issue("frames", "frames and out_every are mutually exclusive"))
    for a, b in [(req.bc.left, req.bc.right), (req.bc.bottom, req.bc.top)]:
        if (a == "periodic") != (b == "periodic"):
            errors.append(
                issue("bc", "Periodic boundaries must be paired on opposite edges")
            )
    if not settings.periodic and "periodic" in req.bc.model_dump().values():
        errors.append(
            issue("bc", "Installed engine does not support periodic boundaries")
        )

    dt_max = stability_limit(req.dx, req.dy, req.D, req.vx, req.vy)
    if dt_max is not None and req.dt > dt_max:
        errors.append(issue("dt", f"dt exceeds the stability limit {dt_max}"))
    if not math.isfinite(req.dt * req.steps):
        errors.append(issue("dt", "Total simulation time exceeds float64 range"))
    if max(nx, ny) > ENGINE_INT_MAX:
        errors.append(
            issue("source", "Grid dimensions exceed the engine's signed integer range")
        )
    ranks = req.processes
    if ranks is None:
        ranks = min(settings.cpu_count, nx * ny)
        while ranks > 1 and not fits(ranks, nx, ny):
            ranks -= 1
    elif ranks > settings.cpu_count or not fits(ranks, nx, ny):
        errors.append(
            issue("processes", "MPI process grid must fit (nx, ny) and cpu_count")
        )

    intervals = (req.frames or 101) - 1
    every = req.out_every or ((req.steps + intervals - 1) // intervals)
    count = 1 + (req.steps + every - 1) // every
    size = 4096 + 8 * nx * ny * count
    resolved = dict(
        nx=nx,
        ny=ny,
        dt=req.dt,
        steps=req.steps,
        out_every=every,
        expected_frames=count,
        processes=ranks,
        dt_max=dt_max,
        estimated_output_bytes=size,
    )
    warnings = (
        [
            "Estimated output and staged input exceed available disk space "
            f"above the {settings.disk_margin_bytes}-byte safety margin"
        ]
        if size + initial_bytes(req, nx, ny) > available_bytes
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
        resolved=None if errors else resolved,
    )


class NowcastParams(RequestModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius_km: float = Field(300, ge=50, le=1000)
    hours: int = Field(6, ge=1, le=12)
    resolution: int = Field(96, ge=16, le=256)
    blend_tau_hours: float = Field(DEFAULT_BLEND_TAU_HOURS, gt=0)

    @model_validator(mode="after")
    def check_projection(self):
        if abs(self.lat) + math.degrees(self.radius_km / 6371) >= 90:
            raise ValueError("Local equirectangular grid must stay short of the poles")
        return self
