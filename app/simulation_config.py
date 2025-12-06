from typing import Literal, Optional

from pydantic import BaseModel


class GridConfig(BaseModel):
    nx: int
    ny: int
    dx: float
    dy: float


class PhysicsConfig(BaseModel):
    D: float
    vx: float
    vy: float


class TimeConfig(BaseModel):
    dt: float
    steps: int
    out_every: int


class BoundaryConfig(BaseModel):
    left: Literal["dirichlet", "neumann", "periodic"]
    right: Literal["dirichlet", "neumann", "periodic"]
    bottom: Literal["dirichlet", "neumann", "periodic"]
    top: Literal["dirichlet", "neumann", "periodic"]


class OutputConfig(BaseModel):
    prefix: str


class ICParams(BaseModel):
    A: float
    sigma_frac: float
    xc_frac: float
    yc_frac: float


class ICConfig(BaseModel):
    preset: str
    file: Optional[str] = None
    params: Optional[ICParams] = None


class SimulationConfig(BaseModel):
    grid: GridConfig
    physics: PhysicsConfig
    time: TimeConfig
    bc: BoundaryConfig
    output: OutputConfig
    ic: ICConfig
