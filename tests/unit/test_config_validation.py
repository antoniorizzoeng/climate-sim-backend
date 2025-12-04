from pydantic import ValidationError

from app.simulation_config import SimulationConfig


def make_valid_config():
    return {
        "grid": {"nx": 512, "ny": 512, "dx": 1.0, "dy": 1.0},
        "physics": {"D": 0.05, "vx": 0.5, "vy": 0.0},
        "time": {"dt": 0.1, "steps": 1000, "out_every": 100},
        "bc": {
            "left": "dirichlet",
            "right": "neumann",
            "bottom": "periodic",
            "top": "dirichlet",
        },
        "output": {"prefix": "dev"},
        "ic": {
            "preset": "gaussian_hotspot",
            "file": "inputs/ic_global.nc",
            "params": {
                "A": 1.0,
                "sigma_frac": 0.05,
                "xc_frac": 0.5,
                "yc_frac": 0.5,
            },
        },
    }


def test_valid_config_parses():
    cfg = SimulationConfig(**make_valid_config())
    assert cfg.grid.nx == 512
    assert cfg.ic.preset == "gaussian_hotspot"


def test_invalid_boundary_raises():
    bad = make_valid_config()
    bad["bc"]["left"] = "wrong"

    try:
        SimulationConfig(**bad)
        assert False, "SimulationConfig must reject invalid boundary type"
    except ValidationError:
        pass


def test_missing_required_field():
    bad = make_valid_config()
    bad.pop("grid")

    try:
        SimulationConfig(**bad)
        assert False, "Missing grid must raise"
    except ValidationError:
        pass
