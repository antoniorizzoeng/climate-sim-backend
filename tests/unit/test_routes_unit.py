from unittest.mock import patch

from app.routes import router
from app.tasks import run_simulation


def test_simulate_route_triggers_task_hardcoded():
    config_yaml_string = """
grid:
  nx: 512
  ny: 512
  dx: 1.0
  dy: 1.0
physics:
  D: 0.05
  vx: 0.5
  vy: 0.0
time:
  dt: 0.1
  steps: 1000
  out_every: 100
bc:
  left: dirichlet
  right: neumann
  bottom: periodic
  top: dirichlet
output:
  prefix: "dev"
ic:
  preset: gaussian_hotspot
  file: "inputs/ic_global.nc"
  params:
    A: 1.0
    sigma_frac: 0.05
    xc_frac: 0.5
    yc_frac: 0.5
"""
    fake_task = type("T", (), {"id": "123"})()

    with patch.object(
        run_simulation, "delay", return_value=fake_task
    ) as mock_delay, patch("builtins.open"):

        func = None

        for r in router.routes:
            if r.path == "/simulate" and r.methods == {"POST"}:
                func = r.endpoint
                break

        if not func:
            raise Exception("Could not find the /simulate POST route function.")

        res = func(payload={"config_yaml": config_yaml_string})

        mock_delay.assert_called_once()

        assert res["task_id"] == "123"
