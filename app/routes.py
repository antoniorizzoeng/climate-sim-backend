import uuid
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException

from .simulation_config import SimulationConfig
from .tasks import run_simulation

MPI_SHARED = Path("/opt/mpi_shared")

router = APIRouter()


@router.post("/simulate")
def simulate(payload: dict):
    if "config_yaml" not in payload:
        raise HTTPException(400, "Missing config_yaml")

    raw_yaml = payload["config_yaml"]

    try:
        data = yaml.safe_load(raw_yaml)
        SimulationConfig(**data)
    except Exception as e:
        raise HTTPException(400, f"Invalid YAML or schema: {e}")

    job_id = str(uuid.uuid4())
    yaml_path = MPI_SHARED / f"{job_id}.yaml"

    try:
        with open(yaml_path, "w") as f:
            f.write(raw_yaml)
    except Exception as e:
        raise HTTPException(500, f"Failed to write config file: {e}")

    task = run_simulation.delay(config_path=str(yaml_path), np_procs=4, extra_args=None)

    return {"task_id": task.id, "job_id": job_id, "config_file": str(yaml_path)}
