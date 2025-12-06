import os
import subprocess
import uuid
from pathlib import Path

from celery import Celery

CELERY_BROKER = os.getenv("CELERY_BROKER", "redis://redis:6379/0")
celery = Celery("app.tasks", broker=CELERY_BROKER)

CORE_BIN = "/usr/local/bin/climate_sim"
MPI_SHARED = Path("/opt/mpi_shared")
HOSTFILE = MPI_SHARED / "hosts" / "hostfile"

MPIUSER_LOCAL_SSH_KEY = Path("/home/mpiuser/.ssh/id_rsa")


@celery.task(bind=True)
def run_simulation(
    self, config_path: str = None, np_procs: int = 4, extra_args: list | None = None
):
    job_id = str(uuid.uuid4())

    if not HOSTFILE.exists():
        return {"status": "failed", "reason": f"mpi hostfile not found at {HOSTFILE}"}

    ssh_agent_config = (
        f"ssh -i {MPIUSER_LOCAL_SSH_KEY} "
        "-o StrictHostKeyChecking=no "
        "-o UserKnownHostsFile=/dev/null"
    )

    cmd = [
        "mpirun",
        "--hostfile",
        str(HOSTFILE),
        "-np",
        str(np_procs),
        "--mca",
        "plm_rsh_agent",
        ssh_agent_config,
        CORE_BIN,
    ]

    if config_path:
        cmd.append(f"--config={config_path}")

    if extra_args:
        cmd.extend(extra_args)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {
            "status": "finished",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "job_id": job_id,
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "failed",
            "returncode": e.returncode,
            "stdout": e.stdout,
            "stderr": e.stderr,
        }
