from fastapi import APIRouter
from .tasks import run_simulation

router = APIRouter()

@router.post("/simulate")
def simulate():
    task = run_simulation.delay()
    return {"task_id": task.id}
