from .worker import celery

@celery.task(bind=True)
def run_simulation(self):
    self.update_state(state="STARTED")
    return {"status": "completed", "file": "output.nc"}
