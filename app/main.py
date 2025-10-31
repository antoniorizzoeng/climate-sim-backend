from fastapi import FastAPI

from .routes import router

app = FastAPI(title="Climate Simulator API")
app.include_router(router)


@app.get("/")
def root():
    return {"message": "Climate Simulation Backend is running 🚀"}
