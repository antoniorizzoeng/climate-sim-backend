from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .events import RequestLogMiddleware
from .manager import RunManager, utcnow
from .routes import router
from .settings import Settings

CONTRACT = Path(__file__).resolve().parents[1] / "api" / "openapi.yaml"


def create_app(settings=None, clock=utcnow):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application):
        manager = RunManager(settings, clock)
        await manager.start()
        application.state.manager = manager
        try:
            yield
        finally:
            await manager.close()

    api = FastAPI(title="climate-sim API", version="1.3.2", lifespan=lifespan)
    api.include_router(router)
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[origin for origin in settings.origins if origin != "*"],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
        expose_headers=[
            "X-Nx",
            "X-Ny",
            "X-Nz",
            "X-I0",
            "X-J0",
            "X-Stride",
            "Retry-After",
            "Content-Disposition",
            "X-Queue-Length",
            "X-Active-Run-Id",
            "X-Disk-Free-Bytes",
        ],
    )

    api.add_middleware(RequestLogMiddleware)

    @api.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(
            {
                "detail": [
                    {k: e[k] for k in ("loc", "type", "msg")} for e in exc.errors()
                ]
            },
            status_code=422,
        )

    @api.exception_handler(Exception)
    async def server_error(request: Request, exc: Exception):
        return JSONResponse({"detail": "Internal server error"}, status_code=500)

    # Serve the canonical schema, including deepObject queries and binary media types.
    with CONTRACT.open() as stream:
        api.openapi_schema = yaml.safe_load(stream)
    return api


app = create_app()
