import asyncio
import base64
import binascii
import math
import os
import re
from pathlib import Path
from typing import Annotated, Literal, get_args
from urllib.parse import quote

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as PathParam
from fastapi import Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from . import netcdf
from .disk import CHUNK_BYTES, DiskSpaceError
from .manager import lookup
from .models import Colormap, NowcastParams, Preset, RunRequest, invalid
from .render import Style, png
from .uploads import read_form

router = APIRouter()
RunIndex = Annotated[int, PathParam(ge=0)]


def manager(request: Request):
    return request.app.state.manager


def style(
    colormap: Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    max_side: Annotated[int | None, Query(ge=16)] = None,
):
    if (vmin is None) != (vmax is None) or (
        vmin is not None
        and not (math.isfinite(vmin) and math.isfinite(vmax) and vmin < vmax)
    ):
        raise invalid(
            "vmin", "Supply finite vmin and vmax together, with vmin < vmax", "query"
        )
    return Style(colormap, vmin, vmax, max_side)


def indices(request: Request):
    result = {}
    for key, value in request.query_params.multi_items():
        if key == "index" or key.startswith("index["):
            match = re.fullmatch(r"index\[([^\]]+)\]", key)
            if not match or not re.fullmatch("[0-9]+", value) or match[1] in result:
                raise invalid(
                    "index",
                    "Use index[dimension]=nonnegative_integer once per dimension",
                    "query",
                )
            significant = value.lstrip("0")
            if len(significant) > 20:
                raise invalid(
                    "index", "Index outside supported dimension range", "query"
                )
            result[match[1]] = int(significant or "0")
    return result


def frame_view(
    axis: Literal["z", "y", "x"] = "z",
    index: Annotated[int | None, Query(alias="slice", ge=0)] = None,
    projection: Literal["none", "max", "mean"] = "none",
):
    return netcdf.FrameView(axis, index, projection)


def window(
    i0: Annotated[int, Query(ge=0)] = 0,
    j0: Annotated[int, Query(ge=0)] = 0,
    w: Annotated[int | None, Query(ge=1)] = None,
    h: Annotated[int | None, Query(ge=1)] = None,
    stride: Annotated[int, Query(ge=1)] = 1,
):
    return (i0, j0, w, h, stride)


def values_response(grid, win):
    i0, j0, _, _, stride = win
    headers = dict(
        zip(
            ("X-Nx", "X-Ny", "X-I0", "X-J0", "X-Stride"),
            map(str, (grid.shape[1], grid.shape[0], i0, j0, stride)),
        )
    )
    values = np.ascontiguousarray(grid, dtype="<f4")
    return Response(
        memoryview(values).cast("B"),
        headers=headers,
        media_type="application/octet-stream",
    )


def page_params(
    limit: Annotated[int, Query(ge=1, le=200)] = 50, cursor: str | None = None
):
    after = None
    if cursor is not None:
        try:
            stamp, identifier = (
                base64.b64decode(cursor, altchars=b"-_", validate=True)
                .decode()
                .split("|")
            )
            after = (stamp, identifier)
        except (ValueError, binascii.Error) as exc:
            raise invalid("cursor", "Invalid pagination cursor", "query") from exc
    return limit, after


def paginate(collection, params):
    def key(item):
        return item["created_at"], item["id"]

    limit, after = params
    ordered = sorted(
        (i for i in collection.records.values() if after is None or key(i) < after),
        key=key,
        reverse=True,
    )
    selected = ordered[:limit]
    cursor = None
    if len(ordered) > limit:
        cursor = base64.urlsafe_b64encode("|".join(key(selected[-1])).encode()).decode()
    return dict(items=selected, next_cursor=cursor)


class Download(StreamingResponse):
    def __init__(self, path, name, media):
        # An open descriptor keeps the contents readable across retention/DELETE.
        self.stream = path.open("rb")
        super().__init__(
            self.chunks(),
            media_type=media,
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(name, safe='')}",
                "Content-Length": str(os.fstat(self.stream.fileno()).st_size),
            },
        )

    async def chunks(self):
        while block := await run_in_threadpool(self.stream.read, CHUNK_BYTES):
            yield block

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.stream.close()


M = Depends(manager)


@router.get("/health")
async def health(m=M):
    return JSONResponse(
        dict(status="ok", engine=m.description),
        headers={
            "X-Queue-Length": str(len(m.queue)),
            "X-Active-Run-Id": m.active or "",
            "X-Disk-Free-Bytes": str(m.disk.free),
        },
    )


@router.get("/config")
async def server_config(m=M):
    boundaries = ["dirichlet", "neumann"]
    if m.settings.periodic:
        boundaries.append("periodic")
    return dict(
        cpu_count=m.settings.cpu_count,
        default_processes=m.settings.cpu_count,
        colormaps=get_args(Colormap),
        presets=get_args(Preset),
        supported_boundary_conditions=boundaries,
        disk_free_bytes=m.disk.free,
        retention=dict(active_days=7, archive_days=30),
    )


async def receive_field(request, m, drawn=False):
    with m.fields.staging() as (identifier, directory):
        name, shape = await read_form(request, directory / "upload.part", m.disk, drawn)
        source = "drawn" if drawn else "upload"
        return await m.store_field(identifier, directory, source, name, shape)


@router.post("/fields", status_code=201, responses={422: {}})
async def upload(request: Request, m=M):
    return await receive_field(request, m)


@router.post("/fields/drawn", status_code=201, responses={422: {}})
async def drawn(request: Request, m=M):
    return await receive_field(request, m, drawn=True)


@router.get("/fields")
async def list_fields(params=Depends(page_params), m=M):
    return paginate(m.fields, params)


@router.get("/fields/{fieldId}", responses={404: {}})
async def get_field(fieldId: str, m=M):
    return lookup(m.fields, fieldId)


@router.delete("/fields/{fieldId}", status_code=204, responses={404: {}, 409: {}})
async def delete_field(fieldId: str, m=M):
    async with m.lock:
        lookup(m.fields, fieldId)
        if m.depends_on(fieldId):
            raise HTTPException(409, "A queued or running simulation uses this field")
        m.fields.remove(fieldId)
    return Response(status_code=204)


@router.get("/fields/{fieldId}/original", responses={404: {}})
async def original(fieldId: str, m=M):
    async with m.lock:
        item = lookup(m.fields, fieldId)
        name = item["name"] + (".nc" if item["source"] == "drawn" else "")
        return Download(m.field_path(item), name, "application/x-netcdf")


async def field_data(m, field_id, var, index, *, window=None, stats=False):
    async with m.lock:
        item = lookup(m.fields, field_id)
        if item["state"] == "archived":
            raise HTTPException(409, "Field is archived")
        try:
            if stats:
                return await asyncio.to_thread(
                    netcdf.field_stats, m.field_path(item), item, var, index, m.disk
                )
            return await asyncio.to_thread(m.field_plane, item, var, index, window)
        except DiskSpaceError as exc:
            raise invalid("var", str(exc), "query") from exc
        except HTTPException as exc:
            for error in exc.detail:
                error["loc"] = ["query", error["loc"][-1]]
            raise


@router.get("/fields/{fieldId}/stats", responses={404: {}, 409: {}, 422: {}})
async def field_stats(
    fieldId: str, var: str | None = None, index=Depends(indices), m=M
):
    async with m.read_slots:
        return await field_data(m, fieldId, var, index, stats=True)


@router.get("/fields/{fieldId}/preview.png", responses={404: {}, 409: {}, 422: {}})
async def field_preview(
    fieldId: str,
    var: str | None = None,
    index=Depends(indices),
    options=Depends(style),
    m=M,
):
    def preview(plane):
        summary = netcdf.stats(plane, var, {})
        return png(plane, options, (summary["min"], summary["max"]))

    async with m.read_slots:
        plane, _ = await field_data(m, fieldId, var, index)
        return Response(await asyncio.to_thread(preview, plane), media_type="image/png")


@router.get("/fields/{fieldId}/values", responses={404: {}, 409: {}, 422: {}})
async def field_values(
    fieldId: str,
    var: str | None = None,
    index=Depends(indices),
    win=Depends(window),
    m=M,
):
    async with m.read_slots:
        plane, _ = await field_data(m, fieldId, var, index, window=win)
        return await asyncio.to_thread(values_response, plane, win)


@router.post("/runs", status_code=202, responses={404: {}, 422: {}})
async def create(req: RunRequest, m=M):
    return await m.submit(req)


@router.post("/runs/validate", responses={422: {}})
async def validate(req: RunRequest, m=M):
    return await m.submit(req, validate=True)


async def run_view(m, run):
    if run["status"] == "running":
        await m.refresh(run)
    return m.view(run)


@router.get("/runs")
async def list_runs(params=Depends(page_params), m=M):
    async with m.lock:
        page = paginate(m.runs, params)
        page["items"] = [await run_view(m, r) for r in page["items"]]
        return page


@router.get("/runs/{runId}", responses={404: {}})
async def get_run(runId: str, m=M):
    async with m.lock:
        return await run_view(m, lookup(m.runs, runId))


@router.delete("/runs/{runId}", status_code=204, responses={404: {}})
async def delete_run(runId: str, m=M):
    await m.delete_run(runId)
    return Response(status_code=204)


@router.post("/runs/{runId}/rerun", status_code=202, responses={404: {}, 422: {}})
async def rerun(runId: str, m=M):
    run = lookup(m.runs, runId)
    return await m.submit(
        RunRequest.model_validate(run["request"]), wind_limit=run.get("_wind_limit")
    )


async def frame(m, run_id, n, reader, *selection):
    async with m.lock:
        run = lookup(m.runs, run_id)
        if run["status"] == "expired" or n >= run["resolved"]["expected_frames"]:
            raise HTTPException(404, "Frame does not exist")
        running = run["status"] == "running"
        if running:
            await m.refresh(run)
        if n >= run["frames"]:
            raise HTTPException(409, "Frame has not been written")
        try:
            plane = await asyncio.to_thread(
                reader, m.runs.root / run_id / "result.nc", n, *selection
            )
        except (OSError, RuntimeError, ValueError, IndexError) as exc:
            if running:
                raise HTTPException(409, "Frame is not yet readable") from exc
            raise
        return plane, run["display_range"]


@router.get("/runs/{runId}/frames/{n}.png", responses={404: {}, 409: {}, 422: {}})
async def frame_png(
    runId: str, n: RunIndex, options=Depends(style), view=Depends(frame_view), m=M
):
    async with m.read_slots:
        plane, limits = await frame(m, runId, n, netcdf.read_frame, None, view)
        content = await asyncio.to_thread(png, plane, options, limits, indexed=True)
        return Response(
            content,
            media_type="image/png",
            headers={"Cache-Control": "private, max-age=31536000, immutable"},
        )


@router.get("/runs/{runId}/frames/{n}/values", responses={404: {}, 409: {}, 422: {}})
async def frame_values(
    runId: str, n: RunIndex, win=Depends(window), view=Depends(frame_view), m=M
):
    async with m.read_slots:
        plane, _ = await frame(m, runId, n, netcdf.read_frame, win, view)
        return await asyncio.to_thread(values_response, plane, win)


@router.get("/runs/{runId}/frames/{n}/volume", responses={404: {}, 409: {}, 422: {}})
async def frame_volume(
    runId: str, n: RunIndex, max_side: Annotated[int, Query(ge=1, le=128)] = 64, m=M
):
    async with m.read_slots:
        volume, _ = await frame(m, runId, n, netcdf.read_volume, max_side)
        nz, ny, nx = volume.shape
        return Response(
            memoryview(volume).cast("B"),
            media_type="application/octet-stream",
            headers={"X-Nz": str(nz), "X-Ny": str(ny), "X-Nx": str(nx)},
        )


def completed(runId: str, m=M):
    run = lookup(m.runs, runId)
    if run["status"] == "expired":
        raise HTTPException(404, "Outputs have expired")
    if run["status"] != "done":
        raise HTTPException(409, "Simulation must finish successfully first")
    return run


@router.get("/runs/{runId}/result.nc", responses={404: {}, 409: {}})
async def result(runId: str, m=M):
    async with m.lock:
        completed(runId, m)
        return Download(
            m.runs.root / runId / "result.nc", f"{runId}.nc", "application/x-netcdf"
        )


@router.get(
    "/runs/{runId}/export.{format}", responses={202: {}, 404: {}, 409: {}, 422: {}}
)
async def export(
    runId: str,
    format: Literal["mp4", "gif"],
    fps: Annotated[int, Query(ge=1, le=60)] = 12,
    options=Depends(style),
    view=Depends(frame_view),
    m=M,
):
    async with m.lock:
        run = completed(runId, m)
        output = m.export(run, format, fps, options, view)
        if isinstance(output, Path):
            return Download(
                output,
                f"{runId}.{format}",
                "video/mp4" if format == "mp4" else "image/gif",
            )
        return JSONResponse(
            {"progress": output}, status_code=202, headers={"Retry-After": "1"}
        )


@router.post("/nowcasts", status_code=202, responses={422: {}})
async def new_nowcast(req: NowcastParams, m=M):
    return m.nowcasts.create(req)


@router.get("/nowcasts", responses={422: {}})
async def list_nowcasts(params=Depends(page_params), m=M):
    page = paginate(m.nowcasts.records, params)
    page["items"] = [m.nowcasts.view(item) for item in page["items"]]
    return page


@router.get("/nowcasts/{id}", responses={404: {}})
async def get_nowcast(id: str, m=M):
    return m.nowcasts.view(lookup(m.nowcasts.records, id))


@router.delete("/nowcasts/{id}", status_code=204, responses={404: {}})
async def delete_nowcast(id: str, m=M):
    async with m.lock:
        await m.nowcasts.delete(lookup(m.nowcasts.records, id))
    return Response(status_code=204)


@router.post("/nowcasts/{id}/refresh", status_code=202, responses={404: {}})
async def refresh_nowcast(id: str, m=M):
    previous = lookup(m.nowcasts.records, id)
    return m.nowcasts.create(NowcastParams(**previous["params"]), fresh=True)
