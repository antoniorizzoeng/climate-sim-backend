import asyncio
import base64
import binascii
import math
import os
import re
import shutil
from pathlib import Path
from typing import Annotated, Literal, get_args
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi import Path as PathParam
from fastapi import Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from . import netcdf
from .manager import lookup
from .models import Colormap, NowcastParams, Preset, RunRequest, invalid
from .render import Style, png

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
            result[match[1]] = int(value)
    return result


def window(
    i0: Annotated[int, Query(ge=0)] = 0,
    j0: Annotated[int, Query(ge=0)] = 0,
    w: Annotated[int | None, Query(ge=1)] = None,
    h: Annotated[int | None, Query(ge=1)] = None,
    stride: Annotated[int, Query(ge=1)] = 1,
):
    return (i0, j0, w, h, stride)


def values_response(plane, win):
    ny, nx = plane.shape
    i0, j0, w, h, stride = win
    end_i, end_j = nx if w is None else i0 + w, ny if h is None else j0 + h
    if i0 >= nx or j0 >= ny or end_i > nx or end_j > ny:
        raise invalid("window", "Window lies outside the plane", "query")
    grid = plane[j0:end_j:stride, i0:end_i:stride]
    headers = dict(
        zip(
            ("X-Nx", "X-Ny", "X-I0", "X-J0", "X-Stride"),
            map(str, (grid.shape[1], grid.shape[0], i0, j0, stride)),
        )
    )
    return Response(
        grid.astype("<f4").tobytes(),
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


def download(path, name, media):
    # Open under the manager lock. An overlapping DELETE can then unlink safely.
    stream = path.open("rb")
    size = os.fstat(stream.fileno()).st_size

    async def chunks():
        try:
            while block := await asyncio.to_thread(stream.read, 256 * 1024):
                yield block
        finally:
            stream.close()

    return StreamingResponse(
        chunks(),
        media_type=media,
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(name, safe='')}",
            "Content-Length": str(size),
        },
    )


M = Depends(manager)


@router.get("/health")
async def health(m=M):
    return dict(status="ok", engine=m.description)


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
        disk_free_bytes=shutil.disk_usage(m.settings.data).free,
        retention=dict(active_days=7, archive_days=30),
    )


class Upload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file: UploadFile


class Drawing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: UploadFile
    nx: int = Field(ge=4)
    ny: int = Field(ge=4)
    name: str = "drawn"


@router.post("/fields", status_code=201, responses={422: {}})
async def upload(form: Annotated[Upload, Form(media_type="multipart/form-data")], m=M):
    return await m.add_field("upload", form.file.filename or "field.nc", form.file.file)


@router.post("/fields/drawn", status_code=201, responses={422: {}})
async def drawn(form: Annotated[Drawing, Form(media_type="multipart/form-data")], m=M):
    return await m.add_field("drawn", form.name, form.data.file, (form.ny, form.nx))


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
    item = lookup(m.fields, fieldId)
    name = item["name"] + (".nc" if item["source"] == "drawn" else "")
    return download(
        m.fields.root / fieldId / "original.nc", name, "application/x-netcdf"
    )


async def field_selection(
    fieldId: str, var: str | None = None, index=Depends(indices), m=M
):
    async with m.lock:
        item = lookup(m.fields, fieldId)
        if item["state"] == "archived":
            raise HTTPException(409, "Field is archived")
        try:
            plane, idx = await asyncio.to_thread(m.field_plane, item, var, index)
        except HTTPException as exc:
            for error in exc.detail:
                error["loc"] = ["query", error["loc"][-1]]
            raise
        return plane, var or item["default_var"], idx


@router.get("/fields/{fieldId}/stats", responses={404: {}, 409: {}, 422: {}})
async def field_stats(selection=Depends(field_selection)):
    return await asyncio.to_thread(netcdf.stats, *selection)


@router.get("/fields/{fieldId}/preview.png", responses={404: {}, 409: {}, 422: {}})
async def field_preview(selection=Depends(field_selection), options=Depends(style)):
    def preview():
        plane = selection[0]
        stats = netcdf.stats(*selection)
        return png(plane, options, (stats["min"], stats["max"]))

    return Response(await asyncio.to_thread(preview), media_type="image/png")


@router.get("/fields/{fieldId}/values", responses={404: {}, 409: {}, 422: {}})
async def field_values(selection=Depends(field_selection), win=Depends(window)):
    return await asyncio.to_thread(values_response, selection[0], win)


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


async def frame(runId: str, n: RunIndex, m=M):
    async with m.lock:
        run = lookup(m.runs, runId)
        if run["status"] == "expired" or n >= run["resolved"]["expected_frames"]:
            raise HTTPException(404, "Frame does not exist")
        if run["status"] == "running":
            await m.refresh(run)
        if n >= run["frames"]:
            raise HTTPException(409, "Frame has not been written")
        plane = await asyncio.to_thread(
            netcdf.read_frame, m.runs.root / runId / "result.nc", n
        )
        return plane, run["display_range"]


@router.get("/runs/{runId}/frames/{n}.png", responses={404: {}, 409: {}, 422: {}})
async def frame_png(selection=Depends(frame), options=Depends(style)):
    plane, limits = selection
    content = await asyncio.to_thread(png, plane, options, limits)
    return Response(
        content,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


@router.get("/runs/{runId}/frames/{n}/values", responses={404: {}, 409: {}, 422: {}})
async def frame_values(selection=Depends(frame), win=Depends(window)):
    return await asyncio.to_thread(values_response, selection[0], win)


def completed(runId: str, m=M):
    run = lookup(m.runs, runId)
    if run["status"] == "expired":
        raise HTTPException(404, "Outputs have expired")
    if run["status"] != "done":
        raise HTTPException(409, "Simulation must finish successfully first")
    return run


@router.get("/runs/{runId}/result.nc", responses={404: {}, 409: {}})
async def result(runId: str, m=M):
    completed(runId, m)
    return download(
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
    m=M,
):
    async with m.lock:
        run = completed(runId, m)
        output = m.export(run, format, fps, options)
        if isinstance(output, Path):
            return download(
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
