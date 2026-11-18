import asyncio
import fcntl
import json
import os
import signal
import threading
from collections import deque
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import psutil
import yaml
from fastapi import HTTPException

from . import models, netcdf
from .disk import DiskBudget, DiskSpaceError
from .events import log_event
from .models import ENGINE_INT_MAX, FieldSource, invalid, issue, resolve
from .nowcasts import Nowcasts
from .render import display_range, encode, export_name
from .sky import wind_config
from .storage import Collection


async def finish_thread(function, *args):
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.gather(task, return_exceptions=True)
        raise


ACTIVE = timedelta(days=7)
LIFETIME = timedelta(days=37)


def utcnow():
    return datetime.now(UTC)


def lookup(collection, identifier):
    record = collection.records.get(identifier)
    if record is None:
        raise HTTPException(404, "Record not found")
    return record


def progress(directory):
    try:
        return json.loads((directory / "progress.json").read_text())
    except FileNotFoundError:
        return {}


def process_tree(directory):
    # A crashed owner can leave a launcher and ranks. Both inherit this cwd
    # and the unique absolute --progress flag, even across MPI process groups.
    flag = f"--progress={directory}/progress.json"
    matches = []
    for proc in psutil.process_iter(["cmdline"]):
        if flag in (proc.info["cmdline"] or []):
            with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                if proc.cwd() == str(directory):
                    matches.append(proc)
    return matches


def reap_remaining(processes):
    def alive(proc):
        try:
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    live = [p for p in set(processes) if alive(p)]
    for p in live:
        with suppress(psutil.NoSuchProcess):
            p.kill()
    _, remaining = psutil.wait_procs(live, timeout=5)
    if any(alive(p) for p in remaining):
        raise RuntimeError("MPI ranks did not exit")


@dataclass
class ExportJob:
    task: asyncio.Task
    cancel: threading.Event
    progress: list[float]


class RunManager:
    def __init__(self, settings, clock=utcnow):
        self.settings = settings
        self.disk = DiskBudget(settings.data, settings.disk_margin_bytes)
        self.read_slots = asyncio.Semaphore(2)
        self.export_slot = asyncio.Semaphore(1)
        self.clock = clock
        self.fields = Collection(settings.data / "fields")
        self.runs = Collection(settings.data / "runs")
        self.queue = deque()
        self.sequence = 0
        self.lock = asyncio.Lock()
        self.wake = asyncio.Event()
        self.process = None
        self.active = None
        self.exports = {}
        self.closing = False
        self.nowcasts = Nowcasts(self, ACTIVE)

    async def start(self):
        self.settings.data.mkdir(parents=True, exist_ok=True)
        self.owner = (self.settings.data / ".owner").open("a")
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fields.load()
            self.runs.load()
            for run in sorted(self.runs.records.values(), key=lambda r: r["_order"]):
                self.sequence = max(run["_order"], self.sequence)
                directory = self.runs.root / run["id"]
                for partial in directory.glob("export-*.partial.*"):
                    partial.unlink()
                if run["status"] == "running":
                    await asyncio.to_thread(reap_remaining, process_tree(directory))
                    # A crash can interrupt creation of result.nc itself.
                    with suppress(
                        OSError, RuntimeError, ValueError, IndexError, KeyError
                    ):
                        await self.refresh(run, strict=True)
                    self.finish(run, "interrupted by restart")
                elif run["status"] == "queued":
                    self.queue.append(run["id"])
            await self.nowcasts.start()
            await self.cleanup()
            self.description = await asyncio.to_thread(self.settings.engine_description)
            self.worker = asyncio.create_task(self.work())
            self.janitor = asyncio.create_task(self.housekeeping())
            self.wake.set()
        except BaseException:
            self.owner.close()
            raise

    async def close(self):
        async with self.lock:
            await self.nowcasts.close()
            self.closing = True
            self.wake.set()
            await self.stop_process()
            await self.stop_exports()
        self.janitor.cancel()
        with suppress(asyncio.CancelledError):
            await self.janitor
        await self.worker
        self.owner.close()

    async def housekeeping(self):
        while True:
            await asyncio.sleep(3600)
            async with self.lock:
                await self.cleanup()

    async def cleanup(self):
        now = self.clock()
        self.nowcasts.cleanup(now)
        for run in list(self.runs.records.values()):
            if run["status"] in ("queued", "running"):
                continue
            finished = datetime.fromisoformat(run["finished_at"])
            if now >= finished + ACTIVE:
                await self.stop_exports(run["id"])
                if now >= finished + LIFETIME:
                    self.remove_run(run)
                else:
                    if run["status"] != "expired":
                        self.update_run(
                            run,
                            status="expired",
                            frames=0,
                            display_range=None,
                            expires_at=(finished + LIFETIME).isoformat(),
                        )
                    for path in (self.runs.root / run["id"]).iterdir():
                        if path.name != "metadata.json":
                            path.unlink()
        for item in list(self.fields.records.values()):
            used = datetime.fromisoformat(item["last_used_at"])
            if now >= used + LIFETIME:
                self.fields.remove(item["id"])
            elif now >= used + ACTIVE and item["state"] == "active":
                self.fields.update(
                    item, state="archived", expires_at=(used + LIFETIME).isoformat()
                )

    def depends_on(self, field_id):
        return any(
            r["status"] in ("running", "queued")
            and r["request"]["source"].get("field_id") == field_id
            for r in self.runs.records.values()
        )

    def touch(self, item):
        now = self.clock()
        self.fields.update(
            item,
            state="active",
            last_used_at=now.isoformat(),
            expires_at=(now + ACTIVE).isoformat(),
        )

    async def store_field(self, identifier, directory, source, name, shape=None):
        raw = directory / "upload.part"
        path = directory / "original.nc"

        def ingest():
            if shape is None:
                raw.replace(path)
            else:
                ny, nx = shape
                size = 4096 + 8 * nx * ny
                if raw.stat().st_size != 4 * nx * ny:
                    raise invalid("data", f"Expected {4 * nx * ny} bytes")
                with self.disk.reserve(size, "drawn NetCDF field", path):
                    netcdf.write_drawing(path, raw, shape)
                raw.unlink()
            return netcdf.inspect_field(path, self.disk)

        try:
            meta = await finish_thread(ingest)
        except DiskSpaceError as exc:
            raise invalid("data" if shape else "file", str(exc)) from exc
        return self.publish_field(identifier, directory, source, name, meta)

    async def add_drawn_field(self, name, values, identifier):
        with self.fields.staging(identifier) as (identifier, directory):
            path = directory / "original.nc"

            def ingest():
                with self.disk.reserve(
                    4096 + 8 * values.size, "drawn NetCDF field", path
                ):
                    netcdf.write_plane(path, values, drawn=True)
                return netcdf.inspect_field(path)

            meta = await finish_thread(ingest)
            return self.publish_field(identifier, directory, "drawn", name, meta)

    def publish_field(self, identifier, directory, source, name, meta):
        now = self.clock()
        item = dict(
            **meta,
            id=identifier,
            source=source,
            name=name,
            size_bytes=(directory / "original.nc").stat().st_size,
            created_at=now.isoformat(),
            last_used_at=now.isoformat(),
            state="active",
            expires_at=(now + ACTIVE).isoformat(),
        )
        self.fields.publish(directory, item)
        return item

    def field_path(self, item):
        return self.fields.root / item["id"] / "original.nc"

    def field_plane(self, item, var, index, window=None):
        return netcdf.read_plane(
            self.field_path(item), item, var, index, window, self.disk
        )

    def assess(self, req, validate):
        src = req.source
        plane = item = None
        if isinstance(src, FieldSource):
            item = lookup(self.fields, src.field_id)
            var, _ = netcdf.selected_indices(item, src.var, src.index)
            nx, ny = var["nx"], var["ny"]
        else:
            nx, ny = src.nx, src.ny
        report = resolve(req, nx, ny, self.settings, self.disk.available)
        if not validate and report["ok"]:
            self.disk.check(
                report["resolved"]["estimated_output_bytes"]
                + models.initial_bytes(req, nx, ny),
                "simulation output and staged input",
            )
        if item is not None:
            if validate:
                summary = netcdf.field_stats(
                    self.field_path(item), item, src.var, src.index, self.disk
                )
                missing = src.fill_value is None and summary["missing_count"]
            elif report["ok"]:
                plane, _ = self.field_plane(item, src.var, src.index)
                mask = ~np.isfinite(plane)
                missing = src.fill_value is None and mask.any()
                if src.fill_value is not None:
                    plane[mask] = src.fill_value
            else:
                missing = False
            if missing:
                report.update(ok=False, resolved=None)
                report["errors"].append(
                    issue("source.fill_value", "Selected plane contains missing cells")
                )
        return report, plane

    async def submit(self, req, validate=False, wind_limit=None):
        async with self.lock:
            return await self.submit_locked(req, validate, wind_limit=wind_limit)

    async def submit_locked(
        self, req, validate=False, wind_limit=None, identifier=None
    ):
        try:
            report, plane = await asyncio.to_thread(self.assess, req, validate)
        except DiskSpaceError as exc:
            if not validate:
                raise invalid("source", str(exc)) from exc
            return dict(
                ok=False,
                resolved=None,
                errors=[issue("source", str(exc))],
                warnings=[str(exc)],
            )
        except HTTPException as exc:
            if not validate:
                raise
            errors = (
                [issue("source.field_id", exc.detail)]
                if exc.status_code == 404
                else exc.detail
            )
            return dict(ok=False, resolved=None, errors=errors, warnings=[])
        if validate:
            return report
        if not report["ok"]:
            raise HTTPException(422, report["errors"])
        resolved = report["resolved"]
        input_size = models.initial_bytes(req, resolved["nx"], resolved["ny"])
        with self.runs.staging(identifier) as (identifier, staging):
            directory = self.runs.root / identifier
            if plane is None:
                ic = {"mode": "preset", "preset": req.source.preset}
            else:
                path = staging / "input.nc"

                def write_input():
                    with self.disk.reserve(input_size, "staged input", path):
                        netcdf.write_plane(path, plane)

                try:
                    await finish_thread(write_input)
                except DiskSpaceError as exc:
                    raise invalid("source", str(exc)) from exc
                ic = {
                    "mode": "file",
                    "var": "u",
                    "path": str(directory / "input.nc"),
                }
            config = dict(
                grid=dict(nx=resolved["nx"], ny=resolved["ny"], dx=req.dx, dy=req.dy),
                physics=dict(D=req.D, vx=req.vx, vy=req.vy),
                bc=req.bc.model_dump(),
                ic=ic,
                time=dict(
                    dt=req.dt,
                    steps=req.steps,
                    out_every=min(ENGINE_INT_MAX, resolved["out_every"]),
                ),
            )
            if wind_limit is not None:
                if not self.nowcasts.supported:
                    raise invalid("source", self.nowcasts.probe_message)
                config["wind"] = wind_config(
                    self.fields.root / req.source.field_id / "wind.nc"
                )
                resolved["dt_max"] = wind_limit
            (staging / "config.yaml").write_text(yaml.safe_dump(config))
            self.sequence += 1
            run = dict(
                id=identifier,
                _order=self.sequence,
                _wind_limit=wind_limit,
                status="queued",
                created_at=self.clock().isoformat(),
                started_at=None,
                finished_at=None,
                expires_at=None,
                request=req.model_dump(exclude_unset=True),
                resolved=resolved,
                step=0,
                time=0.0,
                frames=0,
                elapsed_seconds=None,
                display_range=None,
                message=None,
            )
            if isinstance(req.source, FieldSource):
                self.touch(self.fields.records[req.source.field_id])
            self.runs.publish(staging, run)
            log_event("run_state", run_id=identifier, status="queued", previous=None)
        self.queue.append(identifier)
        self.wake.set()
        return self.view(run)

    def view(self, run):
        position = (
            self.queue.index(run["id"]) + 1 if run["status"] == "queued" else None
        )
        return {
            k: v
            for k, v in dict(run, queue_position=position).items()
            if not k.startswith("_")
        }

    async def refresh(self, run, *, strict=False):
        directory = self.runs.root / run["id"]
        try:
            info = progress(directory)
            if info and info["frames"] >= run["frames"]:
                output = directory / "result.nc"
                if info["frames"] > run["frames"]:
                    if (
                        await asyncio.to_thread(netcdf.frame_count, output)
                        < info["frames"]
                    ):
                        raise RuntimeError(
                            "Published frames are still missing from the NetCDF header"
                        )
                    if not run["frames"]:
                        run["display_range"] = await asyncio.to_thread(
                            display_range, output, run["request"].get("bc", {})
                        )
                run.update({k: info[k] for k in ("step", "time", "frames")})
        except (OSError, RuntimeError, ValueError, IndexError, KeyError):
            if strict:
                raise
            # The writer can publish progress while the growing CDF header
            # is being updated. Keep the last view until the file is readable.
            info = {}
        run["elapsed_seconds"] = (
            self.clock() - datetime.fromisoformat(run["started_at"])
        ).total_seconds()
        return info

    def update_run(self, run, *, status, **changes):
        previous = run["status"]
        self.runs.update(run, status=status, **changes)
        log_event("run_state", run_id=run["id"], status=status, previous=previous)

    def remove_run(self, run):
        self.runs.remove(run["id"])
        log_event(
            "run_state", run_id=run["id"], status="deleted", previous=run["status"]
        )

    def finish(self, run, message):
        now = self.clock()
        self.update_run(
            run,
            status="error" if message else "done",
            finished_at=now.isoformat(),
            expires_at=(now + ACTIVE).isoformat(),
            message=message,
        )

        self.nowcasts.notify()

    async def work(self):
        while True:
            await self.wake.wait()
            with ExitStack() as writes:
                async with self.lock:
                    if self.closing:
                        return
                    if not self.queue:
                        self.wake.clear()
                        continue
                    identifier = self.queue.popleft()
                    run = self.runs.records[identifier]
                    directory = self.runs.root / identifier
                    self.update_run(
                        run, status="running", started_at=self.clock().isoformat()
                    )
                    self.nowcasts.notify()
                    if (
                        run.get("_wind_limit") is not None
                        and not self.nowcasts.supported
                    ):
                        self.finish(run, self.nowcasts.probe_message)
                        continue
                    field_id = run["request"]["source"].get("field_id")
                    if field_id in self.fields.records:
                        self.touch(self.fields.records[field_id])
                    args = [
                        self.settings.mpirun,
                        *self.settings.mpi_args,
                        "-np",
                        str(run["resolved"]["processes"]),
                        self.settings.engine,
                        f"--config={directory}/config.yaml",
                        f"--output={directory}/result.nc",
                        f"--progress={directory}/progress.json",
                    ]
                    try:
                        writes.enter_context(
                            self.disk.reserve(
                                run["resolved"]["estimated_output_bytes"],
                                "simulation output",
                                directory / "result.nc",
                            )
                        )
                        with (directory / "engine.log").open("wb") as log:
                            proc = await asyncio.create_subprocess_exec(
                                *args,
                                cwd=directory,
                                start_new_session=True,
                                stdout=log,
                                stderr=log,
                            )
                    except (OSError, DiskSpaceError) as exc:
                        self.finish(run, str(exc))
                        continue
                    self.active, self.process = identifier, proc
                code = await proc.wait()
                async with self.lock:
                    if identifier in self.runs.records and not self.closing:
                        try:
                            info = await self.refresh(run, strict=True)
                            message = None
                            if (
                                code
                                or info.get("status") != "done"
                                or run["frames"] != run["resolved"]["expected_frames"]
                            ):
                                message = (
                                    info.get("message")
                                    or f"Engine exited with code {code} without completing output"
                                )
                            if message and not info.get("message"):
                                with (directory / "engine.log").open("rb") as log:
                                    log.seek(max(0, log.seek(0, os.SEEK_END) - 2048))
                                    tail = log.read().decode(errors="replace").strip()
                                if tail:
                                    message += f"\n{tail}"
                            self.finish(run, message)
                        except (
                            OSError,
                            RuntimeError,
                            ValueError,
                            IndexError,
                            KeyError,
                        ) as exc:
                            self.finish(run, f"Cannot read engine output: {exc}")
                    self.process = self.active = None

    async def stop_process(self):
        proc = self.process
        if proc is None:
            return
        directory = self.runs.root / self.active
        if proc.returncode is not None:
            await asyncio.to_thread(reap_remaining, process_tree(directory))
            return
        ranks = process_tree(directory)
        with suppress(psutil.NoSuchProcess):
            ranks.extend(psutil.Process(proc.pid).children(recursive=True))
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), 5)
        except TimeoutError:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()
        await asyncio.to_thread(reap_remaining, ranks + process_tree(directory))

    async def delete_run(self, identifier):
        async with self.lock:
            await self.delete_run_locked(identifier)

    async def delete_run_locked(self, identifier):
        run = lookup(self.runs, identifier)
        if self.active == identifier:
            await self.stop_process()
        await self.stop_exports(identifier)
        self.remove_run(run)
        if run["status"] == "queued":
            self.queue.remove(identifier)
        self.nowcasts.notify()

    async def stop_exports(self, identifier=None):
        jobs = {
            key: job
            for key, job in self.exports.items()
            if identifier is None or key.parent.name == identifier
        }
        for job in jobs.values():
            job.cancel.set()
            job.task.cancel()
        await asyncio.gather(
            *(job.task for job in jobs.values()), return_exceptions=True
        )
        for key in jobs:
            del self.exports[key]

    async def encode_export(self, *args):
        async with self.export_slot:
            await finish_thread(encode, *args)

    def export(self, run, fmt, fps, style):
        directory = self.runs.root / run["id"]
        target = directory / export_name(fmt, fps, style)
        job = self.exports.get(target)
        if job and job.task.done():
            del self.exports[target]
            try:
                job.task.result()
            except (RuntimeError, OSError, DiskSpaceError) as exc:
                raise HTTPException(500, f"Export failed: {exc}") from exc
        if target.is_file():
            return target
        if target not in self.exports:
            cancel, fraction = threading.Event(), [0.0]
            task = asyncio.create_task(
                self.encode_export(
                    directory,
                    target,
                    fps,
                    style,
                    run.copy(),
                    cancel,
                    fraction,
                    self.disk,
                )
            )
            self.exports[target] = ExportJob(task, cancel, fraction)
        return self.exports[target].progress[0]
