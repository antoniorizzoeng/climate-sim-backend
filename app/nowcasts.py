"""Durable nowcast ownership, forecast acquisition and ordinary FIFO layer runs."""

import asyncio
import subprocess
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import yaml

from . import netcdf, sky
from .models import DEFAULT_BLEND_TAU_HOURS, NowcastParams
from .open_meteo import ForecastClient
from .storage import Collection

PROBE_ERRORS = (OSError, subprocess.SubprocessError, RuntimeError, IndexError, KeyError)


def probe_frame(settings, plane, bc, vx=0, vy=0, wind=None):
    with TemporaryDirectory(dir=settings.data) as tmp:
        directory = Path(tmp)
        netcdf.write_plane(directory / "input.nc", plane)
        config = dict(
            grid=dict(nx=5, ny=5, dx=1, dy=1),
            physics=dict(D=0, vx=vx, vy=vy),
            ic=dict(mode="file", path="input.nc", var="u"),
            bc=bc,
            time=dict(dt=1, steps=1, out_every=1),
        )
        if wind is not None:
            sky.write_wind(directory / "wind.nc", wind, wind * 0)
            config["wind"] = sky.wind_config("wind.nc")
        (directory / "probe.yaml").write_text(yaml.safe_dump(config))
        subprocess.run(
            [settings.engine, "--config=probe.yaml", "--output=result.nc"],
            cwd=tmp,
            timeout=10,
            check=True,
            capture_output=True,
        )
        return netcdf.read_frame(directory / "result.nc", 1)


def probe_wind(settings):
    """Older engines silently accept unknown flags/YAML. Check a transported pulse."""
    plane = np.zeros((5, 5))
    plane[2, 1] = 1
    try:
        frame = probe_frame(settings, plane, "neumann", wind=np.full((2, 5, 5), 0.5))
        if np.isclose(frame[2, 2], 0.5):
            return None
        return "Installed engine ignores wind.mode=file. Sky nowcasts require engine issue #58"
    except subprocess.CalledProcessError as exc:
        return f"wind.mode=file probe failed (engine exit {exc.returncode})"
    except PROBE_ERRORS as exc:
        return f"Cannot verify engine wind.mode=file support: {exc}"


def probe_periodic(settings):
    plane = np.zeros((5, 5))
    plane[-1, -1] = 1
    edges = dict.fromkeys(("left", "right", "bottom", "top"), "periodic")
    try:
        frame = probe_frame(settings, plane, edges, vx=0.25, vy=0.25)
        return bool(np.allclose([frame[-1, 0], frame[0, -1]], 0.25))
    except PROBE_ERRORS:
        return False


class Nowcasts:
    def __init__(self, manager, retention):
        self.manager = manager
        self.retention = retention
        self.records = Collection(manager.settings.data / "nowcasts")
        self.forecast = ForecastClient(manager.settings.data, manager.clock)
        self.tasks = {}
        self.changed = asyncio.Event()

    @property
    def supported(self):
        return self.probe_message is None

    async def start(self):
        self.records.load()
        self.forecast.start()
        self.probe_message = await asyncio.to_thread(probe_wind, self.manager.settings)
        for record in list(self.records.records.values()):
            if record.get("_deleting"):
                await self.delete(record)
                continue
            if "blend_tau_hours" not in record["params"]:
                self.upgrade(record)
            if record["status"] == "fetching":
                await self.remove_children(record)
                self.finish(
                    record,
                    message="Forecast preparation interrupted by restart. Refresh to retry",
                )
            elif record["status"] in ("queued", "running"):
                self.launch(record)

    def upgrade(self, record):
        params = dict(record["params"], blend_tau_hours=DEFAULT_BLEND_TAU_HOURS)
        site = record["site"]
        if site is not None:
            site = sky.blend_site(site, params["blend_tau_hours"])
        self.records.update(
            record, params=params, site=site, best_window=sky.best_window(site or ())
        )

    async def close(self):
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def cleanup(self, now):
        self.forecast.cleanup()
        for record in list(self.records.records.values()):
            if record["status"] in ("done", "error") and now >= datetime.fromisoformat(
                record["expires_at"]
            ):
                self.records.remove(record["id"])

    def notify(self):
        self.changed.set()
        self.changed = asyncio.Event()

    @staticmethod
    def view(record):
        return {k: v for k, v in record.items() if not k.startswith("_")}

    def create(self, params, fresh=False):
        now = self.manager.clock()
        with self.records.staging() as (identifier, directory):
            record = dict(
                id=identifier,
                status="fetching" if self.supported else "error",
                created_at=now.isoformat(),
                params=params.model_dump(),
                data_time=None,
                layers=[],
                grid=sky.grid(params),
                site=None,
                best_window=None,
                message=self.probe_message,
                expires_at=(now + self.retention).isoformat(),
            )
            self.records.publish(directory, record)
        if self.supported:
            self.launch(record, fresh)
        return self.view(record)

    def launch(self, record, fresh=False):
        task = asyncio.create_task(self.process(record, fresh))
        self.tasks[record["id"]] = task
        task.add_done_callback(lambda _: self.tasks.pop(record["id"], None))

    def finish(self, record, *, message=None, site=None):
        self.records.update(
            record,
            status="error" if message else "done",
            message=message,
            site=site,
            best_window=sky.best_window(site or ()),
            expires_at=(self.manager.clock() + self.retention).isoformat(),
        )

    async def process(self, record, fresh):
        try:
            if record["status"] == "fetching":
                params = NowcastParams(**record["params"])
                start = self.manager.clock().replace(minute=0, second=0, microsecond=0)
                data = await self.forecast.fetch(params, start, fresh)
                async with self.manager.lock:
                    await self.enqueue(record, params, data, start)
            await self.monitor(record)
        except Exception as exc:
            # Keep the error on this job. Other queued runs can proceed.
            async with self.manager.lock:
                if record["status"] == "fetching":
                    await self.remove_children(record)
                self.finish(record, message=f"Sky nowcast failed: {exc}")

    async def enqueue(self, record, params, data, start):
        # Journal intended IDs before publishing any child in another collection.
        fields = [uuid4().hex for _ in sky.LEVELS]
        layers = [
            dict(layer=k, pressure_hpa=v, run_id=uuid4().hex)
            for k, v in sky.LEVELS.items()
        ]
        self.records.update(
            record, _fields=fields, _planned=layers, data_time=start.isoformat()
        )
        for entry, field in zip(layers, fields):
            name = entry["layer"]
            plane, u, v = await asyncio.to_thread(
                sky.layer_inputs, data, name, params.resolution
            )
            await self.manager.add_drawn_field(
                f"Sky nowcast {name}", plane.astype("<f4"), field
            )
            await asyncio.to_thread(self.store_wind, field, u, v)
            req, limit = sky.run_request(params, field, u, v)
            await self.manager.submit_locked(
                req, identifier=entry["run_id"], wind_limit=limit
            )
        centre = len(data["cloud_cover"]) // 2
        model = {k: data[f"cloud_cover_{k}"][centre].tolist() for k in sky.LEVELS}
        model.update(
            total=data["cloud_cover"][centre].tolist(),
            jet_ms=data["wind_speed_250hPa"][centre].tolist(),
        )
        self.records.update(record, layers=layers, _model=model, status="queued")

    def store_wind(self, field, u, v):
        path = self.manager.fields.root / field / "wind.nc"
        size = 4096 + 8 * (u.size + v.size)
        with self.manager.disk.reserve(size, "sky nowcast wind", path):
            sky.write_wind(path, u, v)

    async def monitor(self, record):
        while True:
            async with self.manager.lock:
                changed = self.changed
                runs = [
                    self.manager.runs.records.get(e["run_id"]) for e in record["layers"]
                ]
                for entry, run in zip(record["layers"], runs):
                    if not run or run["status"] in ("error", "expired"):
                        reason = (
                            (run.get("message") or run["status"]) if run else "deleted"
                        )
                        self.finish(record, message=f"{entry['layer']} layer: {reason}")
                        return
                if all(r["status"] == "done" for r in runs):
                    site = await asyncio.to_thread(
                        sky.timeline, record, self.manager.runs.root
                    )
                    self.finish(record, site=site)
                    return
                if record["status"] == "queued" and any(
                    r["status"] != "queued" for r in runs
                ):
                    self.records.update(record, status="running")
            await changed.wait()

    async def remove_children(self, record):
        for entry in record.get("_planned", []):
            if entry["run_id"] in self.manager.runs.records:
                await self.manager.delete_run_locked(entry["run_id"])
        for field in record.get("_fields", []):
            # A user may have independently rerun a layer and retained its input.
            used = any(
                r["request"]["source"].get("field_id") == field
                for r in self.manager.runs.records.values()
            )
            if field in self.manager.fields.records and not used:
                self.manager.fields.remove(field)

    async def delete(self, record):
        self.records.update(record, _deleting=True)
        task = self.tasks.get(record["id"])
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self.remove_children(record)
        self.records.remove(record["id"])
