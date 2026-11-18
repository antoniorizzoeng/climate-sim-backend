import asyncio
import json
import math
from datetime import datetime

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from app import netcdf, nowcasts, sky
from app.main import create_app
from app.manager import RunManager
from tests.conftest import eastward_bank, poll
from tests.unit.test_lifecycle import clean

PARAMS = dict(lat=45, lon=10, radius_km=50, hours=2, resolution=41)


async def parked_worker(manager):
    # Leave ordinary runs in the queue until the test supplies frame data.
    while not manager.closing:
        manager.wake.clear()
        await manager.wake.wait()


@pytest.fixture
def supported(monkeypatch):
    calls = []

    async def fetch(self, params, start, fresh):
        calls.append((start, fresh))
        return eastward_bank(params)

    monkeypatch.setattr(nowcasts, "probe_wind", lambda settings: None)
    monkeypatch.setattr(nowcasts.ForecastClient, "fetch", fetch)
    monkeypatch.setattr(RunManager, "work", parked_worker)
    return calls


def await_state(client, url, status):
    def ready(r):
        item = r.json()
        assert item["status"] != "error" or status == "error", item
        return item["status"] == status

    return poll(client, url, ready).json()


def test_unavailable_and_validation(client):
    for body in (
        {},
        dict(PARAMS, lat=90),
        dict(PARAMS, resolution=257),
        dict(PARAMS, hours=13),
        dict(PARAMS, radius_km=49),
        dict(PARAMS, lon=181),
        dict(PARAMS, blend_tau_hours=0),
        dict(PARAMS, blend_tau_hours=-2),
        dict(PARAMS, blend_tau_hours="2"),
        dict(PARAMS, blend_tau_hours=None),
        dict(PARAMS, other=0),
    ):
        assert client.post("/nowcasts", json=body).status_code == 422
    r = client.post("/nowcasts", json={"lat": 0, "lon": 0})
    assert r.status_code == 202
    item = r.json()
    assert item["status"] == "error" and "wind.mode=file" in item["message"]
    assert item["params"] == dict(
        lat=0,
        lon=0,
        radius_km=300,
        hours=6,
        resolution=96,
        blend_tau_hours=2,
        window="now",
    )
    assert item["best_window"] is None
    assert item["site"] is None and item["layers"] == []
    assert client.get("/nowcasts/missing").status_code == 404
    assert client.post("/nowcasts/missing/refresh").status_code == 404
    assert client.delete("/nowcasts/missing").status_code == 404


def test_queue_frames_comparison_refresh_delete_restart(supported, settings, clock):
    with TestClient(create_app(settings, clock)) as c:
        m = c.app.state.manager
        r = c.post("/nowcasts", json=dict(PARAMS, blend_tau_hours=1.5))
        assert r.status_code == 202 and r.json()["status"] == "fetching"
        url = "/nowcasts/" + r.json()["id"]
        item = await_state(c, url, "queued")
        assert item["site"] is None and item["data_time"] == clock().isoformat()
        assert c.get("/runs").json()["items"] == []
        assert [layer["pressure_hpa"] for layer in item["layers"]] == [850, 500, 300]
        for layer in item["layers"]:
            run = c.get("/runs/" + layer["run_id"]).json()
            directory = m.runs.root / run["id"]
            config = yaml.safe_load((directory / "config.yaml").read_text())
            assert config["wind"] == sky.wind_config(
                m.fields.root / run["request"]["source"]["field_id"] / "wind.nc"
            )
            assert set(config["bc"].values()) == {"neumann"}
            with netcdf.dataset(config["wind"]["path"]) as ds:
                assert ds.data_model == "NETCDF3_64BIT_DATA"
                assert ds["vx"].dimensions == ("time", "y", "x") and ds["vy"].shape == (
                    3,
                    41,
                    41,
                )
                assert ds["vx"][0, 20, 20] > 0
            assert run["resolved"]["expected_frames"] == 13
            assert run["resolved"]["dt_max"] > run["request"]["dt"]
        # Restart with all three runs queued. The retained records supply the model comparison.
    with TestClient(create_app(settings, clock)) as c:
        m = c.app.state.manager
        runs = [m.runs.records[e["run_id"]] for e in item["layers"]]

        async def mark_running():
            m.queue.popleft()
            m.runs.update(runs[0], status="running", started_at=clock().isoformat())
            m.nowcasts.notify()

        c.portal.call(mark_running)
        await_state(c, url, "running")

        async def supply_frames():
            async with m.lock:
                m.queue.clear()
                for layer, run in zip((0.2, 0.3, 0.4), runs):
                    with netcdf.dataset(
                        m.runs.root / run["id"] / "result.nc",
                        "w",
                        format="NETCDF3_64BIT_DATA",
                    ) as ds:
                        for dim, n in (("time", 13), ("y", 41), ("x", 41)):
                            ds.createDimension(dim, n)
                        ds.createVariable("u", "f8", ("time", "y", "x"))[:] = np.full(
                            (13, 41, 41), layer
                        )
                    m.finish(run, None)

        c.portal.call(supply_frames)
        done = await_state(c, url, "done")
        spec = c.get("/openapi.json").json()
        Draft202012Validator(
            {"components": spec["components"], "$ref": "#/components/schemas/Nowcast"}
        ).validate(done)
        rows = done["site"]
        assert len(rows) == 13
        assert rows[0]["nowcast"] == pytest.approx(sky.cover(0.2, 0.3, 0.4))
        assert rows[0]["blend"] == rows[0]["nowcast"]
        assert done["params"]["blend_tau_hours"] == 1.5
        weight = math.exp(-1 / 1.5)
        assert rows[6]["blend"] == pytest.approx(
            sky.cover(
                low=weight * 0.2,
                mid=weight * 0.3,
                high=weight * 0.4,
                total=weight * (1 - 0.8 * 0.7 * 0.6) + (1 - weight) * 0.5,
            )
        )
        assert done["best_window"] is None
        assert rows[3]["model"]["total"] == pytest.approx(0.35)
        assert rows[3]["model"]["low"] == 0
        assert rows[5]["jet_ms"] == 20 and rows[6]["jet_ms"] == 30
        assert rows[-1]["seeing"] == "poor" and rows[-1]["model"][
            "clear"
        ] == pytest.approx(0.2)
        assert (
            datetime.fromisoformat(rows[-1]["time"]) - clock()
        ).total_seconds() == 7200
        # A standalone rerun has its own lifetime and the same file-driven config.
        duplicate = c.post("/runs/" + runs[0]["id"] + "/rerun").json()
        assert (
            yaml.safe_load((m.runs.root / duplicate["id"] / "config.yaml").read_text())[
                "wind"
            ]["interval"]
            == 3600
        )
        fresh = c.post(url + "/refresh").json()
        assert fresh["id"] != done["id"] and fresh["params"] == done["params"]
        await_state(c, "/nowcasts/" + fresh["id"], "queued")
        assert [flag for _, flag in supported] == [False, True]
        page = c.get("/nowcasts?limit=1").json()
        last = c.get("/nowcasts", params={"cursor": page["next_cursor"]}).json()[
            "items"
        ]
        assert len(last) == 1 and last[0]["id"] != page["items"][0]["id"]
        assert c.delete(url).status_code == 204 and c.get(url).status_code == 404
        for run in runs:
            assert c.get("/runs/" + run["id"]).status_code == 404
        assert (
            c.get("/fields/" + duplicate["request"]["source"]["field_id"]).status_code
            == 200
        )


def test_saved_nowcast_gains_blend_on_restart(settings, clock):
    with TestClient(create_app(settings, clock)) as c:
        record = c.post("/nowcasts", json=PARAMS).json()
    path = settings.data / "nowcasts" / record["id"] / "metadata.json"
    del record["params"]["blend_tau_hours"], record["best_window"]
    time = clock().isoformat()
    record.update(
        status="done",
        data_time=time,
        message=None,
        site=[
            dict(
                time=time,
                nowcast=sky.cover(0.1, 0, 0),
                model=sky.cover(0.8, 0, 0),
                jet_ms=15,
                seeing="good",
            )
        ],
    )
    path.write_text(json.dumps(record))
    with TestClient(create_app(settings, clock)) as c:
        upgraded = c.get(f"/nowcasts/{record['id']}").json()
    assert json.loads(path.read_text()) == upgraded
    assert upgraded["params"]["blend_tau_hours"] == 2
    assert upgraded["site"][0]["blend"] == record["site"][0]["nowcast"]
    assert upgraded["best_window"] == dict(
        start=time, end=time, mean_clear=0.9, seeing="good"
    )


def test_fetch_failure_expiry_and_deletion(supported, settings, clock, monkeypatch):
    async def unavailable(self, *args):
        raise ValueError("Forecast unavailable")

    monkeypatch.setattr(nowcasts.ForecastClient, "fetch", unavailable)
    with TestClient(create_app(settings, clock)) as c:
        url = "/nowcasts/" + c.post("/nowcasts", json=PARAMS).json()["id"]
        assert "Forecast unavailable" in await_state(c, url, "error")["message"]
        clock.advance(7)
        clean(c)
        assert c.get(url).status_code == 404

        async def wait_forever(self, *args):
            await asyncio.Event().wait()

        monkeypatch.setattr(nowcasts.ForecastClient, "fetch", wait_forever)
        url = "/nowcasts/" + c.post("/nowcasts", json=PARAMS).json()["id"]
        assert c.delete(url).status_code == 204
        url = "/nowcasts/" + c.post("/nowcasts", json=PARAMS).json()["id"]
    with TestClient(create_app(settings, clock)) as c:
        assert "interrupted" in await_state(c, url, "error")["message"]


def test_partial_submission_and_deletion_recovery(
    supported, settings, clock, monkeypatch
):
    with TestClient(create_app(settings, clock)) as c:
        m = c.app.state.manager

        def broken(*args):
            raise OSError("wind disk full")

        with monkeypatch.context() as ctx:
            ctx.setattr(sky, "write_wind", broken)
            url = "/nowcasts/" + c.post("/nowcasts", json=PARAMS).json()["id"]
            assert "disk full" in await_state(c, url, "error")["message"]
        assert not m.fields.records and not m.runs.records
        item = c.post(url + "/refresh").json()
        url = "/nowcasts/" + item["id"]
        queued = await_state(c, url, "queued")
        first = queued["layers"][0]["run_id"]
        assert c.delete("/runs/" + first).status_code == 204
        assert "deleted" in await_state(c, url, "error")["message"]
    path = settings.data / "nowcasts" / item["id"] / "metadata.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), _deleting=True)))
    with TestClient(create_app(settings, clock)) as c:
        assert c.get(url).status_code == 404 and not c.get("/runs").json()["items"]
