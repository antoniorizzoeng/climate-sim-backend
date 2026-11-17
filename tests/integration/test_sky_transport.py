import asyncio
import os
from datetime import UTC, datetime

import numpy as np
import pytest

from app.models import NowcastParams
from app.open_meteo import ForecastClient
from tests.conftest import eastward_bank, poll


@pytest.mark.integration
def test_eastward_cloud_bank(live, monkeypatch):
    service = live.app.state.manager.nowcasts
    if not service.supported:
        pytest.skip(service.probe_message)

    async def synthetic(params, start, fresh):
        return eastward_bank(params)

    monkeypatch.setattr(service.forecast, "fetch", synthetic)
    accepted = live.post(
        "/nowcasts", json=dict(lat=45, lon=10, radius_km=50, resolution=41, hours=2)
    )
    assert accepted.status_code == 202
    url = "/nowcasts/" + accepted.json()["id"]
    result = poll(live, url, lambda r: r.json()["status"] in ("done", "error")).json()
    assert result["status"] == "done", result["message"]
    for layer in ("low", "mid", "high"):
        cover = [row["nowcast"][layer] for row in result["site"]]
        # Bank started exactly one hour upwind. Allow one frame for numerical diffusion.
        assert cover[0] == pytest.approx(0)
        assert max(cover) > 0.7 and abs(np.argmax(cover) - 6) <= 1
        assert cover[-1] < 0.25
    assert (
        live.get("/runs/" + result["layers"][0]["run_id"] + "/frames/6.png").status_code
        == 200
    )


@pytest.mark.network
@pytest.mark.skipif(
    os.getenv("RUN_NETWORK_TESTS") != "1",
    reason="Set RUN_NETWORK_TESTS=1 to contact Open-Meteo",
)
def test_live_forecast(tmp_path):
    def clock():
        return datetime.now(UTC)

    api = ForecastClient(tmp_path, clock)
    api.start()
    start = clock().replace(minute=0, second=0, microsecond=0)

    params = NowcastParams(lat=45, lon=10, hours=12)
    result = asyncio.run(api.fetch(params, start, fresh=False))
    assert result["cloud_cover"].shape == result["wind_speed_250hPa"].shape == (289, 13)
