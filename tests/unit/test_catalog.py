import base64
import json
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import draw, preset, wait_run


def disk_bytes(path):
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


@pytest.mark.parametrize("kind", ["runs", "fields"])
def test_filtered_pages_renames_and_totals(client, clock, kind, monkeypatch):
    url = "/" + kind
    ids = []
    for name in ("Weather A", "Weather B", "Other"):
        if kind == "runs":
            created = client.post(url, json=preset(name=name)).json()
            wait_run(client, created["id"], "error")
        else:
            created = draw(client, name=name).json()
        ids.append(created["id"])
        clock.time += timedelta(hours=1)

    endpoint = f"{url}/{ids[-1]}"
    for invalid in ("  ", "x" * 121, None):
        assert client.patch(endpoint, json={"name": invalid}).status_code == 422
    assert (
        client.patch(endpoint, json={"name": " Weather C   "}).json()["name"]
        == "Weather C"
    )
    assert client.patch(url + "/missing", json={"name": "valid"}).status_code == 404
    if kind == "fields":
        assert (
            "Other.nc"
            in client.get(endpoint + "/original").headers["Content-Disposition"]
        )
    query = dict(q="wEAther", sort="name_asc", limit=1)
    collection = getattr(client.app.state.manager, kind)
    # Idle catalogue reads never rescan record directories.
    monkeypatch.setattr(
        collection, "measure", lambda *_: pytest.fail("rescanned idle record")
    )
    page = client.get(url, params=query).json()
    assert page["total"] == 3 and page["items"][0]["id"] == ids[0]
    assert page["total_bytes"] == disk_bytes(collection.root)
    cursor = page["next_cursor"]
    assert (
        client.get(url, params=dict(query, limit=2, cursor=cursor)).json()["items"][0][
            "id"
        ]
        == ids[1]
    )
    for change in (
        dict(q="B"),
        dict(sort="size_desc"),
        dict(created_from=clock().isoformat()),
        dict(created_to=clock().isoformat()),
    ):
        assert (
            client.get(url, params=dict(query, cursor=cursor) | change).status_code
            == 422
        )
    assert client.get(url, params=dict(q=ids[0][:10])).json()["total"] == 1
    assert client.get(url, params=dict(q="x" * 121)).status_code == 422
    for date in ("1735689600", "2026-01-01", "2026-01-01T04:00:00"):
        assert client.get(url, params={"created_from": date}).status_code == 422
    for sort in ("name", "size", "created"):
        asc = client.get(url, params=dict(sort=sort + "_asc")).json()["items"]
        desc = client.get(url, params=dict(sort=sort + "_desc")).json()["items"]
        assert asc == list(reversed(desc))
        key = {"name": "name", "size": "size_bytes", "created": "created_at"}[sort]
        assert [r[key] for r in asc] == sorted(r[key] for r in asc)
    a, b = (client.get(f"{url}/{i}").json() for i in ids[:2])
    selected = client.get(
        url, params=dict(created_from=a["created_at"], created_to=b["created_at"])
    ).json()
    assert selected["items"] == [a] and selected["total_bytes"] == a["size_bytes"]
    old = base64.urlsafe_b64encode(f"{b['created_at']}|{b['id']}".encode()).decode()
    assert client.get(url, params={"cursor": old}).json()["items"] == [a]


def test_legacy_records(settings, clock):
    with TestClient(create_app(settings, clock)) as c:
        field = draw(c, name="Temperature").json()
        req = dict(source=dict(kind="field", field_id=field["id"]), dt=1, steps=1)
        run = c.post("/runs", json=req).json()
        wait_run(c, run["id"], "error")
        sky = c.post("/nowcasts", json={"lat": 35, "lon": 12}).json()
    for kind, record in (("fields", field), ("runs", run), ("nowcasts", sky)):
        path = settings.data / kind / record["id"] / "metadata.json"
        meta = json.loads(path.read_text())
        if kind == "runs":
            for key in ("name", "_source_units", "_layer"):
                del meta[key]
            del meta["request"]["units"]
        elif kind == "fields":
            del meta["_original_name"], meta["variables"][0]["spacing"]
        else:
            del meta["night"], meta["params"]["window"], meta["_horizon_seconds"]
        meta.pop("size_bytes", None)
        path.write_text(json.dumps(meta))
    with TestClient(create_app(settings, clock)) as c:
        for kind, record in (("fields", field), ("runs", run)):
            item = c.get(f"/{kind}/{record['id']}").json()
            assert item["size_bytes"] == disk_bytes(settings.data / kind / record["id"])
        assert item["name"] == "Temperature · 01 Jan 00:00"
        assert item["request"]["units"] == {"length": "cell", "time": "t"}
        old = c.get("/nowcasts/" + sky["id"]).json()
        assert old["night"] is None and old["params"]["window"] == "now"
        assert c.get("/fields/" + field["id"]).json()["variables"][0]["spacing"] is None
        assert c.post("/runs/" + run["id"] + "/rerun").status_code == 202
