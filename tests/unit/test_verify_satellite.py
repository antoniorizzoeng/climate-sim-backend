import json
from datetime import UTC, datetime, timedelta

import h5py
import httpx
import netCDF4
import numpy as np
import pytest

from scripts import verify_satellite as vs

START = datetime(2026, 11, 18, 2, tzinfo=UTC)
# NOAA PUG ray intersection at zero-based (y=1870, x=1790) in the
# G18 ACMF s20263220150216 header. See docs/satellite-verification.md.
LAT, LON = 15.679602396850512, -154.80123900734182


@pytest.fixture
def acm(tmp_path):
    path = tmp_path / "acm.nc"
    with netCDF4.Dataset(path, "w") as ds:
        projection = ds.createVariable("goes_imager_projection", "i4")
        projection.setncatts(
            dict(
                grid_mapping_name="geostationary",
                sweep_angle_axis="x",
                longitude_of_projection_origin=-137.0,
                latitude_of_projection_origin=0.0,
                perspective_point_height=35786023.0,
                semi_major_axis=6378137.0,
                semi_minor_axis=6356752.31414,
            )
        )
        for name, first, sign in (("y", 1855, -1), ("x", 1775, 1)):
            ds.createDimension(name, 31)
            var = ds.createVariable(name, "i2", (name,))
            var.scale_factor = np.float32(sign * 0.000056)
            var.add_offset = np.float32(-sign * 0.151844)
            var.set_auto_maskandscale(False)
            var[:] = np.arange(first, first + 31)
        bcm, dqf = (
            ds.createVariable(name, "u1", ("y", "x"), fill_value=255)
            for name in ("BCM", "DQF")
        )
        bcm[:] = dqf[:] = 0
        bcm[15, 15:17] = bcm[15, 18] = 1
        bcm[15, 14] = 255
        dqf[15, 16] = 6
        dqf[14, 15] = 255
    return path


def test_real_header_navigation_and_radius(acm):
    with h5py.File(acm) as ds:
        proj, x, y = vs.grid(ds)
        east, north = proj(LON, LAT)
        assert (
            np.abs(y - north).argmin() + 1855,
            np.abs(x - east).argmin() + 1775,
        ) == (1870, 1790)
        window = vs.read_window(ds, LAT, LON, 25)
        assert window[0][15, 15] < 1e-6
        result = vs.observe(
            (*window[:3], np.full_like(window[0], 5000)), [1, 5, 10, 25], 2000, [300]
        )
    assert result[1]["column_fraction"] == 1
    assert result[5]["valid"] == 14 and result[5]["column_fraction"] == 1 / 14
    assert result[10]["cloudy"] == result[25]["cloudy"] == 2
    assert result[5]["valid"] < result[10]["valid"] < result[25]["valid"]
    assert result[25]["bad_or_missing"] == 3


def test_validity_cutoff_and_empty_disk():
    window = (
        np.array([1, 2, 3, 4, 5]),
        np.array([1, 0, 255, 0, 1]),
        np.array([0, 0, 0, 6, 255]),
    )
    obs = vs.observe((*window, np.zeros(5)), [0.5, 4, 5], 4205, [300])
    assert obs[0.5]["column_fraction"] is None
    assert obs[4]["valid"] == 2 and obs[4]["column_fraction"] == 0.5
    assert obs[4]["above_observer"][300]["fraction"] == 0
    assert obs[5]["bad_or_missing"] == 3 and obs[5]["column_fraction"] is None


@pytest.fixture
def acha(acm, tmp_path):
    path = tmp_path / "acha.nc"
    with h5py.File(path, "w") as ds, h5py.File(acm) as mask:
        mask.copy("goes_imager_projection", ds)
        for axis in ("y", "x"):
            # Tile centers align with ACM local indices 2, 7, ..., 27.
            v = ds.create_dataset(axis, data=mask[axis][2::5])
            for attr in ("scale_factor", "add_offset"):
                v.attrs[attr] = mask[axis].attrs[attr]
        ht = ds.create_dataset("HT", data=np.arange(36, dtype="u2").reshape(6, 6))
        ht.attrs.update(
            dict(
                units="m",
                scale_factor=100.0,
                add_offset=50.0,
                _FillValue=65535,
                valid_range=[0, 100],
            )
        )
        ht[0, 0], ht[0, 1] = 65535, 101
        dqf = ds.create_dataset("DQF", (6, 6), dtype="u1")
        dqf[0, 2:5] = [1, 2, 255]
    return path


def test_coregistration_packing_dqf_and_footprint(acm, acha):
    with h5py.File(acm) as ds:
        *_, lons, lats = vs.read_window(ds, LAT, LON, 25)
    with h5py.File(acha) as ds:
        ht = vs.mapped_heights(ds, lons, lats)
        assert np.isnan(vs.mapped_heights(ds, lons[-1:], lats[-1:])).all()
    assert np.isnan(ht[:5, :25]).all()
    assert np.isnan(ht[:, 30]).all()
    assert np.all(ht[:5, 25:30] == 550)
    assert np.all(ht[5:10, :5] == 650)
    assert np.all(ht[15:20, 15:20] == 2150)


def test_window_cache_matches_remote(acm, acha, tmp_path, monkeypatch):
    calls = []

    def load(url, **opts):
        assert opts["block_size"] and opts["cache_type"] == "blockcache"
        calls.append(url)
        return (acha if vs.ACHA in url else acm).open("rb")

    monkeypatch.setattr(vs.fsspec, "open", load)
    archive = vs.Archive(None, tmp_path)
    scan = vs.Scan(START, "ABI-L2-ACMF/scan.nc")
    height = vs.Scan(START, "ABI-L2-ACHAF/height.nc")
    first = archive.window("G18", scan, LAT, LON, 25, height)
    second = archive.window("G18", scan, LAT, LON, 25, height)
    assert calls == [
        f"https://noaa-goes18.s3.amazonaws.com/{s.key}" for s in (scan, height)
    ]
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    assert np.isnan(archive.window("G18", scan, LAT, LON, 25, None)[3]).all()


def test_nearest_crosses_year_paginates_and_respects_tolerance(tmp_path):
    def listing(request):
        p = request.url.params
        assert request.url.host == "noaa-goes19.s3.amazonaws.com"
        contents = ""
        if p["prefix"] == "ABI-L2-ACMF/2025/365/23/":
            if p.get("continuation-token") == "page2":
                key = "OR_ABI-L2-ACMF-M6_G19_s20253652354000_e20260010004000_c20260010004100.nc"
                contents = f"<Contents><Key>{key}</Key></Contents>"
            else:
                contents = "<Contents><Key>other</Key></Contents><NextContinuationToken>page2</NextContinuationToken>"
        return httpx.Response(
            200, text=f'<ListBucketResult xmlns="urn:s3">{contents}</ListBucketResult>'
        )

    with httpx.Client(transport=httpx.MockTransport(listing)) as client:
        archive = vs.Archive(client, tmp_path)
        at = datetime(2026, 1, 1, tzinfo=UTC)
        assert archive.nearest("G19", at).midpoint == at - timedelta(minutes=1)
        assert archive.nearest("G19", at + timedelta(minutes=9))
        assert archive.nearest("G19", at + timedelta(minutes=9, seconds=1)) is None


def test_nearest_tie_prefers_earlier(monkeypatch):
    archive = vs.Archive(None, None)
    scans = [vs.Scan(START + timedelta(minutes=n), str(n)) for n in (5, -5)]
    monkeypatch.setattr(archive, "listing", lambda *args: scans)
    assert archive.nearest("G18", START).key == "-5"


def test_score_definitions_and_threshold():
    m = vs.metrics([(0.30, 0.30), (0.5, 0.1), (0.25, 0.75), (0.75, 0.5), (0.3, 1)])
    assert m == pytest.approx(
        dict(
            n=5,
            mae_pp=37,
            bias_pp=-11,
            hits=1,
            misses=1,
            false_alarms=2,
            correct_negatives=1,
            hit_rate=0.5,
            false_alarm_rate=2 / 3,
        )
    )
    assert vs.metrics([])["mae_pp"] is None
    assert vs.metrics([(0.4, 0.2)])["false_alarm_rate"] is None
    assert vs.metrics([(0.2, 0.4)])["hit_rate"] is None


def record(minutes=(10, 60, 120, 240, 360, 370), lat=LAT, lon=LON):
    rows = []
    for minute in minutes:
        cloud = dict(low=0.8, mid=0.1, high=0.2, total=0.9)
        rows.append(
            dict(
                time=(START + timedelta(minutes=minute)).isoformat(),
                nowcast=cloud,
                model=dict(cloud, total=0.25),
                blend=dict(cloud, total=0.4),
            )
        )
    return dict(
        status="done",
        params=dict(lat=lat, lon=lon),
        data_time=START.isoformat(),
        site=rows,
    )


@pytest.mark.parametrize(
    "lat,lon,sat,angle",
    [
        (19.82, -155.47, "G18", 31.33),
        (-24.63, -70.4, "G19", 29.29),
        (28.76, -17.89, "G19", 69.90),
        (45, 90, "G19", None),
    ],
)
def test_view_and_refusal(lat, lon, sat, angle):
    actual, center, edge = vs.view(lat, lon, 25)
    if angle is not None:
        assert actual == sat and center == pytest.approx(angle, abs=0.01)
    if (lat, lon) in [(28.76, -17.89), (45, 90)]:
        assert edge > 70
        refused = vs.verify(
            record(lat=lat, lon=lon),
            vs.Archive(None, None),
            [5, 10, 25],
            elevation_m=1000,
        )
        assert refused["status"] == "unreliable view" and refused["samples"] == []


def test_persistence_only_from_data_time_and_pooled_scores(monkeypatch):
    archive = vs.Archive(None, None)
    monkeypatch.setattr(
        archive, "nearest", lambda sat, at, product=vs.ACM: vs.Scan(at, at.isoformat())
    )

    def window(sat, scan, *args):
        cloud = [0, 0, 1] if scan.midpoint == START else [1, 0, 0]
        return (
            np.array([1, 7, 9]),
            np.array(cloud),
            np.zeros(3),
            np.array([100, 100, 3000]),
        )

    monkeypatch.setattr(archive, "window", window)
    rec = vs.verify(record(), archive, [5, 10], 2000)
    assert [s["bucket"] for s in rec["samples"]] == [
        "0-1",
        "1-2",
        "2-4",
        "4-6",
        "4-6",
        ">6",
    ]
    other = vs.verify(record((10, 20)), archive, [5, 10], 2000)
    other["samples"][0]["forecast"]["model"] = 1
    rows = vs.summarize([rec, other], [5, 10])
    assert rows[0]["scores"]["persistence"]["mae_pp"] == 100
    assert rows[5]["scores"]["persistence"]["mae_pp"] == 0
    assert rows[0]["scores"]["model"]["mae_pp"] == 50
    assert rows[0]["requested"] == 3
    summit = vs.summarize([rec, other], [5, 10], margin=300)
    assert summit[0]["scores"]["persistence"]["mae_pp"] == 0
    assert summit[5]["scores"]["persistence"]["mae_pp"] == pytest.approx(100 / 3)


def test_no_init_no_scan_and_bad_quality(monkeypatch):
    archive = vs.Archive(None, None)

    def nearest(sat, at, product=vs.ACM):
        return vs.Scan(at, str(at)) if at >= START + timedelta(minutes=20) else None

    def window(sat, scan, *args):
        return (
            np.array([1]),
            np.array([0]),
            np.array([6 if scan.midpoint.minute == 20 else 0]),
            np.array([np.nan]),
        )

    monkeypatch.setattr(archive, "nearest", nearest)
    monkeypatch.setattr(archive, "window", window)
    r = vs.verify(record((10, 20, 30)), archive, [10], 2000)
    (t,) = vs.summarize([r], [10])
    assert (t["missing_scans"], t["low_valid"], t["bad_or_missing_pixels"]) == (1, 1, 1)
    assert t["scores"]["nowcast"]["n"] == 1 and t["scores"]["persistence"]["n"] == 0
    assert (
        vs.verify(record((10, 20)), archive, [10], 2000)["status"]
        == "no usable observations"
    )


def test_cli_aggregate_and_legacy_blend(tmp_path, monkeypatch, capsys):
    paths = []
    for name in ("first", "second"):
        path = tmp_path / name
        old = record((0, 10))
        for row in old["site"]:
            del row["blend"]
        path.write_text(json.dumps(old))
        paths.append(str(path))
    monkeypatch.setattr(
        vs.Archive, "nearest", lambda self, sat, at, product=vs.ACM: vs.Scan(at, "scan")
    )
    monkeypatch.setattr(
        vs.Archive,
        "window",
        lambda *args: (np.array([1]), np.array([0]), np.array([0]), np.array([np.nan])),
    )
    dest = tmp_path / "out.json"
    vs.main(
        [
            "--aggregate",
            "--elevation-m",
            "1234",
            "--margin-m",
            "750",
            "--json",
            str(dest),
            *paths,
        ]
    )
    payload = json.loads(dest.read_text())
    assert payload["margins_m"] == [750, 0, 1000]
    assert payload["records"][0]["elevation_m"] == 1234
    assert len(payload["summaries"]) == 1
    assert payload["summaries"][0]["table"][0]["scores"]["blend"]["n"] == 4
    assert "Valid/Bad" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["--radius-km", "nan", "unused"], ["missing-record"]])
def test_cli_errors(argv):
    with pytest.raises(SystemExit) as exc:
        vs.main(argv)
    assert exc.value.code == 2


def test_strict_height_threshold_and_missing_height_skip_by_radius():
    window = (
        np.arange(7),
        np.array([0, 1, 1, 1, 1, 0, 1]),
        np.zeros(7),
        np.array([np.nan, 2000, 2300, 2301, 3000, np.nan, np.nan]),
    )
    obs = vs.observe(window, [0.5, 5, 6], 2000, [0, 300, 1000])
    assert obs[0.5]["above_observer"][0]["fraction"] == 0
    assert [obs[5]["above_observer"][m]["fraction"] for m in (0, 300, 1000)] == [
        3 / 6,
        2 / 6,
        0,
    ]
    assert obs[6]["column_fraction"] == 5 / 7
    assert obs[6]["missing_height_pixels"] == 1
    assert all(v["fraction"] is None for v in obs[6]["above_observer"].values())


def test_missing_height_scan_and_reference_specific_persistence(
    acm, tmp_path, monkeypatch
):
    archive = vs.Archive(None, tmp_path)
    monkeypatch.setattr(archive, "remote", lambda *args: acm.open("rb"))
    pairs = []

    def nearest(sat, at, product=vs.ACM):
        if product == vs.ACHA:
            pairs.append(at)
            return None
        return vs.Scan(at + timedelta(minutes=4), "mask")

    monkeypatch.setattr(archive, "nearest", nearest)
    rec = vs.verify(record((0, 10)), archive, [1], 2000)
    assert pairs == [START + timedelta(minutes=m) for m in (4, 14)]
    assert rec["status"] == "column only"
    assert rec["initial"]["height_scan"] is None
    (above,) = vs.summarize([rec], [1], 300)
    assert (above["missing_height_samples"], above["missing_height_pixels"]) == (2, 2)
    assert above["scores"]["model"]["n"] == 0
    (column,) = vs.summarize([rec], [1])
    assert column["scores"]["persistence"]["n"] == 2


@pytest.mark.parametrize(
    "lat,lon,expected",
    [(19.8207, -155.468, 4205), (-24.63, -70.40, 2635), (28.76, -17.89, 2396)],
)
def test_published_elevation_and_override(lat, lon, expected):
    height, source = vs.site_elevation(lat, lon, None, None)
    assert height == expected and "published" in source
    assert vs.site_elevation(lat, lon, 0, None) == (0, "--elevation-m")


@pytest.mark.parametrize("height", [1234.5, None])
def test_elevation_fallback(height):
    def api(request):
        assert request.url.path == "/v1/elevation"
        assert dict(request.url.params) == {"latitude": "19", "longitude": "-155"}
        return httpx.Response(200, json={"elevation": [height]})

    with httpx.Client(transport=httpx.MockTransport(api)) as client:
        if height is None:
            with pytest.raises(ValueError, match="finite site elevation"):
                vs.site_elevation(19, -155, None, client)
        else:
            value, source = vs.site_elevation(19, -155, None, client)
            assert value == height and "Open-Meteo" in source


def test_elevation_assignments():
    sites = {(19.82, -155.47), (-24.63, -70.4)}
    assert vs.elevation_overrides(
        ["19.82,-155.47=4000", "-24.63,-70.40=2600"], sites
    ) == {(19.82, -155.47): 4000, (-24.63, -70.40): 2600}
    for bad in (
        ["4000"],
        ["19,-155=4000"],
        ["19.82,-155.47=4000"] * 2,
        ["19.82,-155.47=nan"],
    ):
        with pytest.raises(ValueError):
            vs.elevation_overrides(bad, sites)


def test_cli_pairs_coarser_height_scans_and_reports_elevation_fallback(
    acm, acha, tmp_path, monkeypatch, capsys
):
    def api(request):
        if request.url.host == "api.open-meteo.com":
            return httpx.Response(200, json={"elevation": [2000]})
        prefix = request.url.params["prefix"]
        product = prefix.split("/")[0]
        # Mask midpoints 02:04 and 02:14 share a single 02:09 height scan.
        times = {vs.ACM: [("00", "08"), ("10", "18")], vs.ACHA: [("05", "13")]}[product]
        keys = [
            f"{prefix}OR_{product}-M6_G18_s202632202{start}000_e202632202{end}000_c1.nc"
            for start, end in times
            if prefix.endswith("/02/")
        ]
        xml = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
        return httpx.Response(200, text=f"<ListBucketResult>{xml}</ListBucketResult>")

    client = httpx.Client(transport=httpx.MockTransport(api))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
    monkeypatch.setattr(
        vs.fsspec,
        "open",
        lambda url, **opts: (acha if vs.ACHA in url else acm).open("rb"),
    )
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record((0, 10))))
    dest = tmp_path / "results.json"
    vs.main(
        [
            "--cache-dir",
            str(tmp_path),
            "--radius-km",
            "1",
            "--json",
            str(dest),
            str(path),
        ]
    )
    payload = json.loads(dest.read_text())
    stdout = capsys.readouterr().out
    assert (
        "2000 m (Open-Meteo Elevation API" in stdout
        and "Any cloud in the column" in stdout
    )
    first, second = payload["records"][0]["samples"]
    assert (
        first["scan"] != second["scan"]
        and first["height_scan"] == second["height_scan"]
    )
    assert [s["height_offset_from_acm_seconds"] for s in (first, second)] == [300, -300]
    rows = [r for r in payload["summaries"][0]["table"] if r["radius_km"] == 1]
    assert [(r["margin_m"], r["mean_observed_fraction"]) for r in rows] == [
        (300, 0),
        (0, 1),
        (1000, 0),
        (None, 1),
    ]
