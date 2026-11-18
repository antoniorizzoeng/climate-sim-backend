# Verifying Sky with GOES cloud mask and cloud-top height

`scripts/verify_satellite.py` scores saved total-cloud forecasts against two satellite references, **cloud above the observer** and **any cloud in the column**. It reads completed Sky metadata or saved API responses containing `params`, `data_time` and `site`. Install `requirements-dev.txt` in the analysis environment.

```sh
python scripts/verify_satellite.py --aggregate --cache-dir artifacts/satellite/cache --json satellite-summary.json \
    ~/skill/nowcasts/*/metadata.json
```

The command prints tables by reference and record and writes JSON. With `--aggregate`, each coordinate pair gets scores pooled across its input records. Each matched forecast time has equal weight, and each record has its own persistence baseline. Supply each forecast record once. The JSON preserves the source paths, forecast totals, selected S3 keys, time offsets, observed fractions, pixel counts, contingency counts and scores. Missing results appear as `null`.

## Site elevation

| Observatory | Latitude | Longitude | Elevation above sea level |
|---|---:|---:|---:|
| Mauna Kea | 19.82 | −155.47 | 4,205 m |
| Paranal | −24.63 | −70.40 | 2,635 m |
| Roque de los Muchachos | 28.76 | −17.89 | 2,396 m |

For one site, `--elevation-m 4205` sets the observer elevation for all its records. For several sites, pass one assignment per override, for example `--elevation-m=19.82,-155.47=4205 --elevation-m=-24.63,-70.4=2635`. An override takes precedence. Otherwise the verifier uses the published elevation within 1 km of a listed coordinate, allowing for rounding in the saved records. For other sites it queries the Open-Meteo Elevation API (Copernicus DEM, 90 m resolution). Both the text and JSON give the elevation and its source. [Open-Meteo Elevation API](https://open-meteo.com/en/docs/elevation-api)

## Reference and alignment

1. Choose the smaller nominal satellite zenith angle of GOES-18 (West, 137.0° W) and GOES-19 (East, 75.2° W). Compute the angle between the local geodetic normal and the ray from sea level to the satellite at its nominal altitude of 35,786 km. Report the angle at the site and the maximum sampled at the site and at one-degree azimuth intervals around the largest requested disk. Flag the entire site as **unreliable view** if either exceeds 70°, the ACM file's upper limit for good-quality retrievals. This conservative rule keeps the same view criterion across radii. For La Palma, the center is at 69.90° and the 25 km disk reaches 70.14°. These slots describe the operational G18/G19 configuration. [NOAA GOES-R Series](https://www.goes-r.gov/)
2. List anonymous HTTPS objects for both `ABI-L2-ACMF` and `ABI-L2-ACHAF` under `YYYY/DDD/HH/`. Decode the UTC scan start and end, including tenths of seconds, from each object key. Match the ACM **midpoint** nearest the forecast valid time within ±10 minutes. Match the ACHA midpoint nearest that ACM midpoint with the same tolerance. Both searches include adjacent UTC hours and days. Ties use the earlier midpoint. A full-disk sweep takes several minutes and the exact acquisition time at a pixel varies. A gap can cause adjacent valid times to share a scan. The JSON keys make those cases identifiable. [GOES Open Data on AWS](https://registry.opendata.aws/noaa-goes/)
3. Construct the CF geostationary projection from each file's `goes_imager_projection`. Unpack the signed, scaled `x` and `y` scan angles and convert radians to projected metres using the file's perspective height. The GOES-19 example has fixed-grid origin 75.0° W and platform longitude 75.2° W. Inverse-project nearby pixel centers, then measure ellipsoidal geodesic distances from the site. Count centers inside each disk with equal weight. The main radius defaults to 10 km (`--radius-km`), with 5 and 25 km sensitivity estimates. [NOAA fixed-grid navigation](https://www.star.nesdis.noaa.gov/atmospheric-composition-training/satellite_data_goes_imager_projection.php)
4. Use `DQF == 0` (good quality) and valid binary `BCM` values. A value of 1 means cloudy or probably cloudy, and 0 means clear or probably clear. Exclude fill values, degraded retrievals and other invalid codes. The **column fraction** is cloudy pixels divided by valid ACM pixels. An empty disk or less than 50% valid yields a missing observation for that radius. Exactly 50% valid qualifies. The table sums valid and rejected pixels across all matched scans, including low-validity disks. The JSON distinguishes a missing scan from a matched scan with insufficient valid pixels. [Example GOES-18 ACM NetCDF with the flag definitions](https://noaa-goes18.s3.amazonaws.com/ABI-L2-ACMF/2026/322/01/OR_ABI-L2-ACMF-M6_G18_s20263220150216_e20263220159525_c20263220200145.nc)
5. Project the center of each ACM pixel into the matched ACHA file's own fixed grid and select the containing height cell. The November files have nominal nadir spacings of 2 km for ACM and 10 km for ACHA. A height cell can represent several cloudy ACM pixels. Accept only ACHA `DQF == 0`, discard HT fill and out-of-range packed values, then apply `scale_factor` and `add_offset` to recover cloud-top geopotential height in metres. [Example GOES-18 ACHA NetCDF with HT and DQF definitions](https://noaa-goes18.s3.amazonaws.com/ABI-L2-ACHAF/2026/322/01/OR_ABI-L2-ACHAF-M6_G18_s20263220100216_e20263220109525_c20263220111250.nc)
6. For **cloud above the observer**, count good cloudy ACM pixels whose top strictly exceeds site elevation plus the margin. Divide by all good ACM pixels. The primary margin is 300 m (`--margin-m`), with 0 and 1,000 m sensitivity tables. If any good cloudy pixel lacks a usable height, skip the entire time sample for that radius in all height-filtered scores. This includes an unmatched ACHA scan. A good clear mask pixel needs no height, so an entirely clear disk can still score. The column reference uses all eligible ACM observations. The text reports affected samples and cloudy ACM pixels per radius and lead. Each sample is counted once per radius, irrespective of how many heights it lacks.

Both products use anonymous 64 KiB HTTP range reads for the HDF5 headers, coordinate axes and relevant compressed data chunks. The default cache is `~/sat-cache`. The invocation above uses a worktree-local cache. Entries store mask, quality, distance and mapped height arrays keyed by both object names, site and maximum radius, so the same cutout supports every margin and site elevation.


The offline navigation fixture preserves the packed axes and projection attributes of the GOES-18 example above. At zero-based full-disk row 1870, column 1790, the NOAA ellipsoidal ray-intersection equations give 15.67960240° N, 154.80123901° W. The regression test uses that geographic position as an independent input and checks both the pixel index and the distance to its center.

## Scores

Lead is measured from the saved `data_time` in UTC. Buckets are [0,1), [1,2), [2,4), [4,6] and (6,∞) hours. Lead zero is included. Records can be created after their initialization time. Operational assessments should also check forecast availability at each valid time.

- **MAE** is mean absolute total-cloud error in percentage points. **Bias** is mean forecast minus observed total cloud, also in percentage points. A positive bias means too much forecast cloud.
- A **clear event** means cloud fraction ≤0.30 for both the observation and forecast. Hit rate is hits divided by observed clear cases. False-alarm rate is false alarms divided by observed non-clear cases. The table and JSON leave rates undefined when their denominator is zero.
- Score saved **nowcast**, **model** and **blend** totals on the same valid observations. Older records without a blend use Sky's original blend routine and the saved decay time or its application default.
- **Persistence** holds the matched satellite fraction at `data_time` constant, separately for each reference, margin and radius. If the initial observation is missing or fails quality control, subsequent usable times still score the three saved forecasts. The table gives the separate persistence sample count. Midpoint matching can select a scan completed or published after initialization, so this is a retrospective persistence benchmark. Its zero-lead error is zero whenever defined.

## Interpreting a night

ACM is a satellite retrieval. It can miss optically thin cirrus. High clouds can be displaced by parallax, particularly at oblique view angles. A nominal 2 km nadir pixel covers more ground off-nadir, and an areal cloud fraction measures a different quantity from a telescope's point or hemispheric view. The column reference includes clouds below the summit. The above-observer reference combines the mask with a coarser top-height retrieval. Sampling across terrain and cloud edges, height retrieval error, and the loss of samples with missing heights all affect its interpretation.

Adjacent ten-minute samples are correlated. For a multi-night claim, choose a primary site, radius, lead and comparison in advance. Calculate paired nightly differences in MAE and estimate uncertainty using independent nights or weather-episode blocks. As a planning example, a two-sided 5% test with 80% power for a 5 percentage-point difference needs about 32 independent nights per site if the nightly paired standard deviation is 10 points, or 71 if it is 15. These are normal approximations. Estimate the actual spread from pilot nights and include both clear and cloudy episodes to estimate the two binary rates.

## Results for 18 November 2026, 02:00–08:00 UTC

Each record requested 37 forecast times. The comparison below gives each retained time equal weight. All matched ACHA and ACM midpoints coincided in this run. Across every lead bucket and radius, the column fractions and scores reproduced the previous run exactly. [Full verifier output by radius, margin and lead](satellite-verification-2026-11-18.txt)

| Site, 10 km disk | Times with complete heights | Cloud above observer, +300 m | Any cloud on those same times | Any cloud over all 37 times |
|---|---:|---:|---:|---:|
| Mauna Kea | 24 | 5.62% | 21.56% | 18.23% |
| Paranal | 25 | 91.82% | 91.82% | 84.66% |
| Roque de los Muchachos | 0 (view rejected) | — | — | — |

At Mauna Kea the height filter strengthens the finding of forecast overcast at a mostly clear summit. The model averaged 100% cloud on the 24 retained times. All three saved forecasts missed every one of the 23 clear events at the 300 m margin. Mean above-observer cloud was 6.75% with a 0 m margin and 0.73% with a 1,000 m margin. Thirteen samples contained a total of 52 cloudy mask pixels lacking a good height.

At Paranal every cloudy pixel on the 25 retained times had a top above 3,635 m, even at the largest margin. At initialization, the ACHA cell containing the site had a good-quality top of 11.10 km. The apparent increase from 84.66% to 91.82% comes entirely from excluding 12 samples with a total of 112 missing cloudy-pixel heights. The blend had the lowest MAE over all 37 column samples. Persistence had the lowest MAE on the retained 25 times for both references. These results support high cloud on the retained samples and leave the cloud heights on the excluded times unresolved. [GOES-19 ACHA initialization scan](https://noaa-goes19.s3.amazonaws.com/ABI-L2-ACHAF/2026/322/01/OR_ABI-L2-ACHAF-M6_G19_s20263220150211_e20263220159520_c20263220202262.nc)

Roque de los Muchachos exceeded the 70° view limit at the edge of its 25 km footprint. Both references therefore lack a score at this site.
