# climate-sim backend

A single-user FastAPI service for uploading NetCDF fields, running the local MPI simulator, reading live frames and exporting animations. The API contract is [api/openapi.yaml](api/openapi.yaml). A running service serves the contract at `/openapi.json` and interactive documentation at `/docs`.

## Local development

Install Python 3.12 with venv support, Open MPI and ffmpeg with libx264. Build the [MPI engine](https://github.com/antoniorizzoeng/climate-sim-mpi-cpp) with CMake, PnetCDF, NetCDF-C and yaml-cpp. Point the backend at the resulting executable.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
export ENGINE_BIN="$HOME/climate-sim/climate-sim-mpi-cpp/build/src/climate_sim"
export MPIRUN_ARGS=--oversubscribe
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Use one API process per data directory. The manager enforces exclusive ownership, runs jobs in submission order and serves independent requests during MPI execution and encoding. On restart it resumes queued jobs and marks interrupted simulations as errors.

## Docker Compose

```bash
docker compose up --build
```

Open http://localhost:8000/docs. The image builds the simulator from the engine repository at main and includes Open MPI and ffmpeg. Compose mounts the named `simulations` volume at `/data`. The service listens on the host loopback interface. Use `docker compose down` to stop it and preserve simulation data.

## Settings

| Variable | Default for local execution | Purpose |
| --- | --- | --- |
| DATA_DIR | ./data | Persistent fields, requests, output and exports. |
| ENGINE_BIN | climate_sim on PATH | Engine executable, optionally an absolute path. Health reports the CMake package version when available and identifies the binary by a SHA-256 fingerprint. |
| MPIRUN | mpirun | Local MPI launcher. |
| MPIRUN_ARGS | empty | Additional shell-quoted launcher arguments, e.g. --oversubscribe. Compose defaults to --oversubscribe. |
| CORS_ORIGINS | http://localhost:3000 | Comma-separated browser origins. |
| ENGINE_PERIODIC | 0 | Set to 1 when the installed engine supports periodic boundary conditions. |

## Run and retrieve a simulation

```bash
curl -s http://localhost:8000/runs -H 'Content-Type: application/json' -d '{
  "source":{"kind":"preset","preset":"gaussian_hotspot","nx":32,"ny":16},
  "processes":2, "dt":0.1, "D":0.2, "steps":25, "frames":8
}'
```

Poll `/runs/{id}` for the readable frame count. Frame indices start at zero. Fetch `/runs/{id}/frames/0.png` for an image or `/runs/{id}/frames/0/values` for little-endian float32 cells starting with the bottom row. After completion, download `/runs/{id}/result.nc` or poll `/runs/{id}/export.mp4`, respecting the Retry-After header. GIF exports use the same interface.

Fields remain active for seven days from last use and archived for another thirty. Archived originals remain downloadable and starting a run reactivates the field. Run outputs expire seven days after completion. A run can be repeated until its record expires on day 37 if the source is still available. Cleanup runs on startup and hourly.

## Checks

```bash
flake8 app tests
pytest -q
```

For the opt-in live data check, run `RUN_NETWORK_TESTS=1 pytest -m network --no-cov`. Test dependencies include pytest-cov. CI enforces 90% branch-inclusive coverage and builds the engine from main for the integration tests. The local suite looks for ENGINE_BIN or the development build shown above. It reports a skip reason if the executable is missing.

## Sky nowcast

Post `{"lat":45,"lon":10}` to `/nowcasts` and poll the returned ID. The default models six hours on a 96 × 96 grid extending 300 km in each direction. The three layer IDs open ordinary run frames, PNGs and exports. Refresh creates a separate forecast with fresh data. Nowcasts expire seven days after completion.

Open-Meteo supplies the initial cloud cover and hourly winds. Only the site and grid coordinates, requested variables and UTC hours are sent, with a generic User-Agent. All simulation data stays local. The client caches responses for an hour, batches up to 64 points, and counts locations and variables against the free non-commercial quota. [Forecast API](https://open-meteo.com/en/docs) · [Rate limits](https://open-meteo.com/en/pricing)

Weather samples span the outer simulation cell centres on an odd regular grid capped at 17 × 17. Sample spacing is about 37 km at the default radius and 125 km at the largest radius. Bilinear interpolation supplies the engine grid. Row zero is south. Low, mid and high clouds follow winds at 850, 500 and 300 hPa, respectively. Meteorological “from” directions become east/north velocities in m/s (1 m/s = 3.6 km/h). Each layer uses zero-gradient boundaries, diffusion of 10 m²/s and a timestep at most 90% of the combined advection–diffusion stability limit. Frames are ten minutes apart.

The timeline compares transported cloud fractions (random overlap for total) with Open-Meteo’s own site forecast, interpolated hourly totals included. Clear fraction estimates unobscured sky area. The hourly 250 hPa wind gives a seeing rule of thumb (good below 20 m/s, fair from 20 through 35, poor above 35). Local turbulence can dominate. Transport is most useful for a few hours. Clouds can form, dissipate or change altitude, and incoming clouds beyond the grid are unknown. Increasing the simulation resolution adds no weather observations. A startup probe checks engine wind-file support before any forecast is fetched.
