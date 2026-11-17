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

Test dependencies include pytest-cov. CI enforces 90% branch-inclusive coverage and builds the engine from main for the integration tests. The local suite looks for ENGINE_BIN or the development build shown above. It reports a skip reason if the executable is missing.
