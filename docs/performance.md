# Frame rendering benchmark

Run this manually from the checkout with the development dependencies installed.

```bash
python -m scripts.benchmark_rendering --samples 9
```

The script creates a 4096 × 4096 float64 frame in a CDF-5 file with the same layout as the engine output. The frame is a sine/cosine surface with a fixed display range. It calls the frame endpoints through FastAPI's in-process test client and measures through receipt of each complete response. Each case warms up once before sampling. The window starts at (1536, 1536), covers 1024 × 1024 cells, and samples every fourth cell along both axes.

| Request | Before | After | Bytes before | Bytes after |
| --- | ---: | ---: | ---: | ---: |
| Native PNG | 1467.28 ms | 297.86 ms | 2,370,274 | 908,735 |
| PNG with `max_side=512` | 876.87 ms | 449.06 ms | 221,709 | 221,709 |
| Native float32 values | 153.61 ms | 157.50 ms | 67,108,864 | 67,108,864 |
| Float32 values, window and stride | 63.46 ms | 7.14 ms | 262,144 | 262,144 |

Medians of nine requests on an AMD Ryzen 5 8500G, pinned to one logical CPU with `taskset -c 2`. Both runs used Python 3.12.3, netCDF4 1.7.5.1.post1, NumPy 2.5.5, Matplotlib 3.11.2, and Pillow 12.3.0. The baseline used the same script with backend revision `8d0af88`. The timings exclude network transfer and reflect a warm filesystem cache.

Opaque native frame PNGs use a lossless indexed palette. Downscaled field previews and frame PNGs read NetCDF row strips and preserve Pillow's BOX filtering of premultiplied colors. Missing cells contribute no color and all-missing blocks are transparent. Field previews with automatic color limits scan the source plane for its finite minimum and maximum before rendering. Windowed values use NetCDF slices before conversion to float32. A native values request still reads and transfers the entire plane.

## Engine memory

Peak anonymous memory from cgroup v2 `memory.stat` (`anon`) on WSL with 12 MPI ranks.

| Cells | Peak engine RAM |
| ---: | ---: |
| 16.8 million | 0.55 GiB |
| 67.1 million | 1.73 GiB (same with 2 or 6 frames) |
| 134.2 million | 3.30 GiB |
| 201.3 million | 4.89 GiB |

The measurements imply roughly 26 bytes per cell and 150 MB fixed overhead. Validation uses 28 bytes per cell (`ENGINE_BYTES_PER_CELL`), 64 MiB per run (`ENGINE_FIXED_BYTES`), and 16 MiB per MPI rank (`ENGINE_BYTES_PER_RANK`). The cell count is `nx × ny × nz`, with `nz = 1` for a plane. For spatially varying NetCDF wind, the estimate adds 8 bytes per cell per component (`WIND_BYTES_PER_CELL_COMPONENT`). Sky supplies two components, `vx` and `vy`. The estimate is independent of output frame count.

The capacity is the lesser of physical RAM and the smallest finite cgroup v2 `memory.max` in the process's hierarchy, minus the configured reserve. Runs execute one at a time, so live memory use does not affect this budget. `ENGINE_MEMORY_RESERVE_BYTES` defaults to 512 MiB and `ENGINE_MEMORY_BYTES` overrides the final capacity.
