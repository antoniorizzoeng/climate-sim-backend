import hashlib
import io
import json
import os
import selectors
import subprocess
from contextlib import suppress
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from matplotlib import colormaps
from PIL import Image

from .disk import CHUNK_BYTES
from .models import Colormap
from .netcdf import FrameView, read_image_plane, rectangles


@dataclass(frozen=True)
class Style:
    colormap: Colormap = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    max_side: int | None = None


@lru_cache(maxsize=len(Colormap.__args__))
def palette(name):
    cmap = colormaps[name]
    rgba = cmap(np.linspace(0, 1, cmap.N), bytes=True)
    bad = np.asarray(cmap(np.nan, bytes=True), dtype=np.uint8)
    return np.vstack((rgba, bad)), rgba.tobytes()


def color_indices(values, style, limits, colors, whole_field):
    lo, hi = (
        (limits or (None, None)) if style.vmin is None else (style.vmin, style.vmax)
    )
    finite = np.isfinite(values)
    opaque = finite.all()
    # Constant fields use the midpoint by default even with a widened run range.
    uniform = (
        whole_field and style.vmin is None and opaque and values.min() == values.max()
    )
    dtype = np.uint8 if opaque else np.uint16
    if lo is None or lo == hi or uniform:
        indices = np.full(values.shape, colors // 2, dtype)
    else:
        indices = np.empty(values.shape, dtype)
        span = hi - lo
        with np.errstate(over="ignore", invalid="ignore"):
            for rect in rectangles(*values.shape, cells=512 * 1024):
                block = values[rect]
                if np.isfinite(span):
                    scaled = np.subtract(block, lo, dtype=np.float64)
                    scaled /= span
                else:
                    scaled = np.multiply(block, 0.5, dtype=np.float64)
                    scaled -= lo * 0.5
                    scaled /= hi * 0.5 - lo * 0.5
                scaled *= colors
                np.clip(scaled, 0, colors - 1, out=scaled)
                indices[rect] = scaled
    if not opaque:
        indices[~finite] = colors
    return indices[::-1], opaque


def display_size(nx, ny, max_side):
    if max_side is None or max(nx, ny) <= max_side:
        return nx, ny
    scale = max_side / max(nx, ny)
    return max(1, round(nx * scale)), max(1, round(ny * scale))


def render(values, style, limits, *, indexed=False, whole_field=True):
    lut, rgba_bytes = palette(style.colormap)
    indices, opaque = color_indices(values, style, limits, len(lut) - 1, whole_field)
    ny, nx = indices.shape
    size = display_size(nx, ny, style.max_side)
    if size != (nx, ny):
        # BOX averages each block in Pillow's premultiplied color space.
        # Process one byte per source pixel at a time.
        channels = [
            Image.fromarray(lut[:, channel][indices]).resize(size, Image.Resampling.BOX)
            for channel in range(3 if opaque else 4)
        ]
        if opaque:
            channels.append(Image.new("L", size, 255))
            return Image.merge("RGBA", channels)
        return Image.merge("RGBa", channels).convert("RGBA")
    if opaque:
        image = Image.fromarray(indices)
        image.putpalette(rgba_bytes, rawmode="RGBA")
        return image if indexed else image.convert("RGBA")
    return Image.fromarray(lut[indices])


def png(values, style, limits, *, indexed=False, whole_field=True):
    buf = io.BytesIO()
    render(values, style, limits, indexed=indexed, whole_field=whole_field).save(
        buf, format="PNG"
    )
    return buf.getvalue()


def export_name(fmt, fps, style, view=FrameView()):
    options = [fps, asdict(style)]
    if view != FrameView():
        options.append(asdict(view))
        if style.vmin is None:
            options.append("whole-volume-scale")
    key = json.dumps(options, sort_keys=True).encode()
    return f"export-{hashlib.sha256(key).hexdigest()}.{fmt}"


def encoder_command(target, fps, size):
    if target.suffix == ".mp4":
        codec = [
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "2",
            "-tune",
            "zerolatency",
            "-movflags",
            "+empty_moov+default_base_moof",
            "-frag_duration",
            "1000000",
            "-f",
            "mp4",
        ]
    else:
        codec = [
            "-filter_complex_threads",
            "1",
            "-filter_complex",
            "split[a][b];[a]palettegen=stats_mode=single[p];[b][p]paletteuse=new=1",
            "-f",
            "gif",
        ]
    return [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{size[0]}x{size[1]}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        *codec,
        "pipe:1",
    ]


def pump(process, frames, stream, count, cancelled, progress, disk):
    remaining = memoryview(b"")
    sent = 0
    with selectors.DefaultSelector() as selector:
        for pipe, mode in [
            (process.stdout, selectors.EVENT_READ),
            (process.stdin, selectors.EVENT_WRITE),
        ]:
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, mode)
        while selector.get_map() and not cancelled.is_set():
            for key, mode in selector.select(timeout=0.1):
                pipe = key.fileobj
                if mode == selectors.EVENT_READ:
                    chunk = os.read(key.fd, CHUNK_BYTES)
                    if chunk:
                        disk.write(stream, chunk, "animation export")
                    else:
                        selector.unregister(pipe)
                else:
                    if not remaining:
                        if sent == count:
                            selector.unregister(pipe)
                            pipe.close()
                            continue
                        remaining = memoryview(next(frames).tobytes())
                    try:
                        remaining = remaining[os.write(key.fd, remaining) :]
                    except BlockingIOError:
                        continue
                    if not remaining:
                        sent += 1
                        progress[0] = sent / count


def wait_encoder(process, cancelled):
    while not cancelled.is_set():
        try:
            status = process.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            continue
        if status:
            raise RuntimeError("ffmpeg failed to encode animation")
        return True
    return False


def stop_encoder(process):
    if process.poll() is None:
        process.kill()
    process.wait()


def finish_mp4(fragmented, target, count, cancelled, disk):
    # The muxer writes an index after the packets. Reserve space for both
    # files during the stream copy, including the sample tables.
    tmp = target.with_suffix(".partial.indexed.mp4")
    size = fragmented.stat().st_size + 64 * count + 4096
    with disk.reserve(size, "MP4 index and stream copy", tmp):
        # Give the muxer seekable descriptors. A surviving child after a hard
        # restart can only write to the old inode, even if its name is reused.
        try:
            with fragmented.open("rb") as source, tmp.open("wb") as stream:
                process = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        "/proc/self/fd/0",
                        "-c",
                        "copy",
                        "-f",
                        "mp4",
                        "/proc/self/fd/1",
                    ],
                    stdin=source,
                    stdout=stream,
                )
                try:
                    if wait_encoder(process, cancelled):
                        tmp.replace(target)
                finally:
                    stop_encoder(process)
        finally:
            tmp.unlink(missing_ok=True)


def encode(
    directory, target, fps, style, run, cancelled, progress, disk, view=FrameView()
):
    tmp = target.with_suffix(".partial" + target.suffix)
    if cancelled.is_set():
        return

    def frame(n):
        plane, whole_field = read_image_plane(
            directory / "result.nc", n, view, style.vmin is None, cancelled
        )
        return render(
            plane, style, run["display_range"], whole_field=whole_field
        ).convert("RGB")

    grid = run["resolved"]
    ny, nx = view.plane_shape((grid["nz"], grid["ny"], grid["nx"]))
    size = display_size(nx, ny, style.max_side)
    process = None
    try:
        with tmp.open("wb", buffering=0) as stream:
            process = subprocess.Popen(
                encoder_command(target, fps, size),
                cwd=directory,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                bufsize=0,
            )

            frames = (frame(n) for n in range(run["frames"]))
            pump(process, frames, stream, run["frames"], cancelled, progress, disk)
        if wait_encoder(process, cancelled):
            if target.suffix == ".mp4":
                finish_mp4(tmp, target, run["frames"], cancelled, disk)
            else:
                tmp.replace(target)
    finally:
        if process is not None:
            stop_encoder(process)
            with suppress(BrokenPipeError):
                process.stdin.close()
            process.stdout.close()
        tmp.unlink(missing_ok=True)
