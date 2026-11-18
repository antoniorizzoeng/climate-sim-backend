import hashlib
import os
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

from .memory import DEFAULT_ENGINE_MEMORY_RESERVE_BYTES, engine_memory_capacity


@dataclass(frozen=True)
class Settings:
    data: Path
    engine: str
    mpirun: str
    mpi_args: tuple[str, ...]
    origins: tuple[str, ...]
    periodic: bool
    cpu_count: int
    disk_margin_bytes: int = 2 * 1024**3
    preview_memory_bytes: int = 256 * 1024**2
    native_image_ram_fraction: float = 0.25
    engine_memory_reserve_bytes: int = DEFAULT_ENGINE_MEMORY_RESERVE_BYTES
    engine_memory_bytes: int | None = None
    engine_nice: int = 10

    @classmethod
    def from_env(cls):
        engine = os.path.expanduser(os.getenv("ENGINE_BIN", "climate_sim"))
        executable = shutil.which(engine)
        launcher = os.path.expanduser(os.getenv("MPIRUN", "mpirun"))
        if os.sep in launcher:
            launcher = str(Path(launcher).resolve())
        cpu_count = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count()
        )
        disk_margin = int(os.getenv("DISK_SAFETY_MARGIN_BYTES", str(2 * 1024**3)))
        if disk_margin < 0:
            raise ValueError("DISK_SAFETY_MARGIN_BYTES must be nonnegative")
        preview_memory = int(os.getenv("PREVIEW_MEMORY_BYTES", str(256 * 1024**2)))
        if preview_memory <= 0:
            raise ValueError("PREVIEW_MEMORY_BYTES must be positive")
        native_fraction = float(os.getenv("NATIVE_IMAGE_RAM_FRACTION", "0.25"))
        if not 0 < native_fraction <= 1:
            raise ValueError("NATIVE_IMAGE_RAM_FRACTION must be in (0, 1]")
        memory_reserve = int(
            os.getenv(
                "ENGINE_MEMORY_RESERVE_BYTES", str(DEFAULT_ENGINE_MEMORY_RESERVE_BYTES)
            )
        )
        if memory_reserve < 0:
            raise ValueError("ENGINE_MEMORY_RESERVE_BYTES must be nonnegative")
        engine_nice = int(os.getenv("ENGINE_NICE", "10"))
        if not 0 <= engine_nice <= 19:
            raise ValueError("ENGINE_NICE must be in 0..19")
        memory_override = os.getenv("ENGINE_MEMORY_BYTES")
        memory_bytes = int(memory_override) if memory_override else None
        if memory_bytes is not None and memory_bytes < 0:
            raise ValueError("ENGINE_MEMORY_BYTES must be nonnegative")
        return cls(
            Path(os.getenv("DATA_DIR", "data")).expanduser().resolve(),
            str(Path(executable).resolve()) if executable else engine,
            launcher,
            tuple(shlex.split(os.getenv("MPIRUN_ARGS", ""))),
            tuple(
                x.strip()
                for x in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
                if x.strip()
            ),
            os.getenv("ENGINE_PERIODIC", "1") == "1",
            cpu_count or 1,
            disk_margin,
            preview_memory,
            native_fraction,
            memory_reserve,
            memory_bytes,
            engine_nice,
        )

    def engine_memory_capacity(self) -> int:
        return engine_memory_capacity(
            self.engine_memory_reserve_bytes, self.engine_memory_bytes
        )

    def engine_description(self):
        try:
            with open(self.engine, "rb") as binary:
                version = hashlib.file_digest(binary, "sha256").hexdigest()[:12]
            prefix = Path(self.engine).parent.parent
            package = "climate-sim-mpi-cpp"
            filename = f"{package}ConfigVersion.cmake"
            for config in (
                prefix / filename,
                prefix / "lib" / "cmake" / package / filename,
            ):
                if config.is_file():
                    match = re.search(
                        r'set\(PACKAGE_VERSION "([^"]+)"\)', config.read_text()
                    )
                    if match:
                        return f"{self.engine} ({match[1]}, sha256 {version})"
            return f"{self.engine} (sha256 {version})"
        except OSError:
            return f"{self.engine} (unavailable)"
