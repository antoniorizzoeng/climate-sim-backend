import hashlib
import os
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path


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
            os.getenv("ENGINE_PERIODIC") == "1",
            cpu_count or 1,
            disk_margin,
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
