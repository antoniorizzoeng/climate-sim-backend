import errno
import os
import shutil
from contextlib import contextmanager
from threading import Lock

CHUNK_BYTES = 256 * 1024


class DiskSpaceError(ValueError):
    pass


class DiskBudget:
    def __init__(self, directory, margin):
        self.directory = directory
        self.margin = margin
        self.lock = Lock()
        self.reservations = {}

    @property
    def free(self):
        return shutil.disk_usage(self.directory).free

    def _available(self):
        outstanding = 0
        for size, path in self.reservations.values():
            allocated = 0
            if path is not None:
                try:
                    allocated = path.stat().st_blocks * 512
                except FileNotFoundError:
                    pass
            outstanding += max(0, size - allocated)
        return self.free - self.margin - outstanding

    @property
    def available(self):
        with self.lock:
            return self._available()

    def _check(self, size, description):
        available = self._available()
        if size > available:
            raise DiskSpaceError(
                f"Insufficient disk space for {description}: "
                f"needs {size} bytes, {max(0, available)} available "
                f"above the {self.margin}-byte safety margin"
            )

    def check(self, size, description):
        with self.lock:
            self._check(size, description)

    @contextmanager
    def reserve(self, size, description, path=None):
        key = object()
        with self.lock:
            self._check(size, description)
            self.reservations[key] = (size, path)
        try:
            yield
        finally:
            with self.lock:
                del self.reservations[key]

    def write(self, stream, data, description):
        with self.lock:
            block = os.statvfs(self.directory).f_frsize
            size = (len(data) + block - 1) // block * block
            self._check(size, description)
            try:
                if stream.write(data) != len(data):
                    raise DiskSpaceError(f"Short write while writing {description}")
            except OSError as exc:
                if exc.errno == errno.ENOSPC:
                    raise DiskSpaceError(
                        f"Disk is full while writing {description}"
                    ) from exc
                raise
