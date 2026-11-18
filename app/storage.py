import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


def sync_dir(directory):
    fd = os.open(directory, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(directory, record):
    tmp = directory / ".metadata.tmp"
    with tmp.open("w") as stream:
        json.dump(record, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(directory / "metadata.json")
    sync_dir(directory)


class Collection:
    def __init__(self, root: Path):
        self.root = root
        self.records = {}
        self.sizes = {}

    def measure(self, identifier):
        # Each record owns a flat directory. An encoder can rename a partial
        # file or the engine can replace progress.json while we sample it.
        size = 0
        with os.scandir(self.root / identifier) as files:
            for entry in files:
                try:
                    size += entry.stat().st_size
                except FileNotFoundError:
                    continue
        self.sizes[identifier] = size

    def load(self):
        self.root.mkdir(exist_ok=True)
        for path in self.root.iterdir():
            if path.name.startswith((".new-", ".delete-")):
                shutil.rmtree(path)
            elif not path.name.startswith("."):
                self.records[path.name] = json.loads(
                    (path / "metadata.json").read_text()
                )
                self.measure(path.name)

    @contextmanager
    def staging(self, identifier=None):
        identifier = identifier or uuid4().hex
        directory = self.root / f".new-{identifier}"
        directory.mkdir()
        try:
            yield identifier, directory
        finally:
            if directory.exists():
                shutil.rmtree(directory)

    def publish(self, directory, record):
        save(directory, record)
        directory.rename(self.root / record["id"])
        sync_dir(self.root)
        self.records[record["id"]] = record
        self.measure(record["id"])

    def update(self, record, **changes):
        updated = dict(record, **changes)
        save(self.root / record["id"], updated)
        record.update(changes)
        self.measure(record["id"])

    def remove(self, identifier):
        directory = self.root / identifier
        trash = self.root / f".delete-{identifier}"
        directory.rename(trash)
        sync_dir(self.root)
        del self.records[identifier]
        del self.sizes[identifier]
        shutil.rmtree(trash)
