from fastapi import UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from python_multipart.multipart import parse_options_header
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile as StarletteUpload
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import ClientDisconnect

from .disk import CHUNK_BYTES, DiskSpaceError
from .models import invalid

BINARY = {b"", b"application/octet-stream"}
NETCDF = BINARY | {
    b"application/x-netcdf",
    b"application/netcdf",
    b"application/x-hdf",
    b"application/x-hdf5",
}


class Upload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file: UploadFile


class Drawing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: UploadFile
    nx: int = Field(ge=4)
    ny: int = Field(ge=4)
    name: str = "drawn"


class DiskUpload(StarletteUpload):
    def __init__(self, upload, path, disk):
        super().__init__(
            path.open("w+b", buffering=0),
            filename=upload.filename,
            headers=upload.headers,
        )
        self.disk = disk

    async def write(self, data):
        await run_in_threadpool(self.disk.write, self.file, data, "field upload")


class FieldMultipart(MultiPartParser):
    def __init__(self, request, path, disk, drawn):
        super().__init__(
            request.headers,
            self.chunks(request),
            max_files=1,
            max_fields=3 if drawn else 0,
        )
        self.path = path
        self.disk = disk
        self.drawn = drawn
        self.complete = False
        self.upload = None

    @staticmethod
    async def chunks(request):
        async for chunk in request.stream():
            for start in range(0, len(chunk), CHUNK_BYTES):
                yield chunk[start : start + CHUNK_BYTES]

    def on_part_begin(self):
        super().on_part_begin()
        self.header_bytes = 0

    def count_header_bytes(self, size):
        self.header_bytes += size
        if self.header_bytes > CHUNK_BYTES:
            raise MultiPartException("Multipart part headers are too large")

    def on_header_field(self, data, start, end):
        self.count_header_bytes(end - start)
        super().on_header_field(data, start, end)

    def on_header_value(self, data, start, end):
        self.count_header_bytes(end - start)
        super().on_header_value(data, start, end)

    def on_headers_finished(self):
        super().on_headers_finished()
        part = self._current_part
        if part.file is None:
            return
        # Starlette parsed the part headers into an empty spool. Write the
        # body directly into this request's staging directory.
        part.file.file.close()
        field = "data" if self.drawn else "file"
        if part.field_name != field:
            raise invalid(field, f"Expected multipart file part {field!r}")
        media, _ = parse_options_header(part.file.content_type)
        if media.lower() not in (BINARY if self.drawn else NETCDF):
            raise invalid(field, "Unsupported file content type")
        self.upload = part.file = DiskUpload(part.file, self.path, self.disk)

    def on_end(self):
        self.complete = True


async def read_form(request, path, disk, drawn):
    media, _ = parse_options_header(request.headers.get("content-type"))
    field = "data" if drawn else "file"
    if media.lower() != b"multipart/form-data":
        raise invalid(field, "Expected multipart/form-data")
    parser = FieldMultipart(request, path, disk, drawn)
    try:
        form = await parser.parse()
        if not parser.complete:
            raise invalid(field, "Incomplete multipart upload")
        if len(form.multi_items()) != len(form):
            raise invalid(field, "Supply each multipart part once")
        if drawn:
            drawing = Drawing.model_validate(dict(form))
            return drawing.name, (drawing.ny, drawing.nx)
        upload = Upload.model_validate(dict(form))
        return upload.file.filename or "field.nc", None
    except (MultiPartException, ClientDisconnect, DiskSpaceError) as exc:
        raise invalid(field, str(exc) or "Upload interrupted") from exc
    except ValidationError as exc:
        raise RequestValidationError(
            [dict(error, loc=("body", *error["loc"])) for error in exc.errors()]
        ) from exc
    finally:
        if parser.upload is not None:
            await parser.upload.close()
