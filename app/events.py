import json
import logging
from datetime import UTC, datetime
from time import perf_counter

logger = logging.getLogger("climate_sim")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def log_event(event, **fields):
    logger.info(
        json.dumps(
            {"timestamp": datetime.now(UTC).isoformat(), "event": event, **fields},
            separators=(",", ":"),
        )
    )


class RequestLogMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = perf_counter()
        status = 500

        async def observed(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, observed)
        finally:
            log_event(
                "request",
                method=scope["method"],
                path=scope["path"],
                status=status,
                duration_ms=round((perf_counter() - started) * 1000, 2),
            )
