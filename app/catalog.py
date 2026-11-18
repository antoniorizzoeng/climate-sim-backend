"""Keyset pagination for the run and field catalogues."""

import base64
import json
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import Query
from pydantic import AwareDatetime, BeforeValidator

from .models import invalid

Sort = Literal[
    "created_desc", "created_asc", "size_desc", "size_asc", "name_asc", "name_desc"
]


def iso_datetime(value):
    if not any(sep in value for sep in ("T", "t", " ")):
        raise ValueError("Expected an ISO date-time with timezone")
    return value


Timestamp = Annotated[AwareDatetime, BeforeValidator(iso_datetime)]


def query(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: str | None = None,
    q: Annotated[str, Query(max_length=120)] = "",
    created_from: Timestamp | None = None,
    created_to: Timestamp | None = None,
    sort: Sort = "created_desc",
):
    return limit, cursor, q, created_from, created_to, sort


def decode(cursor, filters, sort):
    try:
        raw = base64.b64decode(cursor, altchars=b"-_", validate=True).decode()
        if raw.startswith("["):
            saved, value, identifier = json.loads(raw)
            if saved != filters:
                raise ValueError
        else:
            # 1.3 issued timestamp|id markers for the unfiltered descending list.
            if filters[1:] != ["", None, None, "created_desc"]:
                raise ValueError
            value, identifier = raw.split("|")
        if not isinstance(identifier, str):
            raise ValueError
        if sort.startswith("created"):
            value = datetime.fromisoformat(value)
            if value.tzinfo is None:
                raise ValueError
        elif type(value) is not (int if sort.startswith("size") else str):
            raise ValueError
        return value, identifier
    except (ValueError, TypeError, OverflowError) as exc:
        raise invalid("cursor", "Invalid cursor for this query", "query") from exc


def page(kind, records, params, size_of, view):
    limit, cursor, q, lower, upper, sort = params
    filters = [
        kind,
        q,
        *(v.astimezone(UTC).isoformat() if v else None for v in (lower, upper)),
        sort,
    ]
    after = decode(cursor, filters, sort) if cursor is not None else None
    matches = []
    total_bytes = total = 0
    descending = sort.endswith("_desc")
    search = q.casefold()
    order = sort.split("_")[0]
    for item in records:
        stamp = datetime.fromisoformat(item["created_at"])
        if (lower and stamp < lower) or (upper and stamp >= upper):
            continue
        name = item["name"].casefold()
        if search not in name and not item["id"].casefold().startswith(search):
            continue
        size = size_of(item)
        total += 1
        total_bytes += size
        value = {"created": stamp, "size": size, "name": name}[order]
        key = value, item["id"]
        if after is None or (key < after if descending else key > after):
            matches.append((key, item))
    matches.sort(key=lambda entry: entry[0], reverse=descending)
    next_cursor = None
    if len(matches) > limit:
        value, identifier = matches[limit - 1][0]
        marker = [
            filters,
            value.isoformat() if isinstance(value, datetime) else value,
            identifier,
        ]
        next_cursor = base64.urlsafe_b64encode(json.dumps(marker).encode()).decode()
    return dict(
        items=[view(item) for _, item in matches[:limit]],
        next_cursor=next_cursor,
        total=total,
        total_bytes=total_bytes,
    )
