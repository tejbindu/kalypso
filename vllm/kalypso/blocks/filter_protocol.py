"""Strict, side-effect-free validation for block filter responses."""

from __future__ import annotations

import json
from typing import Any

from vllm.kalypso.blocks.errors import BlockOutputError
from vllm.kalypso.blocks.types import FilterDecision


def decision_schema(row_ids: tuple[str, ...]) -> dict[str, Any]:
    """Build the constrained-decoding schema for exactly these row IDs."""
    if not row_ids:
        raise ValueError("a decision schema requires at least one row ID")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("row IDs must be unique")
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": len(row_ids),
                "maxItems": len(row_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": list(row_ids)},
                        "passed": {"type": "boolean"},
                    },
                    "required": ["id", "passed"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant {value}")


def parse_filter_response(
    response: str,
    row_ids: tuple[str, ...],
    *,
    finish_reason: str,
    negate: bool = False,
    block_id: str | None = None,
) -> tuple[FilterDecision, ...]:
    """Validate all decisions atomically and return them in input order."""
    if finish_reason != "stop":
        raise BlockOutputError(
            f"expected finish_reason 'stop', got {finish_reason!r}", block_id
        )
    if not row_ids or len(set(row_ids)) != len(row_ids):
        raise BlockOutputError("submitted row IDs must be nonempty and unique", block_id)
    try:
        parsed = json.loads(
            response,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise BlockOutputError(f"response is not one valid JSON document: {exc}", block_id) from exc

    if type(parsed) is not dict or set(parsed) != {"decisions"}:
        raise BlockOutputError("root must contain only the decisions property", block_id)
    raw_decisions = parsed["decisions"]
    if type(raw_decisions) is not list:
        raise BlockOutputError("decisions must be an array", block_id)

    by_id: dict[str, bool] = {}
    expected = set(row_ids)
    for item in raw_decisions:
        if type(item) is not dict or set(item) != {"id", "passed"}:
            raise BlockOutputError("each decision must contain only id and passed", block_id)
        row_id, passed = item["id"], item["passed"]
        if type(row_id) is not str or row_id not in expected:
            raise BlockOutputError(f"unknown or invalid row ID {row_id!r}", block_id)
        if row_id in by_id:
            raise BlockOutputError(f"duplicate decision for row {row_id}", block_id)
        if type(passed) is not bool:
            raise BlockOutputError(f"passed for row {row_id} must be a Boolean", block_id)
        by_id[row_id] = passed

    if set(by_id) != expected:
        missing = [row_id for row_id in row_ids if row_id not in by_id]
        raise BlockOutputError(f"missing decisions for rows {missing}", block_id)
    return tuple(FilterDecision(row_id, by_id[row_id] ^ negate) for row_id in row_ids)
