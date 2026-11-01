"""Immutable, engine-independent data contracts for block execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.kalypso.blocks.errors import BlockConfigurationError


@dataclass(frozen=True)
class RowMessage:
    role: str
    content: str


@dataclass(frozen=True)
class BlockRow:
    row_id: str
    messages: tuple[RowMessage, ...]


@dataclass(frozen=True)
class BlockEnvelope:
    block_id: str
    parent_block_id: str | None
    rows: tuple[BlockRow, ...]


@dataclass(frozen=True)
class BlockBatch:
    envelope: BlockEnvelope
    active_positions: tuple[int, ...]


@dataclass(frozen=True)
class PreparedBlockCall:
    block: BlockBatch
    prompt_token_ids: tuple[int, ...]
    json_schema: dict[str, Any]
    max_output_tokens: int
    estimated_kv_bytes: int


@dataclass(frozen=True)
class FilterDecision:
    row_id: str
    passed: bool


@dataclass(frozen=True)
class FilterBlockResult:
    block: BlockBatch
    decisions: tuple[FilterDecision, ...]
    request_id: str | None = None
    finish_reason: str | None = None


def source_row(row_id: str, messages: tuple[RowMessage, ...], *, right_data=()) -> BlockRow:
    """Validate and freeze one supported source snapshot."""
    if right_data:
        raise BlockConfigurationError("nonempty right_data is unsupported for block filters")
    if not messages:
        raise BlockConfigurationError("a source row must contain at least one message")
    for message in messages:
        if not isinstance(message, RowMessage):
            raise BlockConfigurationError("source messages must be RowMessage values")
        if message.role not in {"system", "user", "assistant"} or not isinstance(message.content, str):
            raise BlockConfigurationError("source messages require a supported role and text content")
    return BlockRow(row_id=row_id, messages=tuple(messages))


def validate_batch(batch: BlockBatch) -> None:
    """Check the selection invariant before an operator consumes a batch."""
    positions = batch.active_positions
    if len(set(positions)) != len(positions) or any(
        type(position) is not int or position < 0 or position >= len(batch.envelope.rows)
        for position in positions
    ):
        raise ValueError("active_positions must be unique valid envelope positions")
