"""Stable greedy packing and downstream splitting for block filters."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from vllm.kalypso.blocks.errors import BlockSizingError
from vllm.kalypso.blocks.types import (
    BlockBatch,
    BlockEnvelope,
    BlockRow,
    PreparedBlockCall,
)
from vllm.kalypso.sem_ops.block_filter import BlockFilter, BlockRuntime


BlockIdFactory = Callable[[str | None], str]


def _make_batch(
    rows: tuple[BlockRow, ...],
    block_id: str,
    parent_block_id: str | None,
) -> BlockBatch:
    return BlockBatch(
        BlockEnvelope(block_id, parent_block_id, rows),
        tuple(range(len(rows))),
    )


def _greedy_pack(
    rows: Iterable[BlockRow],
    operator: BlockFilter,
    runtime: BlockRuntime,
    new_block_id: BlockIdFactory,
    parent_block_id: str | None,
) -> list[PreparedBlockCall]:
    prepared: list[PreparedBlockCall] = []
    pending: list[BlockRow] = []
    pending_id: str | None = None
    pending_call: PreparedBlockCall | None = None
    max_rows = operator.config.block.max_rows

    for row in rows:
        if len(pending) >= max_rows:
            assert pending_call is not None
            prepared.append(pending_call)
            pending, pending_id, pending_call = [], None, None

        block_id = pending_id or new_block_id(parent_block_id)
        candidate = (*pending, row)
        try:
            candidate_call = operator.prepare(
                _make_batch(candidate, block_id, parent_block_id), runtime
            )
        except BlockSizingError:
            if not pending:
                raise
            assert pending_call is not None
            prepared.append(pending_call)
            pending, pending_id, pending_call = [], None, None
            block_id = new_block_id(parent_block_id)
            try:
                pending_call = operator.prepare(
                    _make_batch((row,), block_id, parent_block_id), runtime
                )
            except BlockSizingError as exc:
                if exc.row_id is None:
                    exc.row_id = row.row_id
                raise
            pending = [row]
            pending_id = block_id
        else:
            pending.append(row)
            pending_id = block_id
            pending_call = candidate_call

    if pending_call is not None:
        prepared.append(pending_call)
    return prepared


def pack_source_rows(
    rows: Iterable[BlockRow],
    operator: BlockFilter,
    runtime: BlockRuntime,
    new_block_id: BlockIdFactory,
) -> list[PreparedBlockCall]:
    """Pack source rows in order, flushing a valid partial tail."""
    return _greedy_pack(rows, operator, runtime, new_block_id, None)


def split_selection(
    batch: BlockBatch,
    operator: BlockFilter,
    runtime: BlockRuntime,
    new_block_id: BlockIdFactory,
) -> list[PreparedBlockCall]:
    """Split one incoming selection stably without merging envelopes."""
    active_rows = tuple(
        batch.envelope.rows[position] for position in batch.active_positions
    )
    if not active_rows:
        return []
    if len(active_rows) <= operator.config.block.max_rows:
        try:
            return [operator.prepare(batch, runtime)]
        except BlockSizingError:
            if len(active_rows) == 1:
                raise
    return _greedy_pack(
        active_rows,
        operator,
        runtime,
        new_block_id,
        batch.envelope.block_id,
    )
