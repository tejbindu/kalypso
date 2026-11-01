"""Source adaptation and execution for block-filter plans."""

from __future__ import annotations

import copy
from collections.abc import Callable

from vllm.kalypso.blocks.errors import BlockConfigurationError
from vllm.kalypso.blocks.packing import pack_source_rows, split_selection
from vllm.kalypso.blocks.types import (
    BlockBatch,
    BlockRow,
    RowMessage,
    source_row,
)
from vllm.kalypso.controller.block_plan import BlockFilterPlan
from vllm.kalypso.sem_ops.block_filter import BlockRuntime


def snapshot_source_contexts(contexts) -> tuple[tuple[BlockRow, ...], dict[str, object]]:
    """Assign query-local stable IDs and preserve original contexts by ID."""
    rows: list[BlockRow] = []
    context_by_id = {}
    for index, context in enumerate(contexts):
        row_id = f"r{index}"
        messages = getattr(context.input, "data", None)
        if not isinstance(messages, list) or not messages:
            raise BlockConfigurationError(
                f"source row {row_id} must contain text messages"
            )
        row_messages = []
        for message in messages:
            if not isinstance(message, dict):
                raise BlockConfigurationError(
                    f"source row {row_id} contains an unsupported message"
                )
            if message.get("type", "text") != "text":
                raise BlockConfigurationError(
                    f"source row {row_id} contains non-text content"
                )
            row_messages.append(
                RowMessage(message.get("role"), message.get("content"))
            )
        row = source_row(
            row_id,
            tuple(row_messages),
            right_data=getattr(context.input, "right_data", ()),
        )
        rows.append(row)
        context_by_id[row_id] = context
    return tuple(rows), context_by_id


async def execute_filter_chain(
    plan: BlockFilterPlan,
    contexts,
    runtime: BlockRuntime,
    new_block_id: Callable[[str | None], str],
) -> list:
    """Run a block-only filter chain and adapt final survivors to contexts."""
    if not plan.operators:
        return list(contexts)
    rows, context_by_id = snapshot_source_contexts(contexts)
    history: dict[str, dict[int, tuple[bool, str]]] = {
        row.row_id: {} for row in rows
    }

    prepared = pack_source_rows(rows, plan.operators[0], runtime, new_block_id)
    current_batches: list[BlockBatch] = []
    for call in prepared:
        result = await plan.operators[0].evaluate(call, runtime)
        current_batches.append(result.block)
        for decision in result.decisions:
            history[decision.row_id][plan.operators[0].position] = (
                decision.passed,
                result.block.envelope.block_id,
            )

    for operator in plan.operators[1:]:
        next_batches = []
        for batch in current_batches:
            calls = split_selection(batch, operator, runtime, new_block_id)
            for call in calls:
                result = await operator.evaluate(call, runtime)
                next_batches.append(result.block)
                for decision in result.decisions:
                    history[decision.row_id][operator.position] = (
                        decision.passed,
                        result.block.envelope.block_id,
                    )
        current_batches = next_batches

    survivor_ids = {
        batch.envelope.rows[position].row_id
        for batch in current_batches
        for position in batch.active_positions
    }
    ordered_ids = sorted(survivor_ids, key=lambda row_id: int(row_id[1:]))
    output_contexts = []
    for row_id in ordered_ids:
        context = copy.copy(context_by_id[row_id])
        context.input = copy.deepcopy(context_by_id[row_id].input)
        context.output = copy.deepcopy(context_by_id[row_id].output)
        context.state = copy.copy(context_by_id[row_id].state)
        for position, (passed, block_id) in sorted(history[row_id].items()):
            context.output.append(
                {
                    "SemFilter": {
                        "passed": passed,
                        "operator_position": position,
                        "block_id": block_id,
                    }
                }
            )
        output_contexts.append(context)
    return output_contexts
