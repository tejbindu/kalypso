"""Source adaptation and execution for block-filter plans."""

from __future__ import annotations

import asyncio
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
    """Run a bounded concurrent filter pipeline and adapt final survivors."""
    if not plan.operators:
        return list(contexts)
    rows, context_by_id = snapshot_source_contexts(contexts)
    history: dict[str, dict[int, tuple[bool, str]]] = {
        row.row_id: {} for row in rows
    }

    queues = [
        asyncio.Queue(maxsize=operator.config.block.queue_capacity)
        for operator in plan.operators
    ]
    stop_counts = [0 for _ in plan.operators]
    final_survivor_ids: set[str] = set()
    stop = object()

    async def publish(stage_index: int, batch: BlockBatch) -> None:
        if stage_index + 1 < len(queues):
            await queues[stage_index + 1].put(batch)
        else:
            final_survivor_ids.update(
                batch.envelope.rows[position].row_id
                for position in batch.active_positions
            )

    async def run_worker(stage_index: int) -> None:
        operator = plan.operators[stage_index]
        queue = queues[stage_index]
        while True:
            item = await queue.get()
            if item is stop:
                queue.task_done()
                stop_counts[stage_index] += 1
                if (
                    stop_counts[stage_index] == operator.config.block.concurrency
                    and stage_index + 1 < len(queues)
                ):
                    downstream = queues[stage_index + 1]
                    for _ in range(
                        plan.operators[stage_index + 1].config.block.concurrency
                    ):
                        await downstream.put(stop)
                return

            try:
                if stage_index == 0:
                    calls = (item,)
                else:
                    batch = item
                    if not batch.active_positions:
                        await publish(stage_index, batch)
                        continue
                    calls = split_selection(
                        batch, operator, runtime, new_block_id
                    )

                for call in calls:
                    acquired = False
                    manager = runtime.memory_manager
                    try:
                        if manager is not None:
                            await manager.allocate(call.estimated_kv_bytes)
                            acquired = True
                        result = await operator.evaluate(call, runtime)
                    finally:
                        if acquired:
                            await manager.release(call.estimated_kv_bytes)
                    for decision in result.decisions:
                        history[decision.row_id][operator.position] = (
                            decision.passed,
                            result.block.envelope.block_id,
                        )
                    await publish(stage_index, result.block)
            finally:
                queue.task_done()

    async def produce_source() -> None:
        first = plan.operators[0]
        for call in pack_source_rows(rows, first, runtime, new_block_id):
            await queues[0].put(call)
        for _ in range(first.config.block.concurrency):
            await queues[0].put(stop)

    tasks = [
        asyncio.create_task(run_worker(stage_index))
        for stage_index, operator in enumerate(plan.operators)
        for _ in range(operator.config.block.concurrency)
    ]
    producer = asyncio.create_task(produce_source())
    tasks.append(producer)
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    ordered_ids = sorted(final_survivor_ids, key=lambda row_id: int(row_id[1:]))
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
