"""Single-request evaluator for a prepared block semantic filter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vllm.kalypso.blocks.config import BlockFilterConfig
from vllm.kalypso.blocks.filter_protocol import parse_filter_response
from vllm.kalypso.blocks.types import (
    BlockBatch,
    FilterBlockResult,
    PreparedBlockCall,
    validate_batch,
)


class TokenCompletion(Protocol):
    """Executor capability needed by this isolated operator."""

    async def complete_tokens(
        self,
        *,
        raw_request,
        prompt_token_ids: tuple[int, ...],
        json_schema: dict,
        max_tokens: int,
        priority: int = 0,
    ):
        """Complete the exact prepared token sequence with constrained output."""


@dataclass(frozen=True)
class BlockRuntime:
    executor: TokenCompletion
    raw_request: object
    priority: int = 0


class BlockFilter:
    """Evaluate one prepared block and publish only a complete selection."""

    def __init__(self, config: BlockFilterConfig, position: int):
        self.config = config
        self.position = position

    async def evaluate(
        self, call: PreparedBlockCall | BlockBatch, runtime: BlockRuntime
    ) -> FilterBlockResult:
        batch = call.block if isinstance(call, PreparedBlockCall) else call
        validate_batch(batch)
        row_ids = tuple(
            batch.envelope.rows[position].row_id for position in batch.active_positions
        )
        if not row_ids:
            return FilterBlockResult(block=batch, decisions=())
        if not isinstance(call, PreparedBlockCall):
            raise TypeError("a nonempty block must be prepared before evaluation")

        completion = await runtime.executor.complete_tokens(
            raw_request=runtime.raw_request,
            prompt_token_ids=call.prompt_token_ids,
            json_schema=call.json_schema,
            max_tokens=call.max_output_tokens,
            priority=runtime.priority,
        )
        decisions = parse_filter_response(
            completion.text,
            row_ids,
            finish_reason=completion.finish_reason,
            negate=self.config.negate,
            block_id=batch.envelope.block_id,
        )
        surviving_positions = tuple(
            position
            for position, decision in zip(batch.active_positions, decisions, strict=True)
            if decision.passed
        )
        selected_batch = BlockBatch(batch.envelope, surviving_positions)
        return FilterBlockResult(
            block=selected_batch,
            decisions=decisions,
            request_id=completion.request_id,
            finish_reason=completion.finish_reason,
        )
