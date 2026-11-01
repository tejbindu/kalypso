"""Single-request evaluator for a prepared block semantic filter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from vllm.kalypso.blocks.config import BlockFilterConfig
from vllm.kalypso.blocks.errors import BlockConfigurationError, BlockSizingError
from vllm.kalypso.blocks.filter_protocol import (
    block_filter_messages,
    decision_schema,
    parse_filter_response,
)
from vllm.kalypso.blocks.types import (
    BlockBatch,
    BlockEnvelope,
    BlockRow,
    FilterBlockResult,
    PreparedBlockCall,
    RowMessage,
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
    tokenizer: Any = None
    chat_template: str | None = None
    max_model_len: int | None = None
    bytes_per_token: int | None = None
    kv_capacity_bytes: int | None = None
    memory_manager: Any = None


class BlockFilter:
    """Evaluate one prepared block and publish only a complete selection."""

    def __init__(self, config: BlockFilterConfig, position: int):
        self.config = config
        self.position = position

    def prepare(self, batch: BlockBatch, runtime: BlockRuntime) -> PreparedBlockCall:
        """Render once with the serving template and budget the exact token IDs."""
        validate_batch(batch)
        if not batch.active_positions:
            raise BlockConfigurationError("empty selections do not need preparation")
        if len(batch.active_positions) > self.config.block.max_rows:
            raise BlockSizingError(
                f"block has {len(batch.active_positions)} active rows; "
                f"limit is {self.config.block.max_rows}"
            )
        if runtime.tokenizer is None:
            raise BlockConfigurationError("serving tokenizer is unavailable", self.position)
        if runtime.max_model_len is None or runtime.max_model_len <= 0:
            raise BlockConfigurationError("served max_model_len is unavailable", self.position)
        if runtime.bytes_per_token is None or runtime.bytes_per_token <= 0:
            raise BlockConfigurationError("KV bytes-per-token estimate is unavailable", self.position)

        max_output_tokens = self.config.block.max_output_tokens
        prompt_limit = min(
            self.config.block.max_prompt_tokens,
            runtime.max_model_len - max_output_tokens,
        )
        if prompt_limit <= 0:
            raise BlockConfigurationError(
                "max_output_tokens leaves no room for an input prompt", self.position
            )

        row_ids = tuple(
            batch.envelope.rows[position].row_id for position in batch.active_positions
        )
        messages = block_filter_messages(self.config.prompt, batch)
        try:
            rendered = runtime.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                chat_template=runtime.chat_template,
            )
            token_ids = tuple(rendered)
        except Exception as exc:
            raise BlockConfigurationError(
                "could not render block prompt with the serving chat template",
                self.position,
            ) from exc
        if not token_ids or any(type(token_id) is not int for token_id in token_ids):
            raise BlockConfigurationError(
                "serving tokenizer did not return a nonempty token-ID sequence",
                self.position,
            )
        if len(token_ids) > prompt_limit:
            row_id = row_ids[0] if len(row_ids) == 1 else None
            raise BlockSizingError(
                f"rendered prompt has {len(token_ids)} tokens; effective limit is "
                f"{prompt_limit} (configured={self.config.block.max_prompt_tokens}, "
                f"served_context={runtime.max_model_len}, output_reserved={max_output_tokens})",
                row_id,
            )
        estimated_kv_bytes = (
            len(token_ids) + max_output_tokens
        ) * runtime.bytes_per_token
        if (
            runtime.kv_capacity_bytes is not None
            and estimated_kv_bytes > runtime.kv_capacity_bytes
        ):
            raise BlockSizingError(
                f"request reserves {estimated_kv_bytes} KV bytes; total capacity is "
                f"{runtime.kv_capacity_bytes}",
                row_ids[0] if len(row_ids) == 1 else None,
            )
        return PreparedBlockCall(
            block=batch,
            prompt_token_ids=token_ids,
            json_schema=decision_schema(row_ids),
            max_output_tokens=max_output_tokens,
            estimated_kv_bytes=estimated_kv_bytes,
        )

    def preflight(self, runtime: BlockRuntime) -> None:
        """Validate static prompt/template/context limits before reading input."""
        probe = BlockBatch(
            BlockEnvelope(
                block_id="preflight",
                parent_block_id=None,
                rows=(BlockRow("r-preflight", (RowMessage("user", ""),)),),
            ),
            active_positions=(0,),
        )
        try:
            self.prepare(probe, runtime)
        except BlockSizingError as exc:
            raise BlockConfigurationError(
                "filter instruction and protocol exceed the configured request limits",
                self.position,
            ) from exc

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
