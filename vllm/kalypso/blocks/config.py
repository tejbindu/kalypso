"""Strict, engine-independent configuration for block semantic filters."""

from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
)

from vllm.kalypso.blocks.errors import BlockConfigurationError

MAX_SUPPORTED_OUTPUT_TOKENS = 16_384


class BlockLimits(BaseModel):
    """Per-stage upper bounds for block formation and execution."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    max_rows: StrictInt = Field(default=32, gt=0)
    max_prompt_tokens: StrictInt = Field(default=4096, gt=0)
    max_output_tokens: StrictInt = Field(
        default=1024, gt=0, le=MAX_SUPPORTED_OUTPUT_TOKENS
    )
    concurrency: StrictInt = Field(default=2, gt=0)
    queue_capacity: StrictInt = Field(default=2, gt=0)


class BlockFilterConfig(BaseModel):
    """Validated arguments for one explicitly block-executed sem_filter."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    prompt: StrictStr = Field(min_length=1)
    execution: StrictStr = "block"
    negate: StrictBool = False
    block: BlockLimits = Field(default_factory=BlockLimits)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a nonempty instruction")
        return value

    @classmethod
    def from_operator_args(
        cls, args: dict[str, Any], operator_position: int | None = None
    ) -> BlockFilterConfig | None:
        """Parse block settings, returning ``None`` for the legacy tuple path.

        Only explicitly opted-in block filters are validated here. The tuple
        planner retains ownership of its existing argument format.
        """
        if "execution" not in args:
            return None
        mode = args["execution"]
        if mode != "block":
            raise BlockConfigurationError(
                f"unknown execution mode {mode!r}", operator_position
            )
        try:
            return cls.model_validate(args)
        except ValidationError as exc:
            # Preserve the Pydantic detail while exposing one stable domain error.
            raise BlockConfigurationError(str(exc), operator_position) from exc
