"""Preflight compilation for the initial block-filter-only plan shape."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.kalypso.blocks.config import BlockFilterConfig
from vllm.kalypso.blocks.errors import BlockConfigurationError
from vllm.kalypso.sem_ops.block_filter import BlockFilter


@dataclass(frozen=True)
class BlockFilterPlan:
    operators: tuple[BlockFilter, ...]


def compile_block_filter_plan(
    logical_ops: list[dict], *, blocking: bool
) -> BlockFilterPlan | None:
    """Recognize supported block plans before tuple lowering or data loading."""
    parsed: list[tuple[int, BlockFilterConfig | None, dict]] = []
    found_block_mode = False
    for position, node in enumerate(logical_ops):
        name = node.get("op")
        args = node.get("args", {})
        if not isinstance(args, dict):
            raise BlockConfigurationError("operator args must be an object", position)
        if name != "sem_filter":
            if "execution" in args:
                raise BlockConfigurationError(
                    "execution is supported only on sem_filter", position
                )
            parsed.append((position, None, args))
            continue

        config = BlockFilterConfig.from_operator_args(args, position)
        found_block_mode |= config is not None
        parsed.append((position, config, args))

    if not found_block_mode:
        return None
    if blocking:
        raise BlockConfigurationError("blocking=True is unsupported with block execution")

    compiled: list[BlockFilter] = []
    for position, config, args in parsed:
        if config is None:
            raise BlockConfigurationError(
                "block filters cannot be mixed with tuple operators", position
            )
        if args.get("cascade") or args.get("llm_cascade"):
            raise BlockConfigurationError("cascade options are unsupported", position)
        compiled.append(BlockFilter(config, position))
    return BlockFilterPlan(tuple(compiled))
