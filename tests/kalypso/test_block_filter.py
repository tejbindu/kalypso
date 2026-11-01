import asyncio
import json
from types import SimpleNamespace

from vllm.kalypso.blocks.config import BlockFilterConfig
from vllm.kalypso.blocks.filter_protocol import decision_schema
from vllm.kalypso.blocks.types import (
    BlockBatch,
    BlockEnvelope,
    BlockRow,
    PreparedBlockCall,
    RowMessage,
)
from vllm.kalypso.sem_ops.block_filter import BlockFilter, BlockRuntime


class FakeExecutor:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def complete_tokens(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            text=self.response, request_id="request-1", finish_reason="stop"
        )


def make_call(active_positions=(0, 1)):
    rows = tuple(
        BlockRow(f"r{index}", (RowMessage("user", f"row {index}"),))
        for index in range(2)
    )
    batch = BlockBatch(BlockEnvelope("b0", None, rows), active_positions)
    if not active_positions:
        return batch
    ids = tuple(rows[position].row_id for position in active_positions)
    return PreparedBlockCall(batch, (1, 2, 3), decision_schema(ids), 64, 256)


def make_filter(negate=False):
    config = BlockFilterConfig.model_validate(
        {"prompt": "keep matching rows", "execution": "block", "negate": negate}
    )
    return BlockFilter(config, position=0)


def test_evaluator_submits_one_structured_call_and_returns_survivors():
    executor = FakeExecutor(
        json.dumps(
            {"decisions": [{"id": "r1", "passed": False}, {"id": "r0", "passed": True}]}
        )
    )
    call = make_call()

    result = asyncio.run(make_filter().evaluate(call, BlockRuntime(executor, object())))

    assert len(executor.calls) == 1
    assert executor.calls[0]["prompt_token_ids"] == call.prompt_token_ids
    assert executor.calls[0]["json_schema"] == call.json_schema
    assert result.block.active_positions == (0,)
    assert [(item.row_id, item.passed) for item in result.decisions] == [
        ("r0", True),
        ("r1", False),
    ]
    assert result.request_id == "request-1"


def test_empty_selection_bypasses_model_request():
    executor = FakeExecutor("not used")
    result = asyncio.run(
        make_filter().evaluate(make_call(()), BlockRuntime(executor, object()))
    )

    assert executor.calls == []
    assert result.block.active_positions == ()
    assert result.decisions == ()


def test_negation_controls_survivors_only_after_validation():
    executor = FakeExecutor(
        '{"decisions":[{"id":"r0","passed":true},{"id":"r1","passed":false}]}'
    )
    result = asyncio.run(
        make_filter(negate=True).evaluate(make_call(), BlockRuntime(executor, object()))
    )

    assert result.block.active_positions == (1,)
    assert [decision.passed for decision in result.decisions] == [False, True]
