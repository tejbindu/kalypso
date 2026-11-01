import asyncio
import json
from types import SimpleNamespace

from vllm.kalypso.controller.block_plan import compile_block_filter_plan
from vllm.kalypso.execution.block_execution import execute_filter_chain
from vllm.kalypso.sem_ops.block_filter import BlockRuntime


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        serialized = json.dumps(messages, ensure_ascii=False)
        return list(range(1, len(serialized) // 20 + 2))


class FakeExecutor:
    def __init__(self, rules):
        self.rules = rules
        self.calls = []

    async def complete_tokens(self, **kwargs):
        ids = kwargs["json_schema"]["properties"]["decisions"]["items"]["properties"]["id"]["enum"]
        prompt = self.rules[len(self.calls)]
        self.calls.append(kwargs)
        body = {"decisions": [{"id": row_id, "passed": row_id in prompt} for row_id in ids]}
        return SimpleNamespace(
            text=json.dumps(body), request_id=f"req-{len(self.calls)}", finish_reason="stop"
        )


def make_context(text, idx):
    return SimpleNamespace(
        input=SimpleNamespace(
            data=[{"role": "user", "type": "text", "content": text}],
            right_data=[],
        ),
        output=[],
        state=SimpleNamespace(idx=idx, raw_request=object()),
    )


def runtime(executor):
    return BlockRuntime(
        executor=executor,
        raw_request=object(),
        tokenizer=FakeTokenizer(),
        max_model_len=4096,
        bytes_per_token=4,
        kv_capacity_bytes=100_000,
    )


def test_source_rows_pack_tails_filter_and_preserve_original_contexts():
    contexts = [make_context(f"same row {i}", idx=7) for i in range(3)]
    executor = FakeExecutor([{"r0"}, {"r2"}, set(), {"r2"}])
    plan = compile_block_filter_plan(
        [
            {
                "op": "sem_filter",
                "args": {
                    "prompt": "first", "execution": "block",
                    "block": {"max_rows": 2},
                },
            },
            {
                "op": "sem_filter",
                "args": {
                    "prompt": "second", "execution": "block",
                    "block": {"max_rows": 2},
                },
            },
        ],
        blocking=False,
    )
    next_block = 0

    def new_block_id(parent_id):
        nonlocal next_block
        block_id = f"b{next_block}"
        next_block += 1
        return block_id

    outputs = asyncio.run(
        execute_filter_chain(plan, contexts, runtime(executor), new_block_id)
    )

    assert len(executor.calls) == 4  # Boundaries are preserved across the chain.
    assert [ctx.state.idx for ctx in outputs] == [7]
    assert outputs[0].input.data == contexts[2].input.data
    assert [item["SemFilter"]["operator_position"] for item in outputs[0].output] == [0, 1]
    assert contexts[2].output == []


def test_duplicate_values_and_duplicate_idx_remain_distinct_rows():
    contexts = [make_context("duplicate", idx=4), make_context("duplicate", idx=4)]
    executor = FakeExecutor([{"r0", "r1"}])
    plan = compile_block_filter_plan(
        [{"op": "sem_filter", "args": {"prompt": "all", "execution": "block"}}],
        blocking=False,
    )
    outputs = asyncio.run(
        execute_filter_chain(plan, contexts, runtime(executor), lambda _parent: "b0")
    )

    assert len(executor.calls) == 1
    assert len(outputs) == 2
    assert [ctx.state.idx for ctx in outputs] == [4, 4]
