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


def test_two_filter_stages_overlap_across_distinct_source_blocks():
    class MarkingTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            marker = 1 if "Predicate: slow" in messages[1]["content"] else 2
            return [marker, 42]

    class OverlapExecutor:
        def __init__(self):
            self.slow_started = asyncio.Event()
            self.release_slow = asyncio.Event()
            self.downstream_started = asyncio.Event()

        async def complete_tokens(self, **kwargs):
            marker = kwargs["prompt_token_ids"][0]
            row_ids = kwargs["json_schema"]["properties"]["decisions"]["items"]["properties"]["id"]["enum"]
            if marker == 1 and row_ids[0] == "r0":
                self.slow_started.set()
                await self.release_slow.wait()
            if marker == 2 and row_ids[0] == "r2":
                self.downstream_started.set()
            return SimpleNamespace(
                text=json.dumps(
                    {"decisions": [{"id": row_id, "passed": True} for row_id in row_ids]}
                ),
                request_id=f"req-{marker}-{row_ids[0]}",
                finish_reason="stop",
            )

    contexts = [make_context(f"row {index}", index) for index in range(4)]
    stage_config = {
        "max_rows": 2,
        "concurrency": 2,
        "queue_capacity": 1,
    }
    plan = compile_block_filter_plan(
        [
            {"op": "sem_filter", "args": {"prompt": "slow", "execution": "block", "block": stage_config}},
            {"op": "sem_filter", "args": {"prompt": "downstream", "execution": "block", "block": stage_config}},
        ],
        blocking=False,
    )
    executor = OverlapExecutor()
    block_runtime = BlockRuntime(
        executor=executor,
        raw_request=object(),
        tokenizer=MarkingTokenizer(),
        max_model_len=4096,
        bytes_per_token=4,
        kv_capacity_bytes=100_000,
    )

    async def run_and_observe():
        run_task = asyncio.create_task(
            execute_filter_chain(plan, contexts, block_runtime, lambda parent: f"b-{parent or 'source'}")
        )
        await asyncio.wait_for(executor.slow_started.wait(), timeout=2)
        await asyncio.wait_for(executor.downstream_started.wait(), timeout=2)
        assert not executor.release_slow.is_set()
        executor.release_slow.set()
        return await run_task

    outputs = asyncio.run(run_and_observe())
    assert len(outputs) == 4
