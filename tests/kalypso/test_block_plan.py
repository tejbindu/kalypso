import pytest

from vllm.kalypso.blocks.errors import BlockConfigurationError
from vllm.kalypso.controller.block_plan import compile_block_filter_plan


def block_filter(prompt="keep the row"):
    return {"op": "sem_filter", "args": {"prompt": prompt, "execution": "block"}}


def test_tuple_plan_is_left_for_existing_planner():
    assert compile_block_filter_plan(
        [{"op": "sem_filter", "args": {"prompt": "keep the row"}}],
        blocking=False,
    ) is None


def test_block_filters_compile_as_a_separate_plan_before_tuple_lowering():
    plan = compile_block_filter_plan(
        [block_filter(), block_filter("keep only recent rows")],
        blocking=False,
    )

    assert plan is not None
    assert [operator.position for operator in plan.operators] == [0, 1]


@pytest.mark.parametrize(
    "logical_ops,blocking",
    [
        ([block_filter(), {"op": "sem_filter", "args": {"prompt": "tuple"}}], False),
        ([block_filter(), {"op": "sem_map", "args": {"prompt": "map"}}], False),
        ([block_filter()], True),
        ([{"op": "sem_filter", "args": {"prompt": "p", "execution": "block", "cascade": True}}], False),
        ([{"op": "join", "args": {"execution": "block"}}], False),
    ],
)
def test_unsupported_block_plan_shapes_fail_during_preflight(logical_ops, blocking):
    with pytest.raises(BlockConfigurationError):
        compile_block_filter_plan(logical_ops, blocking=blocking)
