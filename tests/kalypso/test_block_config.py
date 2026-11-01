import pytest

from vllm.kalypso.blocks.config import BlockFilterConfig
from vllm.kalypso.blocks.errors import BlockConfigurationError


def test_tuple_filter_does_not_enter_block_configuration():
    assert BlockFilterConfig.from_operator_args({"prompt": "keep useful rows"}) is None


def test_block_filter_uses_documented_defaults():
    config = BlockFilterConfig.from_operator_args(
        {"prompt": "keep useful rows", "execution": "block"}
    )

    assert config is not None
    assert config.negate is False
    assert config.block.model_dump() == {
        "max_rows": 32,
        "max_prompt_tokens": 4096,
        "max_output_tokens": 1024,
        "concurrency": 2,
        "queue_capacity": 2,
    }


@pytest.mark.parametrize(
    "args",
    [
        {"prompt": "p", "execution": "batch"},
        {"prompt": "p", "execution": "block", "extra": 1},
        {"prompt": "p", "execution": "block", "negate": 1},
        {"prompt": "p", "execution": "block", "block": {"max_rows": True}},
        {"prompt": "p", "execution": "block", "block": {"max_rows": 1.5}},
        {"prompt": "p", "execution": "block", "block": {"max_rows": 0}},
        {"prompt": "p", "execution": "block", "block": {"unknown": 1}},
        {"prompt": " ", "execution": "block"},
        {"prompt": "", "execution": "block"},
        {"prompt": "p", "execution": "block", "block": {"max_output_tokens": 16385}},
    ],
)
def test_invalid_block_configuration_raises_domain_error(args):
    with pytest.raises(BlockConfigurationError):
        BlockFilterConfig.from_operator_args(args, operator_position=3)


def test_configuration_error_retains_operator_position_and_reason():
    with pytest.raises(BlockConfigurationError) as caught:
        BlockFilterConfig.from_operator_args(
            {"prompt": "p", "execution": "unknown"}, operator_position=2
        )

    assert caught.value.operator_position == 2
    assert "unknown execution mode" in caught.value.reason


def test_block_config_is_immutable():
    config = BlockFilterConfig.from_operator_args(
        {"prompt": "p", "execution": "block"}
    )
    assert config is not None
    with pytest.raises(Exception):
        config.block.max_rows = 4
