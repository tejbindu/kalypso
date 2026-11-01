import json

import pytest

from vllm.kalypso.blocks.errors import BlockOutputError
from vllm.kalypso.blocks.filter_protocol import decision_schema, parse_filter_response


def test_schema_requires_exact_number_of_decisions_and_known_ids():
    schema = decision_schema(("r0", "r1"))
    decisions = schema["properties"]["decisions"]
    assert decisions["minItems"] == decisions["maxItems"] == 2
    assert decisions["items"]["properties"]["id"]["enum"] == ["r0", "r1"]
    assert schema["additionalProperties"] is False


def test_response_is_reordered_and_negated_once():
    result = parse_filter_response(
        '{"decisions":[{"id":"r1","passed":false},{"id":"r0","passed":true}]}',
        ("r0", "r1"),
        finish_reason="stop",
        negate=True,
    )
    assert [(decision.row_id, decision.passed) for decision in result] == [
        ("r0", False),
        ("r1", True),
    ]


@pytest.mark.parametrize(
    "response",
    [
        '{"decisions":[{"id":"r0","passed":true}]}',
        '{"decisions":[{"id":"r0","passed":true},{"id":"r0","passed":false}]}',
        '{"decisions":[{"id":"r0","passed":true},{"id":"r1","passed":0}]}',
        '{"decisions":[{"id":"r0","passed":true},{"id":"r1","passed":"false"}]}',
        '{"decisions":[{"id":"r0","passed":true},{"id":"r1","passed":false,"extra":1}]}',
        '{"decisions":[{"id":"r0","passed":true},{"id":"r1","passed":false}],"extra":1}',
        '{"decisions":[{"id":"r0","passed":true},{"id":"r1","passed":false}]} trailing',
        '{"decisions":[{"id":"r0","passed":true},{"id":"r1","passed":false}],"decisions":[]}',
        '{"decisions":[{"id":"r0","passed":true},{"id":"r1","passed":NaN}]}',
        '```json\n{"decisions":[]}\n```',
    ],
)
def test_invalid_responses_are_rejected_without_partial_results(response):
    with pytest.raises(BlockOutputError):
        parse_filter_response(response, ("r0", "r1"), finish_reason="stop")


def test_unknown_ids_and_non_stop_finish_reason_are_rejected():
    with pytest.raises(BlockOutputError, match="unknown"):
        parse_filter_response(
            json.dumps({"decisions": [{"id": "r2", "passed": True}]}),
            ("r0",),
            finish_reason="stop",
        )
    with pytest.raises(BlockOutputError, match="finish_reason"):
        parse_filter_response(
            '{"decisions":[{"id":"r0","passed":true}]}',
            ("r0",),
            finish_reason="length",
        )


def test_duplicate_keys_are_rejected_at_nested_depth():
    with pytest.raises(BlockOutputError, match="duplicate object key"):
        parse_filter_response(
            '{"decisions":[{"id":"r0","id":"r0","passed":true}]}',
            ("r0",),
            finish_reason="stop",
        )
