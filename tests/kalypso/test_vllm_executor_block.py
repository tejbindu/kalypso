import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.kalypso.blocks.errors import BlockExecutionError
from vllm.kalypso.execution.vllm_executor import VLLMExecutor


SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


def test_token_request_uses_exact_ids_schema_and_neutral_decoding():
    request = VLLMExecutor("test-model")._build_token_request(
        (4, 5, 6), SCHEMA, max_tokens=64, priority=3
    )

    assert request.prompt == [4, 5, 6]
    assert request.structured_outputs.json == SCHEMA
    assert request.max_tokens == 64
    assert request.temperature == 0
    assert request.seed == 42
    assert request.frequency_penalty == 0
    assert request.repetition_penalty == 1
    assert request.priority == 3
    assert request.vllm_xargs == {"pinned": False}


@pytest.mark.parametrize("token_ids,max_tokens", [([], 8), ([1, -1], 8), ([True], 8), ([1], 16385)])
def test_token_request_rejects_invalid_ids_or_output_budget(token_ids, max_tokens):
    with pytest.raises(ValueError):
        VLLMExecutor("test-model")._build_token_request(
            tuple(token_ids), SCHEMA, max_tokens=max_tokens, priority=0
        )


def test_complete_tokens_submits_prepared_request_and_returns_metadata():
    async def fake_create_completion(*, request, raw_request):
        assert request.prompt == [10, 11]
        assert request.structured_outputs.json == SCHEMA
        assert raw_request is sentinel
        return SimpleNamespace(
            status_code=200,
            body=json.dumps(
                {
                    "id": "request-7",
                    "choices": [{"text": '{"ok":true}', "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 4},
                }
            ).encode(),
        )

    sentinel = object()
    with patch(
        "vllm.entrypoints.openai.api_server.create_completion",
        new=fake_create_completion,
    ):
        result = asyncio.run(
            VLLMExecutor("test-model").complete_tokens(
                sentinel,
                (10, 11),
                SCHEMA,
                max_tokens=32,
            )
        )

    assert (result.text, result.request_id, result.finish_reason) == (
        '{"ok":true}',
        "request-7",
        "stop",
    )
    assert result.usage == {"prompt_tokens": 2, "completion_tokens": 4}


def test_complete_tokens_surfaces_serving_error_status_without_response_body():
    async def fake_create_completion(*, request, raw_request):
        return SimpleNamespace(
            status_code=503,
            body=b'{"error":{"message":"private details","code":503}}',
        )

    with patch(
        "vllm.entrypoints.openai.api_server.create_completion",
        new=fake_create_completion,
    ):
        with pytest.raises(BlockExecutionError) as caught:
            asyncio.run(
                VLLMExecutor("test-model").complete_tokens(
                    object(), (10,), SCHEMA, max_tokens=8
                )
            )

    assert caught.value.status_code == 503
    assert "private details" not in str(caught.value)
