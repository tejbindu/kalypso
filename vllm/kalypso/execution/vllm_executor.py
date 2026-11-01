# execution/vllm_executor.py
import json
from fastapi import Request
from vllm.entrypoints.openai.protocol import CompletionRequest, ChatCompletionRequest
from vllm.kalypso.blocks.config import MAX_SUPPORTED_OUTPUT_TOKENS
from vllm.kalypso.blocks.errors import BlockExecutionError
from vllm.kalypso.pin_registry import PinnedRequestRegistry

from .executor import LLMExecutor, CompletionResult


class ClientDisconnectedError(RuntimeError):
    """The semantic query's HTTP client disconnected, so vLLM cancelled the call."""


class VLLMExecutor(LLMExecutor):

    UNPIN_FUNCTION = 'unpin_request'
    LOG = False

    @classmethod
    def _log(cls, message: str):
        if cls.LOG:
            print(message)

    @staticmethod
    def _owner_key(raw_request: Request) -> str:
        return str(id(raw_request))
    
    def __init__(self, model: str):
        self.model = model

    def _build_chat_request(self, message, max_tokens, pin, priority):
        return ChatCompletionRequest(
            model=self.model,
            messages=message,
            max_tokens=min(max_tokens, 16384),
            temperature=0.0,
            seed=42,
            top_p=1.0,
            frequency_penalty=0.5,
            repetition_penalty=1.3,
            priority=priority,
            vllm_xargs={"pinned": pin},
        )


    def _build_request(self, prompt, max_tokens, pin, priority):
        return CompletionRequest(
            model=self.model,
            prompt=prompt,
            max_tokens=min(max_tokens, 16384),
            temperature=0.0,
            seed=42,
            top_p=1.0,
            frequency_penalty=0.5,
            repetition_penalty=1.3,
            priority=priority,
            vllm_xargs={"pinned": pin},
        )

    def _build_token_request(
        self,
        prompt_token_ids: tuple[int, ...],
        json_schema: dict,
        max_tokens: int,
        priority: int,
    ) -> CompletionRequest:
        if not prompt_token_ids:
            raise ValueError("token-ID completion requires a nonempty prompt")
        if any(type(token_id) is not int or token_id < 0 for token_id in prompt_token_ids):
            raise ValueError("prompt_token_ids must contain nonnegative integers")
        if type(max_tokens) is not int or not 0 < max_tokens <= MAX_SUPPORTED_OUTPUT_TOKENS:
            raise ValueError(
                f"max_tokens must be between 1 and {MAX_SUPPORTED_OUTPUT_TOKENS}"
            )
        return CompletionRequest(
            model=self.model,
            prompt=list(prompt_token_ids),
            max_tokens=max_tokens,
            temperature=0.0,
            seed=42,
            top_p=1.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
            priority=priority,
            structured_outputs={"json": json_schema},
            vllm_xargs={"pinned": False},
        )

    async def execute(
        self,
        raw_request: Request,
        prompt,
        max_tokens: int,
        pin: bool = False,
        priority: int = 0
    ) -> CompletionResult:
        llm_func = self.complete if type(prompt) is str else self.chatcomplete
        return await llm_func(
            raw_request=raw_request,
            prompt=prompt,
            max_tokens=max_tokens,
            pin=pin,
            priority=priority,
        )


    async def chatcomplete(
        self,
        raw_request: Request,
        prompt,
        max_tokens: int,
        pin: bool = False,
        priority: int = 0
    ) -> CompletionResult:
        # Import lazily to avoid a cycle while api_server imports Kalypso.
        from vllm.entrypoints.openai.api_server import create_chat_completion

        req = self._build_chat_request(prompt, max_tokens, pin, priority)
        prompt_items = len(prompt) if isinstance(prompt, list) else 1
        self._log(
            "[vllm-exec] "
            f"submit kind=chat "
            f"max_tokens={max_tokens} "
            f"pin={pin} "
            f"priority={priority} "
            f"prompt_items={prompt_items}"
        )

        gen = await create_chat_completion(
            request=req,
            raw_request=raw_request,
        )
        self._log(
            "[vllm-exec] "
            f"returned kind=chat "
            f"response_type={type(gen).__name__}"
        )
        if gen is None:
            # with_cancellation returns None once raw_request is disconnected.
            raise ClientDisconnectedError("client disconnected; LLM call cancelled")

        raw = gen.body.decode("utf-8")
        data = json.loads(raw)
        request_id = data["id"]
        self._log(
            "[vllm-exec] "
            f"decoded kind=chat "
            f"request_id={request_id} "
            f"finish_reason={data['choices'][0].get('finish_reason', '')}"
        )
        if pin:
            PinnedRequestRegistry.instance().add(
                request_id,
                kind="chat",
                max_tokens=max_tokens,
                owner_key=self._owner_key(raw_request),
            )

        return CompletionResult(
            text=data['choices'][0]['message']['content'],
            request_id=request_id,
            finish_reason=data["choices"][0].get("finish_reason", "")
        )


    async def complete(
        self,
        *,
        raw_request: Request,
        prompt,
        max_tokens: int,
        pin: bool = False,
        priority: int = 0
    ) -> CompletionResult:
        # Import lazily to avoid a cycle while api_server imports Kalypso.
        from vllm.entrypoints.openai.api_server import create_completion

        req = self._build_request(prompt, max_tokens, pin, priority)
        prompt_len = len(prompt) if isinstance(prompt, str) else 1
        self._log(
            "[vllm-exec] "
            f"submit kind=completion "
            f"max_tokens={max_tokens} "
            f"pin={pin} "
            f"priority={priority} "
            f"prompt_len={prompt_len}"
        )

        gen = await create_completion(
            request=req,
            raw_request=raw_request,
        )
        self._log(
            "[vllm-exec] "
            f"returned kind=completion "
            f"response_type={type(gen).__name__}"
        )
        if gen is None:
            # with_cancellation returns None once raw_request is disconnected.
            raise ClientDisconnectedError("client disconnected; LLM call cancelled")

        raw = gen.body.decode("utf-8")
        data = json.loads(raw)
        request_id = data["id"]
        self._log(
            "[vllm-exec] "
            f"decoded kind=completion "
            f"request_id={request_id} "
            f"finish_reason={data['choices'][0].get('finish_reason', '')}"
        )
        if pin:
            PinnedRequestRegistry.instance().add(
                request_id,
                kind="completion",
                max_tokens=max_tokens,
                owner_key=self._owner_key(raw_request),
            )

        return CompletionResult(
            text=data["choices"][0]["text"],
            request_id=request_id,
            finish_reason=data["choices"][0].get("finish_reason", ""),
        )

    async def complete_tokens(
        self,
        raw_request: Request,
        prompt_token_ids: tuple[int, ...],
        json_schema: dict,
        max_tokens: int,
        priority: int = 0,
    ) -> CompletionResult:
        """Submit exact prepared IDs with neutral decoding and JSON schema."""
        from vllm.entrypoints.openai.api_server import create_completion

        req = self._build_token_request(
            prompt_token_ids, json_schema, max_tokens, priority
        )
        self._log(
            "[vllm-exec] submit kind=structured-token-completion "
            f"prompt_tokens={len(prompt_token_ids)} max_tokens={max_tokens} "
            f"priority={priority}"
        )
        gen = await create_completion(request=req, raw_request=raw_request)
        if gen is None:
            raise ClientDisconnectedError("client disconnected; LLM call cancelled")

        status_code = getattr(gen, "status_code", 200)
        try:
            data = json.loads(gen.body.decode("utf-8"))
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BlockExecutionError(
                "structured completion returned an unreadable response",
                status_code=status_code,
            ) from exc
        if status_code >= 400 or "error" in data:
            raise BlockExecutionError(
                "structured completion request failed",
                status_code=status_code,
            )
        try:
            choice = data["choices"][0]
            request_id = data["id"]
            text = choice["text"]
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise BlockExecutionError(
                "structured completion response is missing its result",
                status_code=status_code,
            ) from exc
        return CompletionResult(text, request_id, finish_reason)

    async def unpin(self, raw_request: Request, request_id: str):
        engine = raw_request.app.state.engine_client
        self._log(f"[vllm-exec] unpin submit request_id={request_id}")
        await engine.engine_core.call_utility_async(
            VLLMExecutor.UNPIN_FUNCTION,
            request_id,
        )
        self._log(f"[vllm-exec] unpin done request_id={request_id}")
        PinnedRequestRegistry.instance().remove(request_id)

    async def abort(self, raw_request: Request, request_id: str):
        engine = raw_request.app.state.engine_client
        await engine.abort(request_id)
