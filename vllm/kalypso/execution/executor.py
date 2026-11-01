# execution/executor.py
from abc import ABC, abstractmethod


class CompletionResult:
    def __init__(self, text: str, request_id: str, finish_reason, usage=None):
        self.text = text
        self.request_id = request_id
        self.finish_reason = finish_reason
        self.usage = usage


class LLMExecutor(ABC):
    @abstractmethod
    async def complete_tokens(
        self,
        raw_request,
        prompt_token_ids: tuple[int, ...],
        json_schema: dict,
        max_tokens: int,
        priority: int = 0,
    ) -> CompletionResult:
        """Complete exact token IDs using constrained JSON output."""
        pass

    @abstractmethod
    async def complete(
        self,
        raw_request,
        prompt,
        max_tokens: int,
        pin: bool = False,
        priority: int = 0
    ) -> CompletionResult:
        pass

    async def chatcomplete(
        self,
        raw_request,
        prompt,
        max_tokens: int,
        pin: bool = False,
        priority: int = 0
    ) -> CompletionResult:
        pass

    @abstractmethod
    async def unpin(self, request_id: str):
        pass

    @abstractmethod
    async def abort(self, request_id: str):
        pass
