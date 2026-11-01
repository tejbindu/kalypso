"""Query-local, process-independent block execution counters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class BlockExecutionStats:
    query_id: str | None = None
    input_rows: int = 0
    output_rows: int = 0
    input_envelopes: int = 0
    output_envelopes: int = 0
    evaluated_blocks: int = 0
    empty_blocks_bypassed: int = 0
    fit_splits: int = 0
    singleton_blocks: int = 0
    final_tails: int = 0
    calls_submitted: int = 0
    calls_completed: int = 0
    calls_failed: int = 0
    prompt_tokens: int = 0
    reserved_output_tokens: int = 0
    request_wall_seconds: float = 0.0
    admission_wait_seconds: float = 0.0
    queue_wait_seconds: float = 0.0
    queue_peak: dict[str, int] = field(default_factory=dict)
    stage_started: dict[str, float] = field(default_factory=dict)
    stage_last_finish: dict[str, float] = field(default_factory=dict)
    finish_reasons: dict[str, int] = field(default_factory=dict)
    peak_estimated_outstanding_reservation_bytes: int = 0
    first_final_survivor_seconds: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)
