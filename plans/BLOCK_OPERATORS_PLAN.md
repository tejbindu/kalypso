# Block operators: high-level implementation plan

Status: proposal for review. This document describes architecture and implementation milestones; detailed interfaces and code changes belong in the subsequent low-level plan.

## Project goal

Make a block the unit of an LLM invocation and a schedulable unit in Kalypso's query pipeline. Evaluate whether placing multiple tuples in one prompt improves execution beyond vLLM's batching of independent requests.

Support block semantic filter, join, and map, with concurrent execution across operator boundaries. The central demonstration is:

```text
source → block filter → block join → block filter → results
```

The join must start before the first filter finishes, and the final filter must start before the join finishes. Execution must use bounded memory and expose the tradeoffs among block size, latency, throughput, cache reuse, and semantic reliability.

Focus on execution and performance. Start with explicit block-size settings; defer sophisticated optimization. Use a small local model for the proof of concept. Semantic quality remains a diagnostic rather than a prerequisite for the initial execution milestone.

## Existing architecture

The following observations ground this proposal in the current repository:

- `vllm/kalypso/sem_ops/filters.py` and `map.py` accept lists, but execute independent tuple calls. This is batching rather than one LLM invocation per block.
- `vllm/kalypso/controller/plan.py` lowers a logical join into candidate expansion followed by semantic filtering. A block join needs a physical operator that receives both sides before pair expansion.
- `vllm/kalypso/controller/stage.py` and `execution/pipeline_execution.py` already provide asynchronous task scheduling, retries, memory budgeting, and fanout tracking. Their task and result contracts currently center on tuples.
- `vllm/kalypso/context/context.py` combines mutable prompt history, outputs, and per-tuple execution state. Blocks need explicit logical data and shared request ownership.
- The plan materializes source contexts, and joins load the right relation. Streaming between operators can be implemented first; lazy source loading is a separate improvement.
- `vllm/kalypso/execution/vllm_executor.py` is the existing model invocation boundary. Reuse it, extending request options where necessary, rather than initially changing vLLM's core scheduler.

## Challenges and proposed decisions

### Block identity and resizing

`GOAL.md` contains conflicting wording about preserving block boundaries and actively reforming blocks. Treat preservation as the baseline and resizing as an explicit experimental policy.

Filters preserve the original block envelope and record a survivor selection. Joins create descendant blocks with pair lineage. Repacking survivors is a separate operation whose buffering, serialization, and cache effects can be measured. Empty blocks must propagate completion without issuing an LLM request.

Preserving a logical block does not automatically preserve an identical token prefix. Prompt layout and retained data must make any intended reuse possible, and actual reuse must be measured.

### Tuple identity and downstream meaning

Assign stable row IDs independent of prompt position. Preserve duplicate-valued source rows as distinct tuples. Joined rows carry both parent identities, and map outputs attach to their input IDs.

Separate logical row values and lineage from the shared multirow request transcript. Do not copy a whole block conversation into every downstream row. Define how existing row-local history is represented so that map/filter/join composition preserves the intended semantics.

### Response correctness

Require structured responses and validate their contents before publishing outputs:

- Filter: exactly one Boolean decision for every input ID.
- Map: exactly one mapped value for every input ID.
- Join: unique pairs containing IDs from the requested candidate domain.

An empty join result is valid; missing or malformed output is an execution error. Schema-valid join output can still omit true semantic matches, so structural validation does not replace accuracy measurement. The existing permissive Boolean parsing should not be reused for block responses.

Start with explicit failure on invalid output. Add bounded retries, splitting, or tuple fallback only as explicit, observable policies. Validate before committing results so retries cannot publish duplicate outputs.

### Pipeline liveness and memory

Selective filters produce underfilled blocks, while join fanout can produce large outputs. Use bounded handoffs and backpressure, explicit end-of-stream signals, and partial-block flushing. Avoid waiting indefinitely for a full block.

Budget serialized input tokens and an output allowance, not just row counts. Dense join output can grow with the product of the two tile dimensions. Define oversized-row and oversized-request behavior rather than silently truncating data.

Shared pins need ownership across descendants, with cleanup on completion, failure, and cancellation. First establish correct pipelining without explicit pinning, then integrate cache ownership.

### Fair performance comparison

The baseline must retain concurrent independent requests served through vLLM. Fewer requests do not necessarily imply lower latency: longer decoding, GPU utilization, and pipeline startup delays can offset input savings.

Count actual model calls, including auxiliary pinning calls, retries, and fallbacks. Distinguish logical prompt tokens from tokens actually computed after prefix-cache reuse. Use per-query instrumentation where possible; existing process-wide metric deltas need isolated runs for attribution.

## Implementation milestones

### 1. Establish the block contract and configuration

Introduce a block representation containing stable row IDs, logical data, lineage, and selection metadata. Distinguish block execution from relational blocking behavior.

Expose explicit execution mode, row and token limits, join tile dimensions, and resizing policy through the existing query format. Preserve current tuple execution as the baseline.

Outcome: agreed contracts for block inputs, outputs, completion, and errors, ready for a low-level design.

### 2. Implement an isolated block filter

Build one prompt containing multiple identified rows and require structured Boolean decisions. Validate the complete response before returning survivors. Issue one semantic evaluation request for each admitted block in the normal path.

Outcome: the smallest complete vertical slice covering block formation, model invocation, response validation, and survivor production.

### 3. Implement block map

Reuse the block invocation and validation foundation. Return one mapped value per row ID, and define its downstream representation while retaining row-local history where required. Give map its own output budgeting rather than inheriting the current per-tuple estimate.

Outcome: block filter and map compose without leaking unrelated rows' transcripts into downstream inputs.

### 4. Implement a direct block semantic join

Start with one left tuple against a right block, then generalize to configurable k-by-j tiles. Reuse the materialized right relation initially. Schedule tiles lazily and construct joined contexts only for returned matches, avoiding eager creation of every candidate pair.

Preserve parent identities and duplicate semantics. Limit tile size using both context and output budgets. Keep indexed retrieval and cascades outside the first implementation.

Outcome: one semantic evaluation call per join tile in the normal path, with bounded candidate scheduling and valid pair lineage.

### 5. Integrate blocks into pipelined scheduling

Give block operators explicit stage boundaries. Extend the existing scheduling infrastructure with block tasks, bounded handoffs, completion propagation, cancellation, retries, and end-of-stream flushing.

Retain the existing memory manager where practical, using estimates for complete block requests. Avoid planner sampling that eagerly constructs all candidate pairs. Establish pipeline correctness without explicit pinning, then add shared pin ownership and cleanup.

Outcome: a traced filter–join–filter execution demonstrates all three operators overlapping on sufficient input, with bounded buffering and complete cleanup.

### 6. Implement and compare resizing policies

Implement boundary preservation first, followed by explicit split/coalesce operations constrained by rows and tokens. Flush partial blocks at completion and under defined latency or memory conditions.

Compare occupancy improvements against buffering latency, serialization work, and lost prefix reuse. Keep policies configurable rather than introducing an optimizer at this stage.

Outcome: controlled experiments establish when resizing helps isolated operators and whole pipelines.

### 7. Instrument and evaluate throughout

Add instrumentation with the first vertical slice rather than after implementation. Record:

- Actual model request counts, retries, fallbacks, and malformed responses.
- Logical input tokens, computed prefill tokens, and output tokens.
- Input/output cardinalities, join fanout, and block occupancy.
- Stage start/completion times, queueing, time to first result, throughput, and end-to-end latency.
- Peak host and GPU memory and prefix-cache reuse.
- Semantic quality relative to labeled data where available; agreement with tuple execution is a separate measure.

Use deterministic fake-executor tests for routing, ID validation, duplicate-valued rows, empty results, tail flushing, retries, cancellation, fanout, and bounded handoffs. Then run local-model integration experiments using existing workloads, including the Contract NLI filter–join–map workload and a filter–join–filter pipeline.

Compare:

1. Independent tuple requests with vLLM batching.
2. Block calls with materialized operator boundaries.
3. Pipelined block calls with preserved boundaries.
4. Pipelined block calls with resizing.

Sweep block sizes and join tile dimensions, row lengths, shared prompt lengths, filter selectivity, and join fanout. Use controlled synthetic cases where necessary to vary cardinality effects independently of model behavior. Keep model, decoding settings, resource limits, and cache conditions explicit and comparable.

Finally, evaluate Trummer's cost model against measured operator and pipeline performance. This requires reading the referenced paper and reproducing its assumptions and formulas during the evaluation design. Compare its preferred block sizes with empirical latency and throughput results before proposing any adjustment.

## First integration acceptance criteria

- A normal block filter, map, or join tile uses one semantic evaluation request for multiple tuples or candidate pairs.
- Validated outputs retain correct row and pair identities.
- A pipeline trace shows the join starting before the first filter completes and the final filter starting before the join completes.
- Empty outputs and final partial blocks complete correctly.
- Fanout and slow downstream stages do not cause unbounded buffering.
- Failure and cancellation release tasks, reservations, and acquired pins.
- Current tuple execution remains available for comparison.

## Scope for the subsequent low-level plan

Specify concrete types, operator interfaces, planner lowering, task/result contracts, block assembly and flushing rules, response schemas, token admission, pin ownership, configuration fields, instrumentation, and targeted tests.

Keep sophisticated optimization, a new cost model, streaming both sides of the join, and integration with indexed retrieval or cascades outside the initial milestone.
