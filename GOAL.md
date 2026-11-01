Project 1 - block operators
Motivation

Systems such as LOTUS and Kalypso treat LLM calls as operators inside a relational query pipeline. A semantic filter may decide whether each tuple satisfies a natural-language predicate, while a semantic join may determine whether tuples from two relations correspond to one another according to some semantic condition.

A straightforward implementation evaluates these operators at the granularity of individual tuples or tuple pairs. For example, a semantic filter may issue one LLM request per row, while a semantic join may issue one request for each candidate pair. Modern inference engines such as vLLM can batch these requests efficiently, but each request still logically represents a separate LLM invocation and typically repeats the operator instruction and other shared context.

This project will investigate a different execution model: block operators.

A block operator processes multiple tuples using a single LLM call. For example, rather than evaluating a filter independently on 32 rows, Kalypso could construct one prompt containing the filter predicate and all 32 rows and ask the model to return one decision for each row.

Similarly, rather than evaluating a semantic join independently on many candidate pairs, a block join could provide the model with one tuple and a block of candidates, or potentially with blocks from both relations, and ask the model to return all matching pairs.

The idea is related to vectorized query execution in conventional database systems, where operators exchange blocks or vectors of tuples rather than processing one tuple at a time. However, block operators for LLM queries go beyond conventional vectorization: the entire block becomes the unit of a single model invocation.
Goal

The goal of the project is to extend Kalypso with block versions of semantic filter and join operators and study how these operators compose in query pipelines.

The project should answer two broad questions:

    When does evaluating multiple tuples in one LLM call improve performance relative to issuing independent requests through an inference engine such as vLLM?
    How should blocks be managed as they pass through pipelines containing filters and joins that shrink or expand the number of tuples?

The project should focus primarily on execution and performance rather than on developing a sophisticated optimizer.
Block Semantic Filters

A block semantic filter receives a collection of tuples

B = [r1, r2, …, rn]

and evaluates the same semantic predicate over all of them using one LLM request.

For example, instead of issuing requests of the form

r1 - Does this review complain about poor service?
r2 - Does this review complain about poor service?
...

the block operator could issue a single request such as

[r1, r2] - For each review above, determine whether it complains about poor service.

and request a structured result associating each row identifier with a Boolean answer.

The expected advantage is that the operator instruction, schema information, few-shot examples, and other shared prompt context appear only once. Block execution may therefore reduce input-token processing and request overhead beyond what can be obtained through conventional inference batching.

The project should investigate how this benefit changes with block size, row length, prompt length, and filter selectivity.
Block Semantic Joins

Semantic joins provide a more interesting opportunity because conventional pair-at-a-time execution can repeat the same tuples many times.

Suppose a set of tuples a1, a2, …, am is compared with candidate tuples b1, b2, …, bn.

Independent execution creates requests corresponding to

(a, b1), (a, b2), …, (a, bn)

which repeatedly includes a and the join instruction.

A block join could instead make one request containing

[a1, a2, …, ak] [b1, b2, …, bj]

Where k and j are block size parameters, and ask the model to return the IDs of all matching candidates.

This could substantially reduce redundant input representation and the number of LLM calls compared with issuing kj independent pairwise requests. However, it may also create harder reasoning tasks for the model and may reduce accuracy as the number of possible pairings grows.

The project should therefore explore the tradeoff between larger blocks and reliable semantic matching.
Pipelining Across Block Operators

The implementation must support pipelined execution across multiple block operators, rather than executing each operator to completion before starting the next one.

For example, consider a query plan of the form

R → SemanticFilter1 → SemanticJoin(S) → SemanticFilter2.

A non-pipelined implementation could first evaluate the initial filter over all of R, materialize its complete output, then execute the join, materialize the complete join result, and finally evaluate the second filter. The implementation developed in this project should instead allow these operators to execute concurrently as a pipeline.

As soon as SemanticFilter₁ produces enough qualifying tuples, those tuples should become available to the join, even if the filter has not finished processing the rest of R. Similarly, as soon as the join produces enough result tuples, SemanticFilter₂ should be able to start processing them while both upstream operators continue executing.

This requires operators to consume and produce streams of tuples organized into blocks. Importantly, block boundaries should can be assumed to remain unchanged as data moves through the pipeline. Keeping blocks unchanged supports reusing KV cache prefixes. However, this approach can have drawbacks.

The filter–join–filter pipeline is particularly useful for studying this behavior because the operators have different cardinality effects. The first filter may substantially reduce the number of tuples, the join may increase it through join fanout, and the final filter may reduce it again. These changes can lead to underfilled or oversized blocks unless the execution engine actively reforms blocks between operators.

A successful implementation should demonstrate that all three operators can be active concurrently: the join can begin before the first filter has processed its complete input, and the final filter can begin before the join has completed. The evaluation should examine how this pipelined execution interacts with block size, filter selectivity, join fanout, throughput, and end-to-end query latency. It should also demonstrate if an whey block resizing is beneficial.
Correctness Challenges

Block execution also introduces correctness problems that are absent or less significant with one-request-per-tuple execution.

The model may:

    omit answers for some rows;
    return duplicate or incorrect row identifiers;
    mix information from different rows;
    allow one tuple to influence the decision for another;
    miss join matches when too many candidates are presented simultaneously;
    produce malformed structured output.

The evaluation should measure whether accuracy changes as block size increases.
Evaluation

The project should evaluate block operators in the workloads introduced in the Kalypso paper, considering different block sizes.

The primary metrics should include:

    end-to-end query latency;
    number of LLM requests;
    semantic accuracy;
    peak memory usage.

The experiments should study how filter selectivity and join fanout affect downstream block occupancy and overall performance. It should also study in which situations block resizing is beneficial.
Hardware setup

Students will run the experiments using a local vLLM instance on their laptop as a proof of concept. We are not concerned about accuracy, so use the smallest and most resource-efficient LLM model you can run.
Revisiting the Cost Model

Trummer's Implementing Semantic Join Operators Efficiently [1] proposes a cost model for choosing how many tuples from each relation should be included in a single LLM invocation. The model accounts for factors such as prompt size, tuple lengths, expected output size, join selectivity, and the model's context-window constraint.

This project should revisit and evaluate the applicability of this cost model in the Kalypso execution setting. In particular, the students should investigate whether the block sizes favored by the model also lead to good execution performance when the LLM is served locally using an inference engine such as vLLM.

This distinction is important because minimizing token cost does not necessarily minimize execution time. Prefill and decoding have different costs, request size can affect GPU utilization and batching behavior, and multiple block-operator invocations may execute concurrently. The students should therefore compare the predictions of the existing model with empirical measurements of latency and throughput across different block sizes and workloads.

The analysis should also consider how the model interacts with pipelined execution. For example, in a

Filter1 → Join → Filter2

pipeline, a large join block may be efficient for the join itself but delay the production of tuples for the downstream filter. Smaller join blocks may increase the cost of the join while allowing greater overlap between operators. The block size that is best for an isolated operator may therefore differ from the one that performs best in an end-to-end pipeline.

The goal is not necessarily to develop a new cost model. Rather, the students should determine:

    how accurately the existing model predicts useful block sizes in their execution environment;
    which assumptions continue to hold when using vLLM;
    where the model diverges from observed execution behavior;
    whether pipeline effects introduce tradeoffs that are not captured by the operator-level model.

If the evaluation reveals clear and systematic discrepancies, the students may optionally propose a simple adjustment or heuristic that better reflects observed inference or pipeline behavior.

[1] Immanuel Trummer. Implementing Semantic Join Operators Efficiently. arXiv:2510.08489, 2025.
Expected Deliverables

The project should produce:

    A block-oriented operator interface in Kalypso.

    A block semantic filter, join, and map implementations using one LLM call per block.

    Support for composing block operators in a query pipeline.

    A comparison against Kalypso execution using independent requests and vLLM batching.

    An experimental study characterizing the performance and accuracy tradeoffs of block execution.

    An evaluation of Trummer's cost model.

Possible Stretch Goal

If the evaluation identifies clear limitations in the existing model, a possible stretch goal is to propose a new or extended cost model that better predicts end-to-end execution performance in Kalypso. Such a model could incorporate factors such as prefill and decoding cost, vLLM batching behavior, block size, filter selectivity, join fanout, and pipeline overlap. The goal would not be to develop a fully general optimizer, but rather to derive and validate a simple model that explains the observed performance tradeoffs and can guide block-size selection for semantic operators.
