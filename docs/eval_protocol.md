# Native evaluation protocol

The LLM Grounding service exposes a versioned evaluation adapter at `/eval/v1`.
It uses the existing project identifier as `corpus_id`; no second corpus registry
is created. The adapter is intended for benchmark clients and returns the same
provider-neutral `RagRunObservation` used by diagnostics and persisted replay.

## Headers

Clients may send `X-Request-ID` or `Idempotency-Key`. A repeated request identity
returns the persisted observation without running the pipeline again. Clients may
send `X-Rag-Eval-Protocol: 1`; another value is a structured protocol error.

## Endpoints

- `GET /eval/v1/health` reports adapter and dependency configuration.
- `GET /eval/v1/capabilities` advertises implemented features. Claim entailment
  and judging are intentionally not advertised.
- `GET /eval/v1/config-schema` returns the JSON schema for effective configuration.
- `POST /eval/v1/query` runs retrieval, generation, validation, and rendering.
- `POST /eval/v1/retrieve` runs candidate retrieval and final-context construction
  without generation.
- `GET /eval/v1/requests/{request_id}` restores a persisted observation.
- `POST /eval/v1/corpora` creates an isolated benchmark corpus inventory.
- `GET /eval/v1/corpora/{corpus_id}` reads an isolated benchmark corpus inventory.

## Observation semantics

`retrieval` contains ordered `CANDIDATE_RETRIEVAL` and `FINAL_CONTEXT` snapshots.
Items retain stable chunk identities, one-based ranks, scores, and source
provenance. Empty stages use `status: "empty"`; unavailable measurements remain
null rather than being inferred. `trace` durations are measured from a monotonic
clock while wall-clock timestamps support correlation. `usage` distinguishes
provider-reported values from values that were not measured.

Claims and citations are the canonical grounded response mapping. A citation is
`resolved: true` only when its chunk exists in the selected context. The service
does not claim semantic entailment or judgment quality.

Ingestion observations retain parser/chunker identity, extracted evidence,
character counts, source spans, chunk ordering, and parse/chunk/embed/index
traces. Evaluation corpora use separate persistence from production projects;
perturbation metadata and corpus hashes are retained for robustness experiments.
Provider seeds, confidence calibration, multi-hop decomposition, and optional
rerank/fusion/filter/compression stages are explicitly reported unavailable until
their runtime implementations exist.
