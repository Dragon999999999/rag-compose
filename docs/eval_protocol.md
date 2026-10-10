# Native evaluation protocol

The LLM Grounding service exposes a versioned evaluation adapter at `/eval/v1`.
It supports production `corpus_id` values and a separate evaluation-corpus
registry. Evaluation corpora never write production project, document, or chat
tables. The adapter is intended for benchmark clients and returns the same
provider-neutral `RagRunObservation` used by diagnostics and persisted replay.

## Headers

Clients may send `X-Request-ID` or `Idempotency-Key`. A repeated request identity
returns the persisted observation without running the pipeline again; reusing an
identity for different input returns a conflict. Without an explicit identity,
the canonical request fingerprint is deterministic. Clients may send
`X-Rag-Eval-Protocol: 1`; another value is a structured protocol error.

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
- `DELETE /eval/v1/corpora/{corpus_id}` deletes only that isolated inventory.
- `POST /eval/v1/corpora/{corpus_id}/documents` and `/chunks` append controlled
  benchmark inventory and return an operation identity.
- `GET /eval/v1/operations/{op_id}` reports upload status; `/documents`,
  `/chunks`, and `/evidence` read isolated inventories.
- `POST /eval/v1/embeddings` exposes sampled raw vectors only with the configured
  evaluation token; vectors are omitted unless explicitly requested.

## Observation semantics

`retrieval` contains ordered `CANDIDATE_RETRIEVAL` and `FINAL_CONTEXT` snapshots;
`stages` additionally contains generic DAG nodes and item lineage for custom,
parallel, split, merge, and rewrite components. Basic RAG reports only the
components it actually runs.
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
Local process telemetry declares `measurement_scope: process` and never claims
remote-provider CPU/GPU data. Provider seeds, confidence calibration, and
multi-hop decomposition remain explicitly unavailable. Per-request Basic RAG
overrides are validated and recorded without mutating shared configuration.
