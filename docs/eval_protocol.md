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

- `GET /eval/v1/health` performs bounded PostgreSQL, vector-backend, and
  ingestion/evaluation-service checks. It returns the Target Protocol v1
  readiness states `READY`, `DEGRADED`, or `NOT_READY` under the `status` field,
  identifies the target under `target`, and reports dependency-level statuses
  under `details.dependencies`. `READY` and `DEGRADED` use HTTP 200, while
  `NOT_READY` uses HTTP 503. Each dependency reports a redacted error when a
  check fails. The endpoint never executes a RAG query or loads model weights.
- `GET /eval/v1/capabilities` advertises implemented features. Claim entailment
  and judging are intentionally not advertised.
- `GET /eval/v1/config-schema` returns the JSON schema for effective configuration.
- `POST /eval/v1/query` runs retrieval, generation, validation, and rendering.
- `POST /eval/v1/retrieve` runs candidate retrieval and final-context construction
  without generation.
- `GET /eval/v1/requests/{request_id}` restores a persisted observation.
- `GET /eval/v1/observations/{rag_run_id}` restores the original persisted
  observation by run ID without re-executing the query.
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

`retrieval` contains ordered `CANDIDATE_RETRIEVAL`, optional `RERANK`, and
`FINAL_CONTEXT` snapshots. `FINAL_CONTEXT` is the exact context passed to the
generation stage. `stages` additionally contains generic DAG nodes and item
lineage for custom,
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

Browser chat queries and evaluation queries use the same Basic RAG runner and
persist the same canonical observation. Browser response DTOs remain compact;
the full observation is available through the replay endpoint. The current
application has no separate authentication middleware for this protocol, so
deployment-level access control remains responsible for protecting observation
data.
