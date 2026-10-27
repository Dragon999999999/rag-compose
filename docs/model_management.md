# Model configuration and local lifecycle management

Model definitions live in `config/models`. Every definition has the same top-level
shape: `id`, `stage`, `location`, `backend`, `provider` (when applicable), `name`,
and nested `connection`, `source`, `storage`, `runtime`, and `parameters` sections.
The files are validated against `config/models/schema.yaml` before the model
catalog or a pipeline can use them. The schema file is not itself discovered as a
model.

Remote Aqueduct models use `location: remote` and keep endpoint and credential
environment-variable names under `connection`. Credential values are resolved only
when the existing provider makes a request.

Local models use `location: local`. Their `source` can identify a Hugging Face
repository and optional pinned revision, a URL plus SHA-256 digest, or an existing
local directory. `storage.relative_path` is resolved below `MODEL_STORAGE_ROOT`
(default `~/.cache/llm-grounding/models`). Downloads are written to a temporary
directory, verified, and atomically renamed; a process lock prevents duplicate
downloads. Model weights are never removed when an inference container stops.

`runtime.type: docker` identifies an actual inference-server image and its port,
readiness path, environment, device, and resource limits. `LocalModelManager`
loads models on demand through an async lease:

```python
async with manager.acquire("embedding_001") as model:
    vectors = await model.embed(["text to encode"])
```

The manager reuses matching containers, waits for readiness, tracks active leases,
and applies the configured idle timeout. Docker-specific behavior is isolated in
`DockerModelRuntime`; storage and registry implementations can be replaced in
tests or by a future runtime. Remote models do not use Docker or local weight
management.

To register a new model, add one YAML file with the standard structure and point a
component or pipeline at its stable ID. Do not commit downloaded weights or put
credential values in YAML.

The shipped local embedding definitions include MiniLM-L6-v2 (384 dimensions),
EmbeddingGemma (768), Nomic Embed v2 MoE (768), and BGE-M3 (1024). They use the
same CPU-oriented text-embeddings-inference runtime and persistent storage path.
EmbeddingGemma and Nomic declare their required query/document prefixes in model
parameters; the shared embedder applies those prefixes consistently for indexing
and query retrieval. BGE-M3 is currently configured for its dense representation;
the vector contract leaves sparse and multi-vector extensions outside the current
single-vector index boundary.
