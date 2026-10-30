# Model configuration and local lifecycle management

Model definitions live in `config/models`. Every definition has the same
top-level identity fields: `id`, `stage`, `location`, `backend`, and `name`.
Nested `connection`, `source`, `storage`, `runtime`, and `parameters` sections
describe how the model is accessed. The files are validated against
`config/models/schema.yaml` before the model catalog or a pipeline can use them.

The model stage controls which configuration can reference the model:

- `embedding` models create dense, sparse, or multivector document/query
  representations.
- `reasoner` models generate grounded answers.
- `reranker` models score retrieved candidates.
- `validation` models support knowledge-building or specialized chunking.

The CLI displays validation-stage models as knowledge-related models when that is
more useful to a user.

## Remote models

Remote Aqueduct models use `location: remote` and keep the endpoint and credential
environment-variable names under `connection`. Credential values are resolved
only when the provider makes a request. The current remote catalog contains the
Aqueduct Qwen embedding model and the configured reasoning models. The Aqueduct
embedding configuration uses a 240-second request timeout and two retries.

## Local models

Local models use `location: local`. Their `source` can identify a Hugging Face
repository and optional revision, a URL plus SHA-256 digest, or an existing local
directory. `storage.relative_path` is resolved below `MODEL_STORAGE_ROOT`, which
defaults to `~/.cache/llm-grounding/models`. Downloads are written to a temporary
directory, verified, and atomically renamed; a process lock prevents duplicate
downloads. Model weights are not removed when an inference runtime stops.

`runtime.type: in_process` loads the model in the application process. The
`docker` runtime identifies an inference-server image, port, readiness path,
environment, device, and resource limits. `LocalModelManager` loads models on
demand through an async lease:

```python
async with manager.acquire("embedding_001") as model:
    vectors = await model.embed(["text to encode"])
```

The manager reuses matching containers, waits for readiness, tracks active leases,
and applies the configured idle timeout. Docker-specific behavior is isolated in
`DockerModelRuntime`; storage and registry implementations can be replaced in
tests or by another runtime.

The checked-in catalog includes MiniLM, BGE-M3, Nomic, Jina, and ColBERT
embeddings; MiniLM, BGE, Jina, mxbai, and Qwen rerankers; Aqueduct Qwen and
reasoning models; and GLiNER/Qwen local models used by knowledge and chunking
components. ColBERT uses a multivector representation and Qwen/Nomic definitions
include provider-specific runtime parameters such as prefixes or remote-code
requirements.

## Registration and preparation

To register a model, add one YAML file with the standard structure, then point a
component at its stable ID. Use `llm-grounding model inspect MODEL_ID` to review
the credential-free definition and `llm-grounding model download MODEL_ID` to
prepare local weights. Do not commit downloaded weights or put credential values
in YAML.
