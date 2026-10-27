# Component configuration

Standalone component definitions live under `config/components/<family>/`.
Every instance has `id`, `type`, `implementation`, `description`, and `config`.
`requirements` is used when a component declares runtime capabilities. The
family `schema.yaml` files compose `config/components/schema.yaml` using JSON
Schema Draft 2020-12. Schema files are excluded from component discovery.

For example:

```yaml
id: retriever_sparse_postgres_fts
type: sparse
implementation: postgres_fts
description: PostgreSQL full-text retrieval using ts_rank_cd.
config:
  top_k: 20
requirements:
  required: [persisted_chunks, postgres_fts]
```

Component settings are resolved into typed Pydantic runtime models and
synchronized to PostgreSQL during application startup. Pipeline YAML stores
component references and is itself persisted as a resolved snapshot, so query
execution does not need to reread the original YAML files.

Override precedence is:

```text
component defaults -> pipeline -> chat -> request
```

Only settings annotated with `x-override-scopes` in the family schema can be
changed. Chat overrides are stored in `chats.stage_overrides_json`; request
overrides are transient. Both are validated before runtime construction and
are recorded with the effective configuration hash and provenance in run
metadata.

To add a component:

1. Add an instance YAML under the appropriate family and validate its fields in
   that family's `schema.yaml`.
2. Add the constructor to the corresponding typed registry or factory, using
   `implementation` as the lookup key.
3. Declare requirements and override scopes in the schema when applicable.
4. Add or update a pipeline reference and tests.

The application synchronizer stores the validated definition in PostgreSQL;
credentials remain in the existing environment/secrets mechanisms and are not
part of component YAML or execution snapshots.
