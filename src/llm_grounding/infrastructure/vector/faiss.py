"""Lazy, persistent FAISS adapter for the canonical vector-store protocol.

FAISS is intentionally imported only when the backend is used. Each project and
embedding space has a separate index file plus a JSON sidecar containing stable
chunk metadata and raw values needed for reconstruction after restart.
"""

import gc
import json
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any, cast

from llm_grounding.domain.corpus import ChunkRetrievalResult, RetrievalUnit
from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.infrastructure.config.models import VectorBackendConfig
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.observability.embedding import EmbeddingRecord


def _load_faiss() -> Any:
    """Load FAISS lazily and provide an actionable optional-dependency error."""
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is not installed. Install the project's vector dependencies "
            "with `uv sync` before selecting the faiss backend."
        ) from exc
    return faiss


class FaissVectorStore:
    """Maintain project-scoped FAISS indexes without a background service."""

    def __init__(
        self,
        config: VectorBackendConfig,
        repository: PostgresRepository,
        root: str | Path,
    ) -> None:
        """Configure persistent paths; index memory is allocated only on access."""
        self._config = config
        self._repository = repository
        self._root = Path(root).expanduser()
        self._states: dict[Path, dict[str, Any]] = {}

    @property
    def loaded_index_count(self) -> int:
        """Return how many FAISS indexes currently occupy process memory."""
        return len(self._states)

    def _token(self, value: str) -> str:
        """Create a filesystem-safe stable token."""
        return sha256(value.encode()).hexdigest()

    def _path(self, project_id: str, model_identity: str, dimension: int) -> Path:
        """Resolve one project/embedding-space index sidecar pair."""
        space = ":".join(
            (
                model_identity,
                str(dimension),
                self._config.similarity_metric,
                self._config.index_type,
                str(self._config.hnsw_m),
                str(self._config.hnsw_ef_construction),
                str(self._config.hnsw_ef_search),
                str(self._config.nlist),
                str(self._config.nprobe),
            )
        )
        return self._root / f"{self._token(project_id)}_{self._token(space)}.index"

    @staticmethod
    def _metadata_path(index_path: Path) -> Path:
        """Return the JSON sidecar path for an index file."""
        return index_path.with_suffix(".json")

    @staticmethod
    def _record_id(chunk_id: str) -> int:
        """Map a chunk identifier to a stable positive FAISS integer ID."""
        value = int.from_bytes(sha256(chunk_id.encode()).digest()[:8], "big")
        return value & ((1 << 63) - 1)

    @staticmethod
    def _normalize(values: list[list[float]], metric: str) -> list[list[float]]:
        """Normalize copied values when cosine similarity is configured."""
        if metric != "cosine":
            return values
        import numpy as np

        array = np.asarray(values, dtype="float32")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return cast(list[list[float]], (array / norms).tolist())

    def _build_index(self, records: list[dict[str, Any]], dimension: int) -> Any:
        """Construct and train the configured FAISS index from sidecar records."""
        faiss = _load_faiss()
        metric = (
            faiss.METRIC_L2
            if self._config.similarity_metric == "euclidean"
            else faiss.METRIC_INNER_PRODUCT
        )
        if self._config.index_type == "flat":
            base = (
                faiss.IndexFlatL2(dimension)
                if metric == faiss.METRIC_L2
                else faiss.IndexFlatIP(dimension)
            )
        elif self._config.index_type == "hnsw":
            base = faiss.IndexHNSWFlat(dimension, self._config.hnsw_m, metric)
            base.hnsw.efConstruction = self._config.hnsw_ef_construction
            base.hnsw.efSearch = self._config.hnsw_ef_search
        else:
            quantizer = (
                faiss.IndexFlatL2(dimension)
                if metric == faiss.METRIC_L2
                else faiss.IndexFlatIP(dimension)
            )
            nlist = min(self._config.nlist, max(1, len(records)))
            base = faiss.IndexIVFFlat(quantizer, dimension, nlist, metric)
            if records:
                import numpy as np

                training = np.asarray(
                    self._normalize(
                        [record["values"] for record in records],
                        self._config.similarity_metric,
                    ),
                    dtype="float32",
                )
                base.train(training)
            base.nprobe = min(self._config.nprobe, nlist)
        index = faiss.IndexIDMap2(base)
        if records:
            import numpy as np

            values = np.asarray(
                self._normalize(
                    [record["values"] for record in records],
                    self._config.similarity_metric,
                ),
                dtype="float32",
            )
            ids = np.asarray([record["faiss_id"] for record in records], dtype="int64")
            index.add_with_ids(values, ids)
        return index

    def _load_state(self, index_path: Path) -> dict[str, Any] | None:
        """Load one index and sidecar into the bounded in-process cache."""
        cached = self._states.get(index_path)
        if cached is not None:
            cached["last_used"] = monotonic()
            return cached
        metadata_path = self._metadata_path(index_path)
        if not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            records = list(metadata["records"])
            dimension = int(metadata["dimension"])
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Invalid FAISS metadata file: {metadata_path}") from exc
        faiss = _load_faiss()
        if index_path.exists():
            index = faiss.read_index(str(index_path))
        else:
            index = self._build_index(records, dimension)
        state: dict[str, Any] = {
            "metadata": metadata,
            "records": {record["chunk_id"]: record for record in records},
            "index": index,
            "last_used": monotonic(),
        }
        self._states[index_path] = state
        self._evict_idle()
        return state

    def _persist(self, index_path: Path, state: dict[str, Any]) -> None:
        """Write the FAISS index and metadata atomically enough for restart reuse."""
        records = list(state["records"].values())
        if not records:
            self._states.pop(index_path, None)
            index_path.unlink(missing_ok=True)
            self._metadata_path(index_path).unlink(missing_ok=True)
            return
        self._root.mkdir(parents=True, exist_ok=True)
        faiss = _load_faiss()
        temporary_index = index_path.with_suffix(".index.tmp")
        temporary_metadata = self._metadata_path(index_path).with_suffix(".json.tmp")
        faiss.write_index(state["index"], str(temporary_index))
        metadata = dict(state["metadata"])
        metadata["records"] = records
        temporary_metadata.write_text(
            json.dumps(metadata, separators=(",", ":")), encoding="utf-8"
        )
        temporary_index.replace(index_path)
        temporary_metadata.replace(self._metadata_path(index_path))

    def _evict_idle(self) -> None:
        """Release indexes that exceeded the configured idle cache lifetime."""
        if self._config.idle_timeout_seconds == 0:
            stale = tuple(self._states)
        else:
            cutoff = monotonic() - self._config.idle_timeout_seconds
            stale = tuple(
                path
                for path, state in self._states.items()
                if state["last_used"] < cutoff
            )
        for path in stale:
            self._states.pop(path, None)
        if stale:
            gc.collect()

    def _rebuild(self, index_path: Path, state: dict[str, Any]) -> None:
        """Rebuild one index after a mutation and persist the new sidecar."""
        records = list(state["records"].values())
        state["index"] = self._build_index(records, int(state["metadata"]["dimension"]))
        state["last_used"] = monotonic()
        self._persist(index_path, state)

    async def upsert(
        self,
        project_id: str,
        chunks: tuple[RetrievalUnit, ...],
        vectors: tuple[EmbeddingVector, ...],
    ) -> None:
        """Persist and index one compatible project-scoped vector batch."""
        if len(chunks) != len(vectors):
            raise ValueError("Each retrieval chunk needs exactly one vector")
        if not vectors:
            return
        first = vectors[0]
        if any(
            vector.dimension != first.dimension
            or vector.model_identity != first.model_identity
            for vector in vectors
        ):
            raise ValueError("One vector upsert batch must use one embedding space")
        self._evict_idle()
        path = self._path(project_id, first.model_identity, first.dimension)
        state = self._load_state(path)
        if state is None:
            state = {
                "metadata": {
                    "version": 1,
                    "project_id": project_id,
                    "model_identity": first.model_identity,
                    "dimension": first.dimension,
                    "similarity_metric": self._config.similarity_metric,
                    "index_type": self._config.index_type,
                    "records": [],
                },
                "records": {},
                "index": None,
                "last_used": monotonic(),
            }
            self._states[path] = state
        for chunk, vector in zip(chunks, vectors, strict=True):
            state["records"][chunk.chunk_id] = {
                "faiss_id": self._record_id(chunk.chunk_id),
                "chunk_id": chunk.chunk_id,
                "ingestion_id": chunk.ingestion_id,
                "document_id": chunk.document_id,
                "values": list(vector.values),
            }
        self._rebuild(path, state)

    def _paths(self) -> tuple[Path, ...]:
        """List persisted sidecars without loading their FAISS indexes."""
        if not self._root.is_dir():
            return ()
        return tuple(sorted(self._root.glob("*.json")))

    def _read_sidecar(self, path: Path) -> dict[str, Any]:
        """Read metadata without consuming an index-cache entry."""
        try:
            return cast(
                dict[str, Any], json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, UnicodeError, ValueError) as exc:
            raise RuntimeError(f"Invalid FAISS metadata file: {path}") from exc

    async def delete_ingestion(self, ingestion_id: str) -> None:
        """Remove ingestion records from every persisted project index."""
        self._evict_idle()
        for metadata_path in self._paths():
            metadata = self._read_sidecar(metadata_path)
            records = [
                record
                for record in metadata.get("records", [])
                if record.get("ingestion_id") != ingestion_id
            ]
            if len(records) == len(metadata.get("records", [])):
                continue
            path = metadata_path.with_suffix(".index")
            state = self._load_state(path)
            if state is not None:
                state["records"] = {record["chunk_id"]: record for record in records}
                self._rebuild(path, state)

    async def delete_document(self, document_id: str) -> None:
        """Remove document records from every persisted project index."""
        self._evict_idle()
        for metadata_path in self._paths():
            metadata = self._read_sidecar(metadata_path)
            records = [
                record
                for record in metadata.get("records", [])
                if record.get("document_id") != document_id
            ]
            if len(records) == len(metadata.get("records", [])):
                continue
            path = metadata_path.with_suffix(".index")
            state = self._load_state(path)
            if state is not None:
                state["records"] = {record["chunk_id"]: record for record in records}
                self._rebuild(path, state)

    async def associate_ingestion(self, project_id: str, ingestion_id: str) -> bool:
        """Copy existing sidecar vectors into the target project's lazy index."""
        found: dict[tuple[str, str], dict[str, Any]] = {}
        for metadata_path in self._paths():
            metadata = self._read_sidecar(metadata_path)
            for record in metadata.get("records", []):
                if record.get("ingestion_id") == ingestion_id:
                    found[
                        (str(metadata["model_identity"]), str(record["chunk_id"]))
                    ] = record
        if not found:
            return False
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for (model_identity, _), record in found.items():
            grouped.setdefault(
                (model_identity, len(record["values"])), []
            ).append(record)
        for (model_identity, dimension), records in grouped.items():
            path = self._path(project_id, model_identity, dimension)
            state = self._load_state(path)
            if state is None:
                state = {
                    "metadata": {
                        "version": 1,
                        "project_id": project_id,
                        "model_identity": model_identity,
                        "dimension": dimension,
                        "similarity_metric": self._config.similarity_metric,
                        "index_type": self._config.index_type,
                        "records": [],
                    },
                    "records": {},
                    "index": None,
                    "last_used": monotonic(),
                }
                self._states[path] = state
            state["records"].update({record["chunk_id"]: record for record in records})
            self._rebuild(path, state)
        return True

    async def remove_project(self, project_id: str) -> None:
        """Remove all persisted indexes belonging to one project."""
        token = self._token(project_id)
        for metadata_path in self._paths():
            metadata = self._read_sidecar(metadata_path)
            if metadata.get("project_id") != project_id:
                continue
            index_path = metadata_path.with_suffix(".index")
            self._states.pop(index_path, None)
            metadata_path.unlink(missing_ok=True)
            index_path.unlink(missing_ok=True)
        self._root.mkdir(parents=True, exist_ok=True)
        for path in self._root.glob(f"{token}_*"):
            path.unlink(missing_ok=True)

    async def search(
        self,
        project_id: str,
        query: EmbeddingVector,
        top_k: int,
        document_ids: tuple[str, ...] = (),
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Search one lazily loaded FAISS index and hydrate canonical chunks."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self._evict_idle()
        path = self._path(project_id, query.model_identity, query.dimension)
        state = self._load_state(path)
        if state is None:
            return ()
        records = state["records"]
        if not records:
            return ()
        import numpy as np

        query_values = self._normalize(
            [list(query.values)], self._config.similarity_metric
        )
        limit = min(len(records), max(top_k, top_k * (10 if document_ids else 1)))
        distances, ids = state["index"].search(
            np.asarray(query_values, dtype="float32"), limit
        )
        selected: list[tuple[dict[str, Any], float]] = []
        by_id = {record["faiss_id"]: record for record in records.values()}
        for raw_distance, raw_id in zip(distances[0], ids[0], strict=True):
            record = by_id.get(int(raw_id))
            if record is None or (
                document_ids and record["document_id"] not in document_ids
            ):
                continue
            score = (
                1.0 - float(raw_distance)
                if self._config.similarity_metric == "cosine"
                else (
                    -float(raw_distance)
                    if self._config.similarity_metric == "euclidean"
                    else float(raw_distance)
                )
            )
            selected.append((record, score))
            if len(selected) == top_k:
                break
        chunk_ids = tuple(record["chunk_id"] for record, _ in selected)
        chunks = await self._repository.chunks_by_ids(chunk_ids)
        return tuple(
            ChunkRetrievalResult(
                chunks[record["chunk_id"]], rank, "faiss", score
            )
            for rank, (record, score) in enumerate(selected, 1)
            if record["chunk_id"] in chunks
        )

    async def fetch_vectors(
        self, chunk_ids: tuple[str, ...], *, limit: int = 128
    ) -> tuple[EmbeddingRecord, ...]:
        """Fetch bounded persisted vectors without loading unrelated indexes."""
        if not chunk_ids:
            return ()
        if limit < 1 or limit > 128:
            raise ValueError("Embedding sampling limit must be in [1, 128]")
        wanted = set(chunk_ids)
        output: list[EmbeddingRecord] = []
        for metadata_path in self._paths():
            metadata = self._read_sidecar(metadata_path)
            for record in metadata.get("records", []):
                if record.get("chunk_id") not in wanted:
                    continue
                vector = EmbeddingVector(
                    tuple(float(value) for value in record["values"]),
                    str(metadata["model_identity"]),
                )
                output.append(
                    EmbeddingRecord(
                        vector=list(vector.values),
                        vector_norm=vector.vector_norm,
                        embedding_id=vector.embedding_id,
                        source_id=record["chunk_id"],
                        source_type="document_chunk",
                        model_id=vector.model_identity,
                    )
                )
                if len(output) >= limit:
                    return tuple(output)
        return tuple(output)

    async def evict_idle(self) -> None:
        """Explicitly release indexes that exceeded the configured idle timeout."""
        self._evict_idle()

    async def close(self) -> None:
        """Release all loaded FAISS objects while retaining persisted index files."""
        self._states.clear()
        gc.collect()
