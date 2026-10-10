"""Isolated evaluation-corpus lifecycle and deterministic experiment identities."""

import json
from hashlib import sha256
from typing import Any
from uuid import uuid4

from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.observability.models import EvaluationCorpusReference


class EvalCorpusService:
    """Manage benchmark inventories without touching production project tables."""

    def __init__(self, repository: PostgresRepository) -> None:
        """Inject the repository that owns the separate evaluation-corpus table."""
        self._repository = repository

    async def create(
        self,
        *,
        corpus_id: str | None = None,
        parent_corpus_id: str | None = None,
        documents: list[dict[str, Any]] | None = None,
        chunks: list[dict[str, Any]] | None = None,
        perturbation_metadata: dict[str, Any] | None = None,
        version: int = 1,
    ) -> EvaluationCorpusReference:
        """Create an isolated corpus with a deterministic content hash."""
        documents = documents or []
        chunks = chunks or []
        perturbation_metadata = perturbation_metadata or {}
        identifier = corpus_id or f"eval-{uuid4()}"
        corpus = EvaluationCorpusReference(
            corpus_id=identifier,
            parent_corpus_id=parent_corpus_id,
            corpus_hash=self._hash(
                parent_corpus_id, version, documents, chunks, perturbation_metadata
            ),
            version=version,
            documents=documents,
            chunks=chunks,
            perturbation_metadata=perturbation_metadata,
        )
        await self._repository.save_evaluation_corpus(corpus)
        return corpus

    async def get(self, corpus_id: str) -> EvaluationCorpusReference:
        """Return an isolated corpus or raise a client-facing lookup error."""
        corpus = await self._repository.get_evaluation_corpus(corpus_id)
        if corpus is None:
            raise KeyError(f"Unknown evaluation corpus: {corpus_id}")
        return corpus

    async def replace_inventory(
        self,
        corpus_id: str,
        *,
        documents: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        perturbation_metadata: dict[str, Any] | None = None,
    ) -> EvaluationCorpusReference:
        """Replace only an evaluation inventory and increment its version."""
        current = await self.get(corpus_id)
        return await self.create(
            corpus_id=corpus_id,
            parent_corpus_id=current.parent_corpus_id,
            documents=documents,
            chunks=chunks,
            perturbation_metadata=perturbation_metadata
            if perturbation_metadata is not None
            else current.perturbation_metadata,
            version=current.version + 1,
        )

    async def delete(self, corpus_id: str) -> None:
        """Delete an isolated corpus and all of its evaluation-only inventory."""
        deleted = await self._repository.delete_evaluation_corpus(corpus_id)
        if not deleted:
            raise KeyError(f"Unknown evaluation corpus: {corpus_id}")

    async def upload_documents(
        self, corpus_id: str, documents: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Append source-document metadata and record a completed upload operation."""
        current = await self.get(corpus_id)
        corpus = await self.replace_inventory(
            corpus_id,
            documents=[*current.documents, *documents],
            chunks=current.chunks,
        )
        return await self._operation(corpus, "documents", len(documents))

    async def upload_chunks(
        self, corpus_id: str, chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Append canonical chunks without touching production ingestion or vectors."""
        current = await self.get(corpus_id)
        corpus = await self.replace_inventory(
            corpus_id,
            documents=current.documents,
            chunks=[*current.chunks, *chunks],
        )
        return await self._operation(corpus, "chunks", len(chunks))

    async def operation(self, operation_id: str) -> dict[str, Any]:
        """Return one persisted upload-operation status."""
        loader = getattr(self._repository, "get_evaluation_operation", None)
        payload = await loader(operation_id) if loader is not None else None
        if payload is None:
            raise KeyError(f"Unknown evaluation operation: {operation_id}")
        return payload

    async def _operation(
        self, corpus: EvaluationCorpusReference, kind: str, count: int
    ) -> dict[str, Any]:
        """Persist a synchronous operation in the same shape as future async jobs."""
        operation_id = f"eval-op-{uuid4()}"
        payload: dict[str, Any] = {
            "operation_id": operation_id,
            "corpus_id": corpus.corpus_id,
            "operation": kind,
            "status": "completed",
            "count": count,
            "version": corpus.version,
            "corpus_hash": corpus.corpus_hash,
        }
        saver = getattr(self._repository, "save_evaluation_operation", None)
        if saver is not None:
            await saver(operation_id, corpus.corpus_id, "completed", payload)
        return payload

    @staticmethod
    def deterministic_request_id(
        *,
        corpus_id: str,
        benchmark_case_id: str,
        experiment_id: str | None,
        query: str,
        turn_index: int = 0,
    ) -> str:
        """Derive a stable request identity for repeatable benchmark runs."""
        value = json.dumps(
            [corpus_id, benchmark_case_id, experiment_id, query, turn_index],
            separators=(",", ":"),
        )
        return sha256(value.encode()).hexdigest()

    @staticmethod
    def _hash(
        parent_corpus_id: str | None,
        version: int,
        documents: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        perturbation_metadata: dict[str, Any],
    ) -> str:
        """Hash only canonical, sorted JSON and never hash object addresses."""
        value = json.dumps(
            {
                "parent_corpus_id": parent_corpus_id,
                "version": version,
                "documents": documents,
                "chunks": chunks,
                "perturbation_metadata": perturbation_metadata,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(value.encode()).hexdigest()
