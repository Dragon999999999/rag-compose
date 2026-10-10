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
