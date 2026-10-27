"""Project-scoped vector retrieval and simple multi-facet summary selection."""

from llm_grounding.domain.corpus import ChunkRetrievalResult, CorpusRequest, TaskType
from llm_grounding.embedding.models import EmbeddingVector
from llm_grounding.embedding.protocol import Embedder
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.infrastructure.vector.protocol import VectorStore
from llm_grounding.ingestion.models import EmbeddingConfig
from llm_grounding.retrieval.models import (
    RetrievalContext,
    RetrievalRequest,
    RetrievalResponse,
)

SUMMARY_FACETS = (
    "objective and problem addressed",
    "main contribution and method",
    "experiments, data, and evaluation",
    "key results and findings",
    "limitations and conclusion",
)


class VectorRetriever:
    """Embed a query explicitly and search a project-filtered derived vector index."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        top_k: int = 8,
        *,
        embedding_config: EmbeddingConfig | None = None,
    ) -> None:
        """Configure an embedder, vector store, and positive candidate limit."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self._embedder, self._store, self._top_k = embedder, store, top_k
        self.embedding_config = embedding_config
        self._last_query_embedding: EmbeddingVector | None = None

    @property
    def last_query_embedding(self) -> EmbeddingVector | None:
        """Return the most recent query vector for observation instrumentation."""
        return self._last_query_embedding

    @property
    def default_embedder(self) -> Embedder:
        """Return the composition-root embedder used when no override is supplied."""
        return self._embedder

    async def retrieve(
        self,
        request: RetrievalRequest | CorpusRequest,
        context: RetrievalContext | None = None,
        *,
        top_k: int | None = None,
        embedder: Embedder | None = None,
    ) -> RetrievalResponse:
        """Return query-relevant chunks using the canonical retrieval contract.

        ``CorpusRequest`` and keyword overrides remain accepted for existing
        application callers; new code should pass ``RetrievalRequest`` and a
        ``RetrievalContext``.
        """
        canonical = (
            RetrievalRequest.from_corpus(request, top_k=top_k)
            if isinstance(request, CorpusRequest)
            else request
        )
        active_context = context or RetrievalContext()
        if (
            active_context.embedding_config is not None
            and self.embedding_config is not None
            and not active_context.embedding_config.is_compatible_with(
                self.embedding_config
            )
        ):
            raise ValueError(
                "Query embedding configuration is incompatible with the indexed "
                "embedding space"
            )
        active_embedder = embedder or active_context.embedder or self._embedder
        store = active_context.vector_store or self._store
        query = await active_embedder.embed_query(canonical.query)
        self._last_query_embedding = query
        results = await store.search(
            canonical.project_id,
            query,
            canonical.top_k or top_k or self._top_k,
            canonical.document_ids,
        )
        return RetrievalResponse(
            canonical.query,
            results,
            query,
            metadata={
                "retriever": "dense",
                "score_type": "vector_similarity",
                "embedding_identity": query.model_identity,
                "embedding_dimension": query.dimension,
            },
        )


class SummaryRetriever:
    """Retrieve all fitting chunks or fixed deterministic summary facets."""

    def __init__(
        self,
        repository: PostgresRepository,
        vector: VectorRetriever,
        context_chars: int,
        per_facet: int,
    ) -> None:
        """Configure bounded whole-corpus and multi-facet summary selection."""
        if context_chars < 1 or per_facet < 1:
            raise ValueError("Summary context_chars and per_facet must be positive")
        self._repository, self._vector = repository, vector
        self._context_chars, self._per_facet = context_chars, per_facet

    async def retrieve(
        self,
        request: RetrievalRequest | CorpusRequest,
        context: RetrievalContext | None = None,
        *,
        top_k: int | None = None,
        context_chars: int | None = None,
        per_facet: int | None = None,
        embedder: Embedder | None = None,
    ) -> RetrievalResponse:
        """Avoid a naive single 'summarize' query that would bias coverage."""
        canonical = (
            RetrievalRequest.from_corpus(request, top_k=top_k)
            if isinstance(request, CorpusRequest)
            else request
        )
        active_context = context or RetrievalContext()
        if canonical.task_type != TaskType.SUMMARY:
            return await self._vector.retrieve(
                canonical,
                active_context,
                top_k=top_k,
                embedder=embedder,
            )
        repository = active_context.repository or self._repository
        active_context_chars = context_chars or self._context_chars
        active_per_facet = per_facet or self._per_facet
        all_chunks = await repository.chunks_for_project(
            canonical.project_id, canonical.document_ids
        )
        if sum(len(chunk.text) for chunk in all_chunks) <= active_context_chars:
            return RetrievalResponse(
                canonical.query,
                tuple(
                    ChunkRetrievalResult(chunk, rank, "summary_all_chunks", None)
                    for rank, chunk in enumerate(all_chunks, 1)
                ),
            )
        selected: dict[str, ChunkRetrievalResult] = {}
        for facet in SUMMARY_FACETS:
            response = await self._vector.retrieve(
                RetrievalRequest(
                    canonical.project_id,
                    f"{facet}. {canonical.query}",
                    canonical.document_ids,
                    top_k,
                    TaskType.QA,
                ),
                active_context,
                top_k=top_k,
                embedder=embedder,
            )
            for result in response.results[:active_per_facet]:
                selected.setdefault(result.chunk.chunk_id, result)
        ordered = sorted(
            selected.values(),
            key=lambda item: (item.chunk.document_id, item.chunk.order),
        )
        bounded: list[ChunkRetrievalResult] = []
        size = 0
        for result in ordered:
            if bounded and size + len(result.chunk.text) > active_context_chars:
                continue
            bounded.append(result)
            size += len(result.chunk.text)
        return RetrievalResponse(
            canonical.query,
            tuple(
                ChunkRetrievalResult(item.chunk, rank, "summary_facets", item.score)
                for rank, item in enumerate(bounded, 1)
            ),
        )
