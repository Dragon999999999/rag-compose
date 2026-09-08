"""Project-scoped vector retrieval and simple multi-facet summary selection."""

from llm_grounding.domain.corpus import ChunkRetrievalResult, CorpusRequest, TaskType
from llm_grounding.embedding.protocol import Embedder
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.infrastructure.vector.protocol import VectorStore

SUMMARY_FACETS = (
    "objective and problem addressed",
    "main contribution and method",
    "experiments, data, and evaluation",
    "key results and findings",
    "limitations and conclusion",
)


class VectorRetriever:
    """Embed a query explicitly and search a project-filtered derived vector index."""

    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int = 8) -> None:
        """Configure a compatible embedder, vector store, and positive candidate limit."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self._embedder, self._store, self._top_k = embedder, store, top_k

    async def retrieve(
        self, request: CorpusRequest
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Return query-relevant chunks scoped in the store by active project."""
        query = await self._embedder.embed_query(request.task)
        return await self._store.search(
            request.project_id, query, self._top_k, request.document_ids
        )


class SummaryRetriever:
    """Use all fitting corpus chunks or retrieve across fixed deterministic summary facets."""

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
        self, request: CorpusRequest
    ) -> tuple[ChunkRetrievalResult, ...]:
        """Avoid a naive single 'summarize' query that would bias coverage."""
        if request.task_type != TaskType.SUMMARY:
            return await self._vector.retrieve(request)
        all_chunks = await self._repository.chunks_for_project(
            request.project_id, request.document_ids
        )
        if sum(len(chunk.text) for chunk in all_chunks) <= self._context_chars:
            return tuple(
                ChunkRetrievalResult(chunk, rank, "summary_all_chunks", None)
                for rank, chunk in enumerate(all_chunks, 1)
            )
        selected: dict[str, ChunkRetrievalResult] = {}
        for facet in SUMMARY_FACETS:
            results = await self._vector.retrieve(
                CorpusRequest(
                    request.project_id,
                    f"{facet}. {request.task}",
                    TaskType.QA,
                    request.document_ids,
                )
            )
            for result in results[: self._per_facet]:
                selected.setdefault(result.chunk.chunk_id, result)
        ordered = sorted(
            selected.values(),
            key=lambda item: (item.chunk.document_id, item.chunk.order),
        )
        bounded: list[ChunkRetrievalResult] = []
        size = 0
        for result in ordered:
            if bounded and size + len(result.chunk.text) > self._context_chars:
                continue
            bounded.append(result)
            size += len(result.chunk.text)
        return tuple(
            ChunkRetrievalResult(item.chunk, rank, "summary_facets", item.score)
            for rank, item in enumerate(bounded, 1)
        )
