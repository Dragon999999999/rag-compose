"""Genuine BM25 lexical retrieval over canonical PostgreSQL chunks."""

import math
import re

from llm_grounding.domain.corpus import ChunkRetrievalResult
from llm_grounding.infrastructure.database.repositories import PostgresRepository
from llm_grounding.retrieval.models import (
    RetrievalContext,
    RetrievalRequest,
    RetrievalResponse,
)

_TOKEN = re.compile(r"\w+", re.UNICODE)


def bm25_score(
    query_tokens: tuple[str, ...],
    document_tokens: tuple[str, ...],
    document_frequency: dict[str, int],
    document_count: int,
    average_length: float,
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """Calculate Okapi BM25 for one document using Robertson IDF."""
    if not query_tokens or not document_tokens or document_count < 1:
        return 0.0
    frequencies: dict[str, int] = {}
    for token in document_tokens:
        frequencies[token] = frequencies.get(token, 0) + 1
    score = 0.0
    length = len(document_tokens)
    normalization = k1 * (1.0 - b + b * length / max(average_length, 1.0))
    for token in set(query_tokens):
        term_frequency = frequencies.get(token, 0)
        if term_frequency == 0:
            continue
        df = document_frequency.get(token, 0)
        idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
        score += idf * (term_frequency * (k1 + 1.0)) / (term_frequency + normalization)
    return score


class BM25Retriever:
    """Rank project chunks with corpus-local Okapi BM25 scoring."""

    def __init__(
        self,
        repository: PostgresRepository,
        top_k: int = 20,
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        """Configure bounded result count and standard BM25 parameters."""
        if top_k < 1 or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 limits or parameters")
        self._repository = repository
        self._top_k = top_k
        self._k1 = k1
        self._b = b

    async def retrieve(
        self,
        request: RetrievalRequest,
        context: RetrievalContext,
    ) -> RetrievalResponse:
        """Load canonical chunks and return deterministic BM25 rankings."""
        repository = context.repository or self._repository
        chunks = await repository.chunks_for_project(
            request.project_id, request.document_ids
        )
        query_tokens = _tokens(request.query)
        documents = tuple((_tokens(chunk.text), chunk) for chunk in chunks)
        frequencies: dict[str, int] = {}
        for tokens, _chunk in documents:
            for token in set(tokens):
                frequencies[token] = frequencies.get(token, 0) + 1
        average_length = sum(len(tokens) for tokens, _chunk in documents) / max(
            len(documents), 1
        )
        scored = [
            (
                bm25_score(
                    query_tokens,
                    tokens,
                    frequencies,
                    len(documents),
                    average_length,
                    k1=self._k1,
                    b=self._b,
                ),
                chunk,
            )
            for tokens, chunk in documents
        ]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].document_id,
                item[1].order,
                item[1].chunk_id,
            )
        )
        limit = request.top_k or self._top_k
        results = tuple(
            ChunkRetrievalResult(chunk, rank, "bm25", score)
            for rank, (score, chunk) in enumerate(scored[:limit], 1)
        )
        return RetrievalResponse(
            request.query,
            results,
            metadata={
                "retriever": "bm25",
                "score_type": "bm25",
                "k1": self._k1,
                "b": self._b,
                "corpus_size": len(documents),
            },
        )


def _tokens(text: str) -> tuple[str, ...]:
    """Tokenize text consistently for both query and document scoring."""
    return tuple(match.group(0).casefold() for match in _TOKEN.finditer(text))
