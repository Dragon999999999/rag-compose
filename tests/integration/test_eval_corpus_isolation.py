"""Acceptance checks for production/evaluation corpus separation."""

from llm_grounding.application.eval_corpus_service import EvalCorpusService


class IsolatedRepository:
    """Repository exposing only evaluation persistence for this boundary test."""

    def __init__(self) -> None:
        """Start an empty evaluation-only corpus map."""
        self.corpora = {}

    async def save_evaluation_corpus(self, corpus) -> None:
        """Persist only evaluation data."""
        self.corpora[corpus.corpus_id] = corpus

    async def get_evaluation_corpus(self, corpus_id):
        """Return one evaluation corpus when present."""
        return self.corpora.get(corpus_id)


async def test_evaluation_corpus_never_requires_project_storage() -> None:
    """A perturbation corpus can be created with no production project API."""
    repository = IsolatedRepository()
    corpus = await EvalCorpusService(repository).create(corpus_id="evaluation")

    assert corpus.corpus_id in repository.corpora
