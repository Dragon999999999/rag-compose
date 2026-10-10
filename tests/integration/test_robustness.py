"""Acceptance checks for reproducible perturbation metadata."""

from llm_grounding.application.eval_corpus_service import EvalCorpusService


class CorpusRepository:
    """Minimal isolated corpus store for robustness acceptance tests."""

    def __init__(self) -> None:
        """Start an empty isolated corpus map."""
        self.corpora = {}

    async def save_evaluation_corpus(self, corpus) -> None:
        """Save only the supplied evaluation corpus."""
        self.corpora[corpus.corpus_id] = corpus

    async def get_evaluation_corpus(self, corpus_id):
        """Return one isolated corpus when present."""
        return self.corpora.get(corpus_id)


async def test_perturbation_corpus_has_parent_and_stable_hash() -> None:
    """A perturbation is a separate versioned corpus identity."""
    repository = CorpusRepository()
    service = EvalCorpusService(repository)
    corpus = await service.create(
        corpus_id="perturbed",
        parent_corpus_id="original",
        perturbation_metadata={"operation": "typo", "seed": 4},
    )

    assert corpus.parent_corpus_id == "original"
    assert corpus.perturbation_metadata["seed"] == 4
    assert len(corpus.corpus_hash) == 64
