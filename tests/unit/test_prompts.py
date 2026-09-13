"""Test the external prompt library used by Basic RAG reasoning."""

from llm_grounding.infrastructure.prompts import PromptLibrary
from llm_grounding.reasoning.basic_grounded import PromptChunk


def test_grounded_qa_prompt_loads_by_registered_id() -> None:
    """The registry exposes the versioned grounded QA prompt and its template."""
    prompts = PromptLibrary.default()

    metadata = prompts.metadata("grounded_qa_v1")
    rendered = prompts.render(
        "grounded_qa_v1",
        {
            "task": "What does the paper introduce?",
            "history": "(none)",
            "chunks": (
                PromptChunk(
                    "chunk_1",
                    "paper.pdf",
                    "1",
                    "2",
                    "The paper introduces a framework.",
                ),
            ),
        },
    )

    assert metadata.prompt_id == "grounded_qa_v1"
    assert metadata.version == "1.1"
    assert rendered.metadata == metadata
    assert '<chunk ref="chunk_1">' in rendered.text
    assert "The paper introduces a framework." in rendered.text
    assert "Populate `turn_title`" in rendered.text
