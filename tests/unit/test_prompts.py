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
            "generate_chat_name": True,
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
    assert metadata.version == "1.3"
    assert rendered.metadata == metadata
    assert '<chunk ref="chunk_1">' in rendered.text
    assert "The paper introduces a framework." in rendered.text
    assert "Populate `turn_title`" in rendered.text
    assert "Populate `chat_name`" in rendered.text
    assert "first answer in a chat" in rendered.text


def test_grounded_qa_prompt_does_not_rename_a_finalized_chat() -> None:
    """Later turns explicitly tell the model not to propose another chat name."""
    rendered = PromptLibrary.default().render(
        "grounded_qa_v1",
        {
            "task": "Follow up",
            "history": "user: Earlier question",
            "generate_chat_name": False,
            "chunks": (PromptChunk("chunk_1", "paper.pdf", "1", "1", "Evidence."),),
        },
    )

    assert "Set `chat_name` to `null`" in rendered.text
    assert "first answer in a chat" not in rendered.text
