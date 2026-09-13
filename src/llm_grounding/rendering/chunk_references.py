"""Stable human-readable labels for canonical retrieval chunks.

Canonical chunk IDs remain internal persistence identities. This module derives
display-only labels from the original document name and the chunk's immutable
per-ingestion order, so a label remains comparable across chats in a project.
"""

import re
from pathlib import Path

from llm_grounding.domain.corpus import RetrievalUnit

_UNSAFE_LABEL_CHARACTERS = re.compile(r"[^\w]+", flags=re.UNICODE)


def chunk_reference_label(chunk: RetrievalUnit) -> str:
    """Return a stable `Document_15` label for a one-based chunk position.

    Args:
        chunk: Canonical retrieval unit whose document name and ordering are fixed
            for its ingestion representation.

    Returns:
        User-facing label derived from the PDF filename and chunk position.
    """
    document_stem = Path(chunk.document_name).stem
    normalized_stem = _UNSAFE_LABEL_CHARACTERS.sub("_", document_stem).strip("_")
    return f"{normalized_stem or 'document'}_{chunk.order + 1}"
