"""Convert physical sources to structured blocks with stable provenance.

Implementations may read files or remote sources and must preserve source identity,
page positions where known, and deterministic ordering. They must surface missing,
malformed, or unsupported sources as errors and must not construct evidence or
perform reasoning. See docs/architecture.md for the source/knowledge boundary.
"""

from typing import Protocol

from llm_grounding.domain.documents import DocumentSource, ParsedDocument


class DocumentParser(Protocol):
    """Define the interchangeable parsing component contract."""

    async def parse(self, source: DocumentSource) -> ParsedDocument:
        """Parse a source, preserving structure and raising errors for invalid input."""
        ...
