"""Source representations shared by parsing and knowledge construction.

Pages are one-based; block indices and reading-order indices are zero-based.
Bounding boxes use PDF points in the parser's page coordinate system. These
records preserve structure, without assuming that source elements are evidence.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DocumentSource:
    """Identify a local source independently of the technology used to parse it."""

    document_id: str
    path: Path


@dataclass(frozen=True)
class DocumentElement:
    """Preserve one source block and its position in a deterministic reading order."""

    document_id: str
    page_number: int
    block_index: int
    order: int
    text: str
    bounding_box: tuple[float, float, float, float] | None = None
    kind: str = "text"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    """Retain ordered source elements and document identity before indexing."""

    source: DocumentSource
    elements: tuple[DocumentElement, ...]
    page_count: int
    metadata: dict[str, str] = field(default_factory=dict)
