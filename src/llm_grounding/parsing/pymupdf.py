"""Extract PDF text blocks using the original positional reading-order heuristic.

Blocks are grouped into ten-point vertical bands, then sorted left to right.
This does not guarantee correct reading order for arbitrary multi-column PDFs.
PyMuPDF is confined to this concrete adapter; domain records contain no vendor
objects. Calls execute synchronously to avoid PyMuPDF's thread-safety limitations.
"""

from hashlib import sha256

import pymupdf

from llm_grounding.domain.documents import (
    DocumentElement,
    DocumentSource,
    ParsedDocument,
)


class PyMuPDFDocumentParser:
    """Read local PDFs without OCR, retaining one element per nonempty text block."""

    async def parse(self, source: DocumentSource) -> ParsedDocument:
        """Read the PDF and retain page, block, bounding box, and content digest.

        Raises:
            FileNotFoundError: If the source is not an existing file.
            ValueError: If input is not a readable, unencrypted PDF with text.
        """
        path = source.path
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file: {path}")
        content = path.read_bytes()
        elements: list[DocumentElement] = []
        try:
            # PyMuPDF's Document constructor does not publish parameter types.
            with pymupdf.open(stream=content, filetype="pdf") as document:  # type: ignore[no-untyped-call]
                if not document.is_pdf or document.needs_pass:
                    raise ValueError(f"Expected an unencrypted PDF: {path}")
                for page_number, page in enumerate(document, start=1):
                    blocks = page.get_text("blocks")
                    blocks.sort(key=lambda block: (block[1] // 10 * 10, block[0]))
                    for block in blocks:
                        if block[6] != 0 or not block[4].strip():
                            continue
                        elements.append(
                            DocumentElement(
                                document_id=source.document_id,
                                page_number=page_number,
                                block_index=block[5],
                                order=len(elements),
                                text=block[4],
                                bounding_box=tuple(block[:4]),
                            )
                        )
                page_count = len(document)
        except (pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
            raise ValueError(f"Unable to parse PDF: {path}") from exc
        if not elements:
            raise ValueError(
                f"PDF contains no extractable text (OCR is unavailable): {path}"
            )
        return ParsedDocument(
            source=source,
            elements=tuple(elements),
            page_count=page_count,
            metadata={"sha256": sha256(content).hexdigest()},
        )
