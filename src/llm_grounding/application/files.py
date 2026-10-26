"""Application-level validation for user-provided local files."""

from pathlib import Path


def validate_pdf_path(path: Path) -> Path:
    """Resolve an existing PDF path before an ingestion operation begins.

    Raises:
        FileNotFoundError: If the path is missing or is not a file.
        ValueError: If the extension does not identify a PDF.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file: {path}")
    return path
