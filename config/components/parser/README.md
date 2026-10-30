# Parser components

Parser component definitions are validated by `schema.yaml` and use the shared
component fields: `id`, `type`, `implementation`, `description`, `config`,
`requirements`, `input_terms`, and `output_terms`.

The current registered parser is `parser_pymupdf`, which uses PyMuPDF to extract
positioned PDF text blocks and preserve page and source-span provenance. Its
reading order is positional and heuristic; it does not perform OCR. Add another
parser by defining its YAML contract here, registering the implementation in the
component registry, and adding focused ingestion tests.
