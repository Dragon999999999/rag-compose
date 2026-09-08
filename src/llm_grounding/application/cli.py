"""Terminal entry point delegating configuration and execution to the service."""

import argparse
import asyncio
from pathlib import Path

from llm_grounding.application.service import RagService, validate_pdf_path
from llm_grounding.infrastructure.config.loader import load_architecture_config
from llm_grounding.orchestration.runner import RunFailedError


def create_parser() -> argparse.ArgumentParser:
    """Define the normal application command and explicit architecture selection."""
    parser = argparse.ArgumentParser(description="Basic Summary Extraction Baseline")
    parser.add_argument("pdf_file", type=Path, help="Research paper PDF to process")
    parser.add_argument(
        "--architecture",
        type=Path,
        help="Architecture YAML (default: packaged baseline_v0)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the application and print the general answer; report actionable errors."""
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        path = validate_pdf_path(args.pdf_file)
        config = load_architecture_config(args.architecture)
        service = RagService(config)
        run = asyncio.run(service.process_pdf(path))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    except RunFailedError as exc:
        parser.exit(
            1,
            f"error: {exc} Check PDF content, architecture settings, and "
            "AQUEDUCT_API_KEY or AQUEDUCT_API_KEY_FILE; model requests require "
            "a reachable endpoint and valid structured output.\n",
        )
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted.\n")
    if run.answer is None:
        parser.exit(1, "error: Completed run did not produce an answer.\n")
    print(run.answer.text)
    print("\nExtraction successful.")
    return 0
