"""Backward-compatible import path for the reorganized CLI package."""

import argparse

from llm_grounding.cli.main import _run_command, create_phase_one_parser, main

__all__ = ["create_phase_one_parser", "main"]


async def _run_phase_one_command(args: argparse.Namespace) -> int:
    """Preserve the former private dispatcher import for existing callers."""
    return await _run_command(args)
