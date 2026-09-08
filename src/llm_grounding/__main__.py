"""Delegate module execution to the normal application CLI."""

from llm_grounding.application.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
