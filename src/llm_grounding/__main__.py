"""Delegate module execution to the reorganized CLI package."""

from llm_grounding.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
