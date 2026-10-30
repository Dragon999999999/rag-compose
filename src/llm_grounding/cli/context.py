"""Shared composition helpers used by persistent CLI command modules."""

from llm_grounding.application.bootstrap import (
    PhaseOneApplication,
    create_phase_one_application,
)
from llm_grounding.infrastructure.config.loader import (
    load_ingestion_config,
    load_pipeline_config,
)
from llm_grounding.orchestration.grounding_diagnostics import (
    GroundingDiagnostics,
    GroundingDiagnosticsLevel,
    configure_console_grounding_diagnostics,
)


def build_application(args: object) -> PhaseOneApplication:
    """Build the normal application composition without introducing HTTP calls."""
    architecture = load_pipeline_config()
    ingestion = load_ingestion_config(getattr(args, "ingestion", None))
    diagnostics_level = GroundingDiagnosticsLevel(
        getattr(args, "grounding_diagnostics", GroundingDiagnosticsLevel.OFF.value)
    )
    if diagnostics_level != GroundingDiagnosticsLevel.OFF:
        configure_console_grounding_diagnostics()
    pipeline_directory = (
        getattr(args, "directory", None)
        if getattr(args, "command", None) == "pipeline"
        and getattr(args, "pipeline_command", None) == "import"
        else None
    )
    return create_phase_one_application(
        architecture,
        ingestion,
        GroundingDiagnostics(diagnostics_level),
        pipeline_directory=pipeline_directory,
    )


async def start_application(
    args: object, *, synchronize: bool = True
) -> PhaseOneApplication:
    """Build application services, optionally deferring startup synchronization."""
    application = build_application(args)
    if synchronize:
        await application.startup()
    return application
