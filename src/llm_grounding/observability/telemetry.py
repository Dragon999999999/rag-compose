"""Optional local process resource measurements for observability spans."""

import resource
from dataclasses import dataclass
from time import monotonic

from llm_grounding.observability.models import ResourceUsage


@dataclass(frozen=True)
class ProcessSample:
    """Raw process counters used to derive one span's resource delta."""

    monotonic_seconds: float
    cpu_seconds: float
    memory_bytes: int | None


def process_sample() -> ProcessSample:
    """Capture process CPU counters and current/peak resident memory."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return ProcessSample(
        monotonic_seconds=monotonic(),
        cpu_seconds=usage.ru_utime + usage.ru_stime,
        memory_bytes=_resident_memory_bytes() or _max_rss_bytes(usage.ru_maxrss),
    )


def capture_process_usage(
    component_id: str,
    *,
    span_id: str | None = None,
    started: ProcessSample | None = None,
) -> ResourceUsage:
    """Return local process usage without implying remote-provider telemetry."""
    current = process_sample()
    cpu_time_ms = None
    utilization = None
    memory_average = current.memory_bytes
    if started is not None:
        cpu_time_ms = max(0.0, current.cpu_seconds - started.cpu_seconds) * 1000
        wall_seconds = max(
            0.000001, current.monotonic_seconds - started.monotonic_seconds
        )
        utilization = min(100.0, max(0.0, cpu_time_ms / 1000 / wall_seconds * 100))
        if started.memory_bytes is not None and current.memory_bytes is not None:
            memory_average = (started.memory_bytes + current.memory_bytes) // 2
    return ResourceUsage(
        component_id=component_id,
        span_id=span_id,
        cpu_time_ms=cpu_time_ms,
        cpu_utilization_percent=utilization,
        memory_peak_bytes=current.memory_bytes,
        memory_average_bytes=memory_average,
        measurement_scope="process",
        measurement_source="stdlib.resource.procfs",
        metadata={"provider_resource_usage": "unavailable"},
    )


def _resident_memory_bytes() -> int | None:
    """Read current RSS from Linux procfs when available."""
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    value = line.split()[1]
                    return int(value) * 1024
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None
    return None


def _max_rss_bytes(value: int) -> int:
    """Normalize ru_maxrss's platform-specific unit to bytes."""
    return value * 1024 if value else 0
