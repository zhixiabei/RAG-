from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from time import perf_counter
from typing import Any, Callable, Iterator, TypeVar


@dataclass(frozen=True)
class ModelUsageEvent:
    stage: str
    operation: str
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None

    @property
    def reported(self) -> bool:
        return any(
            value is not None
            for value in (self.input_tokens, self.output_tokens, self.total_tokens)
        )


@dataclass(frozen=True)
class TimingEvent:
    """One wall-clock measurement inside a request trace."""

    stage: str
    elapsed_ms: float
    depth: int
    parent: str | None


@dataclass
class TimingCollector:
    events: list[TimingEvent] = field(default_factory=list)

    def record(self, event: TimingEvent) -> None:
        self.events.append(event)

    def summary(self, total_ms: float | None = None) -> dict[str, Any]:
        if total_ms is None:
            roots = [event for event in self.events if event.depth == 0]
            total_ms = max((event.elapsed_ms for event in roots), default=0.0)
        total_ms = max(0.0, float(total_ms))

        by_stage: dict[str, dict[str, Any]] = {}
        for event in self.events:
            stage = by_stage.setdefault(
                event.stage,
                {"calls": 0, "total_ms": 0.0, "max_depth": event.depth},
            )
            stage["calls"] += 1
            stage["total_ms"] += event.elapsed_ms
            stage["max_depth"] = max(stage["max_depth"], event.depth)

        def finalize(values: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
            result = {}
            for stage, value in values.items():
                elapsed = round(value["total_ms"], 2)
                result[stage] = {
                    "calls": value["calls"],
                    "total_ms": elapsed,
                    "share_percent": round(elapsed * 100 / total_ms, 2) if total_ms else 0.0,
                    "max_depth": value["max_depth"],
                }
            return result

        # The shallowest events are the non-overlapping module buckets. A request
        # wrapper is optional, so the base depth is discovered from the trace.
        base_depth = min((event.depth for event in self.events), default=0)
        direct = [event for event in self.events if event.depth == base_depth]
        direct_by_stage: dict[str, dict[str, Any]] = {}
        for event in direct:
            stage = direct_by_stage.setdefault(
                event.stage,
                {"calls": 0, "total_ms": 0.0, "max_depth": event.depth},
            )
            stage["calls"] += 1
            stage["total_ms"] += event.elapsed_ms
            stage["max_depth"] = max(stage["max_depth"], event.depth)

        return {
            "available": bool(self.events),
            "total_ms": round(total_ms, 2),
            "event_count": len(self.events),
            "by_stage": finalize(by_stage),
            "top_level": finalize(direct_by_stage),
        }


@dataclass
class ModelUsageCollector:
    events: list[ModelUsageEvent] = field(default_factory=list)

    def record(self, event: ModelUsageEvent) -> None:
        self.events.append(event)

    def summary(self) -> dict[str, Any]:
        reported = [event for event in self.events if event.reported]
        by_stage: dict[str, dict[str, Any]] = {}
        for event in self.events:
            stage = by_stage.setdefault(
                event.stage,
                {
                    "calls": 0,
                    "reported_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            )
            stage["calls"] += 1
            if event.reported:
                stage["reported_calls"] += 1
                stage["input_tokens"] += event.input_tokens or 0
                stage["output_tokens"] += event.output_tokens or 0
                stage["total_tokens"] += _resolved_total(event)

        return {
            "available": bool(reported),
            "calls": len(self.events),
            "reported_calls": len(reported),
            "unreported_calls": len(self.events) - len(reported),
            "input_tokens": sum(event.input_tokens or 0 for event in reported),
            "output_tokens": sum(event.output_tokens or 0 for event in reported),
            "total_tokens": sum(_resolved_total(event) for event in reported),
            "by_stage": by_stage,
        }


_active_collector: ContextVar[ModelUsageCollector | None] = ContextVar(
    "active_model_usage_collector",
    default=None,
)
_active_stage: ContextVar[str] = ContextVar("active_model_usage_stage", default="unspecified")
_active_timing_collector: ContextVar[TimingCollector | None] = ContextVar(
    "active_timing_collector",
    default=None,
)
_active_timing_stack: ContextVar[tuple[str, ...]] = ContextVar(
    "active_timing_stack",
    default=(),
)


@contextmanager
def collect_model_usage() -> Iterator[ModelUsageCollector]:
    collector = ModelUsageCollector()
    token = _active_collector.set(collector)
    try:
        yield collector
    finally:
        _active_collector.reset(token)


@contextmanager
def model_usage_stage(stage: str) -> Iterator[None]:
    token = _active_stage.set(stage)
    try:
        yield
    finally:
        _active_stage.reset(token)


@contextmanager
def collect_timing() -> Iterator[TimingCollector]:
    collector = TimingCollector()
    collector_token = _active_timing_collector.set(collector)
    stack_token = _active_timing_stack.set(())
    try:
        yield collector
    finally:
        _active_timing_stack.reset(stack_token)
        _active_timing_collector.reset(collector_token)


@contextmanager
def timed_stage(stage: str) -> Iterator[None]:
    collector = _active_timing_collector.get()
    if collector is None:
        yield
        return

    stack = _active_timing_stack.get()
    stack_token = _active_timing_stack.set((*stack, stage))
    started_at = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - started_at) * 1_000
        _active_timing_stack.reset(stack_token)
        collector.record(TimingEvent(
            stage=stage,
            elapsed_ms=elapsed_ms,
            depth=len(stack),
            parent=stack[-1] if stack else None,
        ))


def record_model_usage(
    *,
    operation: str,
    provider: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
) -> None:
    collector = _active_collector.get()
    if collector is None:
        return
    collector.record(ModelUsageEvent(
        stage=_active_stage.get(),
        operation=operation,
        provider=provider,
        model=model,
        input_tokens=_optional_nonnegative_int(input_tokens),
        output_tokens=_optional_nonnegative_int(output_tokens),
        total_tokens=_optional_nonnegative_int(total_tokens),
    ))


F = TypeVar("F", bound=Callable[..., dict[str, Any]])


def capture_request_metrics(operation: F) -> F:
    @wraps(operation)
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        started_at = perf_counter()
        with collect_model_usage() as collector:
            result = operation(*args, **kwargs)
        result["response_time_ms"] = round((perf_counter() - started_at) * 1_000, 2)
        result["token_usage"] = collector.summary()
        return result

    return wrapped  # type: ignore[return-value]


def _resolved_total(event: ModelUsageEvent) -> int:
    if event.total_tokens is not None:
        return event.total_tokens
    return (event.input_tokens or 0) + (event.output_tokens or 0)


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, resolved)
