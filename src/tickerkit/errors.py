"""Exceptions used across tickerkit.

Nothing here is ever allowed to escape the orchestrator; failures are converted into
:class:`tickerkit.models.ProviderError` records.
"""

from __future__ import annotations


class TickerKitError(Exception):
    """Base class for every tickerkit error."""


class ResolutionError(TickerKitError):
    """The requested ticker could not be resolved to a concrete asset."""


class ProviderError(TickerKitError):
    """A provider could not satisfy a capability."""

    def __init__(self, provider: str, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.detail = detail


class UnsupportedCapability(TickerKitError):
    """The provider or exchange does not implement the requested capability."""


class RateLimitExceeded(TickerKitError):
    """A provider rate limit was hit and could not be satisfied within the retries.

    Carries the structured detail a caller needs to react rather than parse a string:
    which source throttled, how long to wait, the advertised budget, and a concrete
    remediation hint. Suppressed calls (short-circuited by the circuit breaker without
    touching the network) set ``suppressed`` so a caller can tell "we were blocked"
    apart from "we were told no".
    """

    def __init__(
        self,
        source: str,
        message: str,
        *,
        retry_after: float | None = None,
        limit: int | None = None,
        remaining: int | None = None,
        reset_at: float | None = None,
        hint: str | None = None,
        suppressed: bool = False,
    ) -> None:
        super().__init__(message)
        self.source = source
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining
        self.reset_at = reset_at
        self.hint = hint
        self.suppressed = suppressed

    def __str__(self) -> str:
        base = super().__str__()
        parts = [base]
        if self.limit is not None:
            parts.append(f"limit {self.limit}/window")
        if self.retry_after is not None:
            parts.append(f"retry after {self.retry_after:.0f}s")
        return " (".join([parts[0], ", ".join(parts[1:]) + ")"]) if len(parts) > 1 else base


class ConfigurationError(TickerKitError):
    """Invalid configuration supplied by the caller."""


__all__ = [
    "ConfigurationError",
    "ProviderError",
    "RateLimitExceeded",
    "ResolutionError",
    "TickerKitError",
    "UnsupportedCapability",
]