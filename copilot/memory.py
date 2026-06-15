"""The memory seam.

This is the ONE interface the rest of the pipeline is allowed to know about.
`context_builder` depends only on `MemoryProvider` — never on any concrete
implementation. The MVP ships `NullMemoryProvider` (recall -> [], remember ->
no-op). A local SQLite-backed provider can be dropped in later behind THIS
EXACT interface with zero changes to the pipeline.

    provider = NullMemoryProvider()        # today
    # provider = SqliteMemoryProvider(...)  # later, your own lib — same ABC
    builder = ContextBuilder(provider)

Do not add methods here without a deliberate decision: every method becomes a
contract that future providers must honour.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class MemoryProvider(ABC):
    @abstractmethod
    def recall(self, query: str, k: int = 5) -> list[str]:
        """Return up to `k` relevant snippets for `query` (most relevant first)."""
        ...

    @abstractmethod
    def remember(self, text: str, meta: dict | None = None) -> None:
        """Persist a snippet (with optional metadata) for later recall."""
        ...


class NullMemoryProvider(MemoryProvider):
    """No-op provider. Returns nothing, stores nothing. The MVP default."""

    def recall(self, query: str, k: int = 5) -> list[str]:
        return []

    def remember(self, text: str, meta: dict | None = None) -> None:
        return None
