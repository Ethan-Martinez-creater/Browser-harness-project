"""Observation normalization.

Converts a raw environment observation (BrowserGym dict) into the harness
`Observation` model. In Phase 0 the model input is text-only: AXTree first,
DOM only when explicitly enabled, with deterministic character truncation.
No LLM-based compression is allowed here.
"""

from __future__ import annotations

import re

from web_harness.core.models import Observation


def truncate_text(text: str, limit: int) -> tuple[str, bool]:
    """Deterministically cut text to `limit` chars, keeping the head.

    Returns (text, was_truncated). The cut marker is appended only when
    truncation happened so the input stays deterministic for the same input.
    """
    if limit <= 0 or len(text) <= limit:
        return text, False
    marker = f"\n[... truncated {len(text) - limit} chars ...]"
    keep = max(0, limit - len(marker))
    return text[:keep] + marker, True


class ObservationNormalizer:
    """Builds harness Observations from adapter-specific raw observations."""

    def __init__(
        self,
        *,
        observation_char_limit: int = 30000,
        include_dom: bool = False,
    ):
        self.observation_char_limit = observation_char_limit
        self.include_dom = include_dom

    def from_axtree_parts(
        self,
        *,
        goal: str,
        url: str,
        axtree_text: str,
        open_pages: list[str],
        last_action: str | None = None,
        last_action_error: str | None = None,
        elapsed_time_s: float = 0.0,
        dom_text: str | None = None,
    ) -> Observation:
        """Build an Observation from already-extracted text parts.

        Environment adapters extract raw text (e.g. from BrowserGym objects);
        this method applies truncation policy only.
        """
        axtree, ax_truncated = truncate_text(axtree_text or "", self.observation_char_limit)
        dom: str | None = None
        if self.include_dom and dom_text:
            dom, dom_trunc = truncate_text(dom_text, self.observation_char_limit)
            ax_truncated = ax_truncated or dom_trunc
        return Observation(
            goal=goal or "",
            url=url or "",
            axtree=axtree if axtree.strip() else None,
            dom=dom,
            open_pages=list(open_pages or []),
            last_action=last_action,
            last_action_error=last_action_error,
            elapsed_time_s=float(elapsed_time_s or 0.0),
            truncated=ax_truncated,
        )


_BID_LINE = re.compile(r"\[([^\]\s]+)\]")


def clickable_bids(axtree: str | None) -> list[str]:
    """All bracketed bids present in a flattened axtree (deterministic order)."""
    if not axtree:
        return []
    seen: list[str] = []
    for m in _BID_LINE.finditer(axtree):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen
