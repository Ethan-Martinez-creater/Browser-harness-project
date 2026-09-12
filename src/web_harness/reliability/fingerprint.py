"""Deterministic observation fingerprints.

A fingerprint answers exactly one question: did the observable page state
change between two points in time? Only stable fields participate (URL,
axtree, optional DOM, open page URLs). Volatile fields (elapsed time, last
action, last action error, screenshot/artifact paths) are excluded by design.
No embeddings, no LLM semantic diffs — deterministic normalization only.
"""

from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel

from web_harness.core.models import Observation


class ObservationFingerprint(BaseModel):
    url_hash: str
    content_hash: str
    combined_hash: str


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(text: str | None) -> str:
    """Deterministic normalization: collapse whitespace noise, keep content."""
    if not text:
        return ""
    return "\n".join(line.rstrip() for line in text.strip().replace("\r\n", "\n").split("\n"))


def compute_fingerprint(observation: Observation) -> ObservationFingerprint:
    url_hash = _sha256(_normalize(observation.url))
    # open_pages keep environment order: tab order is executable state
    # (tab_focus(index) semantics), so [A,B] and [B,A] must differ.
    content_parts = [
        _normalize(observation.axtree),
        _normalize(observation.dom),
        *(_normalize(p) for p in observation.open_pages),
    ]
    content_hash = _sha256("\n\x00\n".join(content_parts))
    combined_hash = _sha256(f"{url_hash}\x00{content_hash}")
    return ObservationFingerprint(
        url_hash=url_hash, content_hash=content_hash, combined_hash=combined_hash
    )


def fingerprint_of(observation: Observation) -> str:
    """Short combined hash (the value stored in state and traces)."""
    return compute_fingerprint(observation).combined_hash[:16]


def transition_signature(
    pre_state_hash: str, action: str, post_state_hash: str
) -> str:
    """Deterministic signature of one state -> action -> state transition."""
    normalized_action = _normalize(action)
    return _sha256(f"{pre_state_hash}\x00{normalized_action}\x00{post_state_hash}")[:16]


def normalize_error_signature(text: str | None, limit: int = 120) -> str:
    """Deterministic, normalized failure signature from an error text.

    Lowercases, collapses whitespace, strips volatile numbers (ports,
    timeouts, bids) so the same error class maps to the same signature.
    Raw exception strings are never used as signatures directly.
    """
    if not text:
        return "none"
    cleaned = " ".join(text.lower().split())
    cleaned = re.sub(r"\d+", "<n>", cleaned)
    return cleaned[:limit]


def extract_action_type(action: str | None) -> str:
    """Stable action function name from an action string.

    `click(bid='13')` -> `click`; unknown/malformed actions normalize to
    `unknown`. Dynamic arguments (bids, values) never enter the signature.
    """
    if not action:
        return "unknown"
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", action)
    return m.group(1).lower() if m else "unknown"
