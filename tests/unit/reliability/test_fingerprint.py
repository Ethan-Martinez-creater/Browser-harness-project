"""Unit tests for deterministic observation fingerprints."""

from web_harness.core.models import Observation
from web_harness.env.fake import make_fake_observation
from web_harness.reliability.fingerprint import (
    compute_fingerprint,
    fingerprint_of,
    transition_signature,
)


def test_fingerprint_stable_for_identical_content():
    a = make_fake_observation(url="http://x/", goal="g")
    b = make_fake_observation(url="http://x/", goal="g")
    assert fingerprint_of(a) == fingerprint_of(b)


def test_fingerprint_changes_with_axtree():
    a = make_fake_observation(url="http://x/")
    b = a.model_copy(deep=True)
    b.axtree = "[1] button 'Changed'"
    assert fingerprint_of(a) != fingerprint_of(b)


def test_fingerprint_ignores_volatile_fields():
    a = make_fake_observation(url="http://x/")
    b = a.model_copy(deep=True)
    b.last_action = "click(bid='1')"
    b.last_action_error = "TimeoutError"
    b.elapsed_time_s = 99.5
    b.screenshot_path = "runs/x/artifacts/shot.png"
    assert fingerprint_of(a) == fingerprint_of(b)


def test_fingerprint_covers_url_and_open_pages():
    a = make_fake_observation(url="http://x/page1")
    b = make_fake_observation(url="http://x/page2")
    assert fingerprint_of(a) != fingerprint_of(b)

    c = make_fake_observation(url="http://x/page1")
    c.open_pages = ["http://x/page1", "http://x/extra"]
    assert fingerprint_of(a) != fingerprint_of(c)


def test_fingerprint_preserves_tab_order():
    """[A,B] and [B,A] must differ: tab order is executable state (R2)."""
    ab = make_fake_observation(url="http://x/a")
    ab.open_pages = ["http://x/a", "http://x/b"]
    ba = make_fake_observation(url="http://x/a")
    ba.open_pages = ["http://x/b", "http://x/a"]
    assert fingerprint_of(ab) != fingerprint_of(ba)


def test_fingerprint_structure():
    fp = compute_fingerprint(make_fake_observation(url="http://x/"))
    assert fp.url_hash and fp.content_hash and fp.combined_hash
    assert fp.url_hash != fp.content_hash


def test_transition_signature_deterministic_and_sensitive():
    s1 = transition_signature("fpA", "click(bid='1')", "fpB")
    s2 = transition_signature("fpA", "click(bid='1')", "fpB")
    assert s1 == s2
    assert transition_signature("fpA", "click(bid='2')", "fpB") != s1
    assert transition_signature("fpC", "click(bid='1')", "fpB") != s1
    assert transition_signature("fpA", "click(bid='1')", "fpC") != s1


def test_fingerprint_handles_empty_observation():
    empty = Observation(goal="", url="")
    assert fingerprint_of(empty)
