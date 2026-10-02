"""Regression: kanban output must coerce int-or-ISO-string timestamps.

Bug class (2026-09-23): the kanban output layer assumed an int epoch, so a
finish time written as an ISO-8601 string (``hermes kanban show`` comments,
``hermes kanban runs``) raised ``TypeError`` inside ``time.localtime``. The
fix extracts a pure ``_coerce_epoch``; the same coercion guards the duration
arithmetic in ``kanban.py`` so a string timestamp cannot become a ``TypeError``
there either.

No change detectors, no reading the source text — these assert behaviour.
"""
from __future__ import annotations

import time
from datetime import datetime

import pytest

from hermes_cli.kanban_output import _coerce_epoch, _fmt_ts

# The real ISO-8601 rows in kanban.db were written in the host's LOCAL time: a
# task's text comments and its epoch task_events rows both render as "10:07" in
# local time. So a NAIVE ISO string must be interpreted as local, not UTC. The
# repo suite pins TZ=UTC (tests/conftest.py), which would hide that difference —
# these tests therefore pin the zone explicitly instead of inheriting the host's.
_LOCAL_TZ = "Asia/Jerusalem"


@pytest.fixture
def local_tz(monkeypatch):
    monkeypatch.setenv("TZ", _LOCAL_TZ)
    time.tzset()
    yield _LOCAL_TZ
    time.tzset()


def test_int_epoch_formats():
    epoch = 1789214427
    assert _coerce_epoch(epoch) == epoch
    assert _fmt_ts(epoch) == time.strftime("%Y-%m-%d %H:%M", time.localtime(epoch))


def test_iso_strings_coerce_to_the_same_instant_as_their_epoch(local_tz):
    # Behaviour contract: the string form and the epoch int describe ONE
    # instant, so they must format identically. The pairs below are real
    # kanban.db task_comments.created_at values and the epoch of the same
    # instant in IDT (+3), which is how the producer wrote them.
    for iso, epoch in [
        ("2026-09-23T10:07:10", 1790147230),
        ("2026-09-23 10:07:10", 1790147230),
        ("2026-09-23T10:07:37", 1790147257),
        ("2026-09-23T10:14:14", 1790147654),
        ("2026-09-23T10:54", 1790150040),
    ]:
        assert _coerce_epoch(iso) is not None
        assert _fmt_ts(iso) == _fmt_ts(int(epoch)), (iso, _fmt_ts(iso), _fmt_ts(int(epoch)))
        assert _coerce_epoch(iso) == epoch, (iso, _coerce_epoch(iso), epoch)


def test_naive_iso_is_local_time_not_utc(local_tz):
    # Pinning this is the point of the whole bug class: reading the naive string
    # as UTC would shift every recovered timestamp by the host's offset.
    assert _coerce_epoch("2026-09-23T10:07:10") == int(
        datetime(2026, 9, 23, 10, 7, 10).timestamp())


def test_iso_string_with_offset_is_preserved(local_tz):
    # An explicit offset must be honoured verbatim, not re-anchored to local.
    assert _coerce_epoch("2026-09-23T10:07:10+00:00") == 1790147230 + 3 * 3600


def test_numeric_string_is_an_epoch():
    assert _coerce_epoch("1789214427") == 1789214427
    assert _coerce_epoch("1789214427.5") == 1789214427


def test_blank_and_absent_render_empty():
    for v in (None, "", "   "):
        assert _coerce_epoch(v) is None
        assert _fmt_ts(v) == ""


def test_zero_matches_legacy_empty():
    assert _fmt_ts(0) == ""


def test_garbage_and_wrong_types_do_not_raise():
    for v in ("not-a-time", "2026-13-45T99:99:99", True, object()):
        assert _fmt_ts(v) == "", v


def test_duration_math_survives_iso_timestamps():
    # The same coercion is reused for run-duration arithmetic in kanban.py,
    # so an ISO started_at/ended_at pair must subtract instead of raising.
    started, ended = _coerce_epoch("2026-09-23T10:07:10"), _coerce_epoch("2026-09-23T10:15:10")
    assert started is not None and ended is not None
    assert max(0, ended - started) == 480


def test_kanban_py_reuses_the_output_coercion():
    # Guard the wiring: kanban.py must USE the shared helper, not re-derive one.
    # Identity (`is`) is unusable here — another test reloads these modules, so
    # kanban may hold a function object from a fresh import of the same source.
    # Definition site is the stable signal.
    from hermes_cli import kanban

    assert kanban._coerce_epoch.__module__ == "hermes_cli.kanban_output"
    assert kanban._coerce_epoch.__qualname__ == "_coerce_epoch"
    # Same behaviour on a real row from the bug report.
    assert kanban._coerce_epoch("2026-09-23T10:07:10") == _coerce_epoch("2026-09-23T10:07:10")