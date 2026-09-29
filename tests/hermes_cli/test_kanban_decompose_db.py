"""Tests for decompose_triage_task — the DB-layer atomic fan-out
from the triage column. LLM-free by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import decompose_triage_task
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_triage(conn, title="rough idea", body=None, assignee=None, tenant=None):
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        tenant=tenant,
        triage=True,
    )


def test_decompose_creates_children_and_promotes_root(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="ship a feature")
        assert kb.get_task(conn, tid).status == "triage"

    children = [
        {"title": "research", "body": "look at prior art", "assignee": "researcher", "parents": []},
        {"title": "build it", "body": "write code", "assignee": "engineer", "parents": [0]},
    ]
    with kbc.connect() as conn:
        child_ids = decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    assert len(child_ids) == 2

    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, child_ids[0])
        c1 = kb.get_task(conn, child_ids[1])

    # Root flipped to todo with orchestrator assignee, gated by children.
    assert root.status == "todo"
    assert root.assignee == "orchestrator"
    # First child has no internal parents → ready on recompute_ready.
    assert c0.status == "ready"
    assert c0.assignee == "researcher"
    # Second child has parents=[0] → stays in todo until c0 completes.
    assert c1.status == "todo"
    assert c1.assignee == "engineer"


def test_decompose_records_audit_comment_and_event(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn)
        child_ids = decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "task A", "assignee": "researcher"}],
            author="alice",
        )
    assert child_ids is not None

    with kbc.connect() as conn:
        comments = kb.list_comments(conn, tid)
        events = kb.list_events(conn, tid)

    assert comments
    assert any(ev.kind == "decomposed" for ev in events)






def test_decompose_children_carry_lineage_header(kanban_home):
    """2026-09-28 paper-trail audit fix: every decomposed child body must open
    with a [Context] lineage header naming the root card, its title, the
    creator session id, and the authoring profile — even when the fan-out
    declares no sibling parents edges (the 54%-orphaned case)."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="root spec",
            session_id="20260928_101915_6aeb00",
            triage=True,
        )
    children = [
        {"title": "orphan-ish child", "body": "do the work", "assignee": "researcher"},
        {"title": "no-body child", "assignee": "engineer"},  # body-less input
    ]
    with kbc.connect() as conn:
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orch",
            children=children, author="decomposer",
        )
    assert child_ids is not None
    with kbc.connect() as conn:
        c0, c1 = (kb.get_task(conn, cid) for cid in child_ids)

    for child in (c0, c1):
        assert child.body.startswith("[Context] decomposed from "), child.body[:80]
        assert f'decomposed from {tid} "root spec"' in child.body
        assert "session 20260928_101915_6aeb00" in child.body
        assert "author decomposer" in child.body
    # Original work spec survives the header.
    assert "do the work" in c0.body
    # A body-less child input yields header-only, not None/empty.
    assert c1.body and c1.body.strip() != ""


def test_decompose_lineage_header_omits_missing_optional_fields(kanban_home):
    """No session stamp and no author → header still forms, without empty
    'session None' / 'author None' fragments."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="bare root")
    with kbc.connect() as conn:
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orch",
            children=[{"title": "solo", "body": "work"}], author=None,
        )
    assert child_ids is not None
    with kbc.connect() as conn:
        body = kb.get_task(conn, child_ids[0]).body
    assert body.startswith("[Context] decomposed from ")
    assert "session" not in body
    assert "author" not in body
    assert "work" in body
