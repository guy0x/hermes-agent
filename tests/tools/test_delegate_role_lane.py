"""Role→lane routing over delegation.task_model_map.

Contract under test:

  * At ``_delegate_depth == 0`` a task whose normalised role matches a
    ``delegation.task_model_map`` lane is built on the lane's {provider, model}
    PAIR instead of the configured pin.
  * At depth >= 1 the lane is IGNORED — children inherit the pin.
  * An unknown role falls back to the pin.
  * ``credentials_cfg`` (internal callers, e.g. /review) always wins over roles.
  * THE SAFETY LINE: a lane entry carrying ``api_key`` or ``base_url`` must be
    REFUSED — a lane moves as a {provider, model} pair and nothing else. Key
    resolution stays inside ``_resolve_delegation_credentials``.

These are written RED against the pre-T1 code and must go GREEN after the
wiring lands in ``_build_children`` (tools/delegate_tool.py).
"""
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import _build_children


def _make_parent(depth=0):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "sk-test-parent"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


# What the configured pin resolves to for these tests (creds as _build_children sees them).
PIN = {"provider": "cheaper-inference", "model": "deepseek-v4-flash-0731",
       "base_url": None, "api_key": "sk-test-parent", "api_mode": "chat_completions"}

LANE_CFG = {
    "task_model_map": {
        "hard_reasoning": {"provider": "openrouter-master", "model": "z-ai/glm-5.3-flash"},
        "code": {"provider": "cheaper-inference", "model": "deepseek-v4-flash-0731"},
    },
}


def _run_build_children(parent, tasks, *, cfg=None, creds=None, allow_role_lanes=True):
    """Call _build_children with child construction stubbed; capture per-child kwargs.

    allow_role_lanes=True by default mirrors a normal dispatch (production computes
    ``allow_role_lanes = credentials_cfg is None`` at the call site — internal
    callers get False, which is what the precedence test exercises).
    """
    captured = []

    def fake_build(**kwargs):
        captured.append({"model": kwargs.get("model"),
                         "override_provider": kwargs.get("override_provider"),
                         "role": kwargs.get("role")})
        child = MagicMock()
        child.session_id = f"sa-fake-{len(captured)}"
        return child

    with patch("tools.delegate_tool._build_child_preserving_parent_tools",
               side_effect=lambda **kw: fake_build(**kw)):
        _build_children(
            task_list=tasks, task_schemas=[None] * len(tasks),
            creds=creds or dict(PIN), top_role="leaf", max_iterations=10,
            parent_agent=parent, routing_cfg=cfg or dict(LANE_CFG),
            live_deleg_id=None, live_writers=[],
            allow_role_lanes=allow_role_lanes,
        )
    return captured


class TestRoleLaneRouting(unittest.TestCase):
    def test_role_routes_to_lane_at_depth_0(self):
        """hard_reasoning at root depth -> the LANE pair, not the pin."""
        parent = _make_parent(depth=0)
        captured = _run_build_children(
            parent, [{"goal": "think hard", "role": "hard_reasoning"}])

        self.assertEqual(captured[0]["override_provider"], "openrouter-master")
        self.assertEqual(captured[0]["model"], "z-ai/glm-5.3-flash")

    def test_unknown_role_falls_back_to_pin(self):
        parent = _make_parent(depth=0)
        captured = _run_build_children(
            parent, [{"goal": "misc", "role": "nonexistent_lane"}])

        self.assertEqual(captured[0]["override_provider"], PIN["provider"])
        self.assertEqual(captured[0]["model"], PIN["model"])

    def test_no_role_gets_pin(self):
        parent = _make_parent(depth=0)
        captured = _run_build_children(parent, [{"goal": "plain task"}])

        self.assertEqual(captured[0]["override_provider"], PIN["provider"])
        self.assertEqual(captured[0]["model"], PIN["model"])

    def test_depth_1_child_ignores_lane(self):
        """A first-level child delegating further inherits the pin, never a lane."""
        parent = _make_parent(depth=1)
        captured = _run_build_children(
            parent, [{"goal": "nested", "role": "hard_reasoning"}])

        self.assertEqual(captured[0]["override_provider"], PIN["provider"])
        self.assertEqual(captured[0]["model"], PIN["model"])

    def test_credentials_cfg_wins_over_role(self):
        """Internal per-call routes (e.g. /review) outrank role lanes.

        Production passes allow_role_lanes=False when credentials_cfg owns the
        route — the flag IS the precedence mechanism. Assert both layers: the
        flag disables lanes, and pin creds land even for a lane-named role.
        """
        parent = _make_parent(depth=0)
        internal = {"provider": "zai", "model": "glm-5.3-flash",
                    "base_url": None, "api_key": "sk-internal", "api_mode": "chat_completions"}
        captured = _run_build_children(
            parent, [{"goal": "review", "role": "hard_reasoning"}],
            creds=dict(internal), allow_role_lanes=False)

        self.assertEqual(captured[0]["override_provider"], "zai")
        self.assertEqual(captured[0]["model"], "glm-5.3-flash")


class TestLaneSafetyLine(unittest.TestCase):
    """THE refusal test: a lane carrying key/url material must be rejected."""

    def test_lane_with_api_key_is_refused(self):
        cfg = {"task_model_map": {
            "evil": {"provider": "p", "model": "m", "api_key": "sk-leak"}}}
        parent = _make_parent(depth=0)
        with self.assertRaises(ValueError) as cm:
            _run_build_children(parent, [{"goal": "x", "role": "evil"}], cfg=cfg)
        self.assertIn("api_key", str(cm.exception))

    def test_lane_with_base_url_is_refused(self):
        cfg = {"task_model_map": {
            "evil": {"provider": "p", "model": "m", "base_url": "https://evil.example"}}}
        parent = _make_parent(depth=0)
        with self.assertRaises(ValueError) as cm:
            _run_build_children(parent, [{"goal": "x", "role": "evil"}], cfg=cfg)
        self.assertIn("base_url", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
