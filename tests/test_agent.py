# -*- coding: utf-8 -*-
"""Agent 硬边界与恢复能力的回归测试。"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.agent import (Agent, State, StepLimitExceeded, ToolRegistry,  # noqa: E402
                       WallTimeout, load_state)
from src.llm import RulePolicy  # noqa: E402


def build_registry():
    reg = ToolRegistry()

    @reg.tool("ping")
    def ping(value=1):
        return {"value": value}

    @reg.tool("finish", terminal=True)
    def finish(reason="done"):
        return {"reason": reason}

    @reg.tool("escalate", terminal=True)
    def escalate(reason):
        return {"escalated": True, "reason": reason}

    @reg.tool("boom")
    def boom():
        raise RuntimeError("工具故障")

    return reg


class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        self.registry = build_registry()

    def test_happy_path(self):
        agent = Agent(RulePolicy([("ping", {"value": 7}), ("finish", {})]),
                      self.registry)
        state = agent.run("任务")
        self.assertTrue(state.done)
        self.assertEqual(state.step, 2)
        self.assertEqual(state.context["last_observation"], {"reason": "done"})

    def test_step_limit(self):
        agent = Agent(RulePolicy([("ping", {})] * 50), self.registry, max_steps=3)
        with self.assertRaises(StepLimitExceeded):
            agent.run("死循环任务")

    def test_tool_error_falls_back_to_escalate(self):
        agent = Agent(RulePolicy([("boom", {})]), self.registry)
        state = agent.run("故障任务")
        self.assertTrue(state.done)
        self.assertEqual(state.trace[0]["tool"], "boom")
        self.assertFalse(state.trace[0]["ok"])
        self.assertEqual(state.trace[1]["tool"], "escalate")

    def test_unknown_tool_is_contained(self):
        class RoguePolicy:
            def __init__(self):
                self.calls = 0

            def decide(self, state, available):
                self.calls += 1
                if self.calls == 1:
                    return "not_registered", {}
                return "escalate", {"reason": "white-list fallback"}

        agent = Agent(RoguePolicy(), self.registry, max_steps=3)
        state = agent.run("越权任务")
        self.assertFalse(state.trace[0]["ok"])
        self.assertIn("not_registered", state.trace[0]["error"])
        self.assertEqual(state.trace[-1]["tool"], "escalate")
        self.assertTrue(state.done)

    def test_wall_timeout(self):
        ticks = [0.0, 10.0]
        clock = lambda: ticks.pop(0) if len(ticks) > 1 else 10.0  # noqa: E731
        agent = Agent(RulePolicy([("ping", {})]), self.registry,
                      wall_timeout=1.0, clock=clock)
        with self.assertRaises(WallTimeout):
            agent.run("超时任务")

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            Agent(RulePolicy([]), self.registry, max_steps=0)
        with self.assertRaises(ValueError):
            Agent(RulePolicy([]), self.registry, wall_timeout=0)


class TestCheckpoint(unittest.TestCase):
    def test_resume_from_file(self):
        registry = build_registry()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            agent = Agent(RulePolicy([("ping", {}), ("finish", {})]),
                          registry, checkpoint_path=path)
            partial = State(task="续跑", step=1, context={"last_observation": {"value": 1}})
            partial.trace.append({"step": 1, "tool": "ping", "ok": True})
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(partial.to_dict(), fh, ensure_ascii=False)
            resumed = load_state(path)
            self.assertEqual(resumed.step, 1)
            state = agent.run("续跑", state=resumed)
            self.assertTrue(state.done)
            self.assertEqual(state.step, 2)

    def test_state_roundtrip(self):
        state = State(task="t", step=3, done=True, context={"k": "v"},
                      trace=[{"step": 1}])
        self.assertEqual(State.from_dict(state.to_dict()), state)


if __name__ == "__main__":
    unittest.main()
