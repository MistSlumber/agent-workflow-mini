# -*- coding: utf-8 -*-
"""最小 Agent 状态机。

工程约束优先于"智能"：最大步数、总时长预算、工具白名单、逐工具异常隔离、
每步持久化检查点。任何策略（包括 LLM）都只能在这些边界内做决策。

这不是 MCP 协议实现，也不是 LangGraph 的替代品：它把状态、决策与工具
三者拆开，用来讲清 Agent 循环里最容易出事的那几个约束。
"""
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional


class AgentError(Exception):
    pass


class StepLimitExceeded(AgentError):
    pass


class WallTimeout(AgentError):
    pass


class UnknownTool(AgentError):
    pass


@dataclass
class State:
    task: str
    step: int = 0
    done: bool = False
    context: Dict[str, Any] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "State":
        return cls(task=data["task"], step=int(data.get("step", 0)),
                   done=bool(data.get("done", False)),
                   context=dict(data.get("context", {})),
                   trace=list(data.get("trace", [])))


class Tool:
    def __init__(self, name: str, fn: Callable, description: str = "",
                 terminal: bool = False):
        if not callable(fn):
            raise TypeError("工具必须是可调用对象")
        self.name = name
        self.fn = fn
        self.description = description
        self.terminal = terminal

    def __call__(self, **kwargs):
        return self.fn(**kwargs)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def tool(self, name: str, **kwargs):
        def decorator(fn):
            return self.register(Tool(name, fn, **kwargs))
        return decorator

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise UnknownTool(name)
        return self._tools[name]

    def names(self) -> List[str]:
        return sorted(self._tools)


class Agent:
    def __init__(self, policy, registry: ToolRegistry, max_steps: int = 8,
                 wall_timeout: float = 5.0,
                 checkpoint_path: Optional[str] = None,
                 clock: Callable[[], float] = time.monotonic):
        if max_steps < 1:
            raise ValueError("max_steps 必须 >= 1")
        if wall_timeout <= 0:
            raise ValueError("wall_timeout 必须 > 0")
        self.policy = policy
        self.registry = registry
        self.max_steps = max_steps
        self.wall_timeout = wall_timeout
        self.checkpoint_path = checkpoint_path
        self.clock = clock

    def run(self, task: str, state: Optional[State] = None) -> State:
        state = state or State(task=task)
        started = self.clock()
        self._save(state)
        while not state.done:
            if state.step >= self.max_steps:
                raise StepLimitExceeded("达到最大步数 %d" % self.max_steps)
            if self.clock() - started > self.wall_timeout:
                raise WallTimeout("超过总时长预算 %.2fs" % self.wall_timeout)

            name, args = self.policy.decide(state, self.registry.names())
            ok, error, observation, terminal = True, "", None, False
            try:
                tool = self.registry.get(name)
                terminal = tool.terminal
                observation = tool(**args)
                state.context["last_observation"] = observation
                state.context.pop("last_error", None)
            except Exception as exc:
                ok, error = False, str(exc)
                state.context["last_error"] = error

            state.step += 1
            state.trace.append({
                "step": state.step, "tool": name, "args": args,
                "ok": ok, "error": error, "observation": observation,
            })
            if ok and terminal:
                state.done = True
            self._save(state)
        return state

    def _save(self, state: State) -> None:
        if not self.checkpoint_path:
            return
        directory = os.path.dirname(self.checkpoint_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.checkpoint_path, "w", encoding="utf-8") as fh:
            json.dump(state.to_dict(), fh, ensure_ascii=False, indent=2)


def load_state(path: str) -> State:
    """从检查点恢复状态，配合 Agent.run(task, state=...) 继续未完成的任务。"""
    with open(path, encoding="utf-8") as fh:
        return State.from_dict(json.load(fh))
