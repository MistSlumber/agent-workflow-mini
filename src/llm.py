# -*- coding: utf-8 -*-
"""可插拔的决策策略。

默认 RulePolicy 完全离线，方便在没有密钥、没有网络的环境复现 Agent 行为。
OpenAICompatiblePolicy 只是接口示例：真正接入模型时，仍由 Agent 层的
最大步数 / 超时 / 工具白名单兜底，模型不能突破这些硬边界。
"""
import json
import os
import urllib.request


class Policy:
    def decide(self, state, available):
        """返回 (tool_name, args_dict)。available 是允许调用的工具名列表。"""
        raise NotImplementedError


def _resolve(item, state):
    if callable(item):
        return item(state)
    name, args = item
    return name, (dict(args(state)) if callable(args) else dict(args))


class RulePolicy(Policy):
    """按预定义计划推进；工具报错时自动转交 escalate。"""

    def __init__(self, plan, escalate_tool="escalate"):
        self.plan = list(plan)
        self.escalate_tool = escalate_tool

    def decide(self, state, available):
        last_error = state.context.get("last_error")
        if last_error and self.escalate_tool in available:
            return self.escalate_tool, {"reason": "tool error: " + str(last_error)}
        if state.step >= len(self.plan):
            if self.escalate_tool in available:
                return self.escalate_tool, {"reason": "plan exhausted"}
            raise RuntimeError("计划已耗尽且没有可用的兜底工具")
        name, args = _resolve(self.plan[state.step], state)
        if name not in available:
            if self.escalate_tool in available:
                return self.escalate_tool, {"reason": "unknown tool: " + name}
            raise RuntimeError("工具不在白名单: " + name)
        return name, args


class OpenAICompatiblePolicy(Policy):
    """调用 OpenAI 兼容端点产出 JSON 决策；任何失败都回退到 RulePolicy。"""

    def __init__(self, fallback, model=None, base_url=None, api_key=None, timeout=20):
        self.fallback = fallback
        self.model = model or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.timeout = timeout

    def decide(self, state, available):
        if not self.api_key:
            return self.fallback.decide(state, available)
        prompt = (
            "只输出 JSON，形如 {\"tool\": \"...\", \"args\": {...}}。"
            "可用工具：%s。当前任务：%s。当前上下文：%s"
            % (", ".join(available), state["task"] if isinstance(state, dict) else state.task,
               json.dumps(state["context"] if isinstance(state, dict) else state.context,
                          ensure_ascii=False)[:800])
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.api_key},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            decision = json.loads(content)
            name, args = decision["tool"], decision.get("args", {})
            if name in available and isinstance(args, dict):
                return name, args
        except Exception:
            pass
        return self.fallback.decide(state, available)
