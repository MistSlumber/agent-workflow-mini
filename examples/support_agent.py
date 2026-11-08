# -*- coding: utf-8 -*-
"""可运行的客服工单 Agent 示例：查询订单 -> 判断可否退款 -> 退款或转人工。

运行：python examples/support_agent.py
不需要 API key、不需要联网。执行后会在 results/ 留下检查点文件。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.agent import Agent, ToolRegistry  # noqa: E402
from src.llm import RulePolicy  # noqa: E402

registry = ToolRegistry()


@registry.tool("lookup_order", description="按订单号查询订单")
def lookup_order(order_id):
    orders = {
        "A-1001": {"order_id": order_id, "amount": 128.0, "days": 3, "status": "paid"},
        "B-2002": {"order_id": order_id, "amount": 899.0, "days": 42, "status": "paid"},
    }
    if order_id not in orders:
        raise KeyError("订单不存在: " + order_id)
    return orders[order_id]


@registry.tool("check_refund", description="按金额与天数判断是否可自动退款")
def check_refund(amount, days):
    return {"eligible": bool(amount < 500 and days <= 7), "amount": amount, "days": days}


@registry.tool("issue_refund", description="执行退款", terminal=True)
def issue_refund(order_id, amount):
    return {"refunded": True, "order_id": order_id, "amount": amount}


@registry.tool("escalate", description="转人工处理", terminal=True)
def escalate(reason):
    return {"escalated": True, "reason": reason}


def refund_decision(state):
    obs = state.context.get("last_observation") or {}
    if obs.get("eligible"):
        return "issue_refund", {"order_id": "A-1001", "amount": obs.get("amount", 0)}
    return "escalate", {"reason": "不满足自动退款条件"}


def build_agent(order_id, checkpoint_name):
    plan = [
        ("lookup_order", {"order_id": order_id}),
        (lambda s: ("check_refund", {
            "amount": (s.context.get("last_observation") or {}).get("amount", 0),
            "days": (s.context.get("last_observation") or {}).get("days", 999),
        })),
        refund_decision,
    ]
    checkpoint = os.path.join(ROOT, "results", checkpoint_name)
    return Agent(RulePolicy(plan), registry, max_steps=6, wall_timeout=5.0,
                 checkpoint_path=checkpoint)


def main():
    for order_id, name in [("A-1001", "checkpoint_auto.json"),
                           ("B-2002", "checkpoint_escalate.json"),
                           ("C-9999", "checkpoint_error.json")]:
        agent = build_agent(order_id, name)
        state = agent.run("处理订单 %s 的退款申请" % order_id)
        print("任务: %s" % order_id)
        for row in state.trace:
            print("  step %d | %-16s | ok=%-5s | %s" % (
                row["step"], row["tool"], row["ok"],
                row["observation"] if row["ok"] else row["error"]))
        print("  done=%s steps=%d\n" % (state.done, state.step))


if __name__ == "__main__":
    main()
