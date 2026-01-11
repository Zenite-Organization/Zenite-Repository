from typing import List, Dict, Any, Optional, TypedDict
import asyncio
from datetime import datetime

from clients.github.github_provider import GitHubProjectProvider
from ai.workflows.estimation_graph import run_estimation_flow
from config.settings import settings
from ai.core.llm_client import LLMClient
import json
from langgraph.graph import StateGraph, END

class SprintPlanningState(TypedDict):
    repo_full_name: str
    installation_id: int
    trigger_issue_node_id: str | None
    options: Dict[str, Any]

    backlog: List[dict]
    enriched: List[dict]
    priorities: List[dict]
    estimates: List[dict]

    capacity_hours: float
    selected: List[dict]

    sprint_title: str
    moved: List[dict]

async def load_backlog(state: SprintPlanningState):
    provider = state["provider"]
    backlog_label = state["options"].get("backlog_label")

    issues = await provider.list_backlog_issues(
        state["repo_full_name"],
        backlog_label
    )

    return {"backlog": issues}


def enrich_context(state: SprintPlanningState):
    enriched = []

    for c in state["backlog"]:
        enriched.append({
            **c,
            "labels": [l.get("name") for l in c.get("labels") or []]
        })

    return {"enriched": enriched}


def prioritize_tasks(state: SprintPlanningState):
    llm = LLMClient()

    prompt = (
        "Você é um Product Owner experiente.\n"
        "Retorne JSON no formato:\n"
        "[{ task_id, priority, reason, dependencies }]\n\n"
        "Tarefas:\n"
    )

    for t in state["enriched"]:
        prompt += f"- id: {t['node_id']} title: {t['title']} labels: {t['labels']}\n"

    resp = llm.send_prompt(prompt)

    try:
        priorities = json.loads(resp)
    except Exception:
        priorities = []

    return {"priorities": priorities}


def estimate_tasks(state: SprintPlanningState):
    cache = {}
    estimated = []

    for task in state["enriched"]:
        key = task.get("node_id")
        if key in cache:
            hours = cache[key]
        else:
            est_state = run_estimation_flow(task.get("body", ""))
            hours = float(
                est_state.get("final_estimation", {}).get("estimate_hours", 0)
            )
            cache[key] = hours

        estimated.append({**task, "estimate_hours": hours})

    return {"estimates": estimated}

async def resolve_capacity(state: SprintPlanningState):
    provider = state["provider"]

    try:
        cap = await provider.get_sprint_capacity(
            state["repo_full_name"],
            state["options"].get("trigger_label")
        )
    except Exception:
        cap = None

    if not cap:
        cap = settings.SPRINT_CAPACITY_HOURS

    return {"capacity_hours": cap}

def select_for_sprint(state: SprintPlanningState):
    prio_map = {
        p["task_id"]: p for p in state.get("priorities", [])
    }

    def score(task):
        p = prio_map.get(task.get("node_id"))
        if not p:
            return 1
        return {"alta": 3, "media": 2, "baixa": 1}.get(
            p.get("priority", "media"), 2
        )

    sorted_tasks = sorted(
        state["estimates"],
        key=lambda t: (-score(t), t.get("created_at") or "")
    )

    selected = []
    used = 0.0

    for t in sorted_tasks:
        if used + t["estimate_hours"] <= state["capacity_hours"]:
            selected.append(t)
            used += t["estimate_hours"]

    return {"selected": selected}


async def assign_iteration(state: SprintPlanningState):
    provider = state["provider"]

    sprint_title = (
        state["options"].get("sprint_title")
        or f"Sprint {datetime.utcnow().isoformat()}"
    )

    moved = []

    for t in state["selected"]:
        try:
            res = await provider.assign_iteration_by_name(
                t["node_id"],
                sprint_title
            )
            moved.append({
                "issue": t["number"],
                "estimate": t["estimate_hours"],
                "iteration": sprint_title
            })
        except Exception as e:
            moved.append({"issue": t["number"], "error": str(e)})

    return {
        "sprint_title": sprint_title,
        "moved": moved
    }




graph = StateGraph(SprintPlanningState)

graph.add_node("load_backlog", load_backlog)
graph.add_node("enrich_context", enrich_context)
graph.add_node("prioritize", prioritize_tasks)
graph.add_node("estimate", estimate_tasks)
graph.add_node("resolve_capacity", resolve_capacity)
graph.add_node("select", select_for_sprint)
graph.add_node("assign_iteration", assign_iteration)

graph.set_entry_point("load_backlog")

graph.add_edge("load_backlog", "enrich_context")
graph.add_edge("enrich_context", "prioritize")
graph.add_edge("prioritize", "estimate")
graph.add_edge("estimate", "resolve_capacity")
graph.add_edge("resolve_capacity", "select")
graph.add_edge("select", "assign_iteration")
graph.add_edge("assign_iteration", END)

planning_graph = graph.compile()
