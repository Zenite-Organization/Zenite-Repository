from typing import List, Dict, Any, Optional, TypedDict
import logging
from ai.workflows.estimation_graph import run_estimation_flow
from ai.core.llm_client import LLMClient
from langgraph.graph import StateGraph, END
from ai.dtos.issues_estimation_dto import IssueEstimationDTO
from ai.agents.prioritize_agent import run_task_prioritization_for_user

class SprintPlanningState(TypedDict):
    issue: Dict[str, Any]
    repo_full_name: str
    installation_id: int
    trigger_issue_node_id: str | None
    options: Dict[str, Any]

    backlog: List[IssueEstimationDTO]
    priorities: List[dict]
    estimates: List[dict]

    capacity_hours: float

    sprint_title: str
    moved: List[dict]
    
logger = logging.getLogger(__name__)


def prioritize_tasks(state: SprintPlanningState):

    llm = LLMClient()

    # Agrupa tarefas por usuário (assignee) a partir do backlog
    tasks_by_user = {}
    for t in state.get("backlog", []):
        # Suporta tanto DTO quanto dict
        if hasattr(t, "assignees"):
            assignees = getattr(t, "assignees", []) or []
            t_dict = t.model_dump() if hasattr(t, "model_dump") else dict(t)
        else:
            assignees = t.get("assignees", []) or []
            t_dict = t
        # Ignora tarefas sem assignee
        if not assignees:
            continue
        for user in assignees:
            tasks_by_user.setdefault(user, []).append(t_dict)

    # Contexto do sprint (issue principal)
    sprint_context = state.get("issue", {})

    all_prioritized = []
    for user, tasks in tasks_by_user.items():
        result = run_task_prioritization_for_user(user, tasks, llm, sprint_context=sprint_context)
        # result["tasks"] já está priorizado pelo agente
        for task in result.get("tasks", []):
            # Adiciona o campo user para rastreio
            task["user"] = user
            all_prioritized.append(task)
    logger.info("[Sprint Planning] tarefas priorizadas=%s", len(all_prioritized))
    return {"priorities": all_prioritized}


def estimate_tasks(state: SprintPlanningState):
    cache = {}
    estimated = []

    backlog = state.get("backlog", []) or []

    # Cria mapa rápido de possíveis ids -> item do backlog
    backlog_map: Dict[str, Any] = {}
    for item in backlog:
        item_dict = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        candidates = set()
        if item_dict.get("issue_number") is not None:
            candidates.add(str(item_dict["issue_number"]))
            candidates.add(f"ISSUE_{item_dict['issue_number']}")
        if item_dict.get("node_id"):
            candidates.add(str(item_dict.get("node_id")))
        if item_dict.get("id"):
            candidates.add(str(item_dict.get("id")))
        for c in candidates:
            backlog_map[c] = item

    # Para cada tarefa priorizada, encontra o backlog correspondente e estima
    for p in state.get("priorities", []):
        task_id = str(p.get("task_id"))
        orig = backlog_map.get(task_id)

        if not orig:
            # tentativa de correspondência flexível
            for k, v in backlog_map.items():
                if task_id.endswith(str(k)) or str(k).endswith(task_id) or task_id in str(k) or str(k) in task_id:
                    orig = v
                    break

        if not orig:
            continue

        item_dict = orig.model_dump() if hasattr(orig, "model_dump") else dict(orig)
        key = item_dict.get("issue_number")

        if key in cache:
            hours = cache[key]
        else:
            est_state = run_estimation_flow(orig)
            hours = float(
                est_state.get("final_estimation", {}).get("estimate_hours", 0)
            )
            cache[key] = hours

        item_dict["estimate_hours"] = hours
        item_dict["priority"] = p.get("priority")
        item_dict["priority_reason"] = p.get("reason")
        estimated.append(item_dict)

    return {"estimates": estimated}


def select_for_sprint(state: SprintPlanningState):

    PRIORITY_WEIGHT = {
        "alta": 3,
        "media": 2,
        "baixa": 1,
    }

    def score(task):
        return PRIORITY_WEIGHT.get(task.get("priority", "media"), 2)

    # Ordena: prioridade desc, depois menor esforço primeiro (melhor packing)
    sorted_tasks = sorted(
        state.get("estimates", []),
        key=lambda t: (-score(t), t.get("estimate_hours", 0))
    )

    selected = []
    used_hours = 0.0
    capacity = state.get("capacity_hours", 0)

    for t in sorted_tasks:
        hours = float(t.get("estimate_hours", 0))
        if used_hours + hours <= capacity:
            selected.append(t)
            used_hours += hours

    return {
        "selected": selected,
        "used_hours": used_hours,
        "capacity_hours": capacity,
        "overflow": [
            t for t in sorted_tasks if t not in selected
        ],
    }





graph = StateGraph(SprintPlanningState)



graph.add_node("prioritize", prioritize_tasks)
graph.add_node("estimate", estimate_tasks)
graph.add_node("select", select_for_sprint)

graph.set_entry_point("prioritize")
graph.add_edge("prioritize", "estimate")
graph.add_edge("estimate", "select")
graph.add_edge("select", END)

planning_graph = graph.compile()

# ------------------------------------------------------------
# API pública
# ------------------------------------------------------------

def run_sprint_planning_flow(dto: IssueEstimationDTO, backlog: Optional[List[Any]] = None, capacity_hours: float = 40.0) -> SprintPlanningState:
    """Executa o fluxo de planning.

    """
    issue_dict = dto.model_dump()

    initial_state: SprintPlanningState = {
        "issue": issue_dict,
        "backlog": backlog or [],
        "capacity_hours": capacity_hours,
    }
    return planning_graph.invoke(initial_state)
