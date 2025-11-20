from typing import TypedDict, List, Any, Dict
from ai.agents.analogical_agent import run_analogical
from ai.agents.heuristic_agent import run_heuristic
from ai.agents.correction_agent import run_correction
from ai.agents.supervisor_agent import combine_estimations
from langgraph.graph import StateGraph, END, START
from ai.core.llm_client import LLMClient
# ------------------------------------------------------------
# Estado compartilhado do grafo
# ------------------------------------------------------------
class Estimation(TypedDict):
    estimate_hours: float
    confidence: float
    justification: str


class EstimationState(TypedDict, total=False):
    issue_description: str
    # similar_issues: List[Any]
    analogical: Estimation
    heuristic: Estimation
    correction: Estimation

    final_estimation: Dict[str, Any]


# ------------------------------------------------------------
# Helpers para padronizar o retorno dos agentes
# ------------------------------------------------------------

def normalize_estimation(res: Dict[str, Any]) -> Estimation:
    """
    Normaliza a saída de qualquer agente para o formato padrão.
    """
    return {
        "estimate_hours": float(res.get("estimate_hours", 0)),
        "confidence": float(res.get("confidence", 0.5)),
        "justification": str(res.get("justification", "")),
    }


# ------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

# def retriever_node(state: EstimationState) -> EstimationState:
#     retriever = Retriever()
#     similar = retriever.search(state["issue_description"])
#     return {"similar_issues": similar}


# def analogical_node(state: EstimationState) -> EstimationState:
#     agent = run_analogical()
#     res = agent.estimate(state["issue_description"], state["similar_issues"])
#     normalized = normalize_estimation(res)
#     return {"analogical": normalized}


def heuristic_node(state: EstimationState) -> EstimationState:
    llm = LLMClient()
    res = run_heuristic(state["issue_description"], llm)
    normalized = normalize_estimation(res)
    return {"heuristic": normalized}


# def correction_node(state: EstimationState) -> EstimationState:
#     agent = run_correction()
#     res = agent.estimate(state["issue_description"], state["similar_issues"])
#     normalized = normalize_estimation(res)
#     return {"correction": normalized}


def supervisor_node(state: EstimationState) -> EstimationState:
    estimations = []
    if "heuristic" in state:
        estimations.append(state["heuristic"])
    if "analogical" in state:
        estimations.append(state["analogical"])
    if "correction" in state:
        estimations.append(state["correction"])
    final = combine_estimations(estimations)
    return {"final_estimation": final}


# ------------------------------------------------------------
# Grafo LangGraph
# ------------------------------------------------------------

graph = StateGraph(EstimationState)


# graph.add_node("retriever", retriever_node)
# graph.add_node("analogical_agent", analogical_node)
graph.add_node("heuristic_agent", heuristic_node)
# graph.add_node("correction_agent", correction_node)
graph.add_node("supervisor", supervisor_node)

# Edges
graph.add_edge(START, "heuristic_agent")
# graph.add_edge("retriever", "analogical_agent")
# graph.add_edge("retriever", "heuristic_agent")
# graph.add_edge("retriever", "correction_agent")

# graph.add_edge("analogical_agent", "supervisor")
graph.add_edge("heuristic_agent", "supervisor")
# graph.add_edge("correction_agent", "supervisor")

graph.add_edge("supervisor", END)

estimation_graph = graph.compile()


# ------------------------------------------------------------
# Função pública
# ------------------------------------------------------------

def run_estimation_flow(issue_description: str) -> EstimationState:
    return estimation_graph.invoke({"issue_description": issue_description})
