from typing import TypedDict, Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END

from ai.core.retriever import Retriever
from ai.agents.analogical_agent import run_analogical
from ai.agents.heuristic_agent import run_heuristic
from ai.agents.supervisor_agent import combine_estimations
from ai.core.llm_client import LLMClient

from ai.core.mock_clients import MockVectorStoreClient
# from ai.core.vector_store import VectorStoreClient

from ai.dtos.issues_estimation_dto import IssueEstimationDTO



class Estimation(TypedDict):
    estimate_hours: float
    confidence: float
    justification: str


class EstimationState(TypedDict, total=False):
    issue: Dict[str, Any]

    issue_description: str
    similar_issues: List[Dict[str, Any]]
    repository_technologies: Dict[str, float]

    analogical: Estimation
    heuristic: Estimation
    final_estimation: Dict[str, Any]



# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def normalize_estimation(res: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "estimate_hours": float(res.get("estimate_hours", 0)),
        "confidence": float(res.get("confidence", 0.5)),
        "justification": str(res.get("justification", "")),
    }


# ------------------------------------------------------------
# Vector Store
# ------------------------------------------------------------

vector_store = MockVectorStoreClient()
# vector_store = VectorStoreClient()


# ------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

def retriever_node(state: EstimationState) -> EstimationState:
    retriever = Retriever(vector_store)

    issue = state["issue"]
    description = issue["description"]  # Fix: use 'description' as per DTO

    similar = retriever.get_similar_issues(description)

    techs = (
        vector_store.get_repository_technologies()
        if hasattr(vector_store, "get_repository_technologies")
        else {}
    )

    print(f"[Retriever] issues carregadas: {len(similar)}")

    return {
        "similar_issues": similar,
        "repository_technologies": techs
    }


def analogical_node(state: EstimationState) -> EstimationState:
    llm = LLMClient()

    issue = state["issue"]

    res = run_analogical(
        issue_context=issue,                     # 🔥 TUDO VAI PRA IA
        similar_issues=state.get("similar_issues", []),
        repository_technologies=state.get("repository_technologies", {}),
        llm=llm
    )

    return {"analogical": normalize_estimation(res)}


def heuristic_node(state: EstimationState) -> EstimationState:
    llm = LLMClient()

    issue = state["issue"]

    res = run_heuristic(
        issue_context=issue,   
        llm=llm
    )

    return {"heuristic": normalize_estimation(res)}


def supervisor_node(state: EstimationState) -> EstimationState:
    estimations = []

    if "heuristic" in state:
        h = dict(state["heuristic"])
        h["source"] = "heuristic"
        estimations.append(h)

    if "analogical" in state:
        a = dict(state["analogical"])
        a["source"] = "analogical"
        estimations.append(a)

    llm = LLMClient()


    final = combine_estimations(
        estimations=estimations,
        llm=llm
    )

    return {"final_estimation": final}


# ------------------------------------------------------------
# Grafo
# ------------------------------------------------------------

graph = StateGraph(EstimationState)

graph.add_node("retriever", retriever_node)
graph.add_node("analogical", analogical_node)
graph.add_node("heuristic", heuristic_node)
graph.add_node("supervisor", supervisor_node)

graph.add_edge(START, "retriever")
graph.add_edge("retriever", "analogical")
graph.add_edge("retriever", "heuristic")
graph.add_edge("analogical", "supervisor")
graph.add_edge("heuristic", "supervisor")
graph.add_edge("supervisor", END)

estimation_graph = graph.compile()


# ------------------------------------------------------------
# API pública
# ------------------------------------------------------------

def run_estimation_flow(dto: IssueEstimationDTO) -> EstimationState:

    issue_dict = dto.model_dump()

    initial_state: EstimationState = {
        "issue": issue_dict,
    }
    
    return estimation_graph.invoke(initial_state)
