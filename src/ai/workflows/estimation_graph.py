import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict, Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END

from ai.core.retriever import Retriever
from ai.agents.analogical_agent import run_analogical
from ai.agents.heuristic_agent import run_heuristic
from ai.agents.supervisor_agent import combine_heuristic_estimations
from ai.core.llm_client import LLMClient
from ai.core.pinecone_vector_store import PineconeVectorStoreClient
from config.settings import settings

from ai.dtos.issues_estimation_dto import IssueEstimationDTO



class Estimation(TypedDict):
    estimated_hours: float
    confidence: float
    justification: str


class EstimationState(TypedDict, total=False):
    issue: Dict[str, Any]
    issue_description: str
    similar_issues: List[Dict[str, Any]]
    repository_technologies: Dict[str, float]
    rag_context_sufficient: bool
    strategy: str
    rag_stats: Dict[str, Any]

    analogical: Estimation
    heuristic: Estimation
    heuristic_candidates: List[Estimation]
    final_estimation: Dict[str, Any]


logger = logging.getLogger(__name__)



# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def normalize_estimation(res: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "estimated_hours": float(res.get("estimated_hours", 0)),
        "confidence": float(res.get("confidence", 0.5)),
        "justification": str(res.get("justification", "")),
    }


def model_from_strategy(strategy: Optional[str]) -> str:
    if strategy == "analogical":
        return "analogical"
    return "heuristic"


# ------------------------------------------------------------
# Vector Store
# ------------------------------------------------------------

vector_store = PineconeVectorStoreClient()


# ------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

def retriever_node(state: EstimationState) -> EstimationState:
    retriever = Retriever(vector_store)

    issue = state["issue"]
    similar = retriever.get_similar_issues(issue)
    min_score = float(settings.RAG_MIN_SCORE_MAIN)
    min_hits = int(settings.RAG_MIN_HITS_MAIN)

    qualified_hits = 0
    for it in similar:
        try:
            score = float(it.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score >= min_score:
            qualified_hits += 1

    rag_context_sufficient = qualified_hits >= min_hits
    strategy = "analogical" if rag_context_sufficient else "heuristic_ensemble"

    techs = (
        vector_store.get_repository_technologies()
        if hasattr(vector_store, "get_repository_technologies")
        else {}
    )

    logger.info("[Retriever] issues carregadas=%s", len(similar))

    return {
        "similar_issues": similar,
        "repository_technologies": techs,
        "rag_context_sufficient": rag_context_sufficient,
        "strategy": strategy,
        "rag_stats": {
            "qualified_hits": qualified_hits,
            "min_hits": min_hits,
            "min_score": min_score,
        },
    }


def analogical_node(state: EstimationState) -> EstimationState:
    llm = LLMClient()

    issue = state["issue"]

    res = run_analogical(
        issue_context=issue,               
        similar_issues=state.get("similar_issues", []),
        repository_technologies=state.get("repository_technologies", {}),
        llm=llm
    )

    return {"analogical": normalize_estimation(res)}


def heuristic_ensemble_node(state: EstimationState) -> EstimationState:
    issue = state["issue"]

    agents = [
        ("p25", 0.25),
        ("p50", 0.50),
        ("p75", 0.75),
        ("p100", 1.00),
    ]

    candidates: List[Estimation] = []

    max_concurrency = int(getattr(settings, "HEURISTIC_ENSEMBLE_MAX_CONCURRENCY", runs) or runs)
    if max_concurrency <= 0:
        max_concurrency = runs
    max_concurrency = max(1, min(max_concurrency, runs))

    def _run_once() -> Dict[str, Any]:
        # Create a client per call to avoid shared-state/thread-safety issues.
        llm = LLMClient()
        return run_heuristic(
            issue_context=issue,
            llm=llm,
            temperature=temperature,
            mode=percentile,
        )
        candidates.append(normalize_estimation(res))

    return {"heuristic_candidates": candidates}


def finalize_analogical_node(state: EstimationState) -> EstimationState:
    analogical = dict(state.get("analogical") or {})
    if not analogical:
        analogical = {
            "estimated_hours": 0.0,
            "confidence": 0.0,
            "justification": "Falha na rota analogical.",
        }
    else:
        justification = str(analogical.get("justification") or "").strip()
        suffix = "Rota analogical escolhida por contexto RAG suficiente."
        analogical["justification"] = f"{justification} {suffix}".strip()
    analogical["estimation_model"] = model_from_strategy(state.get("strategy"))
    return {"final_estimation": analogical}


def supervisor_node(state: EstimationState) -> EstimationState:
    candidates = state.get("heuristic_candidates", [])
    estimations = []

    for idx, candidate in enumerate(candidates, start=1):
        h = dict(candidate)

        # Preserve source se jÃ¡ vier do node anterior
        if "source" not in h:
            h["source"] = f"heuristic_{idx}"

        # Garante que estimated_hours Ã© numÃ©rico
        try:
            h["estimated_hours"] = float(h.get("estimated_hours", 0))
        except (TypeError, ValueError):
            h["estimated_hours"] = 0.0

        # Garante confidence vÃ¡lida
        try:
            conf = float(h.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        h["confidence"] = max(0.0, min(1.0, conf))

        estimations.append(h)

    # Fallback mÃ­nimo
    if not estimations:
        estimations.append(
            {
                "source": "heuristic_fallback",
                "percentile": "p50",
                "estimated_hours": 8.0,
                "confidence": 0.3,
                "justification": "Fallback por falta de candidatos heurÃ­sticos.",
            }
        )

    # ðŸ”¹ Ordena por estimated_hours (garante coerÃªncia para o fallback robusto)
    estimations = sorted(estimations, key=lambda x: x["estimated_hours"])

    llm = LLMClient()

    final = combine_heuristic_estimations(
        estimations=estimations,
        llm=llm,
    )
    final["estimation_model"] = model_from_strategy(state.get("strategy"))

    return {"final_estimation": final}


# ------------------------------------------------------------
# Grafo
# ------------------------------------------------------------

graph = StateGraph(EstimationState)

graph.add_node("retriever", retriever_node)
graph.add_node("analogical", analogical_node)
graph.add_node("heuristic_ensemble", heuristic_ensemble_node)
graph.add_node("supervisor", supervisor_node)
graph.add_node("finalize_analogical", finalize_analogical_node)


def route_after_retriever(state: EstimationState) -> str:
    strategy = state.get("strategy", "heuristic_ensemble")
    return "analogical" if strategy == "analogical" else "heuristic_ensemble"

graph.add_edge(START, "retriever")
graph.add_conditional_edges(
    "retriever",
    route_after_retriever,
    {
        "analogical": "analogical",
        "heuristic_ensemble": "heuristic_ensemble",
    },
)
graph.add_edge("analogical", "finalize_analogical")
graph.add_edge("finalize_analogical", END)
graph.add_edge("heuristic_ensemble", "supervisor")
graph.add_edge("supervisor", END)

estimation_graph = graph.compile()


# ------------------------------------------------------------
# API pÃºblica
# ------------------------------------------------------------

def run_estimation_flow(dto: IssueEstimationDTO) -> EstimationState:

    issue_dict = dto.model_dump()

    initial_state: EstimationState = {
        "issue": issue_dict,
    }
    
    return estimation_graph.invoke(initial_state)

