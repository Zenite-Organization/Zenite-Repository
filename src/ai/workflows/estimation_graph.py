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
    estimate_hours: float
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
        "estimate_hours": float(res.get("estimate_hours", 0)),
        "confidence": float(res.get("confidence", 0.5)),
        "justification": str(res.get("justification", "")),
    }


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
    runs = int(settings.HEURISTIC_ENSEMBLE_RUNS)
    temperature = float(settings.HEURISTIC_ENSEMBLE_TEMPERATURE)
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
        )

    if runs <= 1 or max_concurrency == 1:
        for _ in range(runs):
            res = _run_once()
            candidates.append(normalize_estimation(res))
        return {"heuristic_candidates": candidates}

    results: List[Optional[Estimation]] = [None] * runs
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        future_to_idx = {executor.submit(_run_once): idx for idx in range(runs)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                res = future.result()
            except Exception as exc:
                res = {
                    "estimate_hours": 8,
                    "confidence": 0.3,
                    "justification": "Falha ao executar heurística; valor padrão aplicado.",
                    "error": str(exc),
                }
            results[idx] = normalize_estimation(res)

    candidates = [r for r in results if r is not None]

    return {"heuristic_candidates": candidates}


def finalize_analogical_node(state: EstimationState) -> EstimationState:
    analogical = dict(state.get("analogical") or {})
    if not analogical:
        analogical = {
            "estimate_hours": 0.0,
            "confidence": 0.0,
            "justification": "Falha na rota analogical.",
        }
    else:
        justification = str(analogical.get("justification") or "").strip()
        suffix = "Rota analogical escolhida por contexto RAG suficiente."
        analogical["justification"] = f"{justification} {suffix}".strip()
    return {"final_estimation": analogical}


def supervisor_node(state: EstimationState) -> EstimationState:
    estimations = []
    candidates = state.get("heuristic_candidates", [])
    for idx, candidate in enumerate(candidates, start=1):
        h = dict(candidate)
        h["source"] = f"heuristic_{idx}"
        estimations.append(h)

    if not estimations:
        estimations.append(
            {
                "source": "heuristic_1",
                "estimate_hours": 8.0,
                "confidence": 0.3,
                "justification": "Fallback por falta de candidatos heurísticos.",
            }
        )

    llm = LLMClient()

    final = combine_heuristic_estimations(
        estimations=estimations,
        llm=llm,
    )

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
# API pública
# ------------------------------------------------------------

def run_estimation_flow(dto: IssueEstimationDTO) -> EstimationState:

    issue_dict = dto.model_dump()

    initial_state: EstimationState = {
        "issue": issue_dict,
    }
    
    return estimation_graph.invoke(initial_state)
