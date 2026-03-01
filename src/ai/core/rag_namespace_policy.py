from typing import Any, Dict, List


def is_issue_namespace(namespace: str) -> bool:
    ns = str(namespace or "").strip().lower()
    return bool(ns) and ns.endswith("_issues") and len(ns) > len("_issues")


def project_namespaces(project_key: str) -> List[str]:
    base = project_key.strip().lower()
    if not base:
        return []
    return [f"{base}_issues"]


def extract_project_name(repository: str) -> str:
    raw = str(repository or "").strip().lower()
    if not raw:
        return ""
    if "/" in raw:
        return raw.split("/", 1)[1].strip()
    return raw


def extract_project_issue_namespace(repository: str) -> str:
    project_name = extract_project_name(repository)
    if not project_name:
        return ""
    return f"{project_name}_issues"


def group_issue_namespaces(namespaces: List[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for item in namespaces:
        ns = str(item or "").strip().lower()
        if not is_issue_namespace(ns):
            continue
        if ns not in seen:
            ordered.append(ns)
            seen.add(ns)
    return ordered


def _score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def namespace_quality(
    matches: List[Dict[str, Any]],
    namespaces: List[str],
    min_hits: int,
    min_score: float,
) -> bool:
    if not namespaces:
        return False
    namespace_set = {str(ns or "").strip().lower() for ns in namespaces if str(ns or "").strip()}
    if not namespace_set:
        return False
    qualified_hits = 0
    for match in matches:
        ns = str(match.get("namespace") or "").strip().lower()
        if ns in namespace_set and _score(match.get("score")) >= float(min_score):
            qualified_hits += 1
    return qualified_hits >= int(min_hits)
