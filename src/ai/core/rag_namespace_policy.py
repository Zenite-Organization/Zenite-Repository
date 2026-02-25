from typing import Any, Dict, List


def is_base_namespace(namespace: str) -> bool:
    ns = str(namespace or "").strip().lower()
    return bool(ns) and ("_" not in ns)


def project_namespaces(project_key: str) -> List[str]:
    base = project_key.strip().lower()
    if not base:
        return []
    return [base, f"{base}_comments", f"{base}_changelog"]


def extract_org_namespace(repository: str) -> str:
    raw = str(repository or "").strip().lower()
    if not raw:
        return ""
    if "/" in raw:
        return raw.split("/", 1)[0].strip()
    return raw


def group_namespaces_by_base(namespaces: List[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for item in namespaces:
        ns = str(item or "").strip().lower()
        if not is_base_namespace(ns):
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
