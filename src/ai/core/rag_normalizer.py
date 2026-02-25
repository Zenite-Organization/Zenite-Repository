from typing import Any, Dict, Optional


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _minutes_to_hours(value: Any) -> Optional[float]:
    num = _to_float(value)
    if num is None:
        return None
    return round(num / 60.0, 2)


def _extract_text(metadata: Dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_match(raw: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(raw.get("metadata") or {})
    namespace = str(raw.get("namespace") or "")
    doc_type = str(metadata.get("doc_type") or "").strip().lower()
    if not doc_type:
        if namespace.endswith("_comments"):
            doc_type = "comment"
        elif namespace.endswith("_changelog"):
            doc_type = "changelog"
        else:
            doc_type = "issue"

    issue_id = metadata.get("issue_id")
    if issue_id is not None:
        try:
            issue_id = int(issue_id)
        except (TypeError, ValueError):
            issue_id = str(issue_id)

    estimated_hours = _minutes_to_hours(metadata.get("total_effort_minutes"))

    title = _extract_text(metadata, ["issue_title", "title", "summary"])
    description = _extract_text(
        metadata,
        ["description", "body", "text", "content", "to_string", "from_string"],
    )
    snippet = description[:400] if description else ""

    return {
        "id": str(raw.get("id") or ""),
        "project_key": str(metadata.get("project_key") or "").lower(),
        "namespace": namespace,
        "doc_type": doc_type,
        "issue_id": issue_id,
        "issue_key": metadata.get("issue_key"),
        "title": title,
        "description": description,
        "snippet": snippet,
        "estimated_hours": estimated_hours,
        "score": float(raw.get("score") or 0.0),
        "metadata": metadata,
    }
