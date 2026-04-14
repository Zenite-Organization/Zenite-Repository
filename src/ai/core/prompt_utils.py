from typing import Any, Dict, List


def _clean_text(value: Any, limit: int) -> str:
    text = (value if isinstance(value, str) else str(value or "")).strip()
    text = text.replace("\n", " ")
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def _safe_score(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"


def format_similar_issues(similar: List[Dict]) -> str:
    """
    Formata a lista de issues para colocar no prompt do LLM.
    """
    lines = []
    idx = 1
    for it in similar:
        title = _clean_text(it.get("title"), limit=180)
        est = it.get("total_effort_hours")
        if not title or est is None:
            continue

        score = _safe_score(it.get("score"))
        issue_type = _clean_text(it.get("issue_type") or "unknown", limit=40)
        doc_type = _clean_text(it.get("doc_type") or "issue", limit=20)
        desc = _clean_text(it.get("description"), limit=280)

        parts = [
            f"{idx}. Score: {score}",
            f"Tipo: {issue_type}",
            f"Origem: {doc_type}",
            f"Horas: {est}h",
            f"Titulo: {title}",
        ]
        if desc:
            parts.append(f"Descricao: {desc}")

        lines.append(" | ".join(parts))
        idx += 1

    if not lines:
        return "No valid similar issues with title and estimate."
    return "\n".join(lines)


def build_system_prompt(role_description: str, instruction: str) -> str:
    return f"{role_description}\n\n{instruction}"
