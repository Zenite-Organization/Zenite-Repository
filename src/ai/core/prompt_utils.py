from typing import List, Dict


def format_similar_issues(similar: List[Dict]) -> str:
    """
    Formata a lista de issues para colocar no prompt do LLM.
    """
    lines = []
    idx = 1
    for it in similar:
        raw_title = it.get("title")
        title = (raw_title if isinstance(raw_title, str) else "").strip()[:200].replace("\n", " ")
        est = it.get("total_effort_hours")
        if not title or est is None:
            continue
        score = it.get("score", 0)
        type = it.get("issue_type", "unknown")
        desc_raw = it.get('description', '')
        desc = (desc_raw if isinstance(desc_raw, str) else '')[:400].replace('\n', ' ')
        lines.append(f"{idx}. Score: {score:.2f} | Titulo: {title} | Tipo: {type} | horas consumidas: {est}h | Descrição: {desc}")
        idx += 1

    if not lines:
        return "No valid similar issues with title and estimate."
    return "\n".join(lines)


def build_system_prompt(role_description: str, instruction: str) -> str:
    return f"{role_description}\n\n{instruction}"
