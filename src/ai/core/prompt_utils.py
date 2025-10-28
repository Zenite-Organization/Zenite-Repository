# src/core/prompt_utils.py
from typing import List, Dict

def format_similar_issues(similar: List[Dict]) -> str:
    """
    Formata a lista de issues para colocar no prompt do LLM.
    """
    lines = []
    for idx, it in enumerate(similar, start=1):
        title = it.get("title", "")[:200].replace("\n", " ")
        desc = it.get("description", "")[:600].replace("\n", " ")
        est = it.get("estimated_hours", "N/A")
        real = it.get("real_hours", "N/A")
        lines.append(f"{idx}. {title} | est: {est}h | real: {real}h\n   desc: {desc}")
    return "\n".join(lines)

def build_system_prompt(role_description: str, instruction: str) -> str:
    return f"{role_description}\n\n{instruction}"
