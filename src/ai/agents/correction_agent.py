# src/agents/correction_agent.py
from typing import Dict, Any, List
from ai.core.llm_client import LLMClient
from ai.core.prompt_utils import format_similar_issues, build_system_prompt

SYSTEM_ROLE = "Você é um especialista que calcula padrões de erro entre estimativas e tempo real."
INSTRUCTION = """
Use a lista de issues similares (com estimated_hours e real_hours) para calcular um fator de correção (% de sub/overestima).
Aplique o fator para ajustar uma estimativa base (se fornecida) ou construa uma estimativa ajustada para a nova tarefa.
Retorne JSON: estimate_hours, confidence, justification.
"""

def run_correction(new_issue_text: str, similar_issues: List[Dict[str, Any]], base_estimate: float = None, llm: LLMClient = None) -> Dict[str, Any]:
    prompt = build_system_prompt(SYSTEM_ROLE, INSTRUCTION)
    prompt += "\n\nIssues similareS:\n" + format_similar_issues(similar_issues)
    if base_estimate:
        prompt += f"\n\nEstimativa base: {base_estimate} horas"
    prompt += "\n\nResponda apenas com JSON."
    # Se llm for fornecido use, senão usar fallback
    if llm:
        resp = llm.send_prompt(prompt, temperature=0.0, max_tokens=300)
        import json
        try:
            return json.loads(resp)
        except Exception:
            pass

    # fallback: calc fator medio (real/estimated)
    factors = []
    for it in similar_issues:
        est = it.get("estimated_hours")
        real = it.get("real_hours")
        if est and real:
            factors.append(real / est if est > 0 else 1.0)
    if factors:
        avg_factor = sum(factors) / len(factors)
        if base_estimate:
            adjusted = base_estimate * avg_factor
        else:
            # se não houver base, calcule media dos real_hours
            reals = [it.get("real_hours") for it in similar_issues if it.get("real_hours") is not None]
            adjusted = sum(reals) / len(reals) if reals else 8.0
        return {"estimate_hours": round(adjusted, 1), "confidence": 0.8, "justification": f"Fator medio de correcao: {avg_factor:.2f}x"}
    return {"estimate_hours": 8.0, "confidence": 0.35, "justification": "Fallback: sem dados de correcao."}
