# src/agents/heuristic_agent.py
from typing import Dict, Any
from src.core.llm_client import LLMClient
from src.core.prompt_utils import build_system_prompt

SYSTEM_ROLE = "Você é um analista que avalia complexidade técnica de tickets."
INSTRUCTION = """
Analise a descricao da tarefa e identifique fatores que impactam o tempo:
- integrações externas
- dependencias
- refatoracao de legacy
- ambiente desconhecido
- tamanho do deliverable (pequeno/medio/grande)
Baseando-se nessas heuristicas, estime horas e dê uma justificativa curta. Retorne JSON: estimate_hours, confidence, justification.
"""

def run_heuristic(new_issue_text: str, llm: LLMClient) -> Dict[str, Any]:
    prompt = build_system_prompt(SYSTEM_ROLE, INSTRUCTION)
    prompt += "\n\nTarefa:\n" + new_issue_text
    prompt += "\n\nResponda apenas com JSON."

    resp = llm.send_prompt(prompt, temperature=0.15, max_tokens=300)
    import json
    try:
        return json.loads(resp)
    except Exception:
        # fallback heuristico simples
        low_risk_keywords = ["typo", "ui fix", "docs", "small"]
        high_risk_keywords = ["refactor", "migrate", "integration", "api", "legacy", "database", "performance"]
        text = new_issue_text.lower()
        score = 8.0
        for w in high_risk_keywords:
            if w in text:
                score *= 1.4
        for w in low_risk_keywords:
            if w in text:
                score *= 0.75
        conf = 0.6
        return {"estimate_hours": round(score, 1), "confidence": conf, "justification": "Fallback heurístico baseado em palavras-chave."}
