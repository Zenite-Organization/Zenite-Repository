# src/agents/analogical_agent.py
from typing import Dict, Any, List
from ai.core.llm_client import LLMClient
from ai.core.prompt_utils import format_similar_issues, build_system_prompt

SYSTEM_ROLE = "Você é um especialista que estima esforço com base em tarefas históricas semelhantes."

INSTRUCTION = """
Você recebeu uma nova tarefa (descricao abaixo) e uma lista de issues historicas similares.
Calcule uma estimativa em horas baseada na media ponderada dos 'real_hours' das issues similares.
Retorne um JSON com campos: estimate_hours (float), confidence (0-1), justification (string).
Se não houver issues similares suficientes, indique confidence baixa (ex: 0.4).
"""

def run_analogical(new_issue_text: str, similar_issues: List[Dict[str, Any]], llm: LLMClient) -> Dict[str, Any]:
    prompt = build_system_prompt(SYSTEM_ROLE, INSTRUCTION)
    prompt += "\n\nNova tarefa:\n" + new_issue_text + "\n\nIssues similares:\n"
    prompt += format_similar_issues(similar_issues)

    # Add a deterministic instruction to output only JSON
    prompt += "\n\nPor favor responda apenas com um JSON válido com os campos: estimate_hours, confidence, justification."

    response = llm.send_prompt(prompt, temperature=0.0, max_tokens=300)
    # tentamos parsear JSON da resposta; como fallback, estimamos por média
    import json
    try:
        parsed = json.loads(response)
        return parsed
    except Exception:
        # fallback: calcular média simples
        reals = [i.get("real_hours") for i in similar_issues if i.get("real_hours") is not None]
        if reals:
            avg = sum(reals) / len(reals)
            return {"estimate_hours": avg, "confidence": 0.6, "justification": "Fallback: média dos real_hours das issues similares."}
        return {"estimate_hours": 8.0, "confidence": 0.35, "justification": "Fallback: sem similaridades suficientes."}
