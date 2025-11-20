# src/agents/heuristic_agent.py
from typing import Dict, Any
from ai.core.llm_client import LLMClient
from ai.core.prompt_utils import build_system_prompt
import json
SYSTEM_ROLE = "Você é um analista que avalia complexidade técnica de tickets."
INSTRUCTION = """
Analise a descricao da tarefa e identifique fatores que impactam o tempo:
- integrações externas
- dependencias
- refatoracao de legacy
- ambiente desconhecido
- tamanho do deliverable (pequeno/medio/grande)
Baseando-se nessas heuristicas, estime horas e dê uma justificativa curta.
Retorne um JSON com os seguintes campos e tipos:
    - estimate_hours: float (exemplo: 12.5)
    - confidence: float (exemplo: 0.8)
    - justification: string (exemplo: "Explicação curta da estimativa")
Responda apenas com o JSON, sem texto extra.
"""

def run_heuristic(new_issue_text: str, llm: LLMClient) -> Dict[str, Any]:
    prompt = build_system_prompt(SYSTEM_ROLE, INSTRUCTION)
    prompt += "\n\nTarefa:\n" + new_issue_text
    prompt += "\n\nResponda apenas com JSON."

    resp = llm.send_prompt(prompt, temperature=0.15, max_tokens=300)
    print("Resposta do LLM:", resp)  # Para depuração
    # Remove delimitadores de bloco de código markdown, se existirem
    if resp.strip().startswith("```"):
        # Remove primeira linha (```json ou ```)
        lines = resp.strip().splitlines()
        # Remove linhas que começam e terminam com ```
        lines = [line for line in lines if not line.strip().startswith("```") and not line.strip().endswith("```")]
        resp_clean = "\n".join(lines)
    else:
        resp_clean = resp
    try:
        return json.loads(resp_clean)
    except Exception as e:
        return {"error": "Resposta inválida do LLM", "raw_response": resp, "exception": str(e)}
