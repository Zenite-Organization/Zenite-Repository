from typing import Dict, Any, List
from ai.core.llm_client import LLMClient
from ai.core.prompt_utils import format_similar_issues, build_system_prompt
import json


SYSTEM_ROLE = (
    "Você é um especialista sênior em estimativa de esforço de software, "
    "capaz de analisar tarefas novas com base em histórico de issues semelhantes."
)

INSTRUCTION = """
Você recebeu o contexto completo de uma nova tarefa (issue) e uma lista de issues históricas similares.

Utilize TODAS as informações relevantes do contexto, como:
- título
- descrição
- labels
- repositório
- contexto extra (se houver)

Compare a nova tarefa com as issues similares e calcule uma estimativa de esforço em horas,
priorizando a média ponderada dos campos 'real_hours' das issues históricas.

Retorne APENAS um JSON válido com os campos:
- estimate_hours (float)
- confidence (float entre 0 e 1)
- justification (string)

Regras:
- Se houver poucas issues similares ou baixa similaridade, reduza a confidence (ex: 0.3 a 0.5)
- Explique claramente o raciocínio na justification
- Não inclua texto fora do JSON
"""

def run_analogical(
    issue_context: Dict[str, Any],
    similar_issues: List[Dict[str, Any]],
    repository_technologies: Dict[str, float],
    llm: LLMClient
) -> Dict[str, Any]:

    system_prompt = build_system_prompt(SYSTEM_ROLE, INSTRUCTION)

    issue_text = json.dumps(issue_context, indent=2, ensure_ascii=False)

    prompt = (
        system_prompt
        + "\n\n=== NOVA ISSUE ===\n"
        + issue_text
        + "\n\n=== ISSUES HISTÓRICAS SIMILARES ===\n"
        + format_similar_issues(similar_issues)
    )

    print("[Analogical Agent] Prompt enviado ao LLM:\n", prompt)

    response = llm.send_prompt(
        prompt,
        temperature=0.0,
        max_tokens=400
    )

    # limpeza de markdown ```json
    if response.strip().startswith("```"):
        lines = response.strip().splitlines()
        lines = [
            line for line in lines
            if not line.strip().startswith("```")
        ]
        resp_clean = "\n".join(lines)
    else:
        resp_clean = response

    try:
        return json.loads(resp_clean)
    except Exception as e:
        return {
            "estimate_hours": 0,
            "confidence": 0.0,
            "justification": "Falha ao interpretar resposta do modelo.",
            "error": str(e),
            "raw_response": response
        }
