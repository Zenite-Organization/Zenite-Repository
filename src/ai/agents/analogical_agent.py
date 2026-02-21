from typing import Dict, Any, List
from ai.core.llm_client import LLMClient
from ai.core.json_utils import parse_llm_json_response
from ai.core.prompt_utils import format_similar_issues, build_system_prompt
import json


SYSTEM_ROLE = (
    "Voce e um especialista senior em estimativa de esforco de software, "
    "capaz de analisar tarefas novas com base em historico de issues semelhantes."
)

INSTRUCTION = """
Voce recebeu o contexto completo de uma nova tarefa (issue) e uma lista de issues historicas similares.

Utilize TODAS as informacoes relevantes do contexto, como:
- titulo
- descricao
- labels
- repositorio
- contexto extra (se houver)

Compare a nova tarefa com as issues similares e calcule uma estimativa de esforco em horas,
priorizando a comparacao com os campos 'estimated_hours' das issues historicas.

Retorne APENAS um JSON valido com os campos:
- estimate_hours (float)
- confidence (float entre 0 e 1)
- justification (string)

Regras:
- Se houver poucas issues similares ou baixa similaridade, reduza a confidence (ex: 0.3 a 0.5)
- Explique claramente o raciocinio na justification
- Nao inclua texto fora do JSON
"""


def run_analogical(
    issue_context: Dict[str, Any],
    similar_issues: List[Dict[str, Any]],
    repository_technologies: Dict[str, float],
    llm: LLMClient,
) -> Dict[str, Any]:
    system_prompt = build_system_prompt(SYSTEM_ROLE, INSTRUCTION)

    issue_text = json.dumps(issue_context, indent=2, ensure_ascii=False)

    prompt = (
        system_prompt
        + "\n\n=== NOVA ISSUE ===\n"
        + issue_text
        + "\n\n=== ISSUES HISTORICAS SIMILARES ===\n"
        + format_similar_issues(similar_issues)
    )

    print("[IA][ANALOGICAL] prompt:\n%s" % prompt)
    response = llm.send_prompt(prompt, temperature=0.0, max_tokens=400)

    try:
        parsed = parse_llm_json_response(response)
        return parsed
    except Exception as e:
        print(f"[IA][ANALOGICAL] erro parse: {e}")
        return {
            "estimate_hours": 0,
            "confidence": 0.0,
            "justification": "Falha ao interpretar resposta do modelo.",
            "error": str(e),
            "raw_response": response,
        }
