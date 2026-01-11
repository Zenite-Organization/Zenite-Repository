from typing import Dict, Any
from ai.core.llm_client import LLMClient
from ai.core.prompt_utils import build_system_prompt
import json


SYSTEM_ROLE = (
    "Você é um analista de software sênior especializado em estimativa de esforço "
    "baseada em heurísticas ancoradas e padrões de projeto."
)

INSTRUCTION = """
Você recebeu o CONTEXTO COMPLETO de uma issue de software e um BASELINE DO PROJETO.

Seu trabalho NÃO é chutar horas livremente.

Você deve:
1. Classificar a tarefa em um TAMANHO relativo:
   - XS, S, M, L ou XL
2. Aplicar o baseline de horas correspondente
3. Ajustar a estimativa com base em fatores de risco e incerteza
4. Retornar uma estimativa final ESTÁVEL e JUSTIFICADA

====================
BASELINE DO PROJETO
====================

Tamanhos e faixas típicas:
- XS: 1–2 horas
- S: 3–5 horas
- M: 6–10 horas
- L: 12–20 horas
- XL: 24–40 horas

Regras:
- Bugs simples tendem a XS ou S
- Features pequenas tendem a S ou M
- Integrações, refatorações e POCs tendem a L ou XL
- Se houver alta incerteza, use o topo da faixa
- Nunca extrapole o intervalo do tamanho escolhido

====================
ANÁLISE
====================

Considere:
- tipo da tarefa (bug, feature, infra, etc)
- complexidade técnica
- dependências internas ou externas
- impacto em código legado
- clareza dos requisitos
- risco técnico

====================
SAÍDA
====================

Retorne APENAS um JSON válido com:
- size (string: XS | S | M | L | XL)
- estimate_hours (float)
- confidence (float entre 0 e 1)
- justification (string curta)

Não inclua texto fora do JSON.
"""


def run_heuristic(
    issue_context: Dict[str, Any],
    llm: LLMClient
) -> Dict[str, Any]:

    system_prompt = build_system_prompt(SYSTEM_ROLE, INSTRUCTION)

    issue_text = json.dumps(issue_context, indent=2, ensure_ascii=False)

    prompt = (
        system_prompt
        + "\n\n=== CONTEXTO DA ISSUE ===\n"
        + issue_text
    )

    print("[Heuristic Agent] Prompt enviado ao LLM:\n", prompt)

    response = llm.send_prompt(
        prompt,
        temperature=0.0, 
        max_tokens=350
    )

    # limpeza de ```json
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
            "size": "M",
            "estimate_hours": 8,
            "confidence": 0.3,
            "justification": "Falha ao interpretar resposta do modelo; valor padrão aplicado.",
            "error": str(e),
            "raw_response": response
        }
