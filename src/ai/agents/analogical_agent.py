from typing import Dict, Any, List
from ai.core.llm_client import LLMClient
from ai.core.json_utils import parse_llm_json_response
from ai.core.prompt_utils import format_similar_issues, build_system_prompt
import json


SYSTEM_ROLE = (
    "Você é um especialista sênior em estimativa de esforço de software por analogia."
)

INSTRUCTION = """
Você recebeu:
(1) o contexto completo de uma nova issue
(2) uma lista de issues históricas semelhantes, contendo informacoes como titulo, descricao, tipo da demanda, score (entre 0 e 1) e horas consumidas (tempo que aquela issue levou para ser finalizada)

Objetivo:
Gerar uma estimativa de esforço em horas para a NOVA issue, usando principalmente o histórico.
Use o campo score para ponderar as issues semelhantes.

IMPORTANTE (Duplicatas / Similaridade alta):
1) Se existir ao menos 1 issue histórica com score >= 0.92 e título (ou termos-chave) praticamente idênticos,
   trate essa issue como "âncora" e NÃO a descarte como outlier, mesmo que as horas consumidas seja muito diferente das demais.
2) Quando houver âncora:
   - Faça estimated_hours ficar relativamente próximo das horas consumidas da âncora (ex.: dentro de ~20% a ~40%),
     a menos que existam evidências explícitas no texto da NOVA issue indicando escopo menor/maior.
   - Dê peso dominante à âncora (ex.: 60% a 85%) e distribua o restante entre as próximas mais similares.

Robustez contra outliers (quando NÃO houver âncora):
3) Use agregação robusta nas top-K similares (ex.: K=8 a 15):
   - priorize weighted median ou média ponderada com corte (trim) para reduzir influência de extremos.
   - score deve ser o peso principal (peso cresce não-linearmente; ex.: score^3).

Regras adicionais:
4) Dê preferência a issues com mesmo tipo e labels/componentes semelhantes.
5) Se houver poucos exemplos úteis (ex.: <4 com score >= 0.75), reduza confidence.
6) Confidence deve refletir:
   - quantidade de itens relevantes
   - magnitude do score (top1 e top3)
   - dispersão das horas (se muito disperso, confidence menor), EXCETO quando houver âncora forte.

Retorne APENAS um JSON válido com:
{
  "estimated_hours": float,
  "confidence": float (0..1),
  "justification": "string curta explicando: top similares usados, presença/ausência de âncora, como ponderou, e por que a confiança é X"
}
Não inclua nenhum texto fora do JSON.
"""


def run_analogical(
    issue_context: Dict[str, Any],
    similar_issues: List[Dict[str, Any]],
    repository_technologies: Dict[str, float],
    llm: LLMClient,
) -> Dict[str, Any]:
    system_prompt = build_system_prompt(SYSTEM_ROLE, INSTRUCTION)

    issue_text = json.dumps(issue_context, indent=2, ensure_ascii=False)

    # print("similar issue:", similar_issues)

    prompt = (
        system_prompt
        + "\n\n##NOVA ISSUE\n"
        + issue_text
        + "\n\n##ISSUES HISTORICAS SIMILARES\n"
        + format_similar_issues(similar_issues)
    )
    print("prompt final:", prompt)
    response = llm.send_prompt(prompt, temperature=0.0, max_tokens=400)

    try:
        parsed = parse_llm_json_response(response)
        return parsed
    except Exception as e:
        print(f"[IA][ANALOGICAL] erro parse: {e}")
        return {
            "estimated_hours": 0,
            "confidence": 0.0,
            "justification": "Falha ao interpretar resposta do modelo.",
            "error": str(e),
            "raw_response": response,
        }
