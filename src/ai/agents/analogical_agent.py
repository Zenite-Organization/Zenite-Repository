from typing import Dict, Any, List

from ai.core.llm_client import LLMClient
from ai.core.json_utils import parse_llm_json_response
from ai.core.prompt_utils import format_similar_issues, build_system_prompt
from ai.core.token_usage import coerce_token_usage

import json

AGILE_HOURS_LIMIT = 40

SYSTEM_ROLE = (
    "Você é um especialista sênior em estimativa de esforço de software por analogia, "
    "com foco em boas práticas ágeis e decomposição de trabalho."
)

INSTRUCTION = f"""
Você recebeu:
(1) o contexto completo de uma nova issue

(2) uma lista de issues históricas semelhantes, contendo informações como título, descrição,
tipo da demanda, score (entre 0 e 1) e horas consumidas.

Contexto importante do sistema:
- As issues históricas foram filtradas para manter somente exemplos com até {AGILE_HOURS_LIMIT} horas.
- Isso foi feito para alinhar o histórico às boas práticas ágeis, priorizando tarefas pequenas, refinadas e previsíveis.
- Portanto, a sua estimativa deve tender naturalmente para a faixa de 1 a {AGILE_HOURS_LIMIT} horas.

Objetivo:
Gerar uma estimativa de esforço em horas para a NOVA issue, usando principalmente o histórico.

Use o campo score para ponderar as issues semelhantes.

====================
REGRAS DE ANALOGIA
====================

1) Se existir ao menos 1 issue histórica com score >= 0.92 e título (ou termos-chave) praticamente idênticos,
trate essa issue como "âncora" e NÃO a descarte como outlier, mesmo que as horas consumidas sejam diferentes das demais.

2) Quando houver âncora:
- Faça estimated_hours ficar relativamente próximo das horas consumidas da âncora,
  a menos que existam evidências explícitas no texto da NOVA issue indicando escopo menor ou maior.
- Dê peso dominante à âncora e distribua o restante entre as próximas mais similares.

3) Quando NÃO houver âncora:
- Use agregação robusta nas top-K similares (ex.: K=8 a 15).
- Priorize weighted median ou média ponderada com corte (trim) para reduzir influência de extremos.
- O score deve ser o peso principal (peso pode crescer de forma não linear, como score^3).

4) Dê preferência a issues com:
- mesmo tipo
- labels/componentes semelhantes
- contexto técnico parecido
- escopo comparável

5) Como o histórico foi filtrado em até {AGILE_HOURS_LIMIT}h, trate essa faixa como regime normal.
Isso significa:
- tarefas pequenas e médias devem permanecer dentro desse intervalo
- não extrapole horas sem forte evidência textual
- evite superestimar por cautela genérica

====================
TRATAMENTO DE ISSUES GRANDES
====================

6) É possível que a NOVA issue pareça maior do que o padrão ágil desejado.
Considere que a issue provavelmente exige mais de {AGILE_HOURS_LIMIT}h apenas se houver evidências textuais claras, como:
- múltiplos sistemas ou módulos relevantes
- migração ampla
- refatoração extensa transversal
- rollout grande
- dependências numerosas
- investigação elevada somada à implementação
- escopo claramente épico ou agregador

7) Se concluir que a issue ultrapassa {AGILE_HOURS_LIMIT}h:
- você PODE retornar estimated_hours > {AGILE_HOURS_LIMIT}
- MAS deve deixar explícito na justification que a issue está acima do limite recomendado
- e deve orientar o usuário a refinar/quebrar a demanda em duas ou mais issues menores

8) Se não houver evidência clara de grande porte, mantenha a estimativa em até {AGILE_HOURS_LIMIT}h.

====================
CONFIDENCE
====================

9) Confidence deve refletir:
- quantidade de itens relevantes
- magnitude do score (top1 e top3)
- dispersão das horas
- clareza textual da nova issue
- presença ou ausência de âncora forte

10) Se houver poucos exemplos úteis (ex.: <4 com score >= 0.75), reduza confidence.

====================
SAÍDA
====================

Retorne APENAS um JSON válido com:
{{
  "estimated_hours": float,
  "confidence": float,
  "justification": "string curta explicando os similares usados, presença ou ausência de âncora, como ponderou, e se a issue deve ser quebrada quando ultrapassar {AGILE_HOURS_LIMIT}h"
}}

Regras finais:
- Não inclua texto fora do JSON.
- Não use markdown.
- Prefira estimativas aderentes ao ágil.
- Só ultrapasse {AGILE_HOURS_LIMIT}h com forte evidência textual.
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
        + "\n\n## NOVA ISSUE\n"
        + issue_text
        + "\n\n## ISSUES HISTORICAS SIMILARES\n"
        + format_similar_issues(similar_issues)
    )

    print("[IA][ANALOGICAL] prompt final:", prompt)

    response = llm.send_prompt(prompt, temperature=0.0, max_tokens=450)
    token_usage = coerce_token_usage(getattr(llm, "last_token_usage", None))

    try:
        parsed = parse_llm_json_response(response)

        if not isinstance(parsed, dict):
            raise ValueError("Resposta do modelo não é um JSON objeto.")

        estimated_hours = float(parsed.get("estimated_hours", 0))
        confidence = float(parsed.get("confidence", 0.0))
        justification = str(parsed.get("justification", "")).strip()

        if estimated_hours < 0:
            estimated_hours = 0.0

        confidence = max(0.0, min(1.0, confidence))

        if estimated_hours > AGILE_HOURS_LIMIT:
            split_msg = (
                f" Estimativa acima de {AGILE_HOURS_LIMIT}h; recomenda-se refinar e "
                f"quebrar a demanda em múltiplas issues menores."
            )
            if split_msg.strip() not in justification:
                justification = (justification + split_msg).strip()

        if not justification:
            justification = "Estimativa por analogia baseada nas issues históricas mais similares."

        parsed["estimated_hours"] = round(estimated_hours, 2)
        parsed["confidence"] = round(confidence, 2)
        parsed["justification"] = justification
        parsed["token_usage"] = token_usage
        return parsed

    except Exception as e:
        print(f"[IA][ANALOGICAL] erro parse: {e}")
        return {
            "estimated_hours": 0,
            "confidence": 0.0,
            "justification": "Falha ao interpretar resposta do modelo.",
            "error": str(e),
            "raw_response": response,
            "token_usage": token_usage,
        }