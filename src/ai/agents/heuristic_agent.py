from typing import Dict, Any

from ai.core.llm_client import LLMClient
from ai.core.json_utils import parse_llm_json_response
from ai.core.prompt_utils import build_system_prompt
from ai.core.token_usage import coerce_token_usage

import json

AGILE_HOURS_LIMIT = 40
DEFAULT_FALLBACK_HOURS = 8

BASELINE_REFERENCE = """
Baseline oficial de horas:
- XS = 2h
- S = 4h
- M = 8h
- L = 16h
- XL = 24h
- XXL = 40h

Regra central:
- XXL = limite superior esperado para uma issue saudável.
- Acima de 40h é EXCEÇÃO.
- Só estime acima de 40h quando houver evidência textual forte de que a demanda está grande demais para uma única issue.
"""


SYSTEM_ROLE_SCOPE = (
    "Você é um analista de software sênior especializado em estimativa heurística por escopo funcional, "
    "com foco em tarefas ágeis pequenas, refinadas e previsíveis."
)

INSTRUCTION_SCOPE = f"""
Você recebeu o CONTEXTO COMPLETO de uma issue de software.

Seu papel é estimar o esforço com foco principal em ESCOPO FUNCIONAL.

{BASELINE_REFERENCE}

====================
O QUE ANALISAR
====================

Avalie principalmente:
- quantidade de entregáveis implícitos
- quantidade de requisitos ou regras mencionadas
- número de ações diferentes dentro da mesma issue
- quantidade de áreas, módulos ou partes do sistema afetadas
- se a issue parece uma entrega única ou várias entregas agrupadas

====================
CLASSIFICAÇÃO DE TAMANHO
====================

Classifique a issue em:
XS | S | M | L | XL | XXL

Escolha o size com base no tamanho funcional percebido, e não no medo de risco.

====================
CÁLCULO
====================

estimated_hours = baseline(size) × scale_factor

Para esse modo:
- scale_factor típico entre 0.90 e 1.40
- use valores perto de 1.0 quando o texto for objetivo
- só use fator alto se houver claro acúmulo de escopo funcional
- prefira manter a estimativa em até {AGILE_HOURS_LIMIT}h

====================
EXCEÇÃO > 40H
====================

Se concluir que a issue excede {AGILE_HOURS_LIMIT}h:
- você PODE retornar estimated_hours > {AGILE_HOURS_LIMIT}
- mas apenas com evidência textual forte de escopo excessivo
- a justification DEVE dizer explicitamente que a demanda está acima do limite recomendado
- e DEVE orientar quebrar/refinar em múltiplas issues menores

====================
SAÍDA
====================

Retorne APENAS um JSON válido:
{{
  "mode": "scope",
  "size": "XS|S|M|L|XL|XXL",
  "scale_factor": float,
  "estimated_hours": float,
  "confidence": float,
  "justification": "curta e objetiva"
}}

Não inclua texto fora do JSON.
"""


SYSTEM_ROLE_COMPLEXITY = (
    "Você é um analista de software sênior especializado em estimativa heurística por complexidade técnica."
)

INSTRUCTION_COMPLEXITY = f"""
Você recebeu o CONTEXTO COMPLETO de uma issue de software.

Seu papel é estimar o esforço com foco principal em COMPLEXIDADE TÉCNICA.

{BASELINE_REFERENCE}

====================
O QUE ANALISAR
====================

Avalie principalmente:
- dificuldade técnica da implementação
- integrações
- regras de negócio complexas
- impacto arquitetural
- necessidade de conhecimento especializado
- acoplamento com legado

====================
CLASSIFICAÇÃO DE TAMANHO
====================

Classifique a issue em:
XS | S | M | L | XL | XXL

Escolha o size pelo porte técnico mais plausível da implementação.

====================
CÁLCULO
====================

estimated_hours = baseline(size) × scale_factor

Para esse modo:
- scale_factor típico entre 0.90 e 1.70
- aumente o fator quando houver lógica difícil, integrações ou dependências técnicas relevantes
- não amplifique apenas porque a issue parece genérica ou mal escrita
- prefira manter a estimativa em até {AGILE_HOURS_LIMIT}h

====================
EXCEÇÃO > 40H
====================

Se concluir que a issue excede {AGILE_HOURS_LIMIT}h:
- você PODE retornar estimated_hours > {AGILE_HOURS_LIMIT}
- mas somente se a complexidade técnica realmente sustentar isso
- a justification DEVE informar que a issue ultrapassa o tamanho recomendado
- e DEVE recomendar quebra/refinamento

====================
SAÍDA
====================

Retorne APENAS um JSON válido:
{{
  "mode": "complexity",
  "size": "XS|S|M|L|XL|XXL",
  "scale_factor": float,
  "estimated_hours": float,
  "confidence": float,
  "justification": "curta e objetiva"
}}

Não inclua texto fora do JSON.
"""


SYSTEM_ROLE_UNCERTAINTY = (
    "Você é um analista de software sênior especializado em estimativa heurística por risco e incerteza."
)

INSTRUCTION_UNCERTAINTY = f"""
Você recebeu o CONTEXTO COMPLETO de uma issue de software.

Seu papel é estimar o esforço com foco principal em RISCO E INCERTEZA.

{BASELINE_REFERENCE}

====================
O QUE ANALISAR
====================

Avalie principalmente:
- ambiguidade do texto
- necessidade de investigação
- dependências externas
- dependência de terceiros
- legado e chance de retrabalho
- lacunas de definição
- risco de descoberta durante a execução

====================
CLASSIFICAÇÃO DE TAMANHO
====================

Classifique a issue em:
XS | S | M | L | XL | XXL

Primeiro escolha um size plausível da entrega.
Depois use o fator para refletir incerteza real.

====================
CÁLCULO
====================

estimated_hours = baseline(size) × scale_factor

Para esse modo:
- scale_factor típico entre 1.00 e 1.80
- use fator alto apenas quando a incerteza estiver explicitamente sustentada pelo texto
- não transforme ausência de detalhes em explosão automática de horas
- prefira manter a estimativa em até {AGILE_HOURS_LIMIT}h

====================
EXCEÇÃO > 40H
====================

Se concluir que a issue excede {AGILE_HOURS_LIMIT}h:
- você PODE retornar estimated_hours > {AGILE_HOURS_LIMIT}
- mas apenas se os riscos e as incertezas realmente indicarem demanda grande
- a justification DEVE declarar que a issue está acima do limite recomendado
- e DEVE orientar quebrar/refinar

====================
SAÍDA
====================

Retorne APENAS um JSON válido:
{{
  "mode": "uncertainty",
  "size": "XS|S|M|L|XL|XXL",
  "scale_factor": float,
  "estimated_hours": float,
  "confidence": float,
  "justification": "curta e objetiva"
}}

Não inclua texto fora do JSON.
"""


SYSTEM_ROLE_AGILE_FIT = (
    "Você é um analista de software sênior especializado em granularidade ágil e decomposição de trabalho."
)

INSTRUCTION_AGILE_FIT = f"""
Você recebeu o CONTEXTO COMPLETO de uma issue de software.

Seu papel é estimar o esforço com foco principal em ADERÊNCIA À GRANULARIDADE ÁGIL.

{BASELINE_REFERENCE}

====================
O QUE ANALISAR
====================

Avalie principalmente:
- se a issue parece refinada
- se a issue mistura descoberta + implementação + rollout
- se há sinais de épico condensado
- se a issue combina múltiplos objetivos
- se há escopo excessivo para uma única entrega ágil
- se a demanda deveria ser quebrada em mais de uma issue

====================
CLASSIFICAÇÃO DE TAMANHO
====================

Classifique a issue em:
XS | S | M | L | XL | XXL

Se a issue parecer saudável para o ágil, mantenha a classificação dentro do regime normal.
Se parecer ampla demais, isso deve aparecer tanto no size quanto na justificativa.

====================
CÁLCULO
====================

estimated_hours = baseline(size) × scale_factor

Para esse modo:
- scale_factor típico entre 0.80 e 1.50
- use fator menor quando a issue estiver bem refinada
- use fator maior quando a issue acumular trabalho demais
- prefira manter a estimativa em até {AGILE_HOURS_LIMIT}h

====================
EXCEÇÃO > 40H
====================

Se concluir que a issue excede {AGILE_HOURS_LIMIT}h:
- trate isso como exceção
- você PODE retornar estimated_hours > {AGILE_HOURS_LIMIT}
- a justification DEVE informar explicitamente que a demanda está grande demais para uma única issue
- e DEVE recomendar quebrar/refinar em múltiplas issues

====================
SAÍDA
====================

Retorne APENAS um JSON válido:
{{
  "mode": "agile_fit",
  "size": "XS|S|M|L|XL|XXL",
  "scale_factor": float,
  "estimated_hours": float,
  "confidence": float,
  "justification": "curta e objetiva"
}}

Não inclua texto fora do JSON.
"""


def _normalize_heuristic_output(parsed: Dict[str, Any], mode: str) -> Dict[str, Any]:
    if "mode" not in parsed:
        parsed["mode"] = mode

    try:
        estimated_hours = float(parsed.get("estimated_hours", DEFAULT_FALLBACK_HOURS))
    except (TypeError, ValueError):
        estimated_hours = float(DEFAULT_FALLBACK_HOURS)

    try:
        scale_factor = float(parsed.get("scale_factor", 1.0))
    except (TypeError, ValueError):
        scale_factor = 1.0

    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5

    size = str(parsed.get("size", "M")).strip().upper() or "M"
    justification = str(parsed.get("justification", "")).strip()

    allowed_sizes = {"XS", "S", "M", "L", "XL", "XXL"}
    if size not in allowed_sizes:
        size = "M"

    if estimated_hours < 0:
        estimated_hours = 0.0

    if scale_factor < 0:
        scale_factor = 0.0

    confidence = max(0.0, min(1.0, confidence))

    if estimated_hours > AGILE_HOURS_LIMIT:
        split_msg = (
            f" Estimativa acima de {AGILE_HOURS_LIMIT}h; recomenda-se refinar e "
            f"quebrar a demanda em múltiplas issues menores."
        )
        if split_msg.strip() not in justification:
            justification = (justification + split_msg).strip()

    if not justification:
        justification = "Estimativa heurística gerada com base no contexto técnico da issue."

    parsed["mode"] = mode
    parsed["size"] = size
    parsed["scale_factor"] = round(scale_factor, 2)
    parsed["estimated_hours"] = round(estimated_hours, 2)
    parsed["confidence"] = round(confidence, 2)
    parsed["justification"] = justification
    return parsed


def run_heuristic(
    issue_context: Dict[str, Any],
    llm: LLMClient,
    temperature: float = 0.0,
    mode: str = "complexity",
) -> Dict[str, Any]:
    print(f"[IA][HEURISTIC][{mode}] inicio issue={issue_context.get('issue_number')}")

    if mode == "scope":
        system_role = SYSTEM_ROLE_SCOPE
        instruction = INSTRUCTION_SCOPE
    elif mode == "complexity":
        system_role = SYSTEM_ROLE_COMPLEXITY
        instruction = INSTRUCTION_COMPLEXITY
    elif mode == "uncertainty":
        system_role = SYSTEM_ROLE_UNCERTAINTY
        instruction = INSTRUCTION_UNCERTAINTY
    elif mode == "agile_fit":
        system_role = SYSTEM_ROLE_AGILE_FIT
        instruction = INSTRUCTION_AGILE_FIT
    else:
        raise ValueError(f"Modo heurístico inválido: {mode}")

    system_prompt = build_system_prompt(system_role, instruction)
    issue_text = json.dumps(issue_context, indent=2, ensure_ascii=False)

    prompt = (
        system_prompt
        + "\n\n=== CONTEXTO DA ISSUE ===\n"
        + issue_text
    )

    response = llm.send_prompt(
        prompt,
        temperature=float(temperature),
        max_tokens=400,
    )

    token_usage = coerce_token_usage(getattr(llm, "last_token_usage", None))

    try:
        parsed = parse_llm_json_response(response)
        print(f"[IA][HEURISTIC][{mode}] retorno:", parsed)

        if not isinstance(parsed, dict):
            raise ValueError("Resposta do modelo não é um JSON objeto.")

        parsed = _normalize_heuristic_output(parsed, mode)
        parsed["token_usage"] = token_usage
        return parsed

    except Exception as e:
        print(f"[IA][HEURISTIC][{mode}] erro parse: {e}")
        return {
            "mode": mode,
            "size": "M",
            "scale_factor": 1.0,
            "estimated_hours": DEFAULT_FALLBACK_HOURS,
            "confidence": 0.3,
            "justification": "Falha ao interpretar resposta do modelo; valor padrão aplicado.",
            "error": str(e),
            "raw_response": response,
            "token_usage": token_usage,
        }