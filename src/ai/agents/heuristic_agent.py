from typing import Dict, Any
from ai.core.llm_client import LLMClient
from ai.core.json_utils import parse_llm_json_response
from ai.core.prompt_utils import build_system_prompt
import json


SYSTEM_ROLE_P25 = (
    "Você é um analista de software sênior especializado em estimativa heurística conservadora."
)

INSTRUCTION_P25 = """
Você recebeu o CONTEXTO COMPLETO de uma issue de software.

Seu papel é estimar o esforço representando um cenário OTIMISTA CONTROLADO (aproximadamente p25).
Você assume que riscos médios tendem a não se materializar.

====================
ETAPA 1 — CLASSIFICAÇÃO DE TAMANHO
====================

Classifique como:
XS | S | M | L | XL

Baseline central sugerido:
XS = 2h
S  = 5h
M  = 10h
L  = 20h
XL = 40h

====================
ETAPA 2 — FATORES
====================

Avalie de 1 a 5:
- complexidade técnica
- risco
- incerteza
- escopo (quantidade de componentes afetados)

Para p25:
- Considere apenas riscos CLAROS e explícitos.
- Não amplifique incertezas vagas.
- Use scale_factor > 1 apenas se houver evidência forte de escopo grande.

====================
ETAPA 3 — CÁLCULO
====================

estimated_hours = baseline(size) × scale_factor

Onde:
- scale_factor normalmente entre 0.8 e 1.5
- Pode ultrapassar 1.5 apenas se houver forte evidência de escopo grande.

====================
SAÍDA
====================

Retorne APENAS um JSON válido:

{
  "percentile": "p25",
  "size": "XS|S|M|L|XL",
  "scale_factor": float,
  "estimated_hours": float,
  "confidence": float (0..1),
  "justification": "curta e objetiva"
}

Não inclua texto fora do JSON.
"""

SYSTEM_ROLE_P50 = (
    "Você é um analista de software sênior especializado em estimativa heurística realista."
)

INSTRUCTION_P50 = """
Você recebeu o CONTEXTO COMPLETO de uma issue de software.

Seu papel é estimar o esforço representando o cenário MAIS PROVÁVEL (aproximadamente p50).

====================
ETAPA 1 — CLASSIFICAÇÃO
====================

XS = 2h
S  = 5h
M  = 10h
L  = 20h
XL = 40h

====================
ETAPA 2 — FATORES
====================

Avalie:
- complexidade
- dependências
- legado
- risco
- incerteza
- impacto técnico

Para p50:
- Considere riscos moderados como parcialmente prováveis.
- Use scale_factor proporcional à combinação de escopo + risco.

====================
ETAPA 3 — CÁLCULO
====================

estimated_hours = baseline(size) × scale_factor

Onde:
- scale_factor típico entre 1.0 e 2.5
- Pode ultrapassar 2.5 se o texto indicar escopo grande (migração ampla, múltiplos sistemas, rewrite, rollout extenso, etc.)

====================
SAÍDA
====================

{
  "percentile": "p50",
  "size": "...",
  "scale_factor": float,
  "estimated_hours": float,
  "confidence": float,
  "justification": "curta e objetiva"
}
"""
SYSTEM_ROLE_P75 = (
    "Você é um analista de software sênior especializado em estimativa heurística cautelosa."
)

INSTRUCTION_P75 = """
Você recebeu o CONTEXTO COMPLETO de uma issue.

Seu papel é estimar o esforço representando um cenário CAUTELOSO (aproximadamente p75).
Assuma que riscos moderados tendem a se materializar.

====================
BASELINES
====================

XS = 2h
S  = 5h
M  = 10h
L  = 20h
XL = 40h

====================
REGRAS
====================

- Considere incertezas como custo real.
- Penalize dependências externas.
- Penalize código legado.
- Se houver múltiplos componentes, aumente scale_factor.

====================
CÁLCULO
====================

estimated_hours = baseline × scale_factor

Onde:
- scale_factor típico entre 1.5 e 4.0
- Pode ultrapassar 4.0 se houver evidência de escopo amplo ou integração complexa.

====================
SAÍDA
====================

{
  "percentile": "p75",
  "size": "...",
  "scale_factor": float,
  "estimated_hours": float,
  "confidence": float,
  "justification": "curta e objetiva"
}
"""

SYSTEM_ROLE_P100 = (
    "Você é um analista de software sênior especializado em análise de pior caso plausível."
)

INSTRUCTION_P100 = """
Você recebeu o CONTEXTO COMPLETO de uma issue.

Seu papel é estimar o PIOR CASO PLAUSÍVEL (aproximadamente p100),
mas sem exageros irreais.

====================
BASELINES
====================

XS = 2h
S  = 5h
M  = 10h
L  = 20h
XL = 40h

====================
REGRAS
====================

- Considere que todos riscos moderados e altos se concretizam.
- Se houver incerteza significativa, amplifique.
- Se houver sinais de epic/migração/refatoração ampla, permita escala elevada.

====================
CÁLCULO
====================

estimated_hours = baseline × scale_factor

Onde:
- scale_factor típico entre 2.0 e 6.0
- Pode ultrapassar 6.0 apenas com forte evidência textual de grande escopo.

====================
SAÍDA
====================

{
  "percentile": "p100",
  "size": "...",
  "scale_factor": float,
  "estimated_hours": float,
  "confidence": float,
  "justification": "curta e objetiva"
}
"""

def run_heuristic(
    issue_context: Dict[str, Any],
    llm: LLMClient,
    temperature: float = 0.0,
    mode: str = "p50",   # <- novo parâmetro
) -> Dict[str, Any]:

    print(f"[IA][HEURISTIC][{mode}] inicio issue={issue_context.get('issue_number')}")

    # =========================
    # Seleção de Prompt por modo
    # =========================

    if mode == "p25":
        system_role = SYSTEM_ROLE_P25
        instruction = INSTRUCTION_P25

    elif mode == "p50":
        system_role = SYSTEM_ROLE_P50
        instruction = INSTRUCTION_P50

    elif mode == "p75":
        system_role = SYSTEM_ROLE_P75
        instruction = INSTRUCTION_P75

    elif mode == "p100":
        system_role = SYSTEM_ROLE_P100
        instruction = INSTRUCTION_P100

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
        max_tokens=350
    )

    try:
        parsed = parse_llm_json_response(response)
        print(f"[IA][HEURISTIC][{mode}] retorno:", parsed)

        # opcional: garantir que o percentile volte consistente
        if "percentile" not in parsed:
            parsed["percentile"] = mode

        return parsed

    except Exception as e:
        print(f"[IA][HEURISTIC][{mode}] erro parse: {e}")
        return {
            "percentile": mode,
            "size": "M",
            "scale_factor": 1.0,
            "estimated_hours": 8,
            "confidence": 0.3,
            "justification": "Falha ao interpretar resposta do modelo; valor padrão aplicado.",
            "error": str(e),
            "raw_response": response
        }
