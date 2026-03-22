from typing import Dict, Any, List
from ai.core.llm_client import LLMClient
from ai.core.json_utils import parse_llm_json_response
from ai.core.prompt_utils import format_similar_issues, build_system_prompt
import json


SYSTEM_ROLE = (
    "Você é um especialista sênior em estimativa de esforço de software por analogia."
)

INSTRUCTION = """
Você é um especialista sênior em estimativa de esforço de software por analogia.

Você recebeu:
(1) o contexto completo de uma nova issue
(2) uma lista de issues históricas semelhantes, contendo título, descrição, tipo da demanda, labels/componentes, score (0 a 1) e horas consumidas

Objetivo:
Gerar uma estimativa de esforço em horas para a NOVA issue usando prioritariamente o histórico real mais confiável.

Princípios obrigatórios:
1. O score é a principal medida de relevância.
2. Semelhança textual sozinha NÃO basta para promover uma issue a âncora se o score não atingir o limiar definido.
3. Quando a evidência histórica for fraca, reduza a confidence de forma agressiva.
4. Prefira consistência e robustez a “médias razoáveis”.

====================
REGIMES DE DECISÃO
====================

REGIME A — ÂNCORA FORTE
Use este regime somente se existir ao menos 1 issue com:
- score >= 0.92
- mesmo tipo de demanda, ou tipo altamente compatível
- alta coerência de título, descrição e escopo

Regras:
- Trate essa issue como âncora.
- NÃO descarte essa issue como outlier.
- estimated_hours deve ficar relativamente próximo das horas consumidas da âncora.
- Dê peso dominante à âncora.
- Use as demais issues apenas para ajuste fino.

IMPORTANTE:
- NÃO promova manualmente uma issue com score < 0.92 para âncora, mesmo que pareça muito semelhante.

REGIME B — CLUSTER CONSISTENTE
Use este regime quando NÃO houver âncora forte, mas existirem pelo menos 3 issues úteis com:
- score >= 0.75
- tipo/labels/componentes compatíveis
- horas consumidas relativamente concentradas

Regras:
- Use apenas as top 3 a top 5 issues realmente úteis.
- Identifique o cluster principal de horas.
- Use mediana ponderada ou média ponderada apenas dentro do cluster principal.
- Não deixe 1 caso distante puxar a estimativa.

REGIME C — EVIDÊNCIA FRACA
Use este regime quando:
- não houver âncora forte
- houver menos de 3 issues úteis com score >= 0.75
- ou houver alta dispersão nas horas

Regras:
- Use no máximo as top 2 ou top 3 issues mais relevantes.
- Ignore similares com score < 0.70 para a decisão principal.
- Não use média ampla de muitos casos fracos.
- Se as horas forem muito dispersas, priorize a mediana ponderada dos casos mais próximos.
- Reduza confidence de forma agressiva.

====================
REGRAS GERAIS
====================

1. Dê preferência a issues com:
- mesmo tipo de demanda
- labels/componentes semelhantes
- escopo técnico semelhante

2. Penalize similaridades fracas:
- score < 0.70 não deve definir a estimativa principal
- score entre 0.70 e 0.79 só deve ser usado como apoio
- score >= 0.80 é relevante
- score >= 0.92 pode ser âncora, se houver coerência estrutural

3. Controle de dispersão:
- Se as horas das top similares variarem muito, NÃO use média simples.
- Identifique o grupo dominante e estime com base nele.
- Se não houver grupo dominante, reduza a confidence.

4. Confidence deve refletir:
- presença ou ausência de âncora forte
- quantidade de similares úteis
- score dos top similares
- compatibilidade de tipo/labels/componentes
- dispersão das horas

====================
SAÍDA
====================

Retorne APENAS um JSON válido com:
{
  "estimated_hours": float,
  "confidence": float (0..1),
  "justification": "string curta explicando qual regime foi usado, quais similares pesaram mais, se houve âncora ou cluster dominante, e por que a confiança é esse valor"
}

Regras finais:
- Não invente âncora abaixo do limiar.
- Não use muitos similares fracos para formar uma média artificial.
- Em caso de evidência fraca, prefira baixa confidence a falsa precisão.
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
