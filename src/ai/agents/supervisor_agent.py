from typing import Any, Dict, List, Optional
import json

from ai.core.llm_client import LLMClient
from ai.core.json_utils import parse_llm_json_response
from ai.core.prompt_utils import build_system_prompt


SYSTEM_ROLE = (
    "Voce e um supervisor tecnico senior responsavel por sintetizar "
    "estimativas de esforco provenientes de diferentes estrategias."
)

INSTRUCTION = """
Voce recebera:
- Uma estimativa HEURISTIC (baseline, sem historico)
- Uma estimativa ANALOGICAL (baseada em historico real)

Regras obrigatorias:
1. A estimativa ANALOGICAL e a principal referencia
2. A HEURISTIC serve como ancora e verificacao de sanidade
3. Nunca extrapole fora do intervalo definido pelas estimativas recebidas
4. Se houver forte divergencia, reduza a confianca final
5. Justifique claramente a decisao

Retorne APENAS um JSON valido com:
- estimate_hours (int)
- confidence (float entre 0 e 1)
- justification (string curta e objetiva)

Nao inclua texto fora do JSON.
"""


def _split_estimations(
    estimations: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    out = {}
    for e in estimations:
        src = e.get("source")
        if src:
            out[src.lower()] = e
    return out


def _compute_final_estimate(
    heuristic: Dict[str, Any],
    analogical: Dict[str, Any]
) -> Dict[str, Any]:
    h_val = float(heuristic["estimate_hours"])
    a_val = float(analogical["estimate_hours"])

    low = min(h_val, a_val)
    high = max(h_val, a_val)

    divergence = abs(a_val - h_val) / max(h_val, 1)
    final_estimate = a_val

    if divergence > 0.6:
        final_estimate = (a_val * 0.7) + (h_val * 0.3)

    final_estimate = round(final_estimate)
    base_conf = (heuristic["confidence"] + analogical["confidence"]) / 2

    if divergence < 0.25:
        confidence = min(0.9, base_conf + 0.15)
    elif divergence < 0.5:
        confidence = base_conf
    else:
        confidence = max(0.3, base_conf - 0.2)

    return {
        "estimate_hours": int(final_estimate),
        "confidence": round(confidence, 2),
        "bounds": (low, high),
        "divergence": round(divergence, 2),
    }


def refine_with_llm(
    heuristic: Dict[str, Any],
    analogical: Dict[str, Any],
    computed: Dict[str, Any],
    llm: LLMClient
) -> Dict[str, Any]:
    payload = {
        "heuristic": heuristic,
        "analogical": analogical,
        "decision": computed,
    }

    prompt = build_system_prompt(SYSTEM_ROLE, INSTRUCTION)
    prompt += "\n\nDados:\n"
    prompt += json.dumps(payload, ensure_ascii=False)

    response = llm.send_prompt(
        prompt,
        temperature=0.0,
        max_tokens=300
    )

    parsed = parse_llm_json_response(response)
    print("[IA][SUPERVISOR] retorno llm:", parsed)
    return parsed


def combine_estimations(
    estimations: List[Dict[str, Any]],
    llm: Optional[LLMClient] = None
) -> Dict[str, Any]:
    parts = _split_estimations(estimations)

    heuristic = parts.get("heuristic")
    analogical = parts.get("analogical")

    if not heuristic or not analogical:
        raise ValueError("Supervisor requer heuristic e analogical")

    computed = _compute_final_estimate(heuristic, analogical)
    print(
        "[IA][SUPERVISOR] heuristic=%s analogical=%s computed=%s"
        % (heuristic, analogical, computed)
    )

    if llm:
        return refine_with_llm(heuristic, analogical, computed, llm)

    fallback = {
        "estimate_hours": computed["estimate_hours"],
        "confidence": computed["confidence"],
        "justification": (
            "Estimativa baseada principalmente no historico (analogical), "
            "validada por baseline heuristico. "
            f"Intervalo observado: {computed['bounds'][0]}-{computed['bounds'][1]}h."
        )
    }
    print("[IA][SUPERVISOR] retorno fallback:", fallback)
    return fallback
