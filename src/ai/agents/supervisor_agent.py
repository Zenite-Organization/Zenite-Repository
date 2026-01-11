# src/agents/supervisor_agent.py
from typing import List, Dict, Any, Optional
import json

from ai.core.llm_client import LLMClient
from ai.core.prompt_utils import build_system_prompt


SYSTEM_ROLE = (
    "Você é um supervisor técnico sênior responsável por sintetizar "
    "estimativas de esforço provenientes de diferentes estratégias."
)

INSTRUCTION = """
Você receberá:
- Uma estimativa HEURISTIC (baseline, sem histórico)
- Uma estimativa ANALOGICAL (baseada em histórico real)

Regras obrigatórias:
1. A estimativa ANALOGICAL é a principal referência
2. A HEURISTIC serve como âncora e verificação de sanidade
3. Nunca extrapole fora do intervalo definido pelas estimativas recebidas
4. Se houver forte divergência, reduza a confiança final
5. Justifique claramente a decisão

Retorne APENAS um JSON válido com:
- estimate_hours (int)
- confidence (float entre 0 e 1)
- justification (string curta e objetiva)

Não inclua texto fora do JSON.
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

    # Divergência relativa
    divergence = abs(a_val - h_val) / max(h_val, 1)

    # Ponto inicial: analogical
    final_estimate = a_val

    # Se divergência for muito alta, suaviza levemente
    if divergence > 0.6:
        final_estimate = (a_val * 0.7) + (h_val * 0.3)

    final_estimate = round(final_estimate)

    # Confiança
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

    # limpeza simples
    response = response.strip().strip("```json").strip("```")

    return json.loads(response)


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

    if llm:
        return refine_with_llm(heuristic, analogical, computed, llm)

    # fallback sem LLM
    return {
        "estimate_hours": computed["estimate_hours"],
        "confidence": computed["confidence"],
        "justification": (
            f"Estimativa baseada principalmente no histórico (analogical), "
            f"validada por baseline heurístico. "
            f"Intervalo observado: {computed['bounds'][0]}–{computed['bounds'][1]}h."
        )
    }
