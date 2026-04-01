from typing import Any, Dict, List, Optional
import json
from statistics import median

from ai.core.llm_client import LLMClient
from ai.core.json_utils import parse_llm_json_response
from ai.core.prompt_utils import build_system_prompt

AGILE_HOURS_LIMIT = 40

MODE_WEIGHTS = {
    "scope": 0.30,
    "complexity": 0.30,
    "uncertainty": 0.15,
    "agile_fit": 0.25,
}

SYSTEM_ROLE_SUPERVISOR_HEURISTIC = (
    "Voce e um supervisor tecnico senior especializado em consolidar multiplas estimativas heuristicas "
    "de software, com foco em aderencia a boas praticas ageis."
)

INSTRUCTION_SUPERVISOR_HEURISTIC = f"""
Voce recebera 4 estimativas heuristicas especializadas:
- scope
- complexity
- uncertainty
- agile_fit

Cada uma contem:
- mode
- size
- estimated_hours
- confidence
- justification

Contexto importante:
- O sistema foi recalibrado para metodologia agil.
- O intervalo normal esperado para uma issue saudavel e de 1 a {AGILE_HOURS_LIMIT} horas.
- XXL = {AGILE_HOURS_LIMIT}h.
- Acima de {AGILE_HOURS_LIMIT}h e excecao.

====================
REGRAS
====================

1. Use as 4 perspectivas como visoes complementares, nao como percentis.

2. Priorize principalmente:
- scope
- complexity
- agile_fit

3. uncertainty deve funcionar como moderador de risco,
nao como fonte principal de inflacao.

4. Prefira consolidacoes dentro de 1 a {AGILE_HOURS_LIMIT}h.

5. So aceite resultado final > {AGILE_HOURS_LIMIT}h se:
- duas ou mais estimativas apontarem para acima desse limite
- e as justificativas indicarem claramente que a demanda esta grande demais para uma unica issue

6. Se apenas uma estimativa estiver acima de {AGILE_HOURS_LIMIT}h, trate isso como alerta, nao como consenso.

7. Se houver alta dispersao:
- reduza confidence
- evite extremos injustificados

8. Se o resultado final ultrapassar {AGILE_HOURS_LIMIT}h:
- mantenha a estimativa apenas se houver evidencia forte
- a justification deve declarar que a issue excede o limite agil recomendado
- e orientar refinamento/quebra em mais de uma issue

====================
CALCULO SUGERIDO
====================

Use como referencia aproximada:
- scope: 0.30
- complexity: 0.30
- uncertainty: 0.15
- agile_fit: 0.25

Ajuste levemente conforme:
- coerencia entre as justificativas
- dispersao
- qualidade da definicao da issue

====================
CONFIDENCE
====================

Reduza confidence quando:
- alta divergencia relativa
- justificativas conflitantes
- issue mal definida
- houver sinais de escopo agregador ou mal refinado

====================
SAIDA
====================

Retorne APENAS um JSON valido:
{{
  "estimated_hours": int,
  "confidence": float,
  "justification": "curta e objetiva explicando consolidacao, dispersao e orientacao de quebra quando passar de {AGILE_HOURS_LIMIT}h"
}}

Nao inclua texto fora do JSON.
"""


def _weighted_quantile(values: List[float], weights: List[float], q: float) -> float:
    if not values:
        return 0.0

    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total_w = sum(weights) if weights else 0.0

    if total_w <= 0:
        return median(values)

    target = q * total_w
    cum = 0.0

    for v, w in pairs:
        cum += max(0.0, w)
        if cum >= target:
            return v

    return pairs[-1][0]


def _append_split_warning_if_needed(estimated_hours: int, justification: str) -> str:
    justification = (justification or "").strip()

    if estimated_hours > AGILE_HOURS_LIMIT:
        split_msg = (
            f" Estimativa acima de {AGILE_HOURS_LIMIT}h; recomenda-se refinar e "
            f"quebrar a demanda em múltiplas issues menores."
        )
        if split_msg.strip() not in justification:
            justification = (justification + split_msg).strip()

    return justification


def _collect_mode_map(estimations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mode_map: Dict[str, Dict[str, Any]] = {}

    for e in estimations:
        mode = str(e.get("mode", "")).strip().lower()
        if mode:
            mode_map[mode] = e

    return mode_map


def _safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def _compute_heuristic_ensemble_fallback(
    estimations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    if not estimations:
        return {
            "estimated_hours": 8,
            "confidence": 0.3,
            "justification": "Fallback supervisor: sem entradas heuristicas.",
        }

    mode_map = _collect_mode_map(estimations)

    weighted_sum = 0.0
    total_weight = 0.0
    values: List[float] = []
    confidence_values: List[float] = []
    over_limit_count = 0

    for mode, base_weight in MODE_WEIGHTS.items():
        e = mode_map.get(mode)
        if not e:
            continue

        hours = _safe_float(e.get("estimated_hours", 0), 0.0)
        conf = _safe_float(e.get("confidence", 0.5), 0.5)

        hours = max(0.0, hours)
        conf = max(0.0, min(1.0, conf))

        effective_weight = max(0.05, base_weight * max(0.35, conf))

        weighted_sum += hours * effective_weight
        total_weight += effective_weight
        values.append(hours)
        confidence_values.append(conf)

        if hours > AGILE_HOURS_LIMIT:
            over_limit_count += 1

    if not values or total_weight <= 0:
        return {
            "estimated_hours": 8,
            "confidence": 0.3,
            "justification": "Fallback supervisor: sem dados heurísticos válidos.",
        }

    low = min(values)
    high = max(values)
    avg_conf = sum(confidence_values) / len(confidence_values)

    weights_for_quantile = [max(0.05, c) for c in confidence_values]
    q50 = _weighted_quantile(values, weights_for_quantile, 0.50)
    q75 = _weighted_quantile(values, weights_for_quantile, 0.75)

    weighted_mean = weighted_sum / total_weight
    rel_spread = (high - low) / max(abs(q50), 1.0)

    # Mistura robusta: média ponderada + quantis
    if rel_spread <= 0.35:
        final_hours = 0.70 * weighted_mean + 0.30 * q50
        strategy = "weighted_consensus"
    elif rel_spread <= 0.90:
        final_hours = 0.55 * weighted_mean + 0.25 * q50 + 0.20 * q75
        strategy = "balanced_mix"
    else:
        final_hours = 0.40 * weighted_mean + 0.30 * q50 + 0.30 * q75
        strategy = "cautious_mix"

    # Regra de aderência ágil: >40h só com convergência real
    if over_limit_count < 2 and final_hours > AGILE_HOURS_LIMIT:
        final_hours = float(AGILE_HOURS_LIMIT)
        strategy += "_capped_agile"

    spread_penalty = min(0.35, rel_spread * 0.18)
    over_limit_penalty = 0.05 if over_limit_count == 1 else 0.0

    final_conf = avg_conf - spread_penalty - over_limit_penalty
    final_conf = max(0.20, min(0.95, final_conf))

    result = {
        "estimated_hours": int(round(final_hours)),
        "confidence": round(final_conf, 2),
        "justification": (
            "Consolidacao heuristica robusta "
            f"({strategy}); intervalo observado {round(low, 2)}-{round(high, 2)}h."
        ),
    }
    result["justification"] = _append_split_warning_if_needed(
        result["estimated_hours"],
        result["justification"],
    )
    return result


def combine_heuristic_estimations(
    estimations: List[Dict[str, Any]],
    llm: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    fallback = _compute_heuristic_ensemble_fallback(estimations)

    if not estimations:
        fallback["token_usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        return fallback

    if llm is None:
        llm = LLMClient()

    payload = {
        "heuristic_estimations": estimations,
        "fallback_suggestion": fallback,
        "agile_hours_limit": AGILE_HOURS_LIMIT,
        "mode_weights": MODE_WEIGHTS,
    }

    try:
        messages = [
            {
                "role": "system",
                "content": build_system_prompt(
                    SYSTEM_ROLE_SUPERVISOR_HEURISTIC,
                    INSTRUCTION_SUPERVISOR_HEURISTIC,
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        response = llm.invoke(messages)

        token_usage = getattr(llm, "last_token_usage", None) or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        parsed = parse_llm_json_response(response) if response else {}

        if not isinstance(parsed, dict):
            fallback["token_usage"] = token_usage
            return fallback

        estimated_hours = int(
            round(float(parsed.get("estimated_hours", fallback["estimated_hours"])))
        )
        confidence = float(parsed.get("confidence", fallback["confidence"]))
        justification = str(
            parsed.get("justification", fallback["justification"])
        ).strip()

        if estimated_hours < 0:
            estimated_hours = 0

        confidence = max(0.0, min(1.0, confidence))

        if not justification:
            justification = str(fallback["justification"])

        over_limit_count = 0
        for e in estimations:
            try:
                if float(e.get("estimated_hours", 0)) > AGILE_HOURS_LIMIT:
                    over_limit_count += 1
            except (TypeError, ValueError):
                pass

        if estimated_hours > AGILE_HOURS_LIMIT and over_limit_count < 2:
            estimated_hours = AGILE_HOURS_LIMIT

        justification = _append_split_warning_if_needed(estimated_hours, justification)

        return {
            "estimated_hours": estimated_hours,
            "confidence": round(confidence, 2),
            "justification": justification,
            "token_usage": token_usage,
        }

    except Exception:
        fallback["token_usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        return fallback