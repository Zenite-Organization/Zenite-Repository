from typing import Any, Dict, List, Optional
import json
from statistics import median
from ai.core.llm_client import LLMClient
from ai.core.json_utils import parse_llm_json_response
from ai.core.prompt_utils import build_system_prompt



SYSTEM_ROLE_SUPERVISOR_HEURISTIC = (
    "Voce e um supervisor tecnico senior especializado em consolidar multiplas estimativas heuristicas."
)

INSTRUCTION_SUPERVISOR_HEURISTIC = """
Voce recebera 4 estimativas heuristicas:
- p25
- p50
- p75
- p100

Cada uma contem:
- estimated_hours
- confidence
- justification

====================
REGRAS
====================

1. Garanta monotonicidade:
   p25 <= p50 <= p75 <= p100
   Se nao estiver ordenado, reordene e reduza confidence final.

2. Use p50 como referencia principal.

3. Se dispersao for baixa:
   -> final proximo de p50.

4. Se dispersao for moderada:
   -> incline levemente para p75.

5. Se dispersao for alta:
   -> reduza confidence.
   -> evite extremos injustificados.

6. p100 representa pior caso plausivel, nao deve dominar salvo evidencia forte.

====================
CALCULO SUGERIDO
====================

final_estimate ~=
0.60*p50 + 0.30*p75 + 0.10*p25

Ajuste levemente conforme coerencia das justificativas.

====================
CONFIDENCE
====================

Reduza confidence quando:
- Alta divergencia relativa
- Justificativas conflitantes
- Issue mal definida

====================
SAIDA
====================

Retorne APENAS um JSON valido:

{
  "estimated_hours": int,
  "confidence": float (0..1),
  "justification": "curta e objetiva explicando consolidacao e dispersao"
}

Nao inclua texto fora do JSON.
"""



def _weighted_quantile(values: List[float], weights: List[float], q: float) -> float:
    """Quantil ponderado simples (q em [0,1])."""
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


def _compute_heuristic_ensemble_fallback(
    estimations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    values: List[float] = []
    confidences: List[float] = []

    for e in estimations:
        # hours
        try:
            v = float(e.get("estimated_hours", 0))
        except (TypeError, ValueError):
            v = 0.0
        if v < 0:
            v = 0.0
        values.append(v)

        # confidence
        try:
            c = float(e.get("confidence", 0.5))
        except (TypeError, ValueError):
            c = 0.5
        confidences.append(max(0.0, min(1.0, c)))

    if not values:
        return {
            "estimated_hours": 8,
            "confidence": 0.3,
            "justification": "Fallback supervisor: sem entradas heuristicas.",
        }

    # Ordena por horas (facilita coerencia)
    pairs = sorted(zip(values, confidences), key=lambda x: x[0])
    values = [p[0] for p in pairs]
    confidences = [p[1] for p in pairs]

    low = values[0]
    high = values[-1]
    spread = high - low

    # Base confidence = media
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.5

    # Pesos minimos para evitar zerar contribuicoes
    weights = [max(0.05, c) for c in confidences]

    # Quantis ponderados (aproxima p25/p50/p75 do ensemble)
    q25 = _weighted_quantile(values, weights, 0.25)
    q50 = _weighted_quantile(values, weights, 0.50)
    q75 = _weighted_quantile(values, weights, 0.75)

    # Heuristica "cap p100" (anti-explosao):
    # se o maior valor estiver muito distante do corpo (q75), ele vira pior-caso e nao entra forte na consolidacao.
    # Nao muda as entradas, so evita que 1 retorno extremo domine.
    # p100_eff usado so implicitamente via "high_eff".
    high_eff = high
    if q75 > 0:
        if high > 2.5 * q75:
            high_eff = 2.5 * q75
    else:
        # se q75 ~0 e high alto, claramente e extremo
        if high > 20:
            high_eff = 20.0

    # Recalcula spread efetivo para penalidades (nao para "intervalo observado")
    spread_eff = max(0.0, high_eff - low)

    # Dispersao relativa (com base no q50)
    denom = max(abs(q50), 1.0)
    rel_spread = spread_eff / denom

    # =========================
    # REGIME A: small/normal
    # =========================
    # Se o proprio ensemble indica tarefa pequena:
    # - nao puxe para q75
    # - use q50 quase puro
    small_regime = (q50 <= 8.0 and q75 <= 12.0)

    if small_regime:
        final_hours = q50
        strategy = "small_regime_p50"
        # confianca tende a subir se dispersao for pequena
        small_bonus = 0.05 if rel_spread <= 0.5 else 0.0
    else:
        small_bonus = 0.0

        # =========================
        # REGIME B: geral (misto)
        # =========================
        # Mistura p50/p75 conforme dispersao (sem deixar extremo dominar)
        if rel_spread <= 0.35:
            final_hours = q50
            strategy = "p50"
        elif rel_spread <= 0.90:
            final_hours = 0.65 * q50 + 0.35 * q75
            strategy = "mix_p50_p75"
        else:
            final_hours = 0.50 * q50 + 0.50 * q75
            strategy = "cautious_mix"

    # Penalidade de confianca por dispersao (capada)
    spread_penalty = min(0.35, rel_spread * 0.18)

    # Penalidade adicional se houver "cauda" muito grande (high muito acima de q75)
    tail_penalty = 0.0
    if q75 > 0 and high > 3.0 * q75:
        tail_penalty = 0.07

    final_conf = avg_conf - spread_penalty - tail_penalty + small_bonus
    final_conf = max(0.20, min(0.95, final_conf))

    return {
        "estimated_hours": int(round(final_hours)),
        "confidence": round(final_conf, 2),
        "justification": (
            "Consolidacao heuristica robusta "
            f"({strategy}); intervalo observado {round(low, 2)}-{round(high, 2)}h."
        ),
    }

def combine_heuristic_estimations(
    estimations: List[Dict[str, Any]],
    llm: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """
    Consolida estimativas heuristicas via LLM e aplica fallback deterministico
    quando houver erro, resposta invalida ou ausencia de modelo.
    """
    fallback = _compute_heuristic_ensemble_fallback(estimations)

    if not estimations:
        return fallback

    if llm is None:
        llm = LLMClient()

    payload = {
        "heuristic_estimations": estimations,
        "fallback_suggestion": fallback,
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
        parsed = parse_llm_json_response(response) if response else {}
        if not isinstance(parsed, dict):
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

        return {
            "estimated_hours": estimated_hours,
            "confidence": round(confidence, 2),
            "justification": justification,
        }
    except Exception:
        return fallback


