# src/agents/supervisor_agent.py
from typing import List, Dict, Any
import math

def combine_estimations(estimations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combina estimativas usando média ponderada por confiança.
    estimations: lista de dicts com fields: estimate_hours, confidence, type, justification
    Retorna: final_estimate_hours, confidence (aggregate), summary
    """
    # normalizar confidences
    weighted_sum = 0.0
    weight_total = 0.0
    for e in estimations:
        est = float(e.get("estimate_hours", 0))
        conf = float(e.get("confidence", 0.5))
        # garantir limites
        conf = max(0.01, min(1.0, conf))
        weighted_sum += est * conf
        weight_total += conf

    if weight_total <= 0:
        final = sum([float(e.get("estimate_hours", 0)) for e in estimations]) / max(1, len(estimations))
        agg_conf = 0.5
    else:
        final = weighted_sum / weight_total
        # agregacao simples de confiança: raiz-mean-square das confidences
        agg_conf = math.sqrt(sum([e.get("confidence", 0.5)**2 for e in estimations]) / max(1, len(estimations)))

    # build summary
    lines = []
    for e in estimations:
        lines.append(f"- {e.get('type', 'agent')}: {e.get('estimate_hours')}h (conf {e.get('confidence')}) — {e.get('justification','')[:120]}")
    summary = f"Estimativas combinadas:\n" + "\n".join(lines)
    summary += f"\n\nDecisão final (média ponderada por confiança): {round(final,1)}h"
    return {"final_estimate_hours": round(final, 1), "confidence": round(agg_conf, 2), "summary": summary}
