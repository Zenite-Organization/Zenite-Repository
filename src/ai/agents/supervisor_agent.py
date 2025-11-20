# src/agents/supervisor_agent.py
from typing import List, Dict, Any
import math

def combine_estimations(estimations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combina todas as estimativas recebidas, retornando a média simples dos campos estimate_hours e confidence,
    e a justification do primeiro item.
    """
    print("Combining estimations:", estimations)  # Para depuração
    if not estimations:
        return {"estimate_hours": 0.0, "confidence": 0.5, "justification": "Nenhuma estimativa disponível."}
    total_hours = sum(float(e.get("estimate_hours", 0)) for e in estimations)
    total_conf = sum(float(e.get("confidence", 0.5)) for e in estimations)
    n = len(estimations)
    avg_hours = total_hours / n
    avg_conf = total_conf / n
    justification = str(estimations[0].get("justification", ""))
    return {
        "estimate_hours": round(avg_hours, 2),
        "confidence": round(avg_conf, 2),
        "justification": justification
    }
