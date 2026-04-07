from typing import Dict, Any, List, Tuple
from statistics import median
import math
import json
import re

from ai.core.llm_client import LLMClient
from ai.core.json_utils import parse_llm_json_response
from ai.core.prompt_utils import format_similar_issues, build_system_prompt
from ai.core.token_usage import coerce_token_usage

AGILE_HOURS_LIMIT = 40
DEFAULT_ANALOGICAL_FALLBACK_HOURS = 10.0
USEFUL_SCORE_THRESHOLD = 0.75
GOOD_SCORE_THRESHOLD = 0.85
STRONG_ANCHOR_SCORE = 0.94
GOOD_ANCHOR_SCORE = 0.90
TOP_K = 12

SYSTEM_ROLE = (
    "Você é um especialista sênior em estimativa de esforço de software por analogia, "
    "com foco em boas práticas ágeis, qualidade de retrieval e decomposição de trabalho."
)

INSTRUCTION = f"""
Você recebeu:
(1) o contexto completo de uma nova issue
(2) uma lista de issues históricas semelhantes, contendo informações como título, descrição,
tipo da demanda, score (entre 0 e 1) e horas consumidas
(3) métricas determinísticas calculadas a partir das similares

Contexto importante do sistema:
- As issues históricas foram filtradas para manter somente exemplos com até {AGILE_HOURS_LIMIT} horas.
- Isso foi feito para alinhar o histórico às boas práticas ágeis, priorizando tarefas pequenas, refinadas e previsíveis.
- Portanto, a sua estimativa deve tender naturalmente para a faixa de 1 a {AGILE_HOURS_LIMIT} horas.

Objetivo:
Gerar uma estimativa de esforço em horas para a NOVA issue, usando principalmente o histórico.

====================
REGRAS DE ANALOGIA
====================

1) Use o campo score como peso principal.
   - Scores mais altos devem ter peso significativamente maior.
   - Quando houver métricas determinísticas fornecidas, trate essas métricas como apoio forte para a decisão.

2) Se existir âncora forte:
   - âncora forte = similar com score muito alto e forte proximidade lexical/semântica com a nova issue
   - nesse caso, estimated_hours deve ficar relativamente próximo da âncora
   - só se afaste da âncora quando houver evidência textual clara de escopo maior ou menor

3) Se NÃO houver âncora forte:
   - use agregação robusta nas top-K similares
   - priorize weighted median, weighted mean com trim e coerência entre top1/top3
   - evite ser guiado por um único caso isolado

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

6) Considere que a issue provavelmente exige mais de {AGILE_HOURS_LIMIT}h apenas se houver evidências textuais claras, como:
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
- qualidade do retrieval
- top1_score
- top3_avg_score
- quantidade de exemplos úteis
- dispersão das horas
- presença ou ausência de âncora forte
- coerência entre as similares

10) Reduza confidence quando:
- top1_score estiver abaixo de 0.90
- houver poucos exemplos úteis
- as horas das similares forem muito dispersas
- a nova issue estiver pouco definida
- não houver âncora forte

11) NÃO devolva confidence alta apenas porque o texto parece convincente.
Confidence deve ser coerente com a força do histórico recuperado.

====================
SAÍDA
====================

Retorne APENAS um JSON válido com:
{{
  "estimated_hours": float,
  "confidence": float,
  "justification": "string curta explicando se houve âncora, como os similares foram ponderados, a qualidade do retrieval e se a issue deve ser quebrada quando ultrapassar {AGILE_HOURS_LIMIT}h"
}}

Regras finais:
- Não inclua texto fora do JSON.
- Não use markdown.
- Prefira estimativas aderentes ao ágil.
- Só ultrapasse {AGILE_HOURS_LIMIT}h com forte evidência textual.
- Use as métricas determinísticas como base forte de decisão, não apenas como referência decorativa.
"""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_issue_title(issue: Dict[str, Any]) -> str:
    for key in ("issue_title", "title", "summary", "name"):
        value = issue.get(key)
        if value:
            return str(value)
    return ""


def _extract_issue_description(issue: Dict[str, Any]) -> str:
    for key in ("description", "issue_description", "body", "content"):
        value = issue.get(key)
        if value:
            return str(value)
    return ""


def _extract_issue_hours(issue: Dict[str, Any]) -> float:
    for key in (
        "total_effort_hours",
        "effort_hours",
        "hours",
        "estimated_hours",
        "actual_hours",
        "consumed_hours",
    ):
        value = issue.get(key)
        if value is not None:
            return max(0.0, _safe_float(value, 0.0))
    return 0.0


def _extract_issue_score(issue: Dict[str, Any]) -> float:
    for key in ("score", "similarity", "similarity_score"):
        value = issue.get(key)
        if value is not None:
            return max(0.0, min(1.0, _safe_float(value, 0.0)))
    return 0.0


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9áàâãéèêíïóôõöúçñ\s]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    text = _normalize_text(text)
    tokens = [t for t in text.split(" ") if len(t) >= 3]
    stopwords = {
        "para", "com", "sem", "uma", "por", "das", "dos", "the", "and", "que",
        "issue", "task", "story", "bug", "feat", "fix", "this", "that", "from",
        "into", "over", "under", "refactor", "change", "update"
    }
    return [t for t in tokens if t not in stopwords]


def _jaccard_similarity(a: str, b: str) -> float:
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def _weighted_quantile(values: List[float], weights: List[float], q: float) -> float:
    if not values:
        return 0.0

    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total_w = sum(weights)

    if total_w <= 0:
        return median(values)

    target = q * total_w
    cum = 0.0

    for value, weight in pairs:
        cum += max(0.0, weight)
        if cum >= target:
            return value

    return pairs[-1][0]


def _trimmed_weighted_mean(values: List[float], weights: List[float], trim_ratio: float = 0.15) -> float:
    if not values:
        return 0.0

    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    n = len(pairs)
    trim_n = int(math.floor(n * trim_ratio))

    if n - (2 * trim_n) <= 0:
        trim_n = 0

    trimmed = pairs[trim_n:n - trim_n] if trim_n > 0 else pairs

    total_w = sum(max(0.0, w) for _, w in trimmed)
    if total_w <= 0:
        return sum(v for v, _ in trimmed) / max(1, len(trimmed))

    return sum(v * max(0.0, w) for v, w in trimmed) / total_w


def _has_large_scope_evidence(issue_context: Dict[str, Any]) -> bool:
    text = (
        _extract_issue_title(issue_context)
        + " "
        + _extract_issue_description(issue_context)
    ).lower()

    signals = [
        "migração", "migracao", "rollout", "epic", "épico", "epico",
        "refatoração ampla", "refatoracao ampla", "cross-cutting",
        "múltiplos módulos", "multiplos modulos", "multiple modules",
        "vários sistemas", "varios sistemas", "multiple systems",
        "transversal", "ampla", "wide", "investigation and implementation",
        "descoberta e implementação", "discovery and implementation",
    ]
    return any(signal in text for signal in signals)


def _build_repository_technologies_text(repository_technologies: Dict[str, float]) -> str:
    if not repository_technologies:
        return "N/A"

    sorted_items = sorted(
        repository_technologies.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    return ", ".join(f"{name}:{round(score, 3)}" for name, score in sorted_items)


def _prepare_similar_issues(similar_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared = []
    for issue in similar_issues or []:
        score = _extract_issue_score(issue)
        hours = _extract_issue_hours(issue)
        title = _extract_issue_title(issue)
        desc = _extract_issue_description(issue)

        prepared.append(
            {
                "raw": issue,
                "score": score,
                "hours": hours,
                "title": title,
                "description": desc,
            }
        )

    prepared.sort(key=lambda x: x["score"], reverse=True)
    return prepared[:TOP_K]


def _compute_retrieval_stats(
    issue_context: Dict[str, Any],
    similar_issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prepared = _prepare_similar_issues(similar_issues)
    issue_title = _extract_issue_title(issue_context)

    if not prepared:
        return {
            "top_k_count": 0,
            "useful_count": 0,
            "good_count": 0,
            "top1_score": 0.0,
            "top3_avg_score": 0.0,
            "anchor_hours": None,
            "anchor_score": 0.0,
            "anchor_overlap": 0.0,
            "has_strong_anchor": False,
            "weighted_median_hours": DEFAULT_ANALOGICAL_FALLBACK_HOURS,
            "trimmed_weighted_mean_hours": DEFAULT_ANALOGICAL_FALLBACK_HOURS,
            "deterministic_estimate": DEFAULT_ANALOGICAL_FALLBACK_HOURS,
            "hours_spread": 0.0,
        }

    scores = [item["score"] for item in prepared]
    hours = [item["hours"] for item in prepared]
    weights = [max(0.0001, item["score"] ** 4) for item in prepared]

    useful_count = sum(1 for s in scores if s >= USEFUL_SCORE_THRESHOLD)
    good_count = sum(1 for s in scores if s >= GOOD_SCORE_THRESHOLD)
    top1_score = scores[0]
    top3_avg_score = sum(scores[:3]) / max(1, min(3, len(scores)))

    best_anchor = None
    best_anchor_strength = -1.0

    for item in prepared[:5]:
        overlap = _jaccard_similarity(issue_title, item["title"])
        anchor_strength = (item["score"] * 0.75) + (overlap * 0.25)

        is_anchor_candidate = (
            (item["score"] >= STRONG_ANCHOR_SCORE and overlap >= 0.30)
            or (item["score"] >= GOOD_ANCHOR_SCORE and overlap >= 0.45)
        )

        if is_anchor_candidate and anchor_strength > best_anchor_strength:
            best_anchor = {
                "hours": item["hours"],
                "score": item["score"],
                "overlap": overlap,
                "title": item["title"],
            }
            best_anchor_strength = anchor_strength

    weighted_median_hours = _weighted_quantile(hours, weights, 0.50)
    trimmed_weighted_mean_hours = _trimmed_weighted_mean(hours, weights, trim_ratio=0.15)

    if best_anchor:
        deterministic_estimate = (
            0.62 * best_anchor["hours"]
            + 0.23 * weighted_median_hours
            + 0.15 * trimmed_weighted_mean_hours
        )
    else:
        deterministic_estimate = (
            0.60 * weighted_median_hours
            + 0.40 * trimmed_weighted_mean_hours
        )

    low = min(hours) if hours else 0.0
    high = max(hours) if hours else 0.0
    denom = max(abs(weighted_median_hours), 1.0)
    hours_spread = (high - low) / denom

    return {
        "top_k_count": len(prepared),
        "useful_count": useful_count,
        "good_count": good_count,
        "top1_score": round(top1_score, 4),
        "top3_avg_score": round(top3_avg_score, 4),
        "anchor_hours": None if not best_anchor else round(best_anchor["hours"], 2),
        "anchor_score": 0.0 if not best_anchor else round(best_anchor["score"], 4),
        "anchor_overlap": 0.0 if not best_anchor else round(best_anchor["overlap"], 4),
        "has_strong_anchor": bool(best_anchor),
        "weighted_median_hours": round(weighted_median_hours, 2),
        "trimmed_weighted_mean_hours": round(trimmed_weighted_mean_hours, 2),
        "deterministic_estimate": round(deterministic_estimate, 2),
        "hours_spread": round(hours_spread, 4),
    }


def _confidence_cap_from_stats(stats: Dict[str, Any]) -> float:
    top1 = _safe_float(stats.get("top1_score"), 0.0)
    top3 = _safe_float(stats.get("top3_avg_score"), 0.0)
    useful = int(stats.get("useful_count", 0))
    spread = _safe_float(stats.get("hours_spread"), 0.0)
    has_anchor = bool(stats.get("has_strong_anchor"))

    cap = 0.88

    if has_anchor and top1 >= 0.95 and top3 >= 0.88 and spread <= 1.0:
        cap = 0.92
    elif top1 < 0.90:
        cap = min(cap, 0.74)
    elif top1 < 0.86:
        cap = min(cap, 0.66)
    elif top1 < 0.82:
        cap = min(cap, 0.58)

    if useful < 4:
        cap = min(cap, 0.62)
    if useful < 2:
        cap = min(cap, 0.52)

    if spread > 2.2:
        cap = min(cap, 0.60)
    elif spread > 1.5:
        cap = min(cap, 0.68)

    return max(0.20, min(0.92, cap))


def _postprocess_estimate(
    llm_hours: float,
    llm_confidence: float,
    llm_justification: str,
    stats: Dict[str, Any],
    issue_context: Dict[str, Any],
) -> Tuple[float, float, str]:
    deterministic_estimate = _safe_float(stats.get("deterministic_estimate"), DEFAULT_ANALOGICAL_FALLBACK_HOURS)
    has_anchor = bool(stats.get("has_strong_anchor"))
    top1_score = _safe_float(stats.get("top1_score"), 0.0)
    useful_count = int(stats.get("useful_count", 0))
    spread = _safe_float(stats.get("hours_spread"), 0.0)
    large_scope = _has_large_scope_evidence(issue_context)

    if llm_hours <= 0:
        final_hours = deterministic_estimate
    else:
        if has_anchor:
            final_hours = (0.58 * llm_hours) + (0.42 * deterministic_estimate)
        else:
            final_hours = (0.50 * llm_hours) + (0.50 * deterministic_estimate)

    # Evita extrapolação acima do regime ágil quando não há sinal forte.
    if not large_scope and final_hours > AGILE_HOURS_LIMIT:
        final_hours = float(AGILE_HOURS_LIMIT)

    # Se retrieval for fraco, puxa mais para a estimativa robusta.
    if top1_score < 0.85 or useful_count < 3 or spread > 2.0:
        final_hours = (0.35 * llm_hours) + (0.65 * deterministic_estimate) if llm_hours > 0 else deterministic_estimate

    final_hours = max(0.0, round(final_hours, 2))

    confidence_cap = _confidence_cap_from_stats(stats)
    final_confidence = min(max(0.0, llm_confidence), confidence_cap)

    if llm_confidence <= 0:
        final_confidence = min(0.55, confidence_cap)

    justification = (llm_justification or "").strip()
    if not justification:
        if has_anchor:
            justification = (
                "Estimativa por analogia com âncora forte e agregação robusta das similares mais relevantes."
            )
        else:
            justification = (
                "Estimativa por analogia com agregação robusta das issues históricas mais similares."
            )

    if final_hours > AGILE_HOURS_LIMIT:
        split_msg = (
            f" Estimativa acima de {AGILE_HOURS_LIMIT}h; recomenda-se refinar e "
            f"quebrar a demanda em múltiplas issues menores."
        )
        if split_msg.strip() not in justification:
            justification = (justification + split_msg).strip()

    return final_hours, round(final_confidence, 2), justification


def run_analogical(
    issue_context: Dict[str, Any],
    similar_issues: List[Dict[str, Any]],
    repository_technologies: Dict[str, float],
    llm: LLMClient,
) -> Dict[str, Any]:
    system_prompt = build_system_prompt(SYSTEM_ROLE, INSTRUCTION)
    issue_text = json.dumps(issue_context, indent=2, ensure_ascii=False)
    retrieval_stats = _compute_retrieval_stats(issue_context, similar_issues)

    deterministic_context = {
        "retrieval_stats": retrieval_stats,
        "repository_technologies": _build_repository_technologies_text(repository_technologies),
        "agile_hours_limit": AGILE_HOURS_LIMIT,
    }

    prompt = (
        system_prompt
        + "\n\n## NOVA ISSUE\n"
        + issue_text
        + "\n\n## METRICAS DETERMINISTICAS DE APOIO\n"
        + json.dumps(deterministic_context, ensure_ascii=False, indent=2)
        + "\n\n## ISSUES HISTORICAS SIMILARES\n"
        + format_similar_issues(similar_issues[:TOP_K])
    )

    print("[IA][ANALOGICAL] prompt final:", prompt)

    response = llm.send_prompt(prompt, temperature=0.0, max_tokens=500)
    token_usage = coerce_token_usage(getattr(llm, "last_token_usage", None))

    try:
        parsed = parse_llm_json_response(response)

        if not isinstance(parsed, dict):
            raise ValueError("Resposta do modelo não é um JSON objeto.")

        llm_hours = _safe_float(parsed.get("estimated_hours"), 0.0)
        llm_confidence = _safe_float(parsed.get("confidence"), 0.0)
        llm_justification = str(parsed.get("justification", "")).strip()

        final_hours, final_confidence, final_justification = _postprocess_estimate(
            llm_hours=llm_hours,
            llm_confidence=llm_confidence,
            llm_justification=llm_justification,
            stats=retrieval_stats,
            issue_context=issue_context,
        )

        return {
            "estimated_hours": final_hours,
            "confidence": final_confidence,
            "justification": final_justification,
            "retrieval_stats": retrieval_stats,
            "token_usage": token_usage,
        }

    except Exception as e:
        print(f"[IA][ANALOGICAL] erro parse: {e}")

        fallback_hours, fallback_confidence, fallback_justification = _postprocess_estimate(
            llm_hours=0.0,
            llm_confidence=0.0,
            llm_justification="",
            stats=retrieval_stats,
            issue_context=issue_context,
        )

        return {
            "estimated_hours": fallback_hours,
            "confidence": fallback_confidence,
            "justification": fallback_justification,
            "retrieval_stats": retrieval_stats,
            "error": str(e),
            "raw_response": response,
            "token_usage": token_usage,
        }