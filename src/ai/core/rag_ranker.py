from collections import defaultdict
from typing import Any, Dict, List


PROJECT_WEIGHTS = {
    "mdl": 1.0,
    "mule": 0.9,
    "confserver": 0.85,
}

SOURCE_WEIGHTS = {
    "issue": 1.0,
    "comment": 0.75,
    "changelog": 0.7,
}


def blended_score(item: Dict[str, Any]) -> float:
    p_weight = PROJECT_WEIGHTS.get(item.get("project_key"), 0.8)
    s_weight = SOURCE_WEIGHTS.get(item.get("doc_type"), 0.7)
    base_score = float(item.get("rerank_score", item.get("score", 0.0)) or 0.0)
    return base_score * p_weight * s_weight


def _normalize_tokens(value: Any) -> set[str]:
    text = str(value or "").lower()
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return {token for token in cleaned.split() if len(token) >= 3}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def rerank_issue_context(items: List[Dict[str, Any]], issue_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    query_title_tokens = _normalize_tokens(issue_payload.get("title"))
    query_desc_tokens = _normalize_tokens(issue_payload.get("description"))
    query_type = str(issue_payload.get("issue_type") or "").strip().lower()
    labels_raw = issue_payload.get("labels") or []
    if isinstance(labels_raw, list):
        query_labels = {str(item).strip().lower() for item in labels_raw if str(item).strip()}
    elif isinstance(labels_raw, str):
        query_labels = {item.strip().lower() for item in labels_raw.split(",") if item.strip()}
    else:
        query_labels = set()

    ranked: List[Dict[str, Any]] = []
    for item in items or []:
        title_tokens = _normalize_tokens(item.get("title"))
        desc_tokens = _normalize_tokens(item.get("description") or item.get("snippet"))
        semantic_score = float(item.get("score") or 0.0)
        title_overlap = _jaccard(query_title_tokens, title_tokens)
        desc_overlap = _jaccard(query_desc_tokens, desc_tokens)
        item_type = str(item.get("issue_type") or "").strip().lower()
        item_labels = {
            str(label).strip().lower()
            for label in (item.get("labels") or [])
            if str(label).strip()
        }
        label_overlap = len(query_labels & item_labels)
        type_bonus = 0.06 if query_type and item_type == query_type else 0.0
        label_bonus = min(0.06, 0.03 * label_overlap)
        rerank_score = (
            (semantic_score * 0.82)
            + (title_overlap * 0.12)
            + (desc_overlap * 0.04)
            + type_bonus
            + label_bonus
        )
        enriched = dict(item)
        enriched["semantic_score"] = round(semantic_score, 4)
        enriched["title_overlap"] = round(title_overlap, 4)
        enriched["desc_overlap"] = round(desc_overlap, 4)
        enriched["label_overlap"] = label_overlap
        enriched["rerank_score"] = round(min(1.0, rerank_score), 4)
        enriched["score"] = enriched["rerank_score"]
        ranked.append(enriched)

    return sorted(ranked, key=blended_score, reverse=True)


def join_issue_context(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_issue_id: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    orphans: List[Dict[str, Any]] = []
    for item in items:
        issue_id = item.get("issue_id")
        if issue_id is None:
            orphans.append(item)
            continue
        by_issue_id[issue_id].append(item)

    assembled: List[Dict[str, Any]] = []
    for issue_id, group in by_issue_id.items():
        issues = [it for it in group if it.get("doc_type") == "issue"]
        others = [it for it in group if it.get("doc_type") != "issue"]
        issues.sort(key=blended_score, reverse=True)
        others.sort(key=blended_score, reverse=True)
        if issues:
            anchor = dict(issues[0])
            snippets = [it.get("snippet", "") for it in others[:2] if it.get("snippet")]
            if snippets:
                anchor_desc = anchor.get("description", "")
                anchor["description"] = (anchor_desc + "\n\nContext:\n" + "\n".join(snippets)).strip()
                anchor["snippet"] = anchor.get("description", "")[:400]
            assembled.append(anchor)
        else:
            # keep best non-issue when no anchor exists
            assembled.append(others[0])

    assembled.extend(orphans)
    return assembled


def assemble_context(items: List[Dict[str, Any]], final_size: int) -> List[Dict[str, Any]]:
    ranked = sorted(items, key=blended_score, reverse=True)
    issue_records = [it for it in ranked if it.get("doc_type") == "issue"]
    non_issue_records = [it for it in ranked if it.get("doc_type") != "issue"]

    # issue-first bias: reserve up to 60% for issue anchors
    desired_issue_count = min(len(issue_records), max(1, int(final_size * 0.6)))
    selected = issue_records[:desired_issue_count]
    remaining = final_size - len(selected)
    if remaining > 0:
        selected.extend(non_issue_records[:remaining])
    return selected[:final_size]
