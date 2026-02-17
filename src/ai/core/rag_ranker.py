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
    return float(item.get("score", 0.0)) * p_weight * s_weight


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
