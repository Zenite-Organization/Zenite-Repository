from typing import List


def extract_label_names(labels: List) -> List[str]:
    names: List[str] = []
    for lbl in labels:
        if isinstance(lbl, dict):
            name = lbl.get("name")
        elif hasattr(lbl, "name"):
            name = getattr(lbl, "name")
        else:
            name = lbl
        if name:
            names.append(str(name))
    return names
