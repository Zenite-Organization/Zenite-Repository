from typing import List


def project_namespaces(project_key: str) -> List[str]:
    base = project_key.strip().lower()
    return [base, f"{base}_comments", f"{base}_changelog"]


def parse_fallback_projects(raw: str) -> List[str]:
    return [p.strip().lower() for p in raw.split(",") if p.strip()]
