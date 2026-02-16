"""CLI runner to execute the estimation graph with DTO input."""

import argparse
import json
from typing import List

from ai.dtos.issues_estimation_dto import IssueEstimationDTO
from ai.workflows.estimation_graph import run_estimation_flow


def parse_csv_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run estimation flow with issue fields")
    parser.add_argument("--issue-number", type=int, default=1)
    parser.add_argument("--repo", default="org/repo")
    parser.add_argument("--title", default="Implementar integração com serviço externo")
    parser.add_argument("--body", default="Incluir tratamento de erros, retries e testes.")
    parser.add_argument("--labels", default="estimate,backend")
    parser.add_argument("--assignees", default="alice")
    parser.add_argument("--state", default="OPEN")
    parser.add_argument("--comments", type=int, default=0)
    parser.add_argument("--age-days", type=int, default=1)
    parser.add_argument("--author", default="bot")
    parser.add_argument("--author-role", default="NONE")
    parser.add_argument("--repo-language", default="Python")
    parser.add_argument("--repo-size", type=int, default=0)
    args = parser.parse_args()

    labels = parse_csv_list(args.labels)
    assignees = parse_csv_list(args.assignees)

    dto = IssueEstimationDTO(
        issue_number=args.issue_number,
        repository=args.repo,
        title=args.title,
        description=args.body,
        labels=labels,
        assignees=assignees,
        state=args.state,
        is_open=args.state.upper() == "OPEN",
        comments_count=args.comments,
        age_in_days=args.age_days,
        author_login=args.author,
        author_role=args.author_role,
        repo_language=args.repo_language or None,
        repo_size=args.repo_size,
        is_estimation_issue=any("estimate" in label.lower() for label in labels),
        has_assignee=len(assignees) > 0,
        has_description=bool(args.body),
    )

    result = run_estimation_flow(dto)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
