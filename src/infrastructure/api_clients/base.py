from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional


class ProjectProvider(ABC):
    """Abstraction for project systems (e.g. GitHub Projects V2).

    Implementations should be async and raise exceptions on failures.
    """

    # @abstractmethod
    # async def add_item_by_issue(self, repo_full_name: str, issue_number: int) -> Optional[str]:
    #     """Associate an existing issue to a project and return the project item id.

    #     Args:
    #         repo_full_name: "owner/repo"
    #         issue_number: Issue number in the repository

    #     Returns:
    #         The created/linked project's item id (node id) or None.
    #     """

    # @abstractmethod
    # async def set_item_field(self, project_id: str, item_id: str, field_id: str, value: Any) -> Any:
    #     """Set a field value for a project item.

    #     `value` should be passed in the shape expected by the provider (for GitHub GraphQL,
    #     a dict matching the field type's input).
    #     """

    # @abstractmethod
    # async def clear_item_field(self, project_id: str, item_id: str, field_id: str) -> Any:
    #     """Clear a field value for a project item."""
