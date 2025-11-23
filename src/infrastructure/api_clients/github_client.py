"""API client package for GitHub-related clients.

This module re-exports the project provider implementation so callers can import
from `infrastructure.api_clients.github_client` per the desired project structure.
"""
from .github_provider import GitHubProjectProvider

__all__ = ["GitHubProjectProvider"]
