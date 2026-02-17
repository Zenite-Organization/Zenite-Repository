from typing import Iterable

from domain.webhook_models import WebhookFlow


SUPPORTED_EVENT = "issues"
SUPPORTED_ACTIONS = {"opened", "edited", "labeled"}
ESTIMATION_LABEL = "Estimate"
PLANNING_LABEL = "Planning"


def normalize_labels(labels: Iterable[str]) -> set[str]:
    return {str(label).strip() for label in labels if str(label).strip()}


def decide_flow(event: str, action: str, labels: Iterable[str]) -> WebhookFlow:
    if event != SUPPORTED_EVENT:
        return WebhookFlow.NONE
    if action not in SUPPORTED_ACTIONS:
        return WebhookFlow.NONE

    label_set = normalize_labels(labels)
    if PLANNING_LABEL in label_set:
        return WebhookFlow.PLANNING
    if ESTIMATION_LABEL in label_set:
        return WebhookFlow.ESTIMATION
    return WebhookFlow.NONE
