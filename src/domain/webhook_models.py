from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class WebhookStatus(str, Enum):
    IGNORED = "ignored"
    PROCESSED = "processed"
    ERROR = "error"


class WebhookFlow(str, Enum):
    ESTIMATION = "estimation"
    PLANNING = "planning"
    NONE = "none"


@dataclass
class WebhookResult:
    status: WebhookStatus
    event: str
    action: str
    flow: WebhookFlow = WebhookFlow.NONE
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "event": self.event,
            "action": self.action,
            "flow": self.flow.value,
            "details": self.details,
        }
