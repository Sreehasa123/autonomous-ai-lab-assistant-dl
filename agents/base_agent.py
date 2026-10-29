"""Abstract base agent for specialized deep learning workflow agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict
from core.state import LabWorkflowState


class BaseLabAgent(ABC):
    """Base class for all specialized lab agents in the multi-agent system."""

    def __init__(self, name: str, role_description: str):
        self.name = name
        self.role_description = role_description

    @abstractmethod
    def execute(self, state: LabWorkflowState) -> Dict[str, Any]:
        """Execute the agent's task and return the partial state updates.

        Args:
            state: The current snapshot of the lab workflow state.

        Returns:
            Dict containing state keys to update.
        """
        pass
