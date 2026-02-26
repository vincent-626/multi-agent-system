"""In-session (short-term) conversation memory."""

from __future__ import annotations

from src.schemas import AgentStep


class ShortTermMemory:
    """Holds the agent step history for a single conversation session.

    Not thread-safe — one instance per request.
    """

    def __init__(self) -> None:
        self._steps: list[AgentStep] = []

    def add(self, step: AgentStep) -> None:
        """Append *step* to the history.

        Args:
            step: The :class:`~src.schemas.AgentStep` to record.
        """
        self._steps.append(step)

    def get_history(self) -> list[AgentStep]:
        """Return all recorded steps in insertion order."""
        return list(self._steps)

    def format_for_prompt(self) -> str:
        """Format the step history as readable text for injection into a prompt.

        Returns:
            Multi-line string where each step is one entry, or an empty string
            when there is no history yet.
        """
        if not self._steps:
            return "(no history yet)"
        lines: list[str] = []
        for i, step in enumerate(self._steps, start=1):
            tool_note = f" [tool: {step.tool_used}]" if step.tool_used else ""
            lines.append(
                f"{i}. [{step.agent}] {step.action}{tool_note}\n"
                f"   Input:  {step.input[:300]}\n"
                f"   Output: {step.output[:300]}"
            )
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset the history for a new session."""
        self._steps = []
