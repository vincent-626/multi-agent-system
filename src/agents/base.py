"""BaseAgent — abstract foundation for all agents in the system."""

from abc import ABC

import src.ollama_client as ollama
from src.memory.short_term import ShortTermMemory
from src.schemas import AgentStep


class BaseAgent(ABC):
    """Abstract base class that all agents inherit from.

    Subclasses must implement :meth:`run`.

    Args:
        name:          Human-readable agent name used in trace logs.
        system_prompt: The system prompt injected into every LLM call.
        memory:        A shared :class:`~src.memory.short_term.ShortTermMemory`
                       instance for the current session.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        memory: ShortTermMemory,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.memory = memory

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with *prompt* and this agent's system prompt.

        Args:
            prompt: The user-turn message to send.

        Returns:
            The full model response string.
        """
        return ollama.chat(prompt=prompt, system=self.system_prompt)

    def _log_step(
        self,
        action: str,
        input_text: str,
        output_text: str,
        tool_used: str | None = None,
        thinking: str = "",
    ) -> AgentStep:
        """Record a step to short-term memory and print a trace line.

        Args:
            action:      Short description of what the agent did.
            input_text:  The input given for this step.
            output_text: The output produced.
            tool_used:   Optional tool name that was invoked.

        Returns:
            The created :class:`~src.schemas.AgentStep`.
        """
        step = AgentStep(
            agent=self.name,
            action=action,
            input=input_text,
            output=output_text,
            tool_used=tool_used,
            thinking=thinking,
        )
        self.memory.add(step)
        tool_note = f" [{tool_used}]" if tool_used else ""
        print(f"  [{self.name}] {action}{tool_note}")
        return step
