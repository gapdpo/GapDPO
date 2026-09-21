from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .llm_client import ChatClient
from .prompts import attacker_messages, victim_messages


@dataclass
class AttackerAgent:
    client: ChatClient
    model: str
    temperature: float
    max_tokens: int

    def next_utterance(
        self,
        victim: dict[str, Any],
        attacker: dict[str, Any],
        scenario_type: str,
        attacker_goal: str,
        protected_assets: list[str],
        history: list[dict[str, Any]],
        phase: str,
        turn_instruction: str,
        environmental_context: dict[str, str] | None = None,
        target_outcome_mode: str = "defended_success",
        pressure_style: str = "urgency",
    ) -> str:
        messages = attacker_messages(
            victim,
            attacker,
            scenario_type,
            attacker_goal,
            protected_assets,
            history,
            phase,
            turn_instruction,
            environmental_context,
            target_outcome_mode,
            pressure_style,
        )
        return self.client.chat(messages, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)


@dataclass
class VictimAgent:
    client: ChatClient
    model: str
    temperature: float
    max_tokens: int

    def next_utterance(
        self,
        victim: dict[str, Any],
        scenario_type: str,
        protected_assets: list[str],
        history: list[dict[str, Any]],
        phase: str,
        turn_instruction: str,
        response_mode: str = "defensive",
        environmental_context: dict[str, str] | None = None,
        target_outcome_mode: str = "defended_success",
    ) -> str:
        messages = victim_messages(
            victim,
            scenario_type,
            protected_assets,
            history,
            phase,
            turn_instruction,
            response_mode,
            environmental_context,
            target_outcome_mode,
        )
        return self.client.chat(messages, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)
