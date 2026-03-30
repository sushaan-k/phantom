"""Core data models shared across Phantom modules."""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, enum.Enum):
    """Severity level for a discovered vulnerability."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class AttackCategory(str, enum.Enum):
    """Categories of attacks that Phantom can execute."""

    PROMPT_INJECTION = "prompt_injection"
    GOAL_HIJACKING = "goal_hijacking"
    DATA_EXFILTRATION = "data_exfiltration"
    DENIAL_OF_SERVICE = "denial_of_service"


class OutcomeType(str, enum.Enum):
    """Classification of an attack probe outcome."""

    FULL_BYPASS = "full_bypass"
    PARTIAL_BYPASS = "partial_bypass"
    INFO_LEAK = "info_leak"
    CLEAN_REFUSAL = "clean_refusal"
    ERROR = "error"


class ProbeResult(BaseModel):
    """Result of a single attack probe against the target."""

    probe_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    attack_prompt: str
    response: str
    outcome: OutcomeType
    reward: float = Field(ge=0.0, le=1.0)
    category: AttackCategory
    technique_id: str | None = None
    turn_number: int = 1
    conversation_id: str | None = None
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Finding(BaseModel):
    """A confirmed vulnerability mapped to the MITRE ATLAS framework."""

    finding_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    technique_id: str
    technique_name: str
    tactic: str
    severity: Severity
    attack_prompt: str
    response: str
    reproducibility: float = Field(
        ge=0.0,
        le=1.0,
        description="Success rate over repeated trials",
    )
    remediation: str
    evidence: list[str] = Field(default_factory=list)
    category: AttackCategory
    probe_ids: list[str] = Field(default_factory=list)
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationTurn(BaseModel):
    """A single turn in a multi-turn conversation with the target."""

    role: str = Field(description="Either 'attacker' or 'target'")
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Conversation(BaseModel):
    """A multi-turn conversation session with the target."""

    conversation_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    turns: list[ConversationTurn] = Field(default_factory=list)
    category: AttackCategory = AttackCategory.PROMPT_INJECTION
    outcome: OutcomeType = OutcomeType.CLEAN_REFUSAL
    total_reward: float = 0.0

    def add_turn(self, role: str, content: str) -> None:
        """Append a turn to the conversation.

        Args:
            role: Either 'attacker' or 'target'.
            content: The message content.
        """
        self.turns.append(ConversationTurn(role=role, content=content))

    @property
    def turn_count(self) -> int:
        """Return the number of turns in this conversation."""
        return len(self.turns)

    @property
    def last_response(self) -> str | None:
        """Return the last target response, if any."""
        for turn in reversed(self.turns):
            if turn.role == "target":
                return turn.content
        return None


class AttackAction(BaseModel):
    """An action selected by the RL policy for attack generation."""

    mutation_operator: str = Field(description="Name of the mutation operator to apply")
    strategy: str = Field(
        description="Attack strategy: direct, indirect, or multi_turn"
    )
    escalation: float = Field(
        ge=0.0,
        le=1.0,
        description="Escalation intensity from 0 (cautious) to 1 (aggressive)",
    )
    category: AttackCategory = AttackCategory.PROMPT_INJECTION
    parameters: dict[str, Any] = Field(default_factory=dict)
