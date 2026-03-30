"""Tests for core data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from phantom.models import (
    AttackAction,
    AttackCategory,
    Conversation,
    Finding,
    OutcomeType,
    ProbeResult,
    Severity,
)


class TestProbeResult:
    """Tests for the ProbeResult model."""

    def test_create_with_defaults(self) -> None:
        probe = ProbeResult(
            attack_prompt="test prompt",
            response="test response",
            outcome=OutcomeType.CLEAN_REFUSAL,
            reward=0.0,
            category=AttackCategory.PROMPT_INJECTION,
        )
        assert probe.attack_prompt == "test prompt"
        assert probe.response == "test response"
        assert probe.outcome == OutcomeType.CLEAN_REFUSAL
        assert probe.reward == 0.0
        assert probe.turn_number == 1
        assert probe.probe_id is not None

    def test_reward_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ProbeResult(
                attack_prompt="x",
                response="y",
                outcome=OutcomeType.FULL_BYPASS,
                reward=1.5,
                category=AttackCategory.PROMPT_INJECTION,
            )

        with pytest.raises(ValidationError):
            ProbeResult(
                attack_prompt="x",
                response="y",
                outcome=OutcomeType.FULL_BYPASS,
                reward=-0.1,
                category=AttackCategory.PROMPT_INJECTION,
            )

    def test_serialization_roundtrip(self) -> None:
        probe = ProbeResult(
            attack_prompt="test",
            response="resp",
            outcome=OutcomeType.PARTIAL_BYPASS,
            reward=0.5,
            category=AttackCategory.GOAL_HIJACKING,
        )
        data = probe.model_dump_json()
        restored = ProbeResult.model_validate_json(data)
        assert restored.attack_prompt == probe.attack_prompt
        assert restored.outcome == probe.outcome
        assert restored.reward == probe.reward


class TestFinding:
    """Tests for the Finding model."""

    def test_create_finding(self, sample_finding: Finding) -> None:
        assert sample_finding.technique_id == "AML.T0051.000"
        assert sample_finding.severity == Severity.HIGH
        assert sample_finding.reproducibility == 0.75

    def test_reproducibility_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Finding(
                technique_id="AML.T0051",
                technique_name="Test",
                tactic="Test",
                severity=Severity.LOW,
                attack_prompt="x",
                response="y",
                reproducibility=1.5,
                remediation="fix it",
                category=AttackCategory.PROMPT_INJECTION,
            )


class TestConversation:
    """Tests for the Conversation model."""

    def test_add_turns(self) -> None:
        conv = Conversation()
        conv.add_turn("attacker", "Hello")
        conv.add_turn("target", "Hi there")
        assert conv.turn_count == 2
        assert conv.last_response == "Hi there"

    def test_last_response_no_target(self) -> None:
        conv = Conversation()
        conv.add_turn("attacker", "Hello")
        assert conv.last_response is None

    def test_empty_conversation(self) -> None:
        conv = Conversation()
        assert conv.turn_count == 0
        assert conv.last_response is None


class TestAttackAction:
    """Tests for the AttackAction model."""

    def test_create_action(self, sample_attack_action: AttackAction) -> None:
        assert sample_attack_action.mutation_operator == "synonym_replacement"
        assert sample_attack_action.strategy == "direct"
        assert sample_attack_action.escalation == 0.5

    def test_escalation_bounds(self) -> None:
        with pytest.raises(ValidationError):
            AttackAction(
                mutation_operator="test",
                strategy="direct",
                escalation=1.5,
            )


class TestSeverityEnum:
    """Tests for the Severity enum."""

    @pytest.mark.parametrize(
        "value",
        ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"],
    )
    def test_valid_severities(self, value: str) -> None:
        assert Severity(value).value == value

    def test_invalid_severity(self) -> None:
        with pytest.raises(ValueError):
            Severity("UNKNOWN")


class TestOutcomeType:
    """Tests for the OutcomeType enum."""

    @pytest.mark.parametrize(
        ("outcome", "expected"),
        [
            (OutcomeType.FULL_BYPASS, "full_bypass"),
            (OutcomeType.PARTIAL_BYPASS, "partial_bypass"),
            (OutcomeType.INFO_LEAK, "info_leak"),
            (OutcomeType.CLEAN_REFUSAL, "clean_refusal"),
            (OutcomeType.ERROR, "error"),
        ],
    )
    def test_outcome_values(self, outcome: OutcomeType, expected: str) -> None:
        assert outcome.value == expected
