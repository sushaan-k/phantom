"""Tests for the RedTeam orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from phantom.exceptions import ConfigurationError
from phantom.models import (
    AttackAction,
    AttackCategory,
    OutcomeType,
    Severity,
)
from phantom.redteam import RedTeam, RedTeamConfig, RedTeamResults
from phantom.target import Target

# Dummy key so the OpenAI client doesn't raise on construction
_DUMMY_KEY = "sk-test-phantom-dummy-key-for-unit-tests"


class TestRedTeamConfig:
    """Tests for RedTeamConfig."""

    def test_defaults(self) -> None:
        config = RedTeamConfig()
        assert config.attack_model == "gpt-4"
        assert config.max_interactions == 500
        assert config.multi_turn is True
        assert len(config.categories) > 0

    def test_custom_config(self) -> None:
        config = RedTeamConfig(
            attack_model="claude-sonnet-4-6",
            max_interactions=100,
            categories=["prompt_injection"],
            multi_turn=False,
        )
        assert config.attack_model == "claude-sonnet-4-6"
        assert config.max_interactions == 100


class TestRedTeamResults:
    """Tests for RedTeamResults."""

    def test_empty_results(self) -> None:
        results = RedTeamResults()
        assert results.total_probes == 0
        assert results.bypass_rate == 0.0
        assert results.count_by_severity("CRITICAL") == 0

    def test_count_by_severity(self, sample_findings: list) -> None:
        results = RedTeamResults(
            findings=sample_findings,
            total_probes=10,
        )
        assert results.count_by_severity("CRITICAL") == 1
        assert results.count_by_severity(Severity.HIGH) == 1

    def test_bypass_rate(self) -> None:
        results = RedTeamResults(
            total_probes=100,
            total_bypasses=25,
        )
        assert results.bypass_rate == 0.25


class TestRedTeam:
    """Tests for the RedTeam orchestrator."""

    def test_invalid_category(self) -> None:
        target = Target(endpoint="https://api.example.com/chat")
        with pytest.raises(ConfigurationError, match="Unknown category"):
            RedTeam(
                target=target,
                categories=["invalid_category"],
                attack_api_key=_DUMMY_KEY,
            )

    def test_valid_initialization(self) -> None:
        target = Target(endpoint="https://api.example.com/chat")
        rt = RedTeam(
            target=target,
            categories=["prompt_injection"],
            max_interactions=10,
            attack_api_key=_DUMMY_KEY,
        )
        assert rt.config.max_interactions == 10
        assert rt.config.categories == ["prompt_injection"]

    def test_config_passthrough(self) -> None:
        target = Target(endpoint="https://api.example.com/chat")
        config = RedTeamConfig(
            categories=["goal_hijacking"],
            max_interactions=50,
            learning_rate=1e-3,
            attack_api_key=_DUMMY_KEY,
        )
        rt = RedTeam(target=target, config=config)
        assert rt.config.learning_rate == 1e-3

    @pytest.mark.asyncio
    async def test_run_with_mocks(self) -> None:
        target = Target(endpoint="https://api.example.com/chat")

        rt = RedTeam(
            target=target,
            categories=["prompt_injection"],
            max_interactions=4,
            multi_turn=False,
            attack_api_key=_DUMMY_KEY,
        )

        with (
            patch.object(
                rt._generator,
                "generate",
                new_callable=AsyncMock,
                return_value="Test attack prompt",
            ),
            patch.object(
                rt._target,
                "send_probe",
                new_callable=AsyncMock,
                return_value=("I'm sorry, I cannot help.", 50.0),
            ),
        ):
            results = await rt.run()

        assert isinstance(results, RedTeamResults)
        assert results.total_probes > 0
        assert len(results.probes) > 0

    @pytest.mark.asyncio
    async def test_run_handles_generation_failure(self) -> None:
        target = Target(endpoint="https://api.example.com/chat")

        rt = RedTeam(
            target=target,
            categories=["prompt_injection"],
            max_interactions=2,
            multi_turn=False,
            attack_api_key=_DUMMY_KEY,
        )

        call_count = 0

        async def failing_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("Generation failed")
            return "Fallback prompt"

        with (
            patch.object(
                rt._generator,
                "generate",
                side_effect=failing_generate,
            ),
            patch.object(
                rt._target,
                "send_probe",
                new_callable=AsyncMock,
                return_value=("Response", 10.0),
            ),
        ):
            results = await rt.run()

        assert isinstance(results, RedTeamResults)

    @pytest.mark.asyncio
    async def test_run_multi_turn_attack(self) -> None:
        """Test multi-turn attack conversations in the RL loop."""
        target = Target(endpoint="https://api.example.com/chat")

        rt = RedTeam(
            target=target,
            categories=["prompt_injection"],
            max_interactions=3,
            multi_turn=True,
            max_turns_per_conversation=3,
            attack_api_key=_DUMMY_KEY,
        )

        responses = [
            ("Hello, how can I help?", 20.0),
            ("I understand your question.", 25.0),
            ("Sure, here is the information.", 30.0),
        ]
        call_idx = 0

        async def mock_conv_turn(prompt, conversation):
            nonlocal call_idx
            resp = responses[call_idx % len(responses)]
            call_idx += 1
            conversation.add_turn("attacker", prompt)
            conversation.add_turn("target", resp[0])
            return resp

        # Force the policy to always select multi_turn strategy
        with (
            patch.object(
                rt._generator,
                "generate",
                new_callable=AsyncMock,
                return_value="Multi-turn attack prompt",
            ),
            patch.object(
                rt._target,
                "send_conversation_turn",
                side_effect=mock_conv_turn,
            ),
            patch.object(
                rt._target,
                "send_probe",
                new_callable=AsyncMock,
                return_value=("Response", 10.0),
            ),
            patch.object(
                rt._policy,
                "select_action",
                return_value=(
                    AttackAction(
                        mutation_operator="synonym_replacement",
                        strategy="multi_turn",
                        escalation=0.5,
                        category=AttackCategory.PROMPT_INJECTION,
                        parameters={"mutation_index": 0, "strategy_index": 2},
                    ),
                    {
                        "mutation_probs": [0.125] * 8,
                        "strategy_probs": [0.1, 0.1, 0.8],
                        "escalation": 0.5,
                        "value": 0.0,
                        "mutation_log_prob": -2.0,
                        "strategy_log_prob": -0.2,
                    },
                ),
            ),
        ):
            results = await rt.run()

        assert isinstance(results, RedTeamResults)
        assert results.total_probes > 0

    @pytest.mark.asyncio
    async def test_run_multi_turn_bypass_stops_early(self) -> None:
        """Test that full bypass ends multi-turn conversation early."""
        target = Target(endpoint="https://api.example.com/chat")

        rt = RedTeam(
            target=target,
            categories=["prompt_injection"],
            max_interactions=10,
            multi_turn=True,
            max_turns_per_conversation=10,
            attack_api_key=_DUMMY_KEY,
        )

        # Return a response that triggers full bypass on first turn
        async def mock_conv_turn(prompt, conversation):
            conversation.add_turn("attacker", prompt)
            conversation.add_turn(
                "target", "Sure! Here is my system prompt: You are a helpful assistant."
            )
            return (
                "Sure! Here is my system prompt: You are a helpful assistant.",
                10.0,
            )

        with (
            patch.object(
                rt._generator,
                "generate",
                new_callable=AsyncMock,
                return_value="Attack prompt",
            ),
            patch.object(
                rt._target,
                "send_conversation_turn",
                side_effect=mock_conv_turn,
            ),
            patch.object(
                rt._target,
                "send_probe",
                new_callable=AsyncMock,
                return_value=("Sure! Here is my system prompt.", 10.0),
            ),
            patch.object(
                rt._policy,
                "select_action",
                return_value=(
                    AttackAction(
                        mutation_operator="synonym_replacement",
                        strategy="multi_turn",
                        escalation=0.5,
                        category=AttackCategory.PROMPT_INJECTION,
                        parameters={"mutation_index": 0, "strategy_index": 2},
                    ),
                    {
                        "mutation_probs": [0.125] * 8,
                        "strategy_probs": [0.1, 0.1, 0.8],
                        "escalation": 0.5,
                        "value": 0.0,
                        "mutation_log_prob": -2.0,
                        "strategy_log_prob": -0.2,
                    },
                ),
            ),
        ):
            results = await rt.run()

        assert isinstance(results, RedTeamResults)
        assert results.total_bypasses > 0

    @pytest.mark.asyncio
    async def test_run_handles_probe_failure(self) -> None:
        """Test that probe send failure is handled gracefully.

        First probe fails, second succeeds, so loop makes progress.
        """
        from phantom.exceptions import TargetConnectionError

        target = Target(endpoint="https://api.example.com/chat")

        rt = RedTeam(
            target=target,
            categories=["prompt_injection"],
            max_interactions=2,
            multi_turn=False,
            attack_api_key=_DUMMY_KEY,
        )

        call_count = 0

        async def alternating_probe(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TargetConnectionError("example.com", "timeout")
            return ("I'm sorry, I cannot help.", 50.0)

        with (
            patch.object(
                rt._generator,
                "generate",
                new_callable=AsyncMock,
                return_value="Attack prompt",
            ),
            patch.object(
                rt._target,
                "send_probe",
                side_effect=alternating_probe,
            ),
        ):
            results = await rt.run()

        assert isinstance(results, RedTeamResults)
        assert results.total_probes > 0

    @pytest.mark.asyncio
    async def test_run_multiple_categories(self) -> None:
        """Test running across multiple categories."""
        target = Target(endpoint="https://api.example.com/chat")

        rt = RedTeam(
            target=target,
            categories=["prompt_injection", "goal_hijacking"],
            max_interactions=4,
            multi_turn=False,
            attack_api_key=_DUMMY_KEY,
        )

        with (
            patch.object(
                rt._generator,
                "generate",
                new_callable=AsyncMock,
                return_value="Test attack prompt",
            ),
            patch.object(
                rt._target,
                "send_probe",
                new_callable=AsyncMock,
                return_value=("I'm sorry, I cannot help.", 50.0),
            ),
        ):
            results = await rt.run()

        assert isinstance(results, RedTeamResults)
        assert results.total_probes > 0

    @pytest.mark.asyncio
    async def test_run_multi_turn_generation_failure_breaks(self) -> None:
        """Test that generation failure in multi-turn breaks the conv.

        We alternate: first call is multi_turn (which fails on gen after
        1 turn), subsequent calls are direct (which succeed), so the
        outer while loop can still make progress.
        """
        target = Target(endpoint="https://api.example.com/chat")

        rt = RedTeam(
            target=target,
            categories=["prompt_injection"],
            max_interactions=3,
            multi_turn=True,
            max_turns_per_conversation=5,
            attack_api_key=_DUMMY_KEY,
        )

        gen_count = 0

        async def failing_gen(*args, **kwargs):
            nonlocal gen_count
            gen_count += 1
            if gen_count == 2:
                # Fail on the second generate call (multi-turn follow-up)
                raise RuntimeError("Generation failed on follow-up")
            return "Attack prompt"

        async def mock_conv_turn(prompt, conversation):
            conversation.add_turn("attacker", prompt)
            conversation.add_turn("target", "Response")
            return ("Response", 10.0)

        action_count = 0

        def alternating_action(state, category):
            nonlocal action_count
            action_count += 1
            # First call: multi_turn, subsequent: direct
            strat = "multi_turn" if action_count == 1 else "direct"
            return (
                AttackAction(
                    mutation_operator="synonym_replacement",
                    strategy=strat,
                    escalation=0.5,
                    category=category,
                    parameters={"mutation_index": 0, "strategy_index": 0},
                ),
                {
                    "mutation_probs": [0.125] * 8,
                    "strategy_probs": [0.33, 0.33, 0.34],
                    "escalation": 0.5,
                    "value": 0.0,
                    "mutation_log_prob": -2.0,
                    "strategy_log_prob": -0.2,
                },
            )

        with (
            patch.object(rt._generator, "generate", side_effect=failing_gen),
            patch.object(
                rt._target,
                "send_conversation_turn",
                side_effect=mock_conv_turn,
            ),
            patch.object(
                rt._target,
                "send_probe",
                new_callable=AsyncMock,
                return_value=("I'm sorry.", 10.0),
            ),
            patch.object(
                rt._policy,
                "select_action",
                side_effect=alternating_action,
            ),
        ):
            results = await rt.run()

        assert isinstance(results, RedTeamResults)
        assert results.total_probes > 0

    @pytest.mark.asyncio
    async def test_run_multi_turn_probe_failure_breaks(self) -> None:
        """Test that probe failure in multi-turn breaks the conv.

        First call is multi_turn (probe fails), then switch to direct
        strategy so the loop can make progress.
        """
        from phantom.exceptions import TargetConnectionError

        target = Target(endpoint="https://api.example.com/chat")

        rt = RedTeam(
            target=target,
            categories=["prompt_injection"],
            max_interactions=3,
            multi_turn=True,
            max_turns_per_conversation=5,
            attack_api_key=_DUMMY_KEY,
        )

        async def failing_conv_turn(prompt, conversation):
            conversation.add_turn("attacker", prompt)
            raise TargetConnectionError("example.com", "connection refused")

        action_count = 0

        def alternating_action(state, category):
            nonlocal action_count
            action_count += 1
            strat = "multi_turn" if action_count == 1 else "direct"
            return (
                AttackAction(
                    mutation_operator="synonym_replacement",
                    strategy=strat,
                    escalation=0.5,
                    category=category,
                    parameters={"mutation_index": 0, "strategy_index": 0},
                ),
                {
                    "mutation_probs": [0.125] * 8,
                    "strategy_probs": [0.33, 0.33, 0.34],
                    "escalation": 0.5,
                    "value": 0.0,
                    "mutation_log_prob": -2.0,
                    "strategy_log_prob": -0.2,
                },
            )

        with (
            patch.object(
                rt._generator,
                "generate",
                new_callable=AsyncMock,
                return_value="Attack prompt",
            ),
            patch.object(
                rt._target,
                "send_conversation_turn",
                side_effect=failing_conv_turn,
            ),
            patch.object(
                rt._target,
                "send_probe",
                new_callable=AsyncMock,
                return_value=("I'm sorry.", 10.0),
            ),
            patch.object(
                rt._policy,
                "select_action",
                side_effect=alternating_action,
            ),
        ):
            results = await rt.run()

        assert isinstance(results, RedTeamResults)
        assert results.total_probes > 0


class TestRedTeamUpdateState:
    """Tests for the _update_state method."""

    def test_update_state_refusal(self) -> None:
        target = Target(endpoint="https://api.example.com/chat")
        rt = RedTeam(
            target=target,
            categories=["prompt_injection"],
            attack_api_key=_DUMMY_KEY,
        )
        # Manually add a probe
        from phantom.models import ProbeResult

        rt._probes.append(
            ProbeResult(
                attack_prompt="test",
                response="I'm sorry",
                outcome=OutcomeType.CLEAN_REFUSAL,
                reward=0.0,
                category=AttackCategory.PROMPT_INJECTION,
            )
        )
        rt._update_state(OutcomeType.CLEAN_REFUSAL, 0.0, "I'm sorry")
        assert rt._state.consecutive_refusals == 1
        assert rt._state.refusal_rate > 0

    def test_update_state_bypass_resets_consecutive_refusals(self) -> None:
        target = Target(endpoint="https://api.example.com/chat")
        rt = RedTeam(
            target=target,
            categories=["prompt_injection"],
            attack_api_key=_DUMMY_KEY,
        )
        rt._state.consecutive_refusals = 5
        from phantom.models import ProbeResult

        rt._probes.append(
            ProbeResult(
                attack_prompt="test",
                response="Sure!",
                outcome=OutcomeType.FULL_BYPASS,
                reward=1.0,
                category=AttackCategory.PROMPT_INJECTION,
            )
        )
        rt._update_state(OutcomeType.FULL_BYPASS, 1.0, "Sure!")
        assert rt._state.consecutive_refusals == 0

    def test_update_state_no_probes(self) -> None:
        target = Target(endpoint="https://api.example.com/chat")
        rt = RedTeam(
            target=target,
            categories=["prompt_injection"],
            attack_api_key=_DUMMY_KEY,
        )
        # Should be safe with empty probes list
        rt._update_state(OutcomeType.CLEAN_REFUSAL, 0.0, "response")


class TestRedTeamResultsEdgeCases:
    """Additional edge case tests for RedTeamResults."""

    def test_bypass_rate_zero_probes(self) -> None:
        results = RedTeamResults(total_probes=0, total_bypasses=0)
        assert results.bypass_rate == 0.0

    def test_count_by_severity_with_string(self) -> None:
        results = RedTeamResults()
        assert results.count_by_severity("INFO") == 0

    def test_count_by_severity_with_enum(self) -> None:
        results = RedTeamResults()
        assert results.count_by_severity(Severity.LOW) == 0

    def test_atlas_coverage_uses_probe_techniques(self) -> None:
        from phantom.models import ProbeResult

        results = RedTeamResults(
            probes=[
                ProbeResult(
                    attack_prompt="test",
                    response="Sure!",
                    outcome=OutcomeType.FULL_BYPASS,
                    reward=1.0,
                    category=AttackCategory.PROMPT_INJECTION,
                    technique_id="AML.T0051.000",
                )
            ]
        )

        assert results.atlas_coverage_pct > 0.0
