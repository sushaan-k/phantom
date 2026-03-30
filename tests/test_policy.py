"""Tests for the RL policy network."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from phantom.learner.policy import (
    MUTATION_DIM,
    MUTATION_OPERATORS,
    STATE_DIM,
    STRATEGY_DIM,
    STRATEGY_OPTIONS,
    PolicyNetwork,
    PolicyState,
)
from phantom.models import AttackCategory


class TestPolicyState:
    """Tests for PolicyState."""

    def test_default_state(self, default_policy_state: PolicyState) -> None:
        assert default_policy_state.refusal_rate == 0.0
        assert default_policy_state.bypass_rate == 0.0
        assert default_policy_state.last_reward == 0.0

    def test_to_tensor(self, default_policy_state: PolicyState) -> None:
        tensor = default_policy_state.to_tensor()
        assert tensor.shape == (STATE_DIM,)
        assert tensor.dtype == torch.float32

    def test_to_tensor_with_values(self) -> None:
        state = PolicyState(
            refusal_rate=0.5,
            bypass_rate=0.3,
            info_leak_rate=0.1,
            avg_response_length=0.7,
            turn_number=5,
            last_reward=0.8,
        )
        tensor = state.to_tensor()
        assert tensor[0] == pytest.approx(0.5)
        assert tensor[1] == pytest.approx(0.3)

    def test_to_tensor_normalized_turn(self) -> None:
        state = PolicyState(turn_number=20)
        tensor = state.to_tensor()
        assert tensor[4] <= 1.0


class TestPolicyNetwork:
    """Tests for PolicyNetwork."""

    def test_forward_shape(self, policy_network: PolicyNetwork) -> None:
        state = torch.randn(1, STATE_DIM)
        mut_probs, strat_probs, esc, value = policy_network(state)
        assert mut_probs.shape == (1, MUTATION_DIM)
        assert strat_probs.shape == (1, STRATEGY_DIM)
        assert esc.shape == (1, 1)
        assert value.shape == (1, 1)

    def test_forward_probabilities_sum_to_one(
        self, policy_network: PolicyNetwork
    ) -> None:
        state = torch.randn(1, STATE_DIM)
        mut_probs, strat_probs, _, _ = policy_network(state)
        assert mut_probs.sum().item() == pytest.approx(1.0, abs=1e-5)
        assert strat_probs.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_forward_escalation_bounded(self, policy_network: PolicyNetwork) -> None:
        state = torch.randn(1, STATE_DIM)
        _, _, esc, _ = policy_network(state)
        assert 0.0 <= esc.item() <= 1.0

    def test_forward_1d_input(self, policy_network: PolicyNetwork) -> None:
        state = torch.randn(STATE_DIM)
        mut_probs, strat_probs, esc, value = policy_network(state)
        assert mut_probs.shape == (1, MUTATION_DIM)

    def test_forward_batch(self, policy_network: PolicyNetwork) -> None:
        batch = torch.randn(8, STATE_DIM)
        mut_probs, strat_probs, esc, value = policy_network(batch)
        assert mut_probs.shape == (8, MUTATION_DIM)
        assert strat_probs.shape == (8, STRATEGY_DIM)

    def test_select_action_deterministic(self, policy_network: PolicyNetwork) -> None:
        state = PolicyState()
        action1, _ = policy_network.select_action(state, deterministic=True)
        action2, _ = policy_network.select_action(state, deterministic=True)
        assert action1.mutation_operator == action2.mutation_operator
        assert action1.strategy == action2.strategy

    def test_select_action_returns_valid_action(
        self, policy_network: PolicyNetwork
    ) -> None:
        state = PolicyState(refusal_rate=0.8, bypass_rate=0.1)
        action, info = policy_network.select_action(state)
        assert action.mutation_operator in MUTATION_OPERATORS
        assert action.strategy in STRATEGY_OPTIONS
        assert 0.0 <= action.escalation <= 1.0
        assert "mutation_probs" in info
        assert "strategy_probs" in info
        assert "value" in info

    def test_select_action_with_category(self, policy_network: PolicyNetwork) -> None:
        state = PolicyState()
        action, _ = policy_network.select_action(
            state, category=AttackCategory.DATA_EXFILTRATION
        )
        assert action.category == AttackCategory.DATA_EXFILTRATION

    def test_save_and_load(self, policy_network: PolicyNetwork) -> None:
        state = PolicyState()
        action_before, _ = policy_network.select_action(state, deterministic=True)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        policy_network.save(path)

        new_policy = PolicyNetwork()
        new_policy.load(path)

        action_after, _ = new_policy.select_action(state, deterministic=True)
        assert action_before.mutation_operator == action_after.mutation_operator
        assert action_before.strategy == action_after.strategy

        Path(path).unlink()
