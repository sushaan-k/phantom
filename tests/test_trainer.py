"""Tests for the RL trainer."""

from __future__ import annotations

import pytest

from phantom.learner.policy import PolicyNetwork, PolicyState
from phantom.learner.trainer import RLTrainer, TrainerConfig


class TestTrainerConfig:
    """Tests for TrainerConfig."""

    def test_defaults(self) -> None:
        config = TrainerConfig()
        assert config.learning_rate == 3e-4
        assert config.gamma == 0.99
        assert config.clip_epsilon == 0.2

    def test_custom_values(self) -> None:
        config = TrainerConfig(
            learning_rate=1e-3,
            gamma=0.95,
            batch_size=64,
        )
        assert config.learning_rate == 1e-3
        assert config.batch_size == 64


class TestRLTrainer:
    """Tests for the RLTrainer."""

    @pytest.fixture
    def trainer(self) -> RLTrainer:
        policy = PolicyNetwork()
        config = TrainerConfig(batch_size=4, update_epochs=2)
        return RLTrainer(policy, config)

    def test_initial_state(self, trainer: RLTrainer) -> None:
        assert trainer.total_updates == 0
        assert trainer.buffer_size == 0

    def test_store_experience(self, trainer: RLTrainer) -> None:
        state = PolicyState()
        action_info = {
            "mutation_log_prob": -1.5,
            "strategy_log_prob": -0.8,
            "value": 0.3,
            "mutation_index": 0,
            "strategy_index": 0,
        }
        trainer.store_experience(state, action_info, 0.5, done=False)
        assert trainer.buffer_size == 1

    def test_store_experience_snapshots_state(self, trainer: RLTrainer) -> None:
        state = PolicyState(
            refusal_rate=0.1,
            mutation_history=[0.0] * 8,
            strategy_history=[0.0] * 3,
        )
        action_info = {
            "mutation_log_prob": -1.5,
            "strategy_log_prob": -0.8,
            "value": 0.3,
            "mutation_index": 0,
            "strategy_index": 0,
        }

        trainer.store_experience(state, action_info, 0.5, done=False)
        state.refusal_rate = 0.9
        state.mutation_history[0] = 1.0

        stored = trainer._buffer[0]
        assert stored.state.refusal_rate == 0.1
        assert stored.state.mutation_history[0] == 0.0

    def test_episode_reward_accounting(self, trainer: RLTrainer) -> None:
        state = PolicyState()
        action_info = {
            "mutation_log_prob": -1.0,
            "strategy_log_prob": -0.5,
            "value": 0.2,
            "mutation_index": 0,
            "strategy_index": 0,
        }

        trainer.store_experience(state, action_info, 1.0, done=False)
        trainer.store_experience(state, action_info, 2.0, done=True)
        trainer.store_experience(state, action_info, 0.5, done=False)
        trainer.store_experience(state, action_info, 0.25, done=True)

        stats = trainer.get_stats()
        assert stats["total_episodes"] == 2
        assert stats["mean_episode_reward"] == pytest.approx((3.0 + 0.75) / 2)
        assert stats["max_episode_reward"] == pytest.approx(3.0)

    def test_update_with_insufficient_buffer(self, trainer: RLTrainer) -> None:
        state = PolicyState()
        action_info = {
            "mutation_log_prob": -1.0,
            "strategy_log_prob": -0.5,
            "value": 0.2,
            "mutation_index": 0,
            "strategy_index": 0,
        }
        trainer.store_experience(state, action_info, 0.1, done=True)
        metrics = trainer.update()
        assert metrics == {}

    def test_update_with_sufficient_buffer(self, trainer: RLTrainer) -> None:
        for i in range(8):
            state = PolicyState(
                refusal_rate=0.5,
                bypass_rate=0.1 * i,
                last_reward=0.1 * i,
            )
            action_info = {
                "mutation_log_prob": -1.0 - i * 0.1,
                "strategy_log_prob": -0.5 - i * 0.1,
                "value": 0.1 * i,
                "mutation_index": i % 8,
                "strategy_index": i % 3,
            }
            trainer.store_experience(state, action_info, 0.1 * i, done=(i == 7))

        metrics = trainer.update()
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert trainer.total_updates == 1
        assert trainer.buffer_size == 0

    def test_get_stats(self, trainer: RLTrainer) -> None:
        stats = trainer.get_stats()
        assert stats["total_updates"] == 0
        assert stats["total_episodes"] == 0
        assert stats["mean_episode_reward"] == 0.0

    def test_multiple_updates(self, trainer: RLTrainer) -> None:
        for update_round in range(3):
            for i in range(8):
                state = PolicyState(
                    refusal_rate=0.5 - update_round * 0.1,
                    bypass_rate=0.1 * i,
                )
                action_info = {
                    "mutation_log_prob": -1.0,
                    "strategy_log_prob": -0.5,
                    "value": 0.1 * i,
                    "mutation_index": i % 8,
                    "strategy_index": i % 3,
                }
                trainer.store_experience(state, action_info, 0.2 * i, done=(i == 7))
            trainer.update()

        assert trainer.total_updates == 3
