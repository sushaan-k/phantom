"""Training loop for the RL strategy learner."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from phantom.learner.policy import PolicyNetwork, PolicyState
from phantom.logging import get_logger

logger = get_logger("phantom.learner.trainer")


class TrainerConfig(BaseModel):
    """Configuration for the RL trainer."""

    learning_rate: float = Field(
        default=3e-4, gt=0.0, description="Learning rate for the optimizer"
    )
    gamma: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Discount factor for future rewards",
    )
    gae_lambda: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="GAE lambda for advantage estimation",
    )
    clip_epsilon: float = Field(
        default=0.2,
        gt=0.0,
        description="PPO clipping parameter",
    )
    value_loss_coef: float = Field(
        default=0.5,
        gt=0.0,
        description="Coefficient for the value loss term",
    )
    entropy_coef: float = Field(
        default=0.01,
        ge=0.0,
        description="Coefficient for entropy bonus (exploration)",
    )
    max_grad_norm: float = Field(
        default=0.5,
        gt=0.0,
        description="Maximum gradient norm for clipping",
    )
    update_epochs: int = Field(
        default=4,
        ge=1,
        description="Number of optimization epochs per update",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Minibatch size for updates",
    )


class Experience:
    """A single step of experience for the RL training buffer.

    Args:
        state: The policy state at this step.
        action_info: Dictionary with log probs and value from action selection.
        reward: The reward received.
        done: Whether this is a terminal state.
    """

    def __init__(
        self,
        state: PolicyState,
        action_info: dict[str, Any],
        reward: float,
        done: bool = False,
    ) -> None:
        # Snapshot mutable inputs so later policy/state updates do not
        # rewrite historical experiences in the replay buffer.
        self.state = state.model_copy(deep=True)
        self.action_info = copy.deepcopy(action_info)
        self.reward = reward
        self.done = done


class RLTrainer:
    """PPO-based trainer for the attack policy network.

    Implements Proximal Policy Optimization with generalized advantage
    estimation (GAE) to train the policy network to select effective
    attack strategies.

    Args:
        policy: The PolicyNetwork to train.
        config: Trainer configuration. Uses defaults if not provided.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        config: TrainerConfig | None = None,
    ) -> None:
        self._policy = policy
        self._config = config or TrainerConfig()
        self._optimizer = torch.optim.Adam(
            self._policy.parameters(),
            lr=self._config.learning_rate,
        )
        self._buffer: list[Experience] = []
        self._total_updates = 0
        self._episode_rewards: list[float] = []
        self._current_episode_reward = 0.0

    @property
    def total_updates(self) -> int:
        """Return the total number of policy updates performed."""
        return self._total_updates

    @property
    def buffer_size(self) -> int:
        """Return the number of experiences in the buffer."""
        return len(self._buffer)

    def store_experience(
        self,
        state: PolicyState,
        action_info: dict[str, Any],
        reward: float,
        done: bool = False,
    ) -> None:
        """Store a step of experience in the replay buffer.

        Args:
            state: The policy state at this step.
            action_info: Info dict from PolicyNetwork.select_action().
            reward: The reward received.
            done: Whether this step ends an episode.
        """
        self._buffer.append(Experience(state, action_info, reward, done))
        self._current_episode_reward += reward

        if done:
            self._episode_rewards.append(self._current_episode_reward)
            self._current_episode_reward = 0.0

    def update(self) -> dict[str, float]:
        """Perform a PPO policy update using buffered experiences.

        Computes advantages with GAE, then runs multiple epochs of
        minibatch updates with clipped surrogate loss.

        Returns:
            Dictionary with training metrics (policy_loss, value_loss,
            entropy, mean_advantage).
        """
        if len(self._buffer) < self._config.batch_size:
            logger.debug(
                "skip_update",
                buffer_size=len(self._buffer),
                required=self._config.batch_size,
            )
            return {}

        states, old_log_probs, rewards, values, dones = self._prepare_batch()
        advantages, returns = self._compute_gae(rewards, values, dones)

        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        adv_mean = advantages_tensor.mean()
        adv_std = advantages_tensor.std() + 1e-8
        advantages_tensor = (advantages_tensor - adv_mean) / adv_std

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _epoch in range(self._config.update_epochs):
            indices = np.random.default_rng().permutation(len(self._buffer))

            for start in range(0, len(indices), self._config.batch_size):
                end = start + self._config.batch_size
                batch_idx = indices[start:end]

                batch_states = torch.stack([states[i] for i in batch_idx])
                batch_old_log_probs = torch.tensor(
                    [old_log_probs[i] for i in batch_idx],
                    dtype=torch.float32,
                )
                batch_advantages = advantages_tensor[batch_idx]
                batch_returns = returns_tensor[batch_idx]

                mut_probs, strat_probs, _esc, new_values = self._policy(batch_states)

                new_log_probs = self._compute_log_probs(
                    mut_probs, strat_probs, batch_idx
                )

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self._config.clip_epsilon,
                        1.0 + self._config.clip_epsilon,
                    )
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(
                    new_values.squeeze(-1), batch_returns
                )

                entropy = -(
                    (mut_probs * torch.log(mut_probs + 1e-8)).sum(-1).mean()
                    + (strat_probs * torch.log(strat_probs + 1e-8)).sum(-1).mean()
                )

                loss = (
                    policy_loss
                    + self._config.value_loss_coef * value_loss
                    - self._config.entropy_coef * entropy
                )

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self._policy.parameters(),
                    self._config.max_grad_norm,
                )
                self._optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        self._total_updates += 1
        self._buffer.clear()

        metrics = {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "mean_advantage": float(adv_mean),
            "buffer_episodes": len(self._episode_rewards),
        }

        logger.info("policy_updated", update=self._total_updates, **metrics)
        return metrics

    def _prepare_batch(
        self,
    ) -> tuple[
        list[torch.Tensor],
        list[float],
        list[float],
        list[float],
        list[bool],
    ]:
        """Convert buffered experiences into tensors for training.

        Returns:
            Tuple of (states, old_log_probs, rewards, values, dones).
        """
        states = [exp.state.to_tensor() for exp in self._buffer]
        old_log_probs = [
            exp.action_info.get("mutation_log_prob", 0.0)
            + exp.action_info.get("strategy_log_prob", 0.0)
            for exp in self._buffer
        ]
        rewards = [exp.reward for exp in self._buffer]
        values = [exp.action_info.get("value", 0.0) for exp in self._buffer]
        dones = [exp.done for exp in self._buffer]
        return states, old_log_probs, rewards, values, dones

    def _compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        dones: list[bool],
    ) -> tuple[list[float], list[float]]:
        """Compute generalized advantage estimation.

        Args:
            rewards: Per-step rewards.
            values: Per-step value estimates.
            dones: Per-step terminal flags.

        Returns:
            Tuple of (advantages, returns).
        """
        n = len(rewards)
        advantages = [0.0] * n
        returns = [0.0] * n
        gae = 0.0

        for t in reversed(range(n)):
            next_value = 0.0 if t == n - 1 else values[t + 1]
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self._config.gamma * next_value * mask - values[t]
            gae = delta + self._config.gamma * self._config.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def _compute_log_probs(
        self,
        mut_probs: torch.Tensor,
        strat_probs: torch.Tensor,
        batch_idx: np.ndarray,
    ) -> torch.Tensor:
        """Compute log probabilities for the actions taken.

        Args:
            mut_probs: Mutation probability distributions.
            strat_probs: Strategy probability distributions.
            batch_idx: Indices into the experience buffer.

        Returns:
            Tensor of summed log probabilities.
        """
        log_probs = []
        for i, buf_idx in enumerate(batch_idx):
            exp = self._buffer[buf_idx]
            mut_idx = exp.action_info.get("mutation_index", 0)
            if isinstance(mut_idx, dict):
                mut_idx = mut_idx.get("mutation_index", 0)
            strat_idx = exp.action_info.get("strategy_index", 0)
            if isinstance(strat_idx, dict):
                strat_idx = strat_idx.get("strategy_index", 0)

            lp = torch.log(mut_probs[i, mut_idx] + 1e-8) + torch.log(
                strat_probs[i, strat_idx] + 1e-8
            )
            log_probs.append(lp)

        return torch.stack(log_probs)

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate training statistics.

        Returns:
            Dictionary with cumulative training metrics.
        """
        return {
            "total_updates": self._total_updates,
            "total_episodes": len(self._episode_rewards),
            "mean_episode_reward": (
                float(np.mean(self._episode_rewards)) if self._episode_rewards else 0.0
            ),
            "max_episode_reward": (
                float(np.max(self._episode_rewards)) if self._episode_rewards else 0.0
            ),
            "buffer_size": len(self._buffer),
        }
