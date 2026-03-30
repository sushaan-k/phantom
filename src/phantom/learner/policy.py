"""RL policy network for learning attack strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from phantom.exceptions import PolicyError
from phantom.logging import get_logger
from phantom.models import AttackAction, AttackCategory

logger = get_logger("phantom.learner.policy")

MUTATION_OPERATORS: list[str] = [
    "synonym_replacement",
    "base64_encoding",
    "role_play_framing",
    "language_switching",
    "token_splitting",
    "instruction_nesting",
    "context_overflow",
    "encoding_chain",
]

STRATEGY_OPTIONS: list[str] = [
    "direct",
    "indirect",
    "multi_turn",
]

STATE_DIM = 64
MUTATION_DIM = len(MUTATION_OPERATORS)
STRATEGY_DIM = len(STRATEGY_OPTIONS)
ESCALATION_DIM = 1


class PolicyState(BaseModel):
    """Representation of the environment state for the RL policy.

    Encodes response patterns, filter signatures, and conversation
    context into a fixed-size vector for the policy network.
    """

    refusal_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Recent refusal rate"
    )
    bypass_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Recent bypass rate"
    )
    info_leak_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Recent info leak rate"
    )
    avg_response_length: float = Field(
        default=0.0, ge=0.0, description="Average response length (normalized)"
    )
    turn_number: int = Field(
        default=0, ge=0, description="Current turn in conversation"
    )
    category_index: int = Field(
        default=0, ge=0, description="Index of current attack category"
    )
    mutation_history: list[float] = Field(
        default_factory=lambda: [0.0] * MUTATION_DIM,
        description="Usage counts per mutation operator (normalized)",
    )
    strategy_history: list[float] = Field(
        default_factory=lambda: [0.0] * STRATEGY_DIM,
        description="Usage counts per strategy (normalized)",
    )
    last_reward: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Reward from last probe"
    )
    consecutive_refusals: int = Field(
        default=0, ge=0, description="Number of consecutive refusals"
    )

    def to_tensor(self) -> torch.Tensor:
        """Convert the state to a fixed-size tensor for the policy network.

        Returns:
            A 1-D float tensor of size STATE_DIM.
        """
        raw = [
            self.refusal_rate,
            self.bypass_rate,
            self.info_leak_rate,
            self.avg_response_length,
            min(self.turn_number / 10.0, 1.0),
            self.category_index / max(len(AttackCategory) - 1, 1),
            self.last_reward,
            min(self.consecutive_refusals / 10.0, 1.0),
        ]

        raw.extend(self.mutation_history[:MUTATION_DIM])
        raw.extend(self.strategy_history[:STRATEGY_DIM])

        while len(raw) < STATE_DIM:
            raw.append(0.0)

        return torch.tensor(raw[:STATE_DIM], dtype=torch.float32)


class PolicyNetwork(nn.Module):
    """Neural network policy for selecting attack actions.

    A multi-head architecture that outputs distributions over
    mutation operators, strategies, and an escalation scalar.

    Args:
        state_dim: Dimensionality of the input state vector.
        hidden_dim: Size of hidden layers.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self._state_dim = state_dim
        self._hidden_dim = hidden_dim

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.mutation_head = nn.Sequential(
            nn.Linear(hidden_dim, MUTATION_DIM),
            nn.Softmax(dim=-1),
        )

        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_dim, STRATEGY_DIM),
            nn.Softmax(dim=-1),
        )

        self.escalation_head = nn.Sequential(
            nn.Linear(hidden_dim, ESCALATION_DIM),
            nn.Sigmoid(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the policy network.

        Args:
            state: Input state tensor of shape (batch, STATE_DIM).

        Returns:
            A tuple of (mutation_probs, strategy_probs, escalation, value).
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        shared_features = self.shared(state)
        mutation_probs = self.mutation_head(shared_features)
        strategy_probs = self.strategy_head(shared_features)
        escalation = self.escalation_head(shared_features)
        value = self.value_head(shared_features)

        return mutation_probs, strategy_probs, escalation, value

    def select_action(
        self,
        state: PolicyState,
        category: AttackCategory = AttackCategory.PROMPT_INJECTION,
        deterministic: bool = False,
    ) -> tuple[AttackAction, dict[str, Any]]:
        """Select an attack action given the current state.

        Args:
            state: The current environment state.
            category: The attack category to target.
            deterministic: If True, select the highest-probability actions.
                If False, sample from the distributions.

        Returns:
            A tuple of (AttackAction, info_dict) where info_dict contains
            log probabilities and value estimate for training.
        """
        state_tensor = state.to_tensor()

        with torch.no_grad():
            mut_probs, strat_probs, esc, value = self.forward(state_tensor)

        mut_probs_np = mut_probs.squeeze(0).numpy()
        strat_probs_np = strat_probs.squeeze(0).numpy()
        esc_val = esc.squeeze().item()
        value_val = value.squeeze().item()

        if deterministic:
            mut_idx = int(np.argmax(mut_probs_np))
            strat_idx = int(np.argmax(strat_probs_np))
        else:
            mut_probs_safe = np.maximum(mut_probs_np, 1e-8)
            mut_probs_safe /= mut_probs_safe.sum()
            mut_idx = int(
                np.random.default_rng().choice(MUTATION_DIM, p=mut_probs_safe)
            )

            strat_probs_safe = np.maximum(strat_probs_np, 1e-8)
            strat_probs_safe /= strat_probs_safe.sum()
            strat_idx = int(
                np.random.default_rng().choice(STRATEGY_DIM, p=strat_probs_safe)
            )

        action = AttackAction(
            mutation_operator=MUTATION_OPERATORS[mut_idx],
            strategy=STRATEGY_OPTIONS[strat_idx],
            escalation=round(esc_val, 3),
            category=category,
            parameters={
                "mutation_index": mut_idx,
                "strategy_index": strat_idx,
            },
        )

        info = {
            "mutation_probs": mut_probs_np.tolist(),
            "strategy_probs": strat_probs_np.tolist(),
            "escalation": esc_val,
            "value": value_val,
            "mutation_log_prob": float(np.log(max(mut_probs_np[mut_idx], 1e-8))),
            "strategy_log_prob": float(np.log(max(strat_probs_np[strat_idx], 1e-8))),
            "mutation_index": mut_idx,
            "strategy_index": strat_idx,
        }

        logger.debug(
            "action_selected",
            mutation=action.mutation_operator,
            strategy=action.strategy,
            escalation=action.escalation,
        )

        return action, info

    def save(self, path: str) -> None:
        """Save the policy network weights to disk.

        Args:
            path: File path for the saved state dict.
        """
        torch.save(self.state_dict(), path)
        logger.info("policy_saved", path=path)

    def load(self, path: str) -> None:
        """Load policy network weights from disk.

        Args:
            path: File path to load the state dict from.

        Raises:
            PolicyError: If the file cannot be loaded.
        """
        try:
            state_dict = torch.load(path, weights_only=True)
            self.load_state_dict(state_dict)
            logger.info("policy_loaded", path=path)
        except (OSError, RuntimeError) as exc:
            raise PolicyError(f"Failed to load policy from {path}: {exc}") from exc
