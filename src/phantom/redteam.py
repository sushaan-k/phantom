"""Main RedTeam orchestrator -- coordinates attacks, learning, and reporting."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from phantom.atlas.mapper import ATLASMapper
from phantom.atlas.taxonomy import ATLASTaxonomy
from phantom.attacks.generator import AttackGenerator
from phantom.exceptions import ConfigurationError, PhantomError
from phantom.learner.policy import PolicyNetwork, PolicyState
from phantom.learner.reward import RewardClassifier
from phantom.learner.trainer import RLTrainer, TrainerConfig
from phantom.logging import get_logger
from phantom.models import (
    AttackAction,
    AttackCategory,
    Conversation,
    Finding,
    OutcomeType,
    ProbeResult,
    Severity,
)
from phantom.target import Target

logger = get_logger("phantom.redteam")


class RedTeamConfig(BaseModel):
    """Configuration for a RedTeam assessment run."""

    attack_model: str = Field(
        default="gpt-4",
        description="Model identifier for attack prompt generation",
    )
    attack_api_key: str | None = Field(
        default=None,
        description="API key for the attack model (reads OPENAI_API_KEY if None)",
    )
    attack_api_base: str | None = Field(
        default=None,
        description="Base URL for the attack model API",
    )
    categories: list[str] = Field(
        default_factory=lambda: [
            "prompt_injection",
            "goal_hijacking",
            "data_exfiltration",
        ],
        description="Attack categories to test",
    )
    max_interactions: int = Field(
        default=500,
        ge=1,
        le=50000,
        description="Maximum number of probes to send",
    )
    multi_turn: bool = Field(
        default=True,
        description="Whether to use multi-turn attack strategies",
    )
    max_turns_per_conversation: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum turns per multi-turn conversation",
    )
    learning_rate: float = Field(
        default=3e-4,
        gt=0.0,
        description="Learning rate for the RL policy",
    )
    update_interval: int = Field(
        default=32,
        ge=1,
        description="Number of probes between policy updates",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )


class RedTeamResults(BaseModel):
    """Results from a red-team assessment run."""

    findings: list[Finding] = Field(default_factory=list)
    probes: list[ProbeResult] = Field(default_factory=list)
    total_probes: int = 0
    total_bypasses: int = 0
    novel_attack_count: int = 0
    training_stats: dict[str, Any] = Field(default_factory=dict)

    def count_by_severity(self, severity: str | Severity) -> int:
        """Count findings at a given severity level.

        Args:
            severity: Severity level string or enum value.

        Returns:
            Number of findings at that severity.
        """
        if isinstance(severity, str):
            severity = Severity(severity.upper())
        return sum(1 for f in self.findings if f.severity == severity)

    @property
    def atlas_coverage_pct(self) -> float:
        """Calculate the percentage of ATLAS techniques represented.

        Returns:
            Coverage percentage from 0 to 100.
        """
        tested_techniques = {
            finding.technique_id for finding in self.findings if finding.technique_id
        }
        tested_techniques.update(
            probe.technique_id for probe in self.probes if probe.technique_id
        )
        taxonomy = ATLASTaxonomy()
        taxonomy.load()
        total = len(taxonomy.all_technique_ids)
        if total == 0:
            return 0.0
        return round(len(tested_techniques) / total * 100, 1)

    @property
    def bypass_rate(self) -> float:
        """Calculate the overall bypass rate.

        Returns:
            Bypass rate from 0 to 1.
        """
        if self.total_probes == 0:
            return 0.0
        return round(self.total_bypasses / self.total_probes, 3)


class RedTeam:
    """Main orchestrator for RL-based adversarial red-team assessments.

    Coordinates the attack generator, RL policy learner, target interface,
    and ATLAS mapper to run a full security assessment against an LLM.

    Args:
        target: The Target to assess.
        attack_model: Model identifier for attack generation.
        categories: List of attack category names to test.
        max_interactions: Maximum number of probes to send.
        multi_turn: Whether to enable multi-turn strategies.
        max_turns_per_conversation: Max turns per multi-turn conversation.
        learning_rate: RL policy learning rate.
        config: Optional RedTeamConfig for full configuration.
        attack_api_key: API key for the attack model.
        attack_api_base: Base URL for the attack model API.
    """

    def __init__(
        self,
        target: Target,
        attack_model: str = "gpt-4",
        categories: list[str] | None = None,
        max_interactions: int = 500,
        multi_turn: bool = True,
        max_turns_per_conversation: int = 10,
        learning_rate: float = 3e-4,
        config: RedTeamConfig | None = None,
        attack_api_key: str | None = None,
        attack_api_base: str | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = RedTeamConfig(
                attack_model=attack_model,
                attack_api_key=attack_api_key,
                attack_api_base=attack_api_base,
                categories=categories
                or [
                    "prompt_injection",
                    "goal_hijacking",
                    "data_exfiltration",
                ],
                max_interactions=max_interactions,
                multi_turn=multi_turn,
                max_turns_per_conversation=max_turns_per_conversation,
                learning_rate=learning_rate,
            )

        self._target = target
        self._validate_categories()

        self._policy = PolicyNetwork()
        self._trainer = RLTrainer(
            self._policy,
            TrainerConfig(learning_rate=self._config.learning_rate),
        )
        self._reward_classifier = RewardClassifier()
        self._generator = AttackGenerator(
            attack_model=self._config.attack_model,
            api_key=self._config.attack_api_key,
            api_base=self._config.attack_api_base,
        )
        self._mapper = ATLASMapper()

        self._probes: list[ProbeResult] = []
        self._state = PolicyState()

    def _validate_categories(self) -> None:
        """Validate that all configured categories are recognized.

        Raises:
            ConfigurationError: If an unknown category is specified.
        """
        valid = {c.value for c in AttackCategory}
        for cat in self._config.categories:
            if cat not in valid:
                raise ConfigurationError(
                    "categories",
                    f"Unknown category '{cat}'. Valid: {sorted(valid)}",
                )

    @property
    def config(self) -> RedTeamConfig:
        """Return the red-team configuration."""
        return self._config

    async def run(self) -> RedTeamResults:
        """Execute the full red-team assessment.

        Runs the RL-driven attack loop: generates probes, sends them
        to the target, classifies responses, updates the policy, and
        maps findings to ATLAS.

        Returns:
            A RedTeamResults with all findings and metrics.
        """
        logger.info(
            "assessment_started",
            categories=self._config.categories,
            max_interactions=self._config.max_interactions,
            multi_turn=self._config.multi_turn,
        )

        categories = [AttackCategory(c) for c in self._config.categories]

        interaction_count = 0

        try:
            for category in categories:
                if interaction_count >= self._config.max_interactions:
                    break

                remaining = self._config.max_interactions - interaction_count
                probes_for_category = remaining // max(
                    len(categories) - categories.index(category), 1
                )

                count = await self._run_category(category, probes_for_category)
                interaction_count += count

                logger.info(
                    "category_completed",
                    category=category.value,
                    probes_sent=count,
                    total_so_far=interaction_count,
                )

        except PhantomError:
            raise
        except Exception as exc:
            logger.error("assessment_error", error=str(exc))
            raise PhantomError(f"Assessment failed: {exc}") from exc
        finally:
            await self._cleanup()

        findings = self._mapper.map_probes(self._probes)
        training_stats = self._trainer.get_stats()

        total_bypasses = sum(
            1
            for p in self._probes
            if p.outcome in (OutcomeType.FULL_BYPASS, OutcomeType.PARTIAL_BYPASS)
        )

        novel_count = sum(
            1
            for p in self._probes
            if p.outcome == OutcomeType.FULL_BYPASS and p.reward >= 0.8
        )

        results = RedTeamResults(
            findings=findings,
            probes=self._probes,
            total_probes=len(self._probes),
            total_bypasses=total_bypasses,
            novel_attack_count=novel_count,
            training_stats=training_stats,
        )

        logger.info(
            "assessment_completed",
            total_probes=results.total_probes,
            findings=len(results.findings),
            bypasses=results.total_bypasses,
            bypass_rate=results.bypass_rate,
        )

        return results

    async def _run_category(
        self,
        category: AttackCategory,
        max_probes: int,
    ) -> int:
        """Run probes for a single attack category.

        Args:
            category: The attack category to test.
            max_probes: Maximum number of probes for this category.

        Returns:
            The number of probes sent.
        """
        probe_count = 0
        category_idx = list(AttackCategory).index(category)

        strategies = ["direct"]
        if self._config.multi_turn:
            strategies.extend(["indirect", "multi_turn"])

        while probe_count < max_probes:
            self._state.category_index = category_idx

            action, action_info = self._policy.select_action(self._state, category)

            if action.strategy == "multi_turn" and self._config.multi_turn:
                count = await self._run_multi_turn(action, action_info, category)
                probe_count += count
            else:
                probe = await self._run_single_probe(action, action_info, category)
                if probe:
                    probe_count += 1

            if probe_count > 0 and probe_count % self._config.update_interval == 0:
                self._trainer.update()

        return probe_count

    async def _run_single_probe(
        self,
        action: AttackAction,
        action_info: dict[str, Any],
        category: AttackCategory,
    ) -> ProbeResult | None:
        """Execute a single attack probe.

        Args:
            action: The attack action from the policy.
            action_info: Info dict from action selection.
            category: The attack category.

        Returns:
            The ProbeResult, or None on failure.
        """
        try:
            prompt = await self._generator.generate(action)
        except Exception as exc:
            logger.warning("generation_failed", error=str(exc))
            return None

        try:
            response_text, latency_ms = await self._target.send_probe(prompt)
        except PhantomError as exc:
            logger.warning("probe_failed", error=str(exc))
            return None

        signal = self._reward_classifier.classify(response_text)

        probe = ProbeResult(
            attack_prompt=prompt,
            response=response_text,
            outcome=signal.outcome,
            reward=signal.reward,
            category=category,
            latency_ms=latency_ms,
        )
        self._probes.append(probe)

        self._trainer.store_experience(
            self._state, action_info, signal.reward, done=True
        )
        self._update_state(signal.outcome, signal.reward, response_text)

        return probe

    async def _run_multi_turn(
        self,
        action: AttackAction,
        action_info: dict[str, Any],
        category: AttackCategory,
    ) -> int:
        """Execute a multi-turn attack conversation.

        Args:
            action: The initial attack action.
            action_info: Info dict from action selection.
            category: The attack category.

        Returns:
            Number of probes sent in this conversation.
        """
        conversation = Conversation(category=category)
        probe_count = 0
        episode_reward = 0.0

        for turn in range(self._config.max_turns_per_conversation):
            try:
                prompt = await self._generator.generate(action, conversation)
            except Exception as exc:
                logger.warning(
                    "multi_turn_generation_failed",
                    turn=turn,
                    error=str(exc),
                )
                break

            try:
                response_text, latency_ms = await self._target.send_conversation_turn(
                    prompt, conversation
                )
            except PhantomError as exc:
                logger.warning(
                    "multi_turn_probe_failed",
                    turn=turn,
                    error=str(exc),
                )
                break

            signal = self._reward_classifier.classify(response_text)

            probe = ProbeResult(
                attack_prompt=prompt,
                response=response_text,
                outcome=signal.outcome,
                reward=signal.reward,
                category=category,
                turn_number=turn + 1,
                conversation_id=conversation.conversation_id,
                latency_ms=latency_ms,
            )
            self._probes.append(probe)
            probe_count += 1

            episode_reward += signal.reward
            is_done = (
                turn == self._config.max_turns_per_conversation - 1
                or signal.outcome == OutcomeType.FULL_BYPASS
            )

            self._trainer.store_experience(
                self._state, action_info, signal.reward, done=is_done
            )
            self._update_state(signal.outcome, signal.reward, response_text)

            if signal.outcome == OutcomeType.FULL_BYPASS:
                logger.info(
                    "multi_turn_bypass",
                    turn=turn + 1,
                    conversation_id=conversation.conversation_id,
                )
                break

            action, action_info = self._policy.select_action(self._state, category)

        conversation.total_reward = episode_reward
        return probe_count

    def _update_state(
        self,
        outcome: OutcomeType,
        reward: float,
        response: str,
    ) -> None:
        """Update the policy state based on the latest probe result.

        Args:
            outcome: The outcome of the last probe.
            reward: The reward from the last probe.
            response: The response text from the target.
        """
        n = len(self._probes)
        if n == 0:
            return

        recent = self._probes[-min(n, 20) :]
        refusals = sum(1 for p in recent if p.outcome == OutcomeType.CLEAN_REFUSAL)
        bypasses = sum(
            1
            for p in recent
            if p.outcome in (OutcomeType.FULL_BYPASS, OutcomeType.PARTIAL_BYPASS)
        )
        info_leaks = sum(1 for p in recent if p.outcome == OutcomeType.INFO_LEAK)

        total = len(recent)
        self._state.refusal_rate = refusals / total
        self._state.bypass_rate = bypasses / total
        self._state.info_leak_rate = info_leaks / total
        self._state.avg_response_length = min(len(response) / 2000.0, 1.0)
        self._state.last_reward = reward

        if outcome == OutcomeType.CLEAN_REFUSAL:
            self._state.consecutive_refusals += 1
        else:
            self._state.consecutive_refusals = 0

    async def _cleanup(self) -> None:
        """Clean up resources after the assessment."""
        import contextlib

        with contextlib.suppress(Exception):
            await self._generator.close()
        with contextlib.suppress(Exception):
            await self._target.close()
