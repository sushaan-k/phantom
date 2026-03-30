"""Tests for attack strategies."""

from __future__ import annotations

import pytest

from phantom.attacks.strategies import (
    AttackStrategy,
    DirectStrategy,
    IndirectStrategy,
    MultiTurnStrategy,
    create_strategy,
)
from phantom.models import AttackCategory, Conversation


class TestDirectStrategy:
    """Tests for the DirectStrategy."""

    def test_seed_prompt(self) -> None:
        strategy = DirectStrategy(AttackCategory.PROMPT_INJECTION)
        seed = strategy.generate_seed_prompt()
        assert isinstance(seed, str)
        assert len(seed) > 0

    def test_name(self) -> None:
        strategy = DirectStrategy(AttackCategory.PROMPT_INJECTION)
        assert strategy.name == "direct"

    def test_no_followup(self) -> None:
        strategy = DirectStrategy(AttackCategory.PROMPT_INJECTION)
        conv = Conversation()
        result = strategy.generate_followup(conv, 0.5)
        assert result is None

    @pytest.mark.parametrize("category", list(AttackCategory))
    def test_all_categories_produce_seeds(self, category: AttackCategory) -> None:
        strategy = DirectStrategy(category)
        seed = strategy.generate_seed_prompt()
        assert isinstance(seed, str)
        assert len(seed) > 0

    def test_seeds_vary(self) -> None:
        strategy = DirectStrategy(AttackCategory.PROMPT_INJECTION)
        seeds = {strategy.generate_seed_prompt() for _ in range(10)}
        assert len(seeds) > 1


class TestIndirectStrategy:
    """Tests for the IndirectStrategy."""

    def test_seed_prompt_structure(self) -> None:
        strategy = IndirectStrategy(AttackCategory.PROMPT_INJECTION)
        seed = strategy.generate_seed_prompt()
        assert isinstance(seed, str)
        assert len(seed) > 50

    def test_name(self) -> None:
        strategy = IndirectStrategy(AttackCategory.DATA_EXFILTRATION)
        assert strategy.name == "indirect"

    def test_followup_on_refusal(self) -> None:
        strategy = IndirectStrategy(AttackCategory.PROMPT_INJECTION)
        conv = Conversation()
        conv.add_turn("attacker", "Check this document.")
        conv.add_turn("target", "I'm sorry, I can't process those instructions.")
        followup = strategy.generate_followup(conv, 0.5)
        assert followup is not None
        assert "re-read" in followup.lower() or "instructions" in followup.lower()

    def test_followup_on_success(self) -> None:
        strategy = IndirectStrategy(AttackCategory.PROMPT_INJECTION)
        conv = Conversation()
        conv.add_turn("attacker", "Check this document.")
        conv.add_turn("target", "I found some interesting data in the document.")
        followup = strategy.generate_followup(conv, 0.5)
        assert followup is not None

    def test_followup_stops_after_max_turns(self) -> None:
        strategy = IndirectStrategy(AttackCategory.PROMPT_INJECTION)
        conv = Conversation()
        for _ in range(3):
            conv.add_turn("attacker", "x")
            conv.add_turn("target", "y")
        assert strategy.generate_followup(conv, 0.5) is None


class TestMultiTurnStrategy:
    """Tests for the MultiTurnStrategy."""

    def test_initial_message_is_benign(self) -> None:
        strategy = MultiTurnStrategy(AttackCategory.PROMPT_INJECTION)
        seed = strategy.generate_seed_prompt()
        assert "security researcher" in seed.lower() or "safety" in seed.lower()

    def test_name(self) -> None:
        strategy = MultiTurnStrategy(AttackCategory.PROMPT_INJECTION)
        assert strategy.name == "multi_turn"

    def test_trust_building_phase(self) -> None:
        strategy = MultiTurnStrategy(AttackCategory.PROMPT_INJECTION, max_trust_turns=3)
        conv = Conversation()
        conv.add_turn("attacker", "Hi")
        conv.add_turn("target", "Hello")

        followup = strategy.generate_followup(conv, escalation=0.0)
        assert followup is not None
        assert "experiment" not in followup.lower()

    def test_escalation_phase(self) -> None:
        strategy = MultiTurnStrategy(AttackCategory.PROMPT_INJECTION, max_trust_turns=1)
        conv = Conversation()
        for _ in range(3):
            conv.add_turn("attacker", "Building trust...")
            conv.add_turn("target", "Sure, happy to help.")

        followup = strategy.generate_followup(conv, escalation=0.8)
        assert followup is not None

    def test_high_escalation_reduces_trust_turns(self) -> None:
        strategy = MultiTurnStrategy(AttackCategory.PROMPT_INJECTION, max_trust_turns=4)
        conv = Conversation()
        conv.add_turn("attacker", "Hi")
        conv.add_turn("target", "Hello")
        conv.add_turn("attacker", "Question")
        conv.add_turn("target", "Answer")

        followup = strategy.generate_followup(conv, escalation=0.9)
        assert followup is not None


class TestCreateStrategy:
    """Tests for the strategy factory function."""

    @pytest.mark.parametrize(
        ("name", "cls"),
        [
            ("direct", DirectStrategy),
            ("indirect", IndirectStrategy),
            ("multi_turn", MultiTurnStrategy),
        ],
    )
    def test_create_valid_strategies(
        self,
        name: str,
        cls: type[AttackStrategy],
    ) -> None:
        strategy = create_strategy(name, AttackCategory.PROMPT_INJECTION)
        assert isinstance(strategy, cls)

    def test_create_invalid_strategy(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_strategy("nonexistent", AttackCategory.PROMPT_INJECTION)
