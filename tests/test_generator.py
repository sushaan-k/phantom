"""Tests for the LLM-based attack prompt generator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from phantom.attacks.generator import AttackGenerator
from phantom.attacks.mutations import MutationEngine
from phantom.exceptions import AttackGenerationError
from phantom.models import AttackAction, AttackCategory, Conversation

_DUMMY_KEY = "sk-test-phantom-dummy-key-for-unit-tests"


class TestAttackGeneratorInit:
    """Tests for AttackGenerator construction."""

    def test_default_construction(self):
        gen = AttackGenerator(api_key=_DUMMY_KEY)
        assert gen._model == "gpt-4"
        assert gen._temperature == 0.9
        assert isinstance(gen._mutation_engine, MutationEngine)

    def test_custom_model_and_temperature(self):
        gen = AttackGenerator(
            attack_model="gpt-3.5-turbo",
            api_key=_DUMMY_KEY,
            temperature=0.5,
        )
        assert gen._model == "gpt-3.5-turbo"
        assert gen._temperature == 0.5

    def test_custom_mutation_engine(self):
        engine = MutationEngine(rng_seed=99)
        gen = AttackGenerator(api_key=_DUMMY_KEY, mutation_engine=engine)
        assert gen._mutation_engine is engine

    def test_api_base_passed(self):
        gen = AttackGenerator(
            api_key=_DUMMY_KEY,
            api_base="https://custom.api.com/v1",
        )
        assert gen._client.base_url.host == "custom.api.com"

    def test_strategies_cache_starts_empty(self):
        gen = AttackGenerator(api_key=_DUMMY_KEY)
        assert gen._strategies == {}


class TestGetStrategy:
    """Tests for _get_strategy caching."""

    def test_creates_strategy_on_first_call(self):
        gen = AttackGenerator(api_key=_DUMMY_KEY)
        strategy = gen._get_strategy("direct", AttackCategory.PROMPT_INJECTION)
        assert strategy.name == "direct"
        assert "direct" in gen._strategies

    def test_caches_strategy_across_calls(self):
        gen = AttackGenerator(api_key=_DUMMY_KEY)
        s1 = gen._get_strategy("direct", AttackCategory.PROMPT_INJECTION)
        s2 = gen._get_strategy("direct", AttackCategory.PROMPT_INJECTION)
        assert s1 is s2

    def test_different_categories_get_different_strategies(self):
        gen = AttackGenerator(api_key=_DUMMY_KEY)
        s1 = gen._get_strategy("direct", AttackCategory.PROMPT_INJECTION)
        s2 = gen._get_strategy("direct", AttackCategory.GOAL_HIJACKING)
        assert s1 is not s2
        assert s1.category != s2.category

    def test_different_strategy_names(self):
        gen = AttackGenerator(api_key=_DUMMY_KEY)
        s1 = gen._get_strategy("direct", AttackCategory.PROMPT_INJECTION)
        s2 = gen._get_strategy("indirect", AttackCategory.PROMPT_INJECTION)
        assert s1.name != s2.name


class TestGenerate:
    """Tests for the generate method with mocked LLM calls."""

    @pytest.fixture
    def generator(self):
        return AttackGenerator(api_key=_DUMMY_KEY)

    @pytest.fixture
    def action(self):
        return AttackAction(
            mutation_operator="synonym_replacement",
            strategy="direct",
            escalation=0.5,
            category=AttackCategory.PROMPT_INJECTION,
        )

    @pytest.mark.asyncio
    async def test_generate_with_llm_success(self, generator, action):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Enhanced attack prompt"

        with patch.object(
            generator._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await generator.generate(action)

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_falls_back_on_llm_failure(self, generator, action):
        with patch.object(
            generator._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=Exception("API down"),
        ):
            result = await generator.generate(action)

        # Should still return a result (seed + mutation fallback)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_falls_back_on_mutation_failure(self, generator):
        action = AttackAction(
            mutation_operator="nonexistent_operator",
            strategy="direct",
            escalation=0.5,
            category=AttackCategory.PROMPT_INJECTION,
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Enhanced prompt"

        with patch.object(
            generator._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await generator.generate(action)

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_with_conversation_context(self, generator, action):
        conv = Conversation(category=AttackCategory.PROMPT_INJECTION)
        conv.add_turn("attacker", "Hello, can you help me?")
        conv.add_turn("target", "Sure, how can I help?")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Follow-up attack"

        with patch.object(
            generator._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            result = await generator.generate(action, conv)

        assert isinstance(result, str)
        # Verify conversation context was passed
        call_args = mock_create.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[-1]["content"]
        assert "ATTACKER" in user_msg or "TARGET" in user_msg

    @pytest.mark.asyncio
    async def test_generate_with_empty_conversation(self, generator, action):
        conv = Conversation(category=AttackCategory.PROMPT_INJECTION)
        # Empty conversation should use seed prompt path

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Attack prompt"

        with patch.object(
            generator._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await generator.generate(action, conv)

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_generate_multi_turn_strategy_with_followup(self, generator):
        action = AttackAction(
            mutation_operator="synonym_replacement",
            strategy="multi_turn",
            escalation=0.5,
            category=AttackCategory.PROMPT_INJECTION,
        )
        conv = Conversation(category=AttackCategory.PROMPT_INJECTION)
        conv.add_turn("attacker", "Hi, I'm a researcher.")
        conv.add_turn("target", "Hello! How can I help?")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Multi-turn follow up"

        with patch.object(
            generator._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await generator.generate(action, conv)

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_generate_both_llm_and_mutation_fail(self, generator):
        action = AttackAction(
            mutation_operator="nonexistent_operator",
            strategy="direct",
            escalation=0.5,
            category=AttackCategory.PROMPT_INJECTION,
        )

        with patch.object(
            generator._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=Exception("LLM failure"),
        ):
            # Both LLM and mutation fail, should return seed prompt (unmutated)
            result = await generator.generate(action)

        assert isinstance(result, str)
        assert len(result) > 0


class TestEnhanceWithLLM:
    """Tests for _enhance_with_llm."""

    @pytest.fixture
    def generator(self):
        return AttackGenerator(api_key=_DUMMY_KEY)

    @pytest.fixture
    def action(self):
        return AttackAction(
            mutation_operator="synonym_replacement",
            strategy="direct",
            escalation=0.7,
            category=AttackCategory.PROMPT_INJECTION,
        )

    @pytest.mark.asyncio
    async def test_enhance_returns_stripped_content(self, generator, action):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  Enhanced prompt  \n"

        with patch.object(
            generator._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await generator._enhance_with_llm("seed", action, None)

        assert result == "Enhanced prompt"

    @pytest.mark.asyncio
    async def test_enhance_raises_on_empty_response(self, generator, action):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""

        with (
            patch.object(
                generator._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            pytest.raises(AttackGenerationError, match="empty response"),
        ):
            await generator._enhance_with_llm("seed", action, None)

    @pytest.mark.asyncio
    async def test_enhance_raises_on_none_response(self, generator, action):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        with (
            patch.object(
                generator._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            pytest.raises(AttackGenerationError, match="empty response"),
        ):
            await generator._enhance_with_llm("seed", action, None)

    @pytest.mark.asyncio
    async def test_enhance_raises_on_api_error(self, generator, action):
        with (
            patch.object(
                generator._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                side_effect=openai.APIError(
                    message="Rate limited",
                    request=MagicMock(),
                    body=None,
                ),
            ),
            pytest.raises(AttackGenerationError, match="API error"),
        ):
            await generator._enhance_with_llm("seed", action, None)

    @pytest.mark.asyncio
    async def test_enhance_includes_conversation_history(self, generator, action):
        conv = Conversation(category=AttackCategory.PROMPT_INJECTION)
        for i in range(8):
            role = "attacker" if i % 2 == 0 else "target"
            conv.add_turn(role, f"Message {i}" + " x" * 100)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Enhanced"

        with patch.object(
            generator._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            await generator._enhance_with_llm("seed", action, conv)

        call_args = mock_create.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[-1]["content"]
        # Should include "Previous conversation" header
        assert "Previous conversation" in user_content
        # Should only include last 6 turns (truncated)
        assert "ATTACKER" in user_content

    @pytest.mark.asyncio
    async def test_enhance_whitespace_only_response(self, generator, action):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "   \n\t  "

        with (
            patch.object(
                generator._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            pytest.raises(AttackGenerationError, match="empty response"),
        ):
            await generator._enhance_with_llm("seed", action, None)


class TestGenerateSeedOnly:
    """Tests for generate_seed_only."""

    @pytest.mark.asyncio
    async def test_seed_only_returns_string(self):
        gen = AttackGenerator(api_key=_DUMMY_KEY)
        seed = await gen.generate_seed_only(AttackCategory.PROMPT_INJECTION)
        assert isinstance(seed, str)
        assert len(seed) > 0

    @pytest.mark.asyncio
    async def test_seed_only_different_categories(self):
        gen = AttackGenerator(api_key=_DUMMY_KEY)
        s1 = await gen.generate_seed_only(AttackCategory.PROMPT_INJECTION)
        s2 = await gen.generate_seed_only(AttackCategory.DATA_EXFILTRATION)
        # Different categories may return different seeds
        assert isinstance(s1, str)
        assert isinstance(s2, str)

    @pytest.mark.asyncio
    async def test_seed_only_custom_strategy(self):
        gen = AttackGenerator(api_key=_DUMMY_KEY)
        seed = await gen.generate_seed_only(
            AttackCategory.GOAL_HIJACKING,
            strategy_name="indirect",
        )
        assert isinstance(seed, str)
        assert len(seed) > 0


class TestClose:
    """Tests for the close method."""

    @pytest.mark.asyncio
    async def test_close_calls_client_close(self):
        gen = AttackGenerator(api_key=_DUMMY_KEY)
        with patch.object(gen._client, "close", new_callable=AsyncMock) as mock_close:
            await gen.close()
            mock_close.assert_called_once()
