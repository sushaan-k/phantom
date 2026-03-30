"""Tests for the target interface."""

from __future__ import annotations

import httpx
import pytest
import respx

from phantom.exceptions import TargetResponseError
from phantom.models import Conversation
from phantom.target import Target, TargetConfig


class TestTargetConfig:
    """Tests for TargetConfig validation."""

    def test_default_config(self) -> None:
        config = TargetConfig(endpoint="https://api.example.com/chat")
        assert config.endpoint == "https://api.example.com/chat"
        assert config.auth == {}
        assert config.system_prompt_known is False

    def test_system_prompt_consistency(self) -> None:
        with pytest.raises(ValueError, match="system_prompt must be provided"):
            TargetConfig(
                endpoint="https://api.example.com/chat",
                system_prompt_known=True,
            )

    def test_valid_system_prompt(self) -> None:
        config = TargetConfig(
            endpoint="https://api.example.com/chat",
            system_prompt_known=True,
            system_prompt="You are a helpful assistant.",
        )
        assert config.system_prompt_known is True
        assert config.system_prompt == "You are a helpful assistant."


class TestTarget:
    """Tests for the Target class."""

    def test_create_from_params(self) -> None:
        target = Target(
            endpoint="https://api.example.com/chat",
            auth={"Authorization": "Bearer test"},
        )
        assert target.endpoint == "https://api.example.com/chat"

    def test_create_from_config(self, target_config: TargetConfig) -> None:
        target = Target(config=target_config)
        assert target.endpoint == target_config.endpoint

    def test_build_request_body(self) -> None:
        target = Target(
            endpoint="https://api.example.com/chat",
            system_prompt="You are helpful.",
        )
        body = target._build_request_body("Hello")
        assert "messages" in body
        messages = body["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Hello"

    def test_build_request_body_with_conversation(self) -> None:
        target = Target(endpoint="https://api.example.com/chat")
        conv = Conversation()
        conv.add_turn("attacker", "Hi there")
        conv.add_turn("target", "Hello!")

        body = target._build_request_body("Follow up", conv)
        messages = body["messages"]
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hi there"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hello!"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "Follow up"

    def test_build_request_body_avoids_duplicate_latest_turn(self) -> None:
        target = Target(endpoint="https://api.example.com/chat")
        conv = Conversation()
        conv.add_turn("attacker", "Follow up")

        body = target._build_request_body("Follow up", conv)
        messages = body["messages"]

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Follow up"

    def test_extract_response_success(self) -> None:
        target = Target(endpoint="https://api.example.com/chat")
        data = {"choices": [{"message": {"content": "Hello, how can I help?"}}]}
        result = target._extract_response(data)
        assert result == "Hello, how can I help?"

    def test_extract_response_missing_path(self) -> None:
        target = Target(endpoint="https://api.example.com/chat")
        data = {"result": "some data"}
        with pytest.raises(TargetResponseError, match="not found"):
            target._extract_response(data)

    def test_extract_response_custom_path(self) -> None:
        target = Target(
            endpoint="https://api.example.com/chat",
            response_path="result.text",
        )
        data = {"result": {"text": "Custom response"}}
        result = target._extract_response(data)
        assert result == "Custom response"

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_probe_success(self) -> None:
        respx.post("https://api.example.com/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "I cannot help with that."}}]
                },
            )
        )

        target = Target(endpoint="https://api.example.com/chat")
        response, latency = await target.send_probe("Test prompt")
        assert response == "I cannot help with that."
        assert latency > 0
        await target.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_probe_client_error(self) -> None:
        respx.post("https://api.example.com/chat").mock(
            return_value=httpx.Response(400, text="Bad Request")
        )

        target = Target(endpoint="https://api.example.com/chat")
        with pytest.raises(TargetResponseError, match="400"):
            await target.send_probe("Test prompt")
        await target.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_conversation_turn(self) -> None:
        respx.post("https://api.example.com/chat").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Interesting question."}}]},
            )
        )

        target = Target(endpoint="https://api.example.com/chat")
        conv = Conversation()
        response, _ = await target.send_conversation_turn("What do you think?", conv)
        assert response == "Interesting question."
        assert conv.turn_count == 2
        await target.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_health_check_success(self) -> None:
        respx.post("https://api.example.com/chat").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "OK"}}]},
            )
        )

        target = Target(endpoint="https://api.example.com/chat")
        assert await target.health_check() is True
        await target.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_health_check_failure(self) -> None:
        respx.post("https://api.example.com/chat").mock(
            return_value=httpx.Response(500, text="Server Error")
        )

        target = Target(endpoint="https://api.example.com/chat")
        assert await target.health_check() is False
        await target.close()

    @pytest.mark.asyncio
    async def test_close_idempotent(self) -> None:
        target = Target(endpoint="https://api.example.com/chat")
        await target.close()
        await target.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_probe_server_error_retries(self) -> None:
        """Test that server errors (5xx) are retried."""
        route = respx.post("https://api.example.com/chat")
        route.side_effect = [
            httpx.Response(500, text="Internal Server Error"),
            httpx.Response(500, text="Still broken"),
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Finally ok"}}]},
            ),
        ]

        target = Target(
            endpoint="https://api.example.com/chat",
            max_retries=3,
        )
        response, latency = await target.send_probe("Test prompt")
        assert response == "Finally ok"
        assert latency > 0
        await target.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_probe_server_error_exhausted(self) -> None:
        """Test that exhausted retries raise TargetConnectionError."""
        from phantom.exceptions import TargetConnectionError

        route = respx.post("https://api.example.com/chat")
        route.side_effect = [
            httpx.Response(500, text="Internal Server Error"),
            httpx.Response(502, text="Bad Gateway"),
        ]

        target = Target(
            endpoint="https://api.example.com/chat",
            max_retries=1,
        )
        with pytest.raises(TargetConnectionError, match="Exhausted"):
            await target.send_probe("Test prompt")
        await target.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_probe_timeout_retries(self) -> None:
        """Test that timeouts are retried and eventually raise."""
        from phantom.exceptions import TargetConnectionError

        route = respx.post("https://api.example.com/chat")
        route.side_effect = httpx.ReadTimeout("Timeout")

        target = Target(
            endpoint="https://api.example.com/chat",
            max_retries=1,
            timeout_seconds=1.0,
        )
        with pytest.raises(TargetConnectionError, match="Exhausted"):
            await target.send_probe("Test prompt")
        await target.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_probe_connect_error(self) -> None:
        """Test that connection errors are normalized after retries."""
        from phantom.exceptions import TargetConnectionError

        route = respx.post("https://api.example.com/chat")
        route.side_effect = httpx.ConnectError("Connection refused")

        target = Target(endpoint="https://api.example.com/chat")
        with pytest.raises(TargetConnectionError):
            await target.send_probe("Test prompt")
        await target.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_probe_transport_error_normalized(self) -> None:
        """Test that transport-layer errors are normalized to TargetConnectionError."""
        from phantom.exceptions import TargetConnectionError

        route = respx.post("https://api.example.com/chat")
        route.side_effect = httpx.RemoteProtocolError("Server disconnected")

        target = Target(
            endpoint="https://api.example.com/chat",
            max_retries=1,
        )
        with pytest.raises(TargetConnectionError, match="Exhausted"):
            await target.send_probe("Test prompt")
        await target.close()

    def test_extract_response_list_index_error(self) -> None:
        """Test extract_response with out-of-bounds list index."""
        target = Target(
            endpoint="https://api.example.com/chat",
            response_path="choices.5.message.content",
        )
        data = {"choices": [{"message": {"content": "Hello"}}]}
        with pytest.raises(TargetResponseError, match="failed at index"):
            target._extract_response(data)

    def test_extract_response_non_traversable(self) -> None:
        """Test extract_response when path hits a non-dict/non-list."""
        target = Target(
            endpoint="https://api.example.com/chat",
            response_path="choices.0.message.content.deeper",
        )
        data = {"choices": [{"message": {"content": "Hello"}}]}
        with pytest.raises(TargetResponseError, match="Cannot traverse"):
            target._extract_response(data)

    def test_extract_response_non_string_value(self) -> None:
        """Test extract_response converts non-string to string."""
        target = Target(
            endpoint="https://api.example.com/chat",
            response_path="result.count",
        )
        data = {"result": {"count": 42}}
        result = target._extract_response(data)
        assert result == "42"

    def test_extract_response_list_invalid_index_string(self) -> None:
        """Test extract_response with non-numeric list index."""
        target = Target(
            endpoint="https://api.example.com/chat",
            response_path="choices.abc.message",
        )
        data = {"choices": [{"message": {"content": "Hello"}}]}
        with pytest.raises(TargetResponseError, match="failed at index"):
            target._extract_response(data)

    def test_build_request_body_no_system_prompt(self) -> None:
        """Test request body without system prompt."""
        target = Target(endpoint="https://api.example.com/chat")
        body = target._build_request_body("Hello")
        messages = body["messages"]
        # No system message should be present
        assert all(m["role"] != "system" for m in messages)
        assert messages[-1]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_get_client_creates_new(self) -> None:
        """Test that _get_client creates a client on first call."""
        target = Target(endpoint="https://api.example.com/chat")
        assert target._client is None
        client = await target._get_client()
        assert client is not None
        assert not client.is_closed
        await target.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing(self) -> None:
        """Test that _get_client reuses an open client."""
        target = Target(endpoint="https://api.example.com/chat")
        client1 = await target._get_client()
        client2 = await target._get_client()
        assert client1 is client2
        await target.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_health_check_http_error(self) -> None:
        """Test health_check returns False on HTTP error."""
        route = respx.post("https://api.example.com/chat")
        route.side_effect = httpx.ConnectError("refused")

        target = Target(endpoint="https://api.example.com/chat")
        assert await target.health_check() is False
        await target.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_health_check_4xx_is_true(self) -> None:
        """Test health_check returns True for client errors (< 500)."""
        respx.post("https://api.example.com/chat").mock(
            return_value=httpx.Response(401, text="Unauthorized")
        )

        target = Target(endpoint="https://api.example.com/chat")
        assert await target.health_check() is True
        await target.close()

    def test_config_max_retries_bound(self) -> None:
        """Test that max_retries must be between 0 and 10."""
        with pytest.raises(ValueError):
            TargetConfig(
                endpoint="https://api.example.com/chat",
                max_retries=11,
            )

    def test_create_target_with_custom_template(self) -> None:
        """Test creating Target with a custom request template."""
        target = Target(
            endpoint="https://api.example.com/chat",
            request_template={
                "model": "custom-model",
                "messages": [],
                "temperature": 0.5,
            },
        )
        body = target._build_request_body("hello")
        assert body["model"] == "custom-model"
        assert body["temperature"] == 0.5
