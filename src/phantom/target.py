"""Target interface for communicating with the LLM under test."""

from __future__ import annotations

import time
from typing import Any

import httpx
from pydantic import BaseModel, Field, model_validator

from phantom.exceptions import TargetConnectionError, TargetResponseError
from phantom.logging import get_logger
from phantom.models import Conversation

logger = get_logger("phantom.target")


class TargetConfig(BaseModel):
    """Configuration for the target LLM endpoint."""

    endpoint: str = Field(description="HTTP endpoint URL for the target LLM")
    auth: dict[str, str] = Field(
        default_factory=dict,
        description="Authentication headers (e.g., Authorization: Bearer ...)",
    )
    system_prompt_known: bool = Field(
        default=False,
        description="Whether the system prompt is known to the attacker",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Known system prompt, if available",
    )
    request_template: dict[str, Any] = Field(
        default_factory=lambda: {
            "model": "gpt-4",
            "messages": [],
        },
        description="Template for API requests. Messages will be injected.",
    )
    response_path: str = Field(
        default="choices.0.message.content",
        description="Dot-separated path to extract response text from JSON.",
    )
    timeout_seconds: float = Field(
        default=30.0,
        description="HTTP request timeout in seconds",
    )
    max_retries: int = Field(default=3, ge=0, le=10)

    @model_validator(mode="after")
    def validate_prompt_consistency(self) -> TargetConfig:
        """Ensure system_prompt is set when system_prompt_known is True."""
        if self.system_prompt_known and not self.system_prompt:
            msg = "system_prompt must be provided when system_prompt_known is True"
            raise ValueError(msg)
        return self


class Target:
    """Interface for sending probes to and receiving responses from a target LLM.

    Handles HTTP communication, retry logic, response parsing, and
    conversation state management for multi-turn interactions.

    Args:
        endpoint: HTTP endpoint URL for the target LLM.
        auth: Authentication headers.
        system_prompt_known: Whether the system prompt is known.
        system_prompt: Known system prompt, if available.
        request_template: Template for API requests.
        response_path: Dot-separated path to extract response text.
        timeout_seconds: HTTP request timeout.
        max_retries: Maximum number of retry attempts on failure.
        config: Optional TargetConfig to use instead of individual params.
    """

    def __init__(
        self,
        endpoint: str = "",
        auth: dict[str, str] | None = None,
        system_prompt_known: bool = False,
        system_prompt: str | None = None,
        request_template: dict[str, Any] | None = None,
        response_path: str = "choices.0.message.content",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        config: TargetConfig | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = TargetConfig(
                endpoint=endpoint,
                auth=auth or {},
                system_prompt_known=system_prompt_known,
                system_prompt=system_prompt,
                request_template=request_template or {"model": "gpt-4", "messages": []},
                response_path=response_path,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
        self._client: httpx.AsyncClient | None = None

    @property
    def config(self) -> TargetConfig:
        """Return the target configuration."""
        return self._config

    @property
    def endpoint(self) -> str:
        """Return the target endpoint URL."""
        return self._config.endpoint

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client.

        Returns:
            An httpx AsyncClient instance.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._config.timeout_seconds),
                headers=self._config.auth,
            )
        return self._client

    def _build_request_body(
        self,
        prompt: str,
        conversation: Conversation | None = None,
    ) -> dict[str, Any]:
        """Build the request body for the target API.

        Args:
            prompt: The attack prompt to send.
            conversation: Optional conversation history for multi-turn.

        Returns:
            A dictionary representing the request body.
        """
        body = dict(self._config.request_template)
        messages: list[dict[str, str]] = []

        if self._config.system_prompt:
            messages.append({"role": "system", "content": self._config.system_prompt})

        if conversation:
            for turn in conversation.turns:
                role = "user" if turn.role == "attacker" else "assistant"
                messages.append({"role": role, "content": turn.content})

            if not (
                conversation.turns
                and conversation.turns[-1].role == "attacker"
                and conversation.turns[-1].content == prompt
            ):
                messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": prompt})
        body["messages"] = messages
        return body

    def _extract_response(self, data: dict[str, Any]) -> str:
        """Extract the response text from the API response JSON.

        Args:
            data: The parsed JSON response from the target.

        Returns:
            The extracted response text.

        Raises:
            TargetResponseError: If the response path doesn't resolve.
        """
        parts = self._config.response_path.split(".")
        current: Any = data

        for part in parts:
            if isinstance(current, dict):
                if part not in current:
                    raise TargetResponseError(
                        200,
                        f"Response path '{self._config.response_path}' "
                        f"not found at key '{part}' in {list(current.keys())}",
                    )
                current = current[part]
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    current = current[idx]
                except (ValueError, IndexError) as exc:
                    raise TargetResponseError(
                        200,
                        f"Response path '{self._config.response_path}' "
                        f"failed at index '{part}'",
                    ) from exc
            else:
                raise TargetResponseError(
                    200,
                    f"Cannot traverse '{part}' on type {type(current).__name__}",
                )

        if not isinstance(current, str):
            return str(current)

        return current

    async def send_probe(
        self,
        prompt: str,
        conversation: Conversation | None = None,
    ) -> tuple[str, float]:
        """Send an attack probe to the target and return the response.

        Args:
            prompt: The attack prompt to send.
            conversation: Optional conversation context for multi-turn attacks.

        Returns:
            A tuple of (response_text, latency_ms).

        Raises:
            TargetConnectionError: If the target cannot be reached.
            TargetResponseError: If the target returns an error response.
        """
        client = await self._get_client()
        body = self._build_request_body(prompt, conversation)

        last_error: Exception | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                start_time = time.monotonic()
                response = await client.post(self._config.endpoint, json=body)
                latency_ms = (time.monotonic() - start_time) * 1000

                if response.status_code >= 500:
                    last_error = TargetResponseError(
                        response.status_code, response.text
                    )
                    logger.warning(
                        "target_server_error",
                        status=response.status_code,
                        attempt=attempt + 1,
                    )
                    continue

                if response.status_code >= 400:
                    raise TargetResponseError(response.status_code, response.text)

                data = response.json()
                text = self._extract_response(data)

                logger.debug(
                    "probe_sent",
                    prompt_len=len(prompt),
                    response_len=len(text),
                    latency_ms=round(latency_ms, 1),
                )

                return text, latency_ms

            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning(
                    "target_timeout",
                    attempt=attempt + 1,
                    timeout_s=self._config.timeout_seconds,
                )
                continue
            except httpx.RequestError as exc:
                last_error = exc
                logger.warning(
                    "target_transport_error",
                    attempt=attempt + 1,
                    error=str(exc),
                )
                continue

        raise TargetConnectionError(
            self._config.endpoint,
            f"Exhausted {self._config.max_retries + 1} attempts: {last_error}",
        )

    async def send_conversation_turn(
        self,
        prompt: str,
        conversation: Conversation,
    ) -> tuple[str, float]:
        """Send a prompt as part of a multi-turn conversation.

        Adds the attacker turn to the conversation, sends the probe,
        then adds the target response to the conversation.

        Args:
            prompt: The attack prompt for this turn.
            conversation: The conversation to continue.

        Returns:
            A tuple of (response_text, latency_ms).
        """
        conversation.add_turn("attacker", prompt)
        response_text, latency_ms = await self.send_probe(prompt, conversation)
        conversation.add_turn("target", response_text)
        return response_text, latency_ms

    async def health_check(self) -> bool:
        """Verify connectivity to the target endpoint.

        Returns:
            True if the target responds, False otherwise.
        """
        try:
            client = await self._get_client()
            response = await client.post(
                self._config.endpoint,
                json=self._build_request_body("Hello"),
            )
            return response.status_code < 500
        except (httpx.HTTPError, Exception):
            return False

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
