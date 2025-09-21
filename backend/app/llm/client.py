"""Unified ChatGPT client used across the application."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from openai import OpenAI

from app.config import settings


Message = Dict[str, Any]


@dataclass(slots=True)
class ChatGPTResponse:
    """Simple wrapper mimicking LangChain's response structure."""

    content: str
    raw: Any | None = None


class ChatGPTClient:
    """Lightweight wrapper around the ChatGPT Chat Completions API."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        default_request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or settings.OPENAI_BASE_URL or os.getenv("OPENAI_BASE_URL"),
            organization=
            organization
            or settings.OPENAI_ORGANIZATION
            or os.getenv("OPENAI_ORGANIZATION"),
        )
        self.model = model or settings.OPENAI_MODEL
        self.temperature = temperature
        env_max_tokens = getattr(settings, "OPENAI_MAX_OUTPUT_TOKENS", None)
        effective_max_tokens = max_tokens
        if effective_max_tokens is None:
            effective_max_tokens = env_max_tokens
        if effective_max_tokens is None:
            effective_max_tokens = 4000

        # ``max_tokens`` is kept for backwards compatibility with legacy callers.
        self.max_tokens = effective_max_tokens
        self.max_output_tokens = effective_max_tokens
        self.system_prompt = system_prompt or settings.OPENAI_SYSTEM_PROMPT
        self.default_request_kwargs: Dict[str, Any] = dict(default_request_kwargs or {})

    def invoke(
        self,
        prompt: Union[str, Sequence[Dict[str, Any]]],
        *,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatGPTResponse:
        """Invoke the ChatGPT Chat Completions API and return a simple wrapper."""

        messages = self._prepare_messages(prompt, system_prompt)
        model_name = kwargs.pop("model", self.model)
        temperature = kwargs.pop("temperature", self.temperature)
        max_output_tokens = kwargs.pop(
            "max_output_tokens", kwargs.pop("max_tokens", self.max_output_tokens)
        )

        request_payload: Dict[str, Any] = dict(self.default_request_kwargs)
        request_payload.update(
            {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
            }
        )

        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is None:
            max_tokens = max_output_tokens

        if "max_tokens" not in request_payload and max_tokens is not None:
            request_payload["max_tokens"] = max_tokens

        request_payload.update(kwargs)

        response = self._client.chat.completions.create(**request_payload)
        text = self._extract_text_from_response(response)
        return ChatGPTResponse(text or "", raw=response)

    def _prepare_messages(
        self,
        prompt: Union[str, Sequence[Dict[str, Any]]],
        system_prompt: Optional[str],
    ) -> List[Message]:
        if isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes)):
            normalized = [self._normalize_message(message) for message in prompt]
        else:
            normalized = [self._normalize_message({"role": "user", "content": prompt})]

        effective_system_prompt = system_prompt or self.system_prompt
        system_message = None
        if effective_system_prompt:
            system_message = {
                "role": "system",
                "content": effective_system_prompt,
            }

        if system_message:
            if normalized and normalized[0].get("role") == "system":
                # Replace existing system message if an override is provided
                if system_prompt is not None or not normalized[0].get("content"):
                    normalized[0] = system_message
            else:
                normalized.insert(0, system_message)

        return normalized

    def _normalize_message(self, message: Dict[str, Any]) -> Message:
        role = message.get("role", "user")
        content = message.get("content", "")

        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    if "text" in part:
                        parts.append(str(part.get("text", "")))
                    elif "content" in part:
                        parts.append(str(part.get("content", "")))
                    else:
                        parts.append(str(part))
                else:
                    parts.append(str(part))
            text = "\n".join(filter(None, parts))
            return {"role": role, "content": text}

        if isinstance(content, dict):
            if "text" in content:
                text = str(content.get("text", ""))
            elif "content" in content:
                text = str(content.get("content", ""))
            else:
                text = str(content)
            return {"role": role, "content": text}

        text = str(content)
        return {"role": role, "content": text}

    @staticmethod
    def _extract_text_from_response(response: Any) -> str:
        # Primary extraction path for Chat Completions responses
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")

        for choice in choices or []:
            message = getattr(choice, "message", None)
            if message is None and isinstance(choice, dict):
                message = choice.get("message")

            if not message:
                continue

            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")

            if content:
                return str(content)

            # Some SDK payloads may expose ``delta`` during streaming
            delta = getattr(choice, "delta", None)
            if delta is None and isinstance(choice, dict):
                delta = choice.get("delta")
            if delta and isinstance(delta, dict):
                delta_content = delta.get("content") or delta.get("text")
                if delta_content:
                    return str(delta_content)

        # Fallback to Responses API extraction for backwards compatibility
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)

        collected: List[str] = []

        output = getattr(response, "output", None)
        if output is None and isinstance(response, dict):
            output = response.get("output")

        for item in output or []:
            content = getattr(item, "content", None)
            if content is None and isinstance(item, dict):
                content = item.get("content")

            if not content:
                continue

            if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
                for part in content:
                    text = getattr(part, "text", None)
                    if text is None and isinstance(part, dict):
                        text = part.get("text") or part.get("content")
                    if text:
                        collected.append(str(text))
            else:
                collected.append(str(content))

        if collected:
            return "\n".join(filter(None, collected))

        return ""


@dataclass
class LLMClientConfig:
    """Configuration container for creating a ChatGPT client."""

    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    default_request_kwargs: Dict[str, Any] = field(default_factory=dict)

    def create_client(self) -> ChatGPTClient:
        return ChatGPTClient(
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url,
            organization=self.organization,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt=self.system_prompt,
            default_request_kwargs=self.default_request_kwargs,
        )


def get_llm_client(
    provider: Optional[str] = None,
    *,
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> ChatGPTClient:
    """Return a configured ChatGPT client.

    The provider argument is kept for backward compatibility with legacy call sites.
    """

    normalized_provider = (provider or "chatgpt").lower()
    legacy_providers = {"azure", "gemini"}
    supported_providers = {"chatgpt", "openai"}

    if normalized_provider in legacy_providers:
        warnings.warn(
            "Legacy provider '%s' is deprecated. Falling back to the ChatGPT API." %
            normalized_provider,
            DeprecationWarning,
            stacklevel=2,
        )
    elif normalized_provider not in supported_providers:
        raise ValueError(
            "Only the ChatGPT provider is supported after the migration."
        )

    config = LLMClientConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return config.create_client()


# Default configuration instance
default_config = LLMClientConfig()


def get_configured_client(
    config: Optional[LLMClientConfig] = None,
) -> ChatGPTClient:
    """Create a ChatGPT client from a configuration instance."""

    if config is None:
        config = default_config
    return config.create_client()
