"""Unified ChatGPT client used across the application."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from openai import OpenAI

from app.config import settings


class ChatGPTResponse:
    """Simple wrapper mimicking LangChain's response structure."""

    def __init__(self, content: str) -> None:
        self.content = content


class ChatGPTClient:
    """Lightweight ChatGPT API client with a LangChain-like interface."""
 
    def __init__(
        self,
     *,
        api_key: Optional[str] = None,
             model: Optional[str] = None,
     base_url: Optional[str] = None,
        organization: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4000,
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
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or settings.OPENAI_SYSTEM_PROMPT
        self.default_request_kwargs = default_request_kwargs or {}

    def invoke(
        self,
        prompt: Union[str, Sequence[Dict[str, Any]]],
        *,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatGPTResponse:
        """Invoke the ChatGPT Responses API and return a simple wrapper."""

        input_payload = self._prepare_input(prompt, system_prompt)
        request_payload: Dict[str, Any] = {
            "model": self.model,
            "input": input_payload,
            "temperature": kwargs.pop("temperature", self.temperature),
        }

        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        if max_tokens is not None:
            request_payload["max_output_tokens"] = max_tokens

        request_payload.update(self.default_request_kwargs)
        request_payload.update(kwargs)

        response = self._client.responses.create(**request_payload)
        text = getattr(response, "output_text", None)
        if not text:
            text = self._extract_text_from_response(response)
        return ChatGPTResponse(text)

    def _prepare_input(
        self,
        prompt: Union[str, Sequence[Dict[str, Any]]],
        system_prompt: Optional[str],
    ) -> List[Dict[str, Any]]:
        if isinstance(prompt, Sequence) and prompt and isinstance(prompt[0], dict):
            return [self._normalize_message(message) for message in prompt]

        messages: List[Dict[str, Any]] = []
        combined_system_prompt = system_prompt or self.system_prompt
        if combined_system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": combined_system_prompt,
                        }
                    ],
                }
            )

        messages.append(self._normalize_message({"role": "user", "content": prompt}))
        return messages

    def _normalize_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        role = message.get("role", "user")
        content = message.get("content", "")

        if isinstance(content, list):
            normalized_content: List[Dict[str, Any]] = []
            for part in content:
                if isinstance(part, dict) and "type" in part:
                    normalized_content.append(part)
                else:
                    normalized_content.append({"type": "text", "text": str(part)})
            return {"role": role, "content": normalized_content}

        if isinstance(content, str):
            text = content
        else:
            text = str(content)
        return {"role": role, "content": [{"type": "text", "text": text}]}

    @staticmethod
    def _extract_text_from_response(response: Any) -> str:
        outputs = getattr(response, "output", None)
        if not outputs:
            return ""

        collected: List[str] = []
        for item in outputs:
            if getattr(item, "type", None) != "message":
                continue
            for part in getattr(item, "content", []) or []:
                if getattr(part, "type", None) == "text":
                    collected.append(getattr(part, "text", ""))
        return "".join(collected)


@dataclass
class LLMClientConfig:
    """Configuration container for creating a ChatGPT client."""

    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4000
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
    max_tokens: int = 4000,
    **kwargs: Any,
) -> ChatGPTClient:
    """Return a configured ChatGPT client.

    The provider argument is kept for backward compatibility. Any value other than
    ``None``, ``"chatgpt"`` or ``"openai"`` will raise a ``ValueError``.
    """

    if provider and provider not in {"chatgpt", "openai"}:
        raise ValueError("Only the ChatGPT provider is supported after the migration.")

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
