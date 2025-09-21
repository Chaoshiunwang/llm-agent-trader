"""Tests for the ChatGPT client wrapper."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIENT_MODULE_PATH = PROJECT_ROOT / "backend" / "app" / "llm" / "client.py"

# Provide lightweight stand-ins for optional dependencies when running unit tests
if "openai" not in sys.modules:
    dummy_openai = types.ModuleType("openai")
    dummy_openai.OpenAI = MagicMock()
    sys.modules["openai"] = dummy_openai

# Bootstrap a minimal package structure so the client module can be imported
app_module = sys.modules.setdefault("app", types.ModuleType("app"))

if "app.config" not in sys.modules:
    config_module = types.ModuleType("app.config")

    class _Settings:
        OPENAI_API_KEY = None
        OPENAI_MODEL = "gpt-4.1-mini"
        OPENAI_BASE_URL = None
        OPENAI_ORGANIZATION = None
        OPENAI_SYSTEM_PROMPT = None

    config_module.settings = _Settings()
    sys.modules["app.config"] = config_module
    setattr(app_module, "config", config_module)

llm_package = sys.modules.setdefault("app.llm", types.ModuleType("app.llm"))
llm_package.__path__ = []  # type: ignore[attr-defined]
setattr(app_module, "llm", llm_package)

spec = importlib.util.spec_from_file_location("app.llm.client", CLIENT_MODULE_PATH)
assert spec and spec.loader
llm_client = importlib.util.module_from_spec(spec)
sys.modules["app.llm.client"] = llm_client
spec.loader.exec_module(llm_client)

ChatGPTClient = llm_client.ChatGPTClient
ChatGPTResponse = llm_client.ChatGPTResponse
get_llm_client = llm_client.get_llm_client


@pytest.fixture()
def mock_openai(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the OpenAI client constructor and return the mock instance."""

    mock_client = MagicMock()
    mock_completions = MagicMock()
    mock_completions.create = MagicMock()
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat
    mock_api = MagicMock(return_value=mock_client)
    monkeypatch.setattr(llm_client, "OpenAI", mock_api)
    return mock_client


def _mock_chat_response(content: str = "ok") -> MagicMock:
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message = {"role": "assistant", "content": content}
    mock_response.choices = [mock_choice]
    return mock_response


def test_get_llm_client_uses_chatgpt_api(mock_openai: MagicMock) -> None:
    """Ensure ``get_llm_client`` constructs and invokes the ChatGPT API."""

    mock_response = _mock_chat_response("Test response")
    mock_openai.chat.completions.create.return_value = mock_response

    client = get_llm_client(model="gpt-test", temperature=0.5, max_tokens=200)
    response = client.invoke("Generate a trading decision", metadata={"source": "unit"})

    mock_openai.chat.completions.create.assert_called_once()
    _, kwargs = mock_openai.chat.completions.create.call_args

    assert kwargs["model"] == "gpt-test"
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 200
    assert kwargs["metadata"] == {"source": "unit"}

    payload = kwargs["messages"]
    assert isinstance(payload, list)
    user_message = next((m for m in reversed(payload) if m["role"] == "user"), None)
    assert user_message is not None
    assert user_message["role"] == "user"
    assert user_message["content"] == "Generate a trading decision"

    assert isinstance(response, ChatGPTResponse)
    assert response.content == "Test response"


def test_chatgpt_client_merges_system_prompts(mock_openai: MagicMock) -> None:
    """System prompts are applied to list-based prompts as well as strings."""

    mock_openai.chat.completions.create.return_value = _mock_chat_response("Applied")

    client = ChatGPTClient(system_prompt="Default system")
    response = client.invoke(
        [
            {"role": "user", "content": "Hello"},
        ],
        system_prompt="Override",
    )

    _, kwargs = mock_openai.chat.completions.create.call_args
    system_message, user_message = kwargs["messages"]

    assert system_message["role"] == "system"
    assert system_message["content"] == "Override"
    assert user_message["role"] == "user"
    assert response.content == "Applied"
    assert response.raw is mock_openai.chat.completions.create.return_value


def test_chatgpt_client_accepts_legacy_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy provider names should fall back to ChatGPT with a warning."""

    issued: list[str] = []

    def _warn(message: str, *_: Any, **__: Any) -> None:
        issued.append(message)

    mock_client = MagicMock()
    mock_completions = MagicMock()
    mock_completions.create = MagicMock()
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat
    mock_api = MagicMock(return_value=mock_client)
    monkeypatch.setattr(llm_client, "OpenAI", mock_api)
    monkeypatch.setattr(llm_client.warnings, "warn", _warn)

    mock_response = _mock_chat_response("Legacy ok")
    mock_client.chat.completions.create.return_value = mock_response

    client = get_llm_client(provider="azure")
    _ = client.invoke("Hello")

    assert issued
    assert "azure" in issued[0]
    assert mock_client.chat.completions.create.called
