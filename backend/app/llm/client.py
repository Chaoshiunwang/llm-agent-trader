"""
Unified LLM client for all modules
"""

import os
from typing import Literal, Optional, Union

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI

from app.config import settings


def _detect_available_provider() -> Literal["azure", "gemini"]:
    """
    Auto-detect which LLM provider to use based on available API keys in .env

    Returns:
        "gemini" if only GOOGLE_API_KEY is available and uncommented
        "azure" if AZURE_OPENAI_API_KEY is available and uncommented
        Prioritizes Azure when both are available
        Defaults to "azure" for backward compatibility
    """
    # Read directly from environment variables (not from settings)
    # This allows users to comment out keys in .env to disable providers
    google_api_key = os.getenv("GOOGLE_API_KEY")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")

    # Check if keys are properly set and not placeholder values
    azure_available = (
        azure_api_key
        and azure_api_key.strip()
        and azure_api_key not in ["your_azure_api_key_here", "", "none", "null"]
    )

    gemini_available = (
        google_api_key
        and google_api_key.strip()
        and google_api_key not in ["your_google_api_key_here", "", "none", "null"]
    )

    # Priority logic: Azure first (for backward compatibility), then Gemini
    if azure_available:
        return "azure"
    elif gemini_available:
        return "gemini"
    else:
        # Default to azure for backward compatibility
        return "azure"


def get_llm_client(
    provider: Optional[Literal["azure", "gemini"]] = None,
    temperature: float = 0.1,
    max_tokens: int = 4000,
    **kwargs,
) -> Union[AzureChatOpenAI, ChatGoogleGenerativeAI]:
    """
    Get unified LLM client with automatic provider detection

    Args:
        provider: LLM provider - "azure" for Azure OpenAI or "gemini" for Google Gemini
                 If None, auto-detects based on available API keys in .env
        temperature: Temperature parameter
        max_tokens: Maximum number of tokens
        **kwargs: Other provider-specific parameters

    Returns:
        LLM client instance (AzureChatOpenAI or ChatGoogleGenerativeAI)
    """
    # Auto-detect provider if not specified
    if provider is None:
        provider = _detect_available_provider()

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            **kwargs,
        )
    else:  # azure
        return AzureChatOpenAI(
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            api_key=settings.AZURE_OPENAI_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )


class LLMClientConfig:
    """LLM Client Configuration Class"""

    def __init__(
        self,
        provider: Optional[Literal["azure", "gemini"]] = None,
        # Azure OpenAI specific
        deployment_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        # Gemini specific
        google_api_key: Optional[str] = None,
        model: str = None,
        # Common parameters
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ):
        # Auto-detect provider if not specified
        self.provider = (
            provider if provider is not None else _detect_available_provider()
        )

        # Azure OpenAI configuration
        self.deployment_name = deployment_name or settings.AZURE_OPENAI_DEPLOYMENT_NAME
        self.endpoint = endpoint or settings.AZURE_OPENAI_ENDPOINT
        self.api_version = api_version or settings.AZURE_OPENAI_API_VERSION
        self.api_key = api_key or settings.AZURE_OPENAI_API_KEY

        # Gemini configuration
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model or settings.GEMINI_MODEL

        # Common parameters
        self.temperature = temperature
        self.max_tokens = max_tokens

    def create_client(self) -> Union[AzureChatOpenAI, ChatGoogleGenerativeAI]:
        """Create client instance based on provider"""
        if self.provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=self.model,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                google_api_key=self.google_api_key,
            )
        else:  # Azure OpenAI
            return AzureChatOpenAI(
                azure_deployment=self.deployment_name,
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )


# Default configuration instance
default_config = LLMClientConfig()


def get_configured_client(
    config: Optional[LLMClientConfig] = None,
) -> Union[AzureChatOpenAI, ChatGoogleGenerativeAI]:
    """
    Get client using configuration

    Args:
        config: LLM client configuration

    Returns:
        Configured LLM client (Azure OpenAI or Google Gemini)
    """
    if config is None:
        config = default_config
    return config.create_client()
