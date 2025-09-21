 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/backend/app/llm/client.py b/backend/app/llm/client.py
index 4d5a248cf1d40b4cbc6db89d76340fe6259c759a..3016992933be01d6b22b97b647daac6d512da206 100644
--- a/backend/app/llm/client.py
+++ b/backend/app/llm/client.py
@@ -1,199 +1,201 @@
-"""
-Unified LLM client for all modules
-"""
+"""Unified ChatGPT client used across the application."""
+
+from __future__ import annotations
 
 import os
-from typing import Literal, Optional, Union
+from dataclasses import dataclass, field
+from typing import Any, Dict, List, Optional, Sequence, Union
 
-from langchain_google_genai import ChatGoogleGenerativeAI
-from langchain_openai import AzureChatOpenAI, ChatOpenAI
+from openai import OpenAI
 
 from app.config import settings
 
 
-def _detect_available_provider() -> Literal["azure", "gemini", "openai"]:
-    """
-    Auto-detect which LLM provider to use based on available API keys in .env
-
-    Returns:
-        "gemini" if only GOOGLE_API_KEY is available and uncommented
-        "openai" if only OPENAI_API_KEY is available and uncommented
-        "azure" if AZURE_OPENAI_API_KEY is available and uncommented
-        Prioritizes Azure > OpenAI > Gemini when multiple are available
-        Defaults to "azure" for backward compatibility
-    """
-    google_api_key = os.getenv("GOOGLE_API_KEY")
-    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
-    openai_api_key = os.getenv("OPENAI_API_KEY")
-
-    azure_available = (
-        azure_api_key
-        and azure_api_key.strip()
-        and azure_api_key not in ["your_azure_api_key_here", "", "none", "null"]
-    )
-    openai_available = (
-        openai_api_key
-        and openai_api_key.strip()
-        and openai_api_key not in ["your_openai_api_key_here", "", "none", "null"]
-    )
-    gemini_available = (
-        google_api_key
-        and google_api_key.strip()
-        and google_api_key not in ["your_google_api_key_here", "", "none", "null"]
-    )
+class ChatGPTResponse:
+    """Simple wrapper mimicking LangChain's response structure."""
 
-    # Priority logic: Azure > OpenAI > Gemini
-    if azure_available:
-        return "azure"
-    elif openai_available:
-        return "openai"
-    elif gemini_available:
-        return "gemini"
-    else:
-        # Default to azure for backward compatibility
-        return "azure"
+    def __init__(self, content: str) -> None:
+        self.content = content
 
 
-def get_llm_client(
-    provider: Optional[Literal["azure", "gemini", "openai"]] = None,
-    temperature: float = 0.1,
-    max_tokens: int = 4000,
-    **kwargs,
-) -> Union[AzureChatOpenAI, ChatGoogleGenerativeAI, ChatOpenAI]:
-    """
-    Get unified LLM client with automatic provider detection
-
-    Args:
-        provider: LLM provider - "azure" for Azure OpenAI, "openai" for OpenAI, "gemini" for Google Gemini
-                 If None, auto-detects based on available API keys in .env
-        temperature: Temperature parameter
-        max_tokens: Maximum number of tokens
-        **kwargs: Other provider-specific parameters
-
-    Returns:
-        LLM client instance (AzureChatOpenAI, ChatOpenAI, or ChatGoogleGenerativeAI)
-    """
-    if provider is None:
-        provider = _detect_available_provider()
-
-    if provider == "gemini":
-        return ChatGoogleGenerativeAI(
-            model=settings.GEMINI_MODEL,
-            temperature=temperature,
-            max_output_tokens=max_tokens,
-            google_api_key=os.getenv("GOOGLE_API_KEY"),
-            **kwargs,
-        )
-    elif provider == "openai":
-        return ChatOpenAI(
-            model=settings.OPENAI_MODEL,
-            openai_api_key=os.getenv("OPENAI_API_KEY"),
-            temperature=temperature,
-            max_tokens=max_tokens,
-            **kwargs,
-        )
-    else:  # azure
-        return AzureChatOpenAI(
-            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
-            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
-            api_version=settings.AZURE_OPENAI_API_VERSION,
-            api_key=settings.AZURE_OPENAI_API_KEY,
-            temperature=temperature,
-            max_tokens=max_tokens,
-            **kwargs,
-        )
-
-
-class LLMClientConfig:
-    """LLM Client Configuration Class"""
+class ChatGPTClient:
+    """Lightweight ChatGPT API client with a LangChain-like interface."""
 
     def __init__(
         self,
-        provider: Optional[Literal["azure", "gemini", "openai"]] = None,
-        # Azure OpenAI specific
-        deployment_name: Optional[str] = None,
-        endpoint: Optional[str] = None,
-        api_version: Optional[str] = None,
+        *,
         api_key: Optional[str] = None,
-        # Gemini specific
-        google_api_key: Optional[str] = None,
         model: Optional[str] = None,
-        # OpenAI specific
-        openai_api_key: Optional[str] = None,
-        openai_model: Optional[str] = None,
-        # Common parameters
+        base_url: Optional[str] = None,
+        organization: Optional[str] = None,
         temperature: float = 0.1,
         max_tokens: int = 4000,
-    ):
-        self.provider = (
-            provider if provider is not None else _detect_available_provider()
+        system_prompt: Optional[str] = None,
+        default_request_kwargs: Optional[Dict[str, Any]] = None,
+    ) -> None:
+        self._client = OpenAI(
+            api_key=api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"),
+            base_url=base_url or settings.OPENAI_BASE_URL or os.getenv("OPENAI_BASE_URL"),
+            organization=
+                organization
+                or settings.OPENAI_ORGANIZATION
+                or os.getenv("OPENAI_ORGANIZATION"),
         )
+        self.model = model or settings.OPENAI_MODEL
+        self.temperature = temperature
+        self.max_tokens = max_tokens
+        self.system_prompt = system_prompt or settings.OPENAI_SYSTEM_PROMPT
+        self.default_request_kwargs = default_request_kwargs or {}
 
-        # Azure OpenAI configuration
-        self.deployment_name = deployment_name or settings.AZURE_OPENAI_DEPLOYMENT_NAME
-        self.endpoint = endpoint or settings.AZURE_OPENAI_ENDPOINT
-        self.api_version = api_version or settings.AZURE_OPENAI_API_VERSION
-        self.api_key = api_key or settings.AZURE_OPENAI_API_KEY
-
-        # Gemini configuration
-        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
-        self.model = model or (
-            settings.GEMINI_MODEL if self.provider == "gemini" else
-            settings.OPENAI_MODEL if self.provider == "openai" else
-            settings.AZURE_OPENAI_DEPLOYMENT_NAME
+    def invoke(
+        self,
+        prompt: Union[str, Sequence[Dict[str, Any]]],
+        *,
+        system_prompt: Optional[str] = None,
+        **kwargs: Any,
+    ) -> ChatGPTResponse:
+        """Invoke the ChatGPT Responses API and return a simple wrapper."""
+
+        input_payload = self._prepare_input(prompt, system_prompt)
+        request_payload: Dict[str, Any] = {
+            "model": self.model,
+            "input": input_payload,
+            "temperature": kwargs.pop("temperature", self.temperature),
+        }
+
+        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
+        if max_tokens is not None:
+            request_payload["max_output_tokens"] = max_tokens
+
+        request_payload.update(self.default_request_kwargs)
+        request_payload.update(kwargs)
+
+        response = self._client.responses.create(**request_payload)
+        text = getattr(response, "output_text", None)
+        if not text:
+            text = self._extract_text_from_response(response)
+        return ChatGPTResponse(text)
+
+    def _prepare_input(
+        self,
+        prompt: Union[str, Sequence[Dict[str, Any]]],
+        system_prompt: Optional[str],
+    ) -> List[Dict[str, Any]]:
+        if isinstance(prompt, Sequence) and prompt and isinstance(prompt[0], dict):
+            return [self._normalize_message(message) for message in prompt]
+
+        messages: List[Dict[str, Any]] = []
+        combined_system_prompt = system_prompt or self.system_prompt
+        if combined_system_prompt:
+            messages.append(
+                {
+                    "role": "system",
+                    "content": [
+                        {
+                            "type": "text",
+                            "text": combined_system_prompt,
+                        }
+                    ],
+                }
+            )
+
+        messages.append(self._normalize_message({"role": "user", "content": prompt}))
+        return messages
+
+    def _normalize_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
+        role = message.get("role", "user")
+        content = message.get("content", "")
+
+        if isinstance(content, list):
+            normalized_content: List[Dict[str, Any]] = []
+            for part in content:
+                if isinstance(part, dict) and "type" in part:
+                    normalized_content.append(part)
+                else:
+                    normalized_content.append({"type": "text", "text": str(part)})
+            return {"role": role, "content": normalized_content}
+
+        if isinstance(content, str):
+            text = content
+        else:
+            text = str(content)
+        return {"role": role, "content": [{"type": "text", "text": text}]}
+
+    @staticmethod
+    def _extract_text_from_response(response: Any) -> str:
+        outputs = getattr(response, "output", None)
+        if not outputs:
+            return ""
+
+        collected: List[str] = []
+        for item in outputs:
+            if getattr(item, "type", None) != "message":
+                continue
+            for part in getattr(item, "content", []) or []:
+                if getattr(part, "type", None) == "text":
+                    collected.append(getattr(part, "text", ""))
+        return "".join(collected)
+
+
+@dataclass
+class LLMClientConfig:
+    """Configuration container for creating a ChatGPT client."""
+
+    api_key: Optional[str] = None
+    model: Optional[str] = None
+    base_url: Optional[str] = None
+    organization: Optional[str] = None
+    temperature: float = 0.1
+    max_tokens: int = 4000
+    system_prompt: Optional[str] = None
+    default_request_kwargs: Dict[str, Any] = field(default_factory=dict)
+
+    def create_client(self) -> ChatGPTClient:
+        return ChatGPTClient(
+            api_key=self.api_key,
+            model=self.model,
+            base_url=self.base_url,
+            organization=self.organization,
+            temperature=self.temperature,
+            max_tokens=self.max_tokens,
+            system_prompt=self.system_prompt,
+            default_request_kwargs=self.default_request_kwargs,
         )
 
-        # OpenAI configuration
-        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
-        self.openai_model = openai_model or settings.OPENAI_MODEL
 
-        # Common parameters
-        self.temperature = temperature
-        self.max_tokens = max_tokens
+def get_llm_client(
+    provider: Optional[str] = None,
+    *,
+    temperature: float = 0.1,
+    max_tokens: int = 4000,
+    **kwargs: Any,
+) -> ChatGPTClient:
+    """Return a configured ChatGPT client.
 
-    def create_client(self) -> Union[AzureChatOpenAI, ChatGoogleGenerativeAI, ChatOpenAI]:
-        """Create client instance based on provider"""
-        if self.provider == "gemini":
-            return ChatGoogleGenerativeAI(
-                model=self.model,
-                temperature=self.temperature,
-                max_output_tokens=self.max_tokens,
-                google_api_key=self.google_api_key,
-            )
-        elif self.provider == "openai":
-            return ChatOpenAI(
-                model=self.openai_model,
-                openai_api_key=self.openai_api_key,
-                temperature=self.temperature,
-                max_tokens=self.max_tokens,
-            )
-        else:  # Azure OpenAI
-            return AzureChatOpenAI(
-                azure_deployment=self.deployment_name,
-                azure_endpoint=self.endpoint,
-                api_version=self.api_version,
-                api_key=self.api_key,
-                temperature=self.temperature,
-                max_tokens=self.max_tokens,
-            )
+    The provider argument is kept for backward compatibility. Any value other than
+    ``None``, ``"chatgpt"`` or ``"openai"`` will raise a ``ValueError``.
+    """
+
+    if provider and provider not in {"chatgpt", "openai"}:
+        raise ValueError("Only the ChatGPT provider is supported after the migration.")
+
+    config = LLMClientConfig(
+        temperature=temperature,
+        max_tokens=max_tokens,
+        **kwargs,
+    )
+    return config.create_client()
 
 
 # Default configuration instance
 default_config = LLMClientConfig()
 
 
 def get_configured_client(
     config: Optional[LLMClientConfig] = None,
-) -> Union[AzureChatOpenAI, ChatGoogleGenerativeAI, ChatOpenAI]:
-    """
-    Get client using configuration
+) -> ChatGPTClient:
+    """Create a ChatGPT client from a configuration instance."""
 
-    Args:
-        config: LLM client configuration
-
-    Returns:
-        Configured LLM client (Azure OpenAI, Google Gemini, or OpenAI)
-    """
     if config is None:
         config = default_config
     return config.create_client()
 
EOF
)
