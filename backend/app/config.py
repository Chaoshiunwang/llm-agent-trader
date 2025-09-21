 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/backend/app/config.py b/backend/app/config.py
index 0b425fe4707d3cc56412807bd85d29f142b6e32d..132c0e9e97e627950bbfbd222b763bb294f6b656 100644
--- a/backend/app/config.py
+++ b/backend/app/config.py
@@ -18,49 +18,46 @@ if env_path.exists():
 else:
     # Fallback: load from current directory
     load_dotenv(".env")
 
 
 class Settings(BaseSettings):
     """Application settings."""
 
     # API settings
     API_V1_STR: str = "/api/v1"
     PROJECT_NAME: str = "LLM Stock Backtesting Dashboard"
 
     # CORS settings
     BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000"]
 
     # Database settings (future use)
     DATABASE_URL: Optional[str] = None
 
     # External API settings
     YFINANCE_TIMEOUT: int = 30
 
     # Backtesting settings
     DEFAULT_INITIAL_CAPITAL: float = 10000.0
     DEFAULT_COMMISSION: float = 0.001
 
    # Azure OpenAI settings
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "gpt-4.1"

    # Google Gemini settings
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash"
    # ChatGPT API settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4.1-mini"
    OPENAI_BASE_URL: Optional[str] = None
    OPENAI_ORGANIZATION: Optional[str] = None
    OPENAI_SYSTEM_PROMPT: Optional[str] = None
 
     # Additional .env settings
     REDIS_URL: Optional[str] = None
     BACKEND_URL: Optional[str] = None
     FRONTEND_URL: Optional[str] = None
     DEBUG: bool = False
     LOG_LEVEL: str = "INFO"
 
     class Config:
         env_file = "../../.env"
         env_file_encoding = "utf-8"
         case_sensitive = True
 
 
 settings = Settings()
 
EOF
)
