"""
rllm.config
-----------
Single source of truth for configuration. Values come from environment
variables or ``backend/.env`` (see ``.env.example`` for documentation).

Nothing in here performs network I/O; it is safe to import anywhere.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Provider routing ────────────────────────────────────────────────
    # Comma-separated, tried in order. A provider is skipped automatically
    # when it has no key (or, for Ollama / LM Studio, when *_ENABLED is false).
    LLM_PROVIDER_ORDER: str = "gemini,groq,openrouter,ollama"

    # Google Gemini — free key: https://aistudio.google.com/apikey
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.6-flash"            # answer / judge
    GEMINI_FAST_MODEL: str = "gemini-3.5-flash-lite"  # consistency samples
    GEMINI_RPM: int = 5
    GEMINI_FAST_RPM: int = 10
    # Thinking tokens count against max_output_tokens. "low" keeps latency
    # and blank-answer risk down. Set to "" to leave the model default.
    GEMINI_THINKING_LEVEL: str = "low"

    # Groq — free key: https://console.groq.com/keys
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_FAST_MODEL: str = "llama-3.1-8b-instant"
    GROQ_RPM: int = 30
    GROQ_FAST_RPM: int = 30

    # OpenRouter — free key: https://openrouter.ai/keys (use ":free" models)
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "meta-llama/llama-3.3-70b-instruct:free"
    OPENROUTER_FAST_MODEL: str = "meta-llama/llama-3.3-70b-instruct:free"
    OPENROUTER_RPM: int = 20
    OPENROUTER_FAST_RPM: int = 20

    # Ollama — local, no key, unlimited: https://ollama.com
    OLLAMA_ENABLED: bool = False
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3:4b"
    OLLAMA_FAST_MODEL: str = ""          # empty = same as OLLAMA_MODEL
    OLLAMA_NUM_CTX: int = 4096           # context window; bigger = slower on CPU
    OLLAMA_THINK: bool = False
    OLLAMA_MAX_TOKENS: int = 700         # cap per reply — CPU generation is slow           # reasoning mode for qwen3/deepseek-r1 (slower, uses more tokens)

    # LM Studio — local, no key: start the server in LM Studio (Developer tab)
    # or with `lms server start`. Model names come from `lms ls` / GET /v1/models.
    LMSTUDIO_ENABLED: bool = False
    LMSTUDIO_BASE_URL: str = "http://localhost:1234/v1"
    LMSTUDIO_MODEL: str = "meta-llama-3.1-8b-instruct"
    LMSTUDIO_FAST_MODEL: str = ""        # empty = same as LMSTUDIO_MODEL

    # ── LLM call behaviour ──────────────────────────────────────────────
    LLM_TIMEOUT_SECS: float = 90.0
    LLM_MAX_RETRIES: int = 3
    # If a provider asks us to wait longer than this, fail over instead
    # (only when another provider is available).
    LLM_MAX_WAIT_SECS: float = 25.0

    # ── Retrieval ───────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    TOP_K: int = 6
    CANDIDATE_POOL: int = 24
    # Below this reranker relevance (0–1) for every passage, decline without
    # calling the LLM. Set to 0 to always ask the model.
    MIN_RELEVANCE: float = 0.02
    CHUNK_WORDS: int = 200
    CHUNK_OVERLAP_WORDS: int = 40

    # ── Pipeline ────────────────────────────────────────────────────────
    NUM_SAMPLES: int = 2
    SAMPLE_TEMPERATURE: float = 0.8
    ANSWER_MAX_TOKENS: int = 1200
    CONFIDENCE_THRESHOLD: float = 0.60

    # Calibration weights (renormalised over whichever signals are present)
    W_SUPPORT: float = 0.35
    W_GROUNDING: float = 0.15
    W_CITATION: float = 0.10
    W_CONSISTENCY: float = 0.15
    W_REFLECTION: float = 0.15
    W_RETRIEVAL: float = 0.10

    # ── Storage / server ────────────────────────────────────────────────
    DATA_DIR: str = "data"
    MAX_UPLOAD_MB: int = 25
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"
    LOG_LEVEL: str = "INFO"

    # Pasted keys often carry stray whitespace/newlines.
    @field_validator("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", mode="before")
    @classmethod
    def _strip(cls, v):
        return (v or "").strip()

    # ── Derived helpers ─────────────────────────────────────────────────
    @property
    def data_path(self) -> Path:
        p = Path(self.DATA_DIR)
        return p if p.is_absolute() else BACKEND_DIR / p

    @property
    def provider_order(self) -> List[str]:
        return [p.strip().lower() for p in self.LLM_PROVIDER_ORDER.split(",") if p.strip()]

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
