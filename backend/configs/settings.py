"""
backend/configs/settings.py
----------------------------
Pydantic-settings configuration — Google Gemini API.

Free-tier RPM limits (as of 2026) for this API key:
  gemini-3.6-flash        5 RPM    ← main model (only model available on this key)
  gemini-3.5-flash-lite   higher   ← sampler model (higher free-tier quota)
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Gemini API ────────────────────────────────────────────────────
    GEMINI_API_KEY: str = ""

    # ── Model names ───────────────────────────────────────────────────
    # gemini-3.6-flash      — 5 RPM free tier  (only model available on this key)
    # gemini-3.5-flash-lite — higher free-tier quota for sampling calls
    ANSWER_MODEL:     str = "gemini-3.6-flash"
    VERIFIER_MODEL:   str = "gemini-3.6-flash"
    REFLECTION_MODEL: str = "gemini-3.6-flash"
    SAMPLER_MODEL:    str = "gemini-3.5-flash-lite"   # higher quota for 2× sampling
    IMPROVER_MODEL:   str = "gemini-3.6-flash"

    # ── Retrieval ─────────────────────────────────────────────────────
    TOP_K_RETRIEVE:  int   = 6
    HYBRID_ALPHA:    float = 0.5      # 0=BM25-only, 1=vector-only
    KB_INDEX_PATH:   str   = "data/index/knowledge"
    UPLOAD_DIR:      str   = "data/uploads"

    # ── Sampling ─────────────────────────────────────────────────────
    NUM_SAMPLES:  int   = 2
    TEMPERATURE:  float = 0.6
    TOP_P:        float = 0.95

    # ── Confidence weights ────────────────────────────────────────────
    EVIDENCE_WEIGHT:      float = 0.50
    CONSISTENCY_WEIGHT:   float = 0.25
    REFLECTION_WEIGHT:    float = 0.25
    CONFIDENCE_THRESHOLD: float = 0.65

    # ── Server ────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    def require_gemini_key(self) -> str:
        if not self.GEMINI_API_KEY:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set.\n"
                "Add it to backend/.env:\n  GEMINI_API_KEY=AIza..."
            )
        return self.GEMINI_API_KEY

    # ── Backwards-compat shim ─────────────────────────────────────────
    def require_groq_key(self) -> str:
        """Old agents call require_groq_key() — redirect to Gemini."""
        return self.require_gemini_key()


settings = Settings()

# Old flat-name imports still work
GROQ_API_KEY              = settings.GEMINI_API_KEY
RESPONSE_GENERATOR_MODEL  = settings.ANSWER_MODEL
FACT_CHECKER_MODEL        = settings.VERIFIER_MODEL
REFLECTION_MODEL          = settings.REFLECTION_MODEL
CONSISTENCY_MODEL         = settings.SAMPLER_MODEL
RESPONSE_IMPROVER_MODEL   = settings.IMPROVER_MODEL
NUM_SAMPLES               = settings.NUM_SAMPLES
TEMPERATURE               = settings.TEMPERATURE
TOP_P                     = settings.TOP_P
TOP_K                     = 50
CONFIDENCE_THRESHOLD      = settings.CONFIDENCE_THRESHOLD
