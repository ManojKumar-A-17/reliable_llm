"""
configs/config.py
-----------------
LEGACY SHIM — all real config now lives in configs/settings.py.

This file is kept so older code that does `from configs.config import X`
continues to work without changes during the migration.
"""
from configs.settings import (   # noqa: F401  (re-export everything)
    settings,
    GROQ_API_KEY,
    MULTI_LLM_MODELS,
    RESPONSE_GENERATOR_MODEL,
    FACT_CHECKER_MODEL,
    REFLECTION_MODEL,
    CONSISTENCY_MODEL,
    RESPONSE_IMPROVER_MODEL,
    NUM_SAMPLES,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    CONFIDENCE_THRESHOLD,
)
