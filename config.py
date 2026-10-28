import os

RANDOM_SEED = 42
EMBEDDING_DIM = 128

SYMBOLIC_WEIGHTS = {
    "mild": 0.2,
    "moderate": 0.5,
    "high": 0.8,
    "critical": 1.0
}

CONTRAINDICATION_ESCALATION = 0.2

THETA_CONFLICT = 0.3
THETA_FALLBACK = 0.5
CONFIDENCE_MARGIN = 0.2

ALPHA_WEIGHTS = {
    "rule": 0.40,
    "context": 0.35,
    "data": 0.25
}

MODIFIERS = {
    "gamma_age": 0.012,
    "gamma_renal": 0.015,
    "gamma_poly": 0.08,
    "age_threshold": 65,
    "egfr_reference": 90,
    "poly_reference": 5
}

LLM_CONFIG = {
    "provider": os.getenv("LLM_PROVIDER", "openai"),
    "model": os.getenv("LLM_MODEL", "gpt-4"),
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "base_url": os.getenv("LLM_BASE_URL", None),
    "temperature": 0.1,
    "max_tokens": 256
}

RULES_PATH = "data/rules_data.json"
TWOSIDES_URL = "https://nsides.io/data/"
TWOSIDES_MIN_COUNT = 5
TWOSIDES_MIN_PRR = 1.5

PROFILES = {
    "A": {"name": "Healthy young adult", "n": 20, "age_range": (25, 40), "egfr_range": (91, 120),
          "n_conditions": 0, "n_meds_range": (2, 3)},
    "B": {"name": "Frail elderly", "n": 30, "age_range": (75, 90), "egfr_range": (45, 75),
          "n_conditions": (2, 4), "n_meds_range": (5, 7)},
    "C": {"name": "Moderate renal impairment", "n": 25, "age_range": (50, 70), "egfr_range": (30, 59),
          "n_conditions": (1, 3), "n_meds_range": (4, 6)},
    "D": {"name": "Chronic polypharmacy", "n": 25, "age_range": (55, 75), "egfr_range": (60, 110),
          "n_conditions": (4, 6), "n_meds_range": (8, 12)}
}

TRAJECTORY_MONTHS = 24
