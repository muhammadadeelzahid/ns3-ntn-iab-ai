"""Shared paths and defaults for the ai_llm package."""

from pathlib import Path

AI_LLM_DIR = Path(__file__).resolve().parent
CONFIG_DIR = AI_LLM_DIR / "config"

DEFAULT_SEMANTIC_SCALER = str(CONFIG_DIR / "feature_scaler_semantic.json")
DEFAULT_NETWORK_SCALER = str(CONFIG_DIR / "feature_scaler_network.json")

# 6-rung ladder aligned with metadata CSV labels (250k .. 12000k) and DashClient::rates.
NS3_RATE_LADDER_BPS = [250_000, 750_000, 1_500_000, 3_000_000, 6_000_000, 12_000_000]
NUM_LADDER_RUNGS = len(NS3_RATE_LADDER_BPS)
