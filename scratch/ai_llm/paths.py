"""Shared paths and defaults for the ai_llm package."""

from pathlib import Path

AI_LLM_DIR = Path(__file__).resolve().parent
CONFIG_DIR = AI_LLM_DIR / "config"

DEFAULT_SEMANTIC_SCALER = str(CONFIG_DIR / "feature_scaler_semantic.json")
DEFAULT_NETWORK_SCALER = str(CONFIG_DIR / "feature_scaler_network.json")
