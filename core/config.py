"""System configuration and constants for the Autonomous AI Lab Assistant."""

import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = BASE_DIR / "workspace"
DATASET_DIR = WORKSPACE_DIR / "datasets"
EXPERIMENT_DIR = WORKSPACE_DIR / "experiments"
SCRIPT_DIR = WORKSPACE_DIR / "scripts"


class SystemConfig(BaseModel):
    default_device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    max_training_timeout_sec: int = 600
    max_hpo_trials: int = 5
    default_batch_size: int = 32
    max_retries_per_agent: int = 3
    llm_model_name: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")


config = SystemConfig()
