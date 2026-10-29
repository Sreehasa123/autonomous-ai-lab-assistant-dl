"""Experiment storage and persistence manager for the Autonomous AI Lab Assistant."""

import json
import uuid
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

STORAGE_DIR = Path(__file__).resolve().parent / "workspace"
STORAGE_FILE = STORAGE_DIR / "experiments_history.json"


def _ensure_storage_exists():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    if not STORAGE_FILE.exists():
        with open(STORAGE_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


def get_all_experiments() -> List[Dict[str, Any]]:
    """Retrieve all saved experiments sorted by last updated timestamp."""
    _ensure_storage_exists()
    try:
        with open(STORAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return sorted(
                data, key=lambda x: x.get("updated_at", ""), reverse=True
            )
    except Exception as e:
        print(f"Error loading experiments: {e}")
        return []


def get_experiment_by_id(exp_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single experiment by ID."""
    experiments = get_all_experiments()
    for exp in experiments:
        if exp.get("id") == exp_id:
            return exp
    return None


def save_or_update_experiment(exp_data: Dict[str, Any]) -> str:
    """Save a new experiment or update an existing one by its ID."""
    _ensure_storage_exists()
    experiments = get_all_experiments()

    exp_id = exp_data.get("id")
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not exp_id:
        exp_id = str(uuid.uuid4())[:8]
        exp_data["id"] = exp_id
        exp_data["created_at"] = now_str
        exp_data["updated_at"] = now_str
        experiments.append(exp_data)
    else:
        # Update existing
        found = False
        for i, item in enumerate(experiments):
            if item.get("id") == exp_id:
                exp_data["updated_at"] = now_str
                # preserve created_at
                exp_data["created_at"] = item.get("created_at", now_str)
                experiments[i] = exp_data
                found = True
                break
        if not found:
            exp_data["created_at"] = now_str
            exp_data["updated_at"] = now_str
            experiments.append(exp_data)

    with open(STORAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2)

    return exp_id


def delete_experiment(exp_id: str) -> bool:
    """Delete an experiment by its ID."""
    _ensure_storage_exists()
    experiments = get_all_experiments()
    initial_len = len(experiments)
    experiments = [e for e in experiments if e.get("id") != exp_id]

    if len(experiments) < initial_len:
        with open(STORAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(experiments, f, indent=2)
        return True
    return False
