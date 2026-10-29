"""Entry point for the Autonomous AI Lab Assistant."""

import argparse
import uuid
from pathlib import Path
from core.state import LabWorkflowState
from core.config import DATASET_DIR, EXPERIMENT_DIR, SCRIPT_DIR


def initialize_environment():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)


def create_initial_state(
    dataset_path: str, task_description: str
) -> LabWorkflowState:
    return {
        "project_id": str(uuid.uuid4())[:8],
        "task_description": task_description,
        "current_phase": "INIT",
        "iteration": 1,
        "max_iterations": 3,
        "dataset_profile": None,
        "preprocessing_plan": None,
        "model_spec": None,
        "training_artifacts": None,
        "diagnostic_result": None,
        "report_markdown": None,
        "error_log": [],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous AI Lab Assistant for Deep Learning"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to raw dataset (CSV or image folder)",
        required=False,
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Image classification task with automatic model architecture selection",
        help="Description of the learning objective",
    )
    args = parser.parse_args()

    initialize_environment()
    print("=" * 60)
    print("🤖 Autonomous AI Lab Assistant - Deep Learning Multi-Agent Engine")
    print("=" * 60)

    state = create_initial_state(args.data or "", args.task)
    print(f"Initialized Experiment Run: {state['project_id']}")
    print(f"Goal: {state['task_description']}")
    print("\nReady to run multi-agent graph.")


if __name__ == "__main__":
    main()
