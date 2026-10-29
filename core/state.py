"""Shared state and data models for the Autonomous AI Lab Assistant workflow."""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class DatasetProfile(BaseModel):
    modality: Literal["tabular", "vision", "text"] = "tabular"
    num_samples: int = 0
    num_features_or_channels: int = 0
    num_classes: Optional[int] = None
    target_column: Optional[str] = None
    class_distribution: Dict[str, int] = Field(default_factory=dict)
    missing_ratio: Dict[str, float] = Field(default_factory=dict)
    data_leakage_risks: List[str] = Field(default_factory=list)
    raw_data_path: str = ""
    sample_preview: List[Dict[str, Any]] = Field(default_factory=list)


class PreprocessingPlan(BaseModel):
    transform_pipeline_code: str = ""
    train_split_path: str = ""
    val_split_path: str = ""
    test_split_path: str = ""
    input_shape: List[int] = Field(default_factory=list)
    batch_size: int = 32
    normalization_params: Dict[str, Any] = Field(default_factory=dict)
    augmentations_applied: List[str] = Field(default_factory=list)


class ModelSpecification(BaseModel):
    architecture_name: str = ""
    backbone_type: str = ""
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    loss_function: str = "CrossEntropyLoss"
    optimizer: str = "AdamW"
    model_script_path: str = ""
    reasoning: str = ""


class TrainingArtifacts(BaseModel):
    best_checkpoint_path: str = ""
    history: Dict[str, List[float]] = Field(default_factory=dict)
    epochs_trained: int = 0
    training_duration_seconds: float = 0.0
    hardware_usage: Dict[str, float] = Field(default_factory=dict)
    early_stopped: bool = False


class DiagnosticResult(BaseModel):
    test_metrics: Dict[str, float] = Field(default_factory=dict)
    confusion_matrix_path: Optional[str] = None
    xai_saliency_paths: List[str] = Field(default_factory=list)
    failure_patterns: List[str] = Field(default_factory=list)
    diagnosis_category: Literal[
        "healthy",
        "underfitting",
        "overfitting",
        "class_imbalance_collapse",
        "exploding_gradients",
    ] = "healthy"
    actionable_recommendations: List[str] = Field(default_factory=list)


class LabWorkflowState(TypedDict):
    project_id: str
    task_description: str
    current_phase: str
    iteration: int
    max_iterations: int
    dataset_profile: Optional[DatasetProfile]
    preprocessing_plan: Optional[PreprocessingPlan]
    model_spec: Optional[ModelSpecification]
    training_artifacts: Optional[TrainingArtifacts]
    diagnostic_result: Optional[DiagnosticResult]
    report_markdown: Optional[str]
    error_log: List[str]
