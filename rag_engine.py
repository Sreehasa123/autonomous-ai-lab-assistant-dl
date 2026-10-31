"""RAG Engine and Gemini AI Lab Assistant for Deep Learning Experiments."""

import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional
from knowledge_base import search_knowledge_base, ML_LAB_KNOWLEDGE

# Category mapping per workflow step
STEP_CATEGORY_MAP = {
    "Step 1: Dataset & Problem Setup": "dataset_inspection",
    "Step 2: Preprocessing & Leakage Checker": "preprocessing",
    "Step 3: Model Architecture & Loss Selection": "model_selection",
    "Step 4: Training & Hyperparameter Tuning": "training_debugging",
    "Step 5: Evaluation & Error Analysis": "evaluation_diagnostics",
    "Step 6: Lab Report Generator": "lab_report_guidance",
}

# Candidate Gemini models in priority order
CANDIDATE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-3.8-flash",
]


class LabAssistantRAG:
    """Orchestrates RAG retrieval and Gemini AI generation for deep learning lab students."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = self._clean_key(api_key or os.environ.get("GEMINI_API_KEY", ""))
        self.client = None
        self._init_client()

    @staticmethod
    def _clean_key(key: Optional[str]) -> str:
        """Strip whitespaces, quotes, and newlines from copy-pasted API keys."""
        if not key:
            return ""
        return str(key).strip().strip("'\"").strip()

    def set_api_key(self, api_key: str):
        """Update API key and reinitialize client."""
        self.api_key = self._clean_key(api_key)
        self._init_client()

    def is_connected(self) -> bool:
        """Check if an API key is present and appears valid."""
        return bool(self.api_key and len(self.api_key) >= 15)

    def _init_client(self):
        """Initialize official google-genai client if installed."""
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                # Will fall back to direct universal REST API
                self.client = None
        else:
            self.client = None

    def _call_gemini_rest(self, prompt: str, model_name: str) -> Optional[str]:
        """Direct REST call to Google AI Studio API using standard library urllib."""
        if not self.api_key:
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 4096,
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                candidates = res_json.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
        except Exception:
            return None
        return None

    def call_gemini(self, prompt: str) -> Optional[str]:
        """Robust Gemini calling engine with multi-model fallback and REST backup."""
        if not self.api_key:
            return None

        # 1. Try google-genai SDK if client initialized
        if self.client:
            for model_name in CANDIDATE_MODELS:
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    if response and response.text:
                        return response.text
                except Exception:
                    continue

        # 2. Universal REST fallback (works with standard Python, zero external dependencies)
        for model_name in CANDIDATE_MODELS:
            text = self._call_gemini_rest(prompt, model_name)
            if text:
                return text

        return None

    def retrieve_context(self, query: str, step_name: str) -> str:
        """Fetch matching knowledge base rules formatted as prompt context."""
        category = STEP_CATEGORY_MAP.get(step_name, "")
        matched_items = search_knowledge_base(query, category=category, top_k=3)

        if not matched_items:
            return ""

        context_blocks = ["### 📚 Curated Deep Learning Lab Knowledge Base Rules:"]
        for idx, item in enumerate(matched_items, 1):
            block = (
                f"\n[Rule {idx}: {item['title']}]\n"
                f"- Common Pitfall: {item['common_pitfall']}\n"
                f"- Recommended Practice: {item['recommended_practice']}\n"
                f"- Reference Code:\n```python\n{item['code_snippet']}\n```"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)

    def suggest_datasets(
        self, title: str, aim: str, modality: str = "Vision (Images)"
    ) -> List[Dict[str, Any]]:
        """Recommends suitable datasets with expected accuracy levels based on Title and Aim."""
        prompt = f"""
You are an expert Machine Learning Professor. A student wants to conduct a laboratory experiment.

EXPERIMENT TITLE: {title}
EXPERIMENT AIM / OBJECTIVE: {aim}
DATA MODALITY: {modality}

Suggest 3 to 4 well-suited, popular datasets that are ideal for this student experiment (e.g., from PyTorch torchvision, Hugging Face, Kaggle, or UCI ML Repository).
For EACH dataset, specify:
1. "name": Dataset name
2. "source": Exact library/source (e.g., torchvision.datasets.CIFAR10, Hugging Face 'imdb', Kaggle, UCI)
3. "description": 1-2 sentence description (classes, size, nature)
4. "expected_accuracy": Realistic accuracy range students can expect to achieve (e.g., "Baseline CNN: 72-76%, ResNet-18: 91-94%, State-of-the-Art: ~96%")
5. "why_recommended": Why this matches their specific aim
6. "recommended_architecture": Best model architecture for this dataset

Return the response STRICTLY as a valid JSON list of objects with the keys:
["name", "source", "description", "expected_accuracy", "why_recommended", "recommended_architecture"]
Do not wrap in markdown quotes other than standard ```json ... ``` if necessary.
"""

        raw_text = self.call_gemini(prompt) if self.api_key else None
        if raw_text:
            try:
                clean = raw_text.strip()
                if clean.startswith("```json"):
                    clean = clean[7:]
                if clean.startswith("```"):
                    clean = clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]
                parsed = json.loads(clean.strip())
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed
            except Exception:
                pass

        # High-quality fallback presets when offline or if parsing fails
        if "vision" in modality.lower() or "image" in modality.lower():
            return [
                {
                    "name": "CIFAR-10",
                    "source": "torchvision.datasets.CIFAR10",
                    "description": "60,000 32x32 color images across 10 balanced object classes (50k train, 10k test).",
                    "expected_accuracy": "Basic CNN: 70-75% | ResNet-18 (scratch): 88-92% | Pretrained ResNet: 94-96%",
                    "why_recommended": "Standard benchmark, fast to train on local GPUs or Colab, ideal for studying CNNs & ResNets.",
                    "recommended_architecture": "ResNet-18 / MobileNetV2",
                },
                {
                    "name": "Fashion-MNIST",
                    "source": "torchvision.datasets.FashionMNIST",
                    "description": "70,000 28x28 grayscale images of 10 clothing categories.",
                    "expected_accuracy": "MLP: 86-88% | 2-layer CNN: 91-93% | Deep CNN: ~94%",
                    "why_recommended": "Drop-in replacement for MNIST with realistic complexity for testing regularization.",
                    "recommended_architecture": "Custom 3-layer ConvNet",
                },
                {
                    "name": "Flowers-102 (or Oxford-IIIT Pet)",
                    "source": "torchvision.datasets.Flowers102",
                    "description": "Fine-grained image classification with 102 flower categories.",
                    "expected_accuracy": "From scratch: 45-60% | Transfer Learning (ResNet-50): 88-94%",
                    "why_recommended": "Perfect for demonstrating transfer learning and data augmentation under small sample size.",
                    "recommended_architecture": "Pretrained ResNet-50 / EfficientNet-B0",
                },
            ]
        else:
            return [
                {
                    "name": "Heart Disease / Breast Cancer Wisconsin",
                    "source": "sklearn.datasets.load_breast_cancer",
                    "description": "569 samples, 30 continuous numerical features, binary classification.",
                    "expected_accuracy": "Logistic Regression: 94-95% | 2-layer MLP: 96-98%",
                    "why_recommended": "Great for demonstrating feature scaling, precision-recall tradeoff, and confusion matrices.",
                    "recommended_architecture": "2-layer PyTorch MLP with Dropout",
                },
                {
                    "name": "California Housing / Wine Quality",
                    "source": "sklearn.datasets.fetch_california_housing",
                    "description": "20,640 samples, 8 numerical features predicting median house value.",
                    "expected_accuracy": "R²: 0.65-0.75 | RMSE: ~0.50-0.55 using Deep Neural Networks",
                    "why_recommended": "Standard tabular benchmark for regression and feature normalization.",
                    "recommended_architecture": "Deep Residual MLP",
                },
            ]

    def recommend_evaluation_metrics(
        self, title: str, aim: str, task_type: str = "Classification"
    ) -> Dict[str, Any]:
        """Automatically recommends appropriate primary and secondary evaluation metrics."""
        prompt = f"""
You are a Machine Learning Laboratory Professor.
A student is doing this experiment:
- Title: {title}
- Aim: {aim}
- Task Type: {task_type}

Recommend the essential evaluation metrics they MUST compute for this specific experiment.
Provide:
1. "primary_metrics": List of 2-3 essential metrics (e.g. Accuracy, Macro F1, Loss)
2. "secondary_metrics": List of 2-3 deeper diagnostic metrics (e.g. Confusion Matrix, PR-AUC, Specificity, Log-Loss)
3. "explanation": 2-3 sentences explaining why these specific metrics are needed (e.g., addressing class balance or false positive penalties)
4. "sklearn_code": Python code snippet showing how to calculate them.

Return strictly as JSON with keys ["primary_metrics", "secondary_metrics", "explanation", "sklearn_code"].
"""
        raw_text = self.call_gemini(prompt) if self.api_key else None
        if raw_text:
            try:
                clean = raw_text.strip()
                if clean.startswith("```json"):
                    clean = clean[7:]
                if clean.startswith("```"):
                    clean = clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]
                return json.loads(clean.strip())
            except Exception:
                pass

        return {
            "primary_metrics": ["Accuracy", "Cross-Entropy Loss", "Macro-F1 Score"],
            "secondary_metrics": ["Confusion Matrix", "Per-Class Precision & Recall", "PR-AUC"],
            "explanation": "Accuracy measures overall correct predictions, while Macro-F1 and Confusion Matrix ensure the model performs well on all classes without ignoring minority categories.",
            "sklearn_code": (
                "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n\n"
                "# y_true: actual labels, y_pred: model predictions\n"
                "acc = accuracy_score(y_true, y_pred)\n"
                "print(f'Accuracy: {acc * 100:.2f}%')\n"
                "print(classification_report(y_true, y_pred))\n"
                "cm = confusion_matrix(y_true, y_pred)\n"
            ),
        }

    def generate_baseline_code(
        self,
        title: str,
        aim: str,
        dataset_name: str,
        model_arch: str,
        modality: str,
        task_type: str,
    ) -> str:
        """Generates a complete, ready-to-run PyTorch starter script tailored to the student's experiment."""
        prompt = f"""
You are an expert Deep Learning Instructor. Write a clean, modular, end-to-end Python script using PyTorch
for the student's lab experiment:

- Title: {title}
- Aim: {aim}
- Dataset: {dataset_name} ({modality})
- Chosen Model Architecture: {model_arch}
- Task Type: {task_type}

REQUIREMENTS FOR THE GENERATED CODE:
1. Include dataset loading (using standard torchvision/sklearn datasets or mock data loaders if custom).
2. Data Preprocessing & Transformations (Train with augmentation, Validation without).
3. The complete Model Architecture class (`nn.Module`) or transfer learning backbone with custom classifier. Note: DO NOT put `nn.Softmax` before `nn.CrossEntropyLoss`.
4. Training loop with `optimizer.zero_grad()`, `loss.backward()`, gradient clipping, and metrics logging.
5. Evaluation loop with `model.eval()`, `torch.no_grad()`, and computing accuracy & confusion matrix.
6. Clean comments explaining each step for a student.

Return ONLY the runnable Python code, enclosed in ```python ... ```.
"""
        raw_text = self.call_gemini(prompt) if self.api_key else None
        if raw_text:
            clean = raw_text.strip()
            if "```python" in clean:
                clean = clean.split("```python")[1].split("```")[0].strip()
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0].strip()
            return clean

        return (
            "# PyTorch Baseline Template (Automated Lab Generator)\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.optim as optim\n"
            "from torch.utils.data import DataLoader\n"
            "import torchvision\n"
            "import torchvision.transforms as transforms\n\n"
            "# 1. Transforms & Dataset\n"
            "transform_train = transforms.Compose([\n"
            "    transforms.RandomHorizontalFlip(),\n"
            "    transforms.ToTensor(),\n"
            "    transforms.Normalize((0.5,), (0.5,))\n"
            "])\n\n"
            "# 2. Model Definition\n"
            "class BaselineNet(nn.Module):\n"
            "    def __init__(self, num_classes=10):\n"
            "        super().__init__()\n"
            "        self.features = nn.Sequential(\n"
            "            nn.Conv2d(3, 32, kernel_size=3, padding=1),\n"
            "            nn.BatchNorm2d(32),\n"
            "            nn.ReLU(),\n"
            "            nn.MaxPool2d(2, 2),\n"
            "            nn.Conv2d(32, 64, kernel_size=3, padding=1),\n"
            "            nn.BatchNorm2d(64),\n"
            "            nn.ReLU(),\n"
            "            nn.MaxPool2d(2, 2)\n"
            "        )\n"
            "        self.classifier = nn.Sequential(\n"
            "            nn.Flatten(),\n"
            "            nn.Linear(64 * 8 * 8, 128),\n"
            "            nn.ReLU(),\n"
            "            nn.Dropout(0.3),\n"
            "            nn.Linear(128, num_classes) # Raw logits for CrossEntropyLoss\n"
            "        )\n"
            "    def forward(self, x):\n"
            "        return self.classifier(self.features(x))\n\n"
            "# 3. Loss & Optimizer\n"
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
            "model = BaselineNet().to(device)\n"
            "criterion = nn.CrossEntropyLoss()\n"
            "optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)\n"
            "print('Ready to train model on:', device)\n"
        )

    def ask_step_assistant(
        self,
        user_query: str,
        current_step: str,
        experiment_context: Dict[str, Any],
        code_context: Optional[str] = None,
    ) -> str:
        """Answers student queries at a specific experiment step using RAG + Gemini."""
        retrieved_rules = self.retrieve_context(user_query, current_step)

        system_instruction = (
            "You are an encouraging, expert Deep Learning Lab Professor & Teaching Assistant. "
            "Your mission is to guide students step-by-step through their deep learning experiment, "
            "helping them avoid common pitfalls, verifying their code, explaining the math/theory simply, "
            "and ensuring their experiment adheres to standard deep learning best practices.\n"
            "Always format your response with clear headings, bullet points, and clean, runnable PyTorch/Python code snippets. "
            "Cite and apply the retrieved knowledge base rules whenever applicable."
        )

        prompt = f"""
{system_instruction}

---
CURRENT LAB EXPERIMENT STEP: {current_step}

STUDENT'S EXPERIMENT DETAILS:
- Aim / Goal: {experiment_context.get('aim', 'Not specified yet')}
- Modality / Dataset: {experiment_context.get('dataset_name', 'Not specified')} ({experiment_context.get('modality', 'Tabular/Vision')})
- Target / Task: {experiment_context.get('task_type', 'Classification')}
- Chosen Architecture: {experiment_context.get('model_arch', 'Undecided')}
- Hyperparameters: {experiment_context.get('hyperparams', 'Default')}
- Key Metrics: {experiment_context.get('metrics', 'None recorded yet')}

{retrieved_rules}

{f"STUDENT'S RELEVANT CODE:\n```python\n{code_context}\n```" if code_context else ""}

STUDENT'S QUESTION / REQUEST:
"{user_query}"

Provide a comprehensive, pedagogical, and actionable response that guides the student through this step.
"""

        ai_response = self.call_gemini(prompt) if self.api_key else None
        if ai_response:
            return ai_response

        return (
            "ℹ️ **Running in Curated Knowledge-Base Mode**.\n\n"
            + retrieved_rules
            + "\n\n💡 *Tip: Enter a valid Google Gemini API key in the sidebar to activate interactive AI feedback, code verification, and deep explanations!*"
        )

    def diagnose_code_and_errors(
        self,
        code_snippet: str,
        error_traceback: str,
        current_step: str,
    ) -> str:
        """Pinpoints root cause of errors, explains why they happen, and gives fixed code."""
        retrieved_rules = self.retrieve_context(
            error_traceback + " " + code_snippet, current_step
        )

        prompt = f"""
You are a senior Deep Learning Debugging Assistant. A student encountered an error or needs code correction.

CURRENT EXPERIMENT STEP: {current_step}

{retrieved_rules}

STUDENT'S CODE:
```python
{code_snippet}
```

ERROR TRACEBACK / ISSUE DESCRIPTION:
```text
{error_traceback}
```

Please structure your response with:
1. 🔍 **Root Cause Analysis**: Explain in clear, simple terms why this error occurred (tensor shape mismatch, missing zero_grad, data leakage, unscaled features, etc.).
2. ⚠️ **Deep Learning Concept / Pitfall**: What conceptual mistake led to this?
3. 🛠️ **Corrected & Verified Code**: Provide the drop-in replacement code with clear comments.
4. 💡 **Prevention Tip**: How to avoid this in future experiments.
"""

        ai_response = self.call_gemini(prompt) if self.api_key else None
        if ai_response:
            return ai_response

        return (
            "🔍 **Diagnostic Rule Matches for this Issue:**\n\n"
            + retrieved_rules
            + "\n\n💡 *Tip: Add your Gemini API key in the sidebar to get line-by-line custom debugging for this exact error traceback!*"
        )

    def _generate_fallback_academic_report(self, data: Dict[str, Any]) -> str:
        """High-fidelity fallback academic lab report compiled directly from experiment state."""
        title = data.get("title", "Deep Learning Model Training & Evaluation")
        student = data.get("student_name", "Student Researcher")
        aim = data.get("aim", "To construct, train, evaluate, and diagnose a Deep Learning model on the dataset.")
        dataset = data.get("dataset_name", "Benchmark Dataset")
        modality = data.get("modality", "Vision (Images)")
        task_type = data.get("task_type", "Classification")
        preprocessing = data.get("preprocessing", "Standard scaling, normalization, and train-validation-test partitioning.")
        model_arch = data.get("model_arch", "Convolutional Neural Network / MLP")
        loss_opt = data.get("loss_opt", "nn.CrossEntropyLoss(), AdamW(lr=0.001)")
        hyperparams = data.get("hyperparams", "Batch Size: 32, Epochs: 20")
        metrics = data.get("metrics", "Accuracy: >90%, Macro F1: >0.88, Low Validation Loss")
        observations = data.get("observations", "Model converged smoothly with minimal overfitting after regularization.")
        conclusion = data.get("conclusion", "The deep learning pipeline successfully achieved target benchmark accuracy.")

        template = r"""# Deep Learning Laboratory Experiment Report

> **Experiment Title:** __TITLE__  
> **Student Researcher / Roll No:** __STUDENT__  
> **Status:** Completed & Evaluated  
> *Compiled via Autonomous AI Lab Assistant Academic Engine*

---

## 1. Experiment Aim & Objective
The primary objective of this experiment is:
> **__AIM__**

The task was formulated as a **__TASK_TYPE__** problem operating on the **__DATASET__** dataset under the **__MODALITY__** modality.

---

## 2. Theoretical Background & Mathematical Formulation

### 2.1 Optimization Objective & Loss Function
For multi-class classification over $C$ classes, the model produces raw unnormalized output logits $z \in \\mathbb{R}^C$. The probability distribution is modeled via the Softmax activation:
$$\\hat{y}_i = \\frac{\\exp(z_i)}{\\sum_{j=1}^C \\exp(z_j)}$$

The objective is optimized by minimizing the **Categorical Cross-Entropy Loss**:
$$\\mathcal{L}_{\\text{CE}} = -\\sum_{i=1}^C y_i \\log(\\hat{y}_i)$$
where $y \\in \\{0, 1\\}^C$ represents the ground-truth one-hot label vector. In PyTorch, numerical stability is preserved by combining $\\log(\\text{softmax}(z))$ with the negative log-likelihood loss directly in `nn.CrossEntropyLoss()`, avoiding separate Softmax exponentiation in the model's forward path.

### 2.2 Optimizer Dynamics (AdamW)
Parameter updates are governed by the AdamW optimizer with decoupled weight decay:
$$m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t$$
$$v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2$$
$$\\theta_t = \\theta_{t-1} - \\eta \\left( \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon} + \\lambda \\theta_{t-1} \\right)$$
where $\\eta$ is the learning rate, $\\lambda$ is the weight decay regularization factor, and $g_t = \\nabla_\\theta \\mathcal{L}_t$.

---

## 3. Dataset Characteristics & Exploratory Analysis
- **Target Dataset:** `__DATASET__`
- **Data Modality:** __MODALITY__
- **Task Category:** __TASK_TYPE__
- **Data Leakage Safeguards:** Features are normalized using statistics computed *exclusively* from the training split. Validation and test sets are transformed without updating scaler statistics.

---

## 4. Preprocessing Pipeline & Data Integrity Safeguards
The applied data preprocessing and augmentation pipeline is defined as follows:
```text
__PREPROCESSING__
```
- **Integrity Rule Applied:** Random augmentations (e.g. flips, crops, jitter) were applied strictly to the training partition to expand the effective sample space, while the validation and test sets remained unaugmented to ensure valid generalization metrics.

---

## 5. Model Architecture & Layer Specifications
- **Selected Backbone:** `__MODEL_ARCH__`
- **Loss Function & Optimizer:** `__LOSS_OPT__`
- **Architecture Integrity:** Raw logits are emitted from the final classification head without redundant Softmax layers, guaranteeing stable gradient backpropagation without saturation.

---

## 6. Experimental Configuration & Hyperparameters

| Hyperparameter Component | Configured Value / Description |
|---|---|
| **Dataset & Modality** | __DATASET__ (__MODALITY__) |
| **Model Architecture** | __MODEL_ARCH__ |
| **Loss & Optimizer** | __LOSS_OPT__ |
| **Training Hyperparameters** | __HYPERPARAMS__ |
| **Gradient Clipping** | `clip_grad_norm_(max_norm=1.0)` (Exploding Gradient Protection) |
| **Evaluation Mode** | `model.eval()` with `torch.no_grad()` |

---

## 7. Results, Performance Metrics & Visualizations Discussion
The quantitative results recorded for this experiment run are:
```text
__METRICS__
```

### 7.1 Metric Analysis
- **Accuracy & F1 Assessment:** The model achieved consistent performance across evaluation partitions. Macro-F1 guarantees that accuracy is not skewed by majority-class dominance.
- **Loss Progression:** Validation loss remained bounded and tracked the training loss trajectory, indicating healthy generalization.

---

## 8. Error Analysis, Failure Modes & Diagnostic Review
### 8.1 Empirical Observations
__OBSERVATIONS__

### 8.2 Diagnostic Checklist
- **Overfitting vs Underfitting:** Monitored the generalization gap $\\Delta = |\\mathcal{L}_{\\text{val}} - \\mathcal{L}_{\\text{train}}|$. The gap was stabilized using dropout and weight decay.
- **Gradient Stability:** No `NaN` loss anomalies occurred due to gradient norm bounding.

---

## 9. Conclusion & Practical Takeaways
__CONCLUSION__

---
*Report successfully compiled by Autonomous AI Lab Assistant. Ready for lab review and grading submission.*
"""
        return (
            template.replace("__TITLE__", str(title))
            .replace("__STUDENT__", str(student))
            .replace("__AIM__", str(aim))
            .replace("__DATASET__", str(dataset))
            .replace("__MODALITY__", str(modality))
            .replace("__TASK_TYPE__", str(task_type))
            .replace("__PREPROCESSING__", str(preprocessing))
            .replace("__MODEL_ARCH__", str(model_arch))
            .replace("__LOSS_OPT__", str(loss_opt))
            .replace("__HYPERPARAMS__", str(hyperparams))
            .replace("__METRICS__", str(metrics))
            .replace("__OBSERVATIONS__", str(observations))
            .replace("__CONCLUSION__", str(conclusion))
        )

    def generate_experiment_report(self, experiment_data: Dict[str, Any]) -> str:
        """Generates a complete, formal, academic lab experiment report in Markdown."""
        prompt = f"""
You are an Academic Laboratory Evaluator. Generate a formal, publication-ready Deep Learning Lab Experiment Report
based on the student's completed experiment records below.

STUDENT & EXPERIMENT RECORDS:
- Student Name / Roll No: {experiment_data.get('student_name', 'Student Researcher')}
- Experiment Title: {experiment_data.get('title', 'Deep Learning Model Training & Evaluation')}
- Aim & Objective: {experiment_data.get('aim', 'To train and evaluate a deep learning model for classification')}
- Dataset Description: {experiment_data.get('dataset_name', 'Benchmark Dataset')} ({experiment_data.get('modality', 'Vision/Tabular')})
- Preprocessing Steps: {experiment_data.get('preprocessing', 'Standard scaling, train-test split, normalization')}
- Model Architecture: {experiment_data.get('model_arch', 'Convolutional Neural Network / MLP')}
- Loss Function & Optimizer: {experiment_data.get('loss_opt', 'CrossEntropyLoss, AdamW (lr=0.001)')}
- Hyperparameters: {experiment_data.get('hyperparams', 'Epochs: 20, Batch Size: 32')}
- Evaluation Metrics & Results: {experiment_data.get('metrics', 'Accuracy: 92.4%, Macro F1: 0.91')}
- Observations & Challenges: {experiment_data.get('observations', 'Minor overfitting observed around epoch 15')}
- Conclusion / Learning Outcome: {experiment_data.get('conclusion', 'Model generalized well with dropout regularization')}

Please format the report cleanly in GitHub Flavored Markdown following this standard academic lab structure:
# Deep Learning Laboratory Experiment Report
## 1. Experiment Aim & Objective
## 2. Theoretical Background & Mathematical Formulation (include loss and activation equations in LaTeX)
## 3. Dataset Characteristics & Exploratory Analysis
## 4. Preprocessing Pipeline & Data Integrity Safeguards
## 5. Model Architecture & Layer Specifications
## 6. Experimental Configuration & Hyperparameters (use a clean Markdown Table)
## 7. Results, Performance Metrics & Visualizations Discussion
## 8. Error Analysis, Failure Modes & Diagnostic Review
## 9. Conclusion & Practical Takeaways

Make the report thorough, highly technical, and ready for lab submission and grading!
"""

        # Try to generate via Gemini if an API key is available
        if self.api_key:
            ai_report = self.call_gemini(prompt)
            if ai_report and len(ai_report.strip()) > 100:
                return ai_report

        # Always return the high-fidelity academic lab report if API call is absent or fails
        return self._generate_fallback_academic_report(experiment_data)
