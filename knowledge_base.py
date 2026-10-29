"""Curated Knowledge Base for Deep Learning Lab Experiments.

Provides verified domain rules, best practices, common student pitfalls, and
code snippets across the full ML lifecycle.
"""

from typing import List, Dict, Any

ML_LAB_KNOWLEDGE: List[Dict[str, Any]] = [
    # ------------------ STEP 1: DATASET INSPECTION & EDA ------------------
    {
        "id": "eda-01",
        "category": "dataset_inspection",
        "title": "Handling Missing Data and Avoid Naive Imputation",
        "keywords": [
            "missing values",
            "imputation",
            "nulls",
            "nan",
            "data cleaning",
            "eda",
        ],
        "common_pitfall": "Imputing test set with mean/median computed on the entire dataset before splitting (causes data leakage) or blindly dropping rows with missing values when missingness carries signal.",
        "recommended_practice": "Always split data into train and test FIRST. Fit imputers (SimpleImputer, KNNImputer) only on the training set, then transform both train and test. Drop features only if >60% values are missing.",
        "code_snippet": (
            "# Correct Workflow: Split first, then fit imputer\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.impute import SimpleImputer\n\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
            "imputer = SimpleImputer(strategy='median')\n"
            "X_train_imp = imputer.fit_transform(X_train)  # Fit ONLY on train\n"
            "X_test_imp = imputer.transform(X_test)        # Transform test with train stats\n"
        ),
    },
    {
        "id": "eda-02",
        "category": "dataset_inspection",
        "title": "Class Imbalance Pitfalls & Metric Traps",
        "keywords": [
            "class imbalance",
            "accuracy paradox",
            "f1 score",
            "auc",
            "minority class",
            "imbalance",
        ],
        "common_pitfall": "Using standard accuracy to evaluate imbalanced datasets (e.g. 95% negative, 5% positive where a naive model predicting all negatives gets 95% accuracy while completely failing the minority class).",
        "recommended_practice": "Use StratifiedKFold or stratified train_test_split. Evaluate with Macro-F1, Precision-Recall AUC (PR-AUC), and Confusion Matrix. In loss functions, apply class weights (weight parameter in CrossEntropyLoss) or Focal Loss.",
        "code_snippet": (
            "# PyTorch: Class Weighted Cross Entropy\n"
            "import torch\n"
            "import torch.nn as nn\n\n"
            "# weights = total_samples / (n_classes * class_samples)\n"
            "class_weights = torch.tensor([0.2, 0.8], dtype=torch.float32)\n"
            "criterion = nn.CrossEntropyLoss(weight=class_weights)\n"
        ),
    },
    {
        "id": "eda-03",
        "category": "dataset_inspection",
        "title": "Data Leakage Identification",
        "keywords": [
            "data leakage",
            "target leakage",
            "id columns",
            "temporal leakage",
            "split",
        ],
        "common_pitfall": "Leaving identifiers (CustomerID, Timestamp, RowNumber) in feature matrices, or performing feature selection/PCA before splitting data.",
        "recommended_practice": "Remove unique ID columns and target proxies. Never let any transformation or feature calculation use test split data.",
        "code_snippet": (
            "# Drop ID columns and verify target correlation\n"
            "X = df.drop(columns=['id', 'customer_id', 'target'])\n"
            "y = df['target']\n"
        ),
    },
    # ------------------ STEP 2: PREPROCESSING & FEATURE PIPELINES ------------------
    {
        "id": "prep-01",
        "category": "preprocessing",
        "title": "Feature Scaling for Neural Networks (StandardScaler vs MinMax)",
        "keywords": [
            "scaling",
            "standardscaler",
            "minmaxscaler",
            "normalization",
            "gradients",
        ],
        "common_pitfall": "Feeding unscaled or differently-scaled numerical features into neural networks, causing gradient descent to oscillate and diverge or slow down.",
        "recommended_practice": "Always scale continuous features to zero-mean and unit variance (StandardScaler) or [0, 1] (MinMaxScaler). Neural network activation functions (ReLU, Sigmoid) behave best when input features are standardized.",
        "code_snippet": (
            "from sklearn.preprocessing import StandardScaler\n\n"
            "scaler = StandardScaler()\n"
            "X_train_scaled = scaler.fit_transform(X_train)\n"
            "X_test_scaled = scaler.transform(X_test)\n"
        ),
    },
    {
        "id": "prep-02",
        "category": "preprocessing",
        "title": "Data Augmentation for Vision Models",
        "keywords": [
            "data augmentation",
            "torchvision transforms",
            "random crop",
            "horizontal flip",
            "image preprocessing",
        ],
        "common_pitfall": "Applying data augmentation to the validation or test sets. Augmentation is ONLY for training to simulate real-world variance.",
        "recommended_practice": "Define separate transform pipelines: training pipeline with augmentations (RandomCrop, RandomHorizontalFlip, ColorJitter) and validation/test pipeline with only deterministic Resize/CenterCrop and Normalize.",
        "code_snippet": (
            "import torchvision.transforms as T\n\n"
            "train_transform = T.Compose([\n"
            "    T.RandomResizedCrop(224),\n"
            "    T.RandomHorizontalFlip(),\n"
            "    T.ToTensor(),\n"
            "    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n"
            "])\n\n"
            "val_transform = T.Compose([\n"
            "    T.Resize(256),\n"
            "    T.CenterCrop(224),\n"
            "    T.ToTensor(),\n"
            "    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n"
            "])\n"
        ),
    },
    # ------------------ STEP 3: MODEL ARCHITECTURE & LOSS SELECTION ------------------
    {
        "id": "model-01",
        "category": "model_selection",
        "title": "PyTorch nn.CrossEntropyLoss and Softmax Pitfall",
        "keywords": [
            "crossentropyloss",
            "softmax",
            "logits",
            "runtime error",
            "loss computation",
        ],
        "common_pitfall": "Adding `nn.Softmax()` at the end of the PyTorch model forward pass while also using `nn.CrossEntropyLoss()`. This double-softmaxes the probabilities, drastically flattening gradients and ruining convergence.",
        "recommended_practice": "In PyTorch, `nn.CrossEntropyLoss()` combines `LogSoftmax()` and `NLLLoss()` internally. The model forward pass MUST return raw unnormalized logits.",
        "code_snippet": (
            "import torch.nn as nn\n\n"
            "class Classifier(nn.Module):\n"
            "    def __init__(self, in_features, num_classes):\n"
            "        super().__init__()\n"
            "        self.fc = nn.Sequential(\n"
            "            nn.Linear(in_features, 128),\n"
            "            nn.ReLU(),\n"
            "            nn.Dropout(0.3),\n"
            "            nn.Linear(128, num_classes) # DO NOT put Softmax here!\n"
            "        )\n"
            "    def forward(self, x):\n"
            "        return self.fc(x) # Returns raw logits\n\n"
            "criterion = nn.CrossEntropyLoss() # Applies Softmax internally\n"
        ),
    },
    {
        "id": "model-02",
        "category": "model_selection",
        "title": "Selecting the Right Backbone: ResNet vs ViT vs MLP",
        "keywords": [
            "architecture selection",
            "resnet",
            "vit",
            "mlp",
            "transfer learning",
            "backbone",
        ],
        "common_pitfall": "Training a Vision Transformer (ViT) from scratch on a small dataset (<50,000 images). ViTs lack inductive bias (translation invariance) and will severely overfit without huge pre-training data.",
        "recommended_practice": "For small/medium vision datasets, use pre-trained ResNet-18/50 or EfficientNet with transfer learning (fine-tuning). For tabular datasets, start with a 2-3 layer MLP or Tree-based baseline (XGBoost/LightGBM) before TabNet.",
        "code_snippet": (
            "import torchvision.models as models\n"
            "import torch.nn as nn\n\n"
            "# Transfer Learning with ResNet-18\n"
            "model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)\n"
            "# Freeze earlier layers if dataset is very small\n"
            "for param in model.parameters():\n"
            "    param.requires_grad = False\n"
            "# Replace final FC layer\n"
            "model.fc = nn.Linear(model.fc.in_features, num_classes)\n"
        ),
    },
    # ------------------ STEP 4: TRAINING & HYPERPARAMETER TUNING ------------------
    {
        "id": "train-01",
        "category": "training_debugging",
        "title": "Preventing Exploding Gradients & Loss is NaN",
        "keywords": [
            "loss is nan",
            "exploding gradients",
            "gradient clipping",
            "nan loss",
            "learning rate too high",
        ],
        "common_pitfall": "Setting learning rate too high (e.g. 0.1 without warmup or scheduler) or missing gradient clipping in recurrent/deep networks, causing loss to suddenly become NaN.",
        "recommended_practice": "Always clip gradients using `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`. Start with AdamW at lr=1e-3 or 3e-4.",
        "code_snippet": (
            "# Safe Training Step with Gradient Clipping\n"
            "optimizer.zero_grad()\n"
            "outputs = model(inputs)\n"
            "loss = criterion(outputs, targets)\n"
            "loss.backward()\n"
            "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Prevents exploding gradients\n"
            "optimizer.step()\n"
        ),
    },
    {
        "id": "train-02",
        "category": "training_debugging",
        "title": "Learning Rate Schedulers & Early Stopping",
        "keywords": [
            "early stopping",
            "learning rate scheduler",
            "reduceonplateau",
            "overfitting",
            "patience",
        ],
        "common_pitfall": "Training for a fixed high number of epochs without monitoring validation loss, resulting in a heavily overfitted final checkpoint.",
        "recommended_practice": "Implement Early Stopping with a patience of 5-10 epochs based on validation loss, and save only the best model checkpoint (`best_model.pt`). Pair with `ReduceLROnPlateau` or `CosineAnnealingLR`.",
        "code_snippet": (
            "from torch.optim.lr_scheduler import ReduceLROnPlateau\n\n"
            "scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)\n"
            "# In epoch loop after validation:\n"
            "scheduler.step(val_loss)\n"
        ),
    },
    {
        "id": "train-03",
        "category": "training_debugging",
        "title": "Forgetting optimizer.zero_grad() and model.eval()",
        "keywords": [
            "zero_grad",
            "model.eval()",
            "model.train()",
            "gradient accumulation bug",
            "dropout test error",
        ],
        "common_pitfall": "Forgetting `optimizer.zero_grad()` accumulates gradients across batches, resulting in erratic step directions. Forgetting `model.eval()` during validation causes Dropout and BatchNorm to behave as training steps.",
        "recommended_practice": "Always call `optimizer.zero_grad()` before `loss.backward()`. Wrap validation in `model.eval()` and `with torch.no_grad():`.",
        "code_snippet": (
            "# Standard Validation Loop\n"
            "model.eval()\n"
            "with torch.no_grad():\n"
            "    for inputs, labels in val_loader:\n"
            "        outputs = model(inputs)\n"
            "        val_loss += criterion(outputs, labels).item()\n"
            "model.train() # Remember to switch back for training!\n"
        ),
    },
    # ------------------ STEP 5: EVALUATION & DIAGNOSTICS ------------------
    {
        "id": "eval-01",
        "category": "evaluation_diagnostics",
        "title": "Diagnosing Overfitting vs Underfitting from Loss Curves",
        "keywords": [
            "overfitting",
            "underfitting",
            "loss curves",
            "bias variance",
            "generalization gap",
        ],
        "common_pitfall": "Misinterpreting training curve gaps or not plotting training vs validation loss on the same graph.",
        "recommended_practice": "Examine the gap: If Train Loss drops low but Val Loss rises, model is Overfitting (Add Dropout 0.3-0.5, L2 weight decay 1e-4, data augmentation, or simplify model). If both Train and Val Loss stay stubbornly high, model is Underfitting (Increase model capacity/depth, lower learning rate, or add better features).",
        "code_snippet": (
            "import matplotlib.pyplot as plt\n\n"
            "plt.figure(figsize=(8, 4))\n"
            "plt.plot(train_losses, label='Train Loss')\n"
            "plt.plot(val_losses, label='Val Loss')\n"
            "plt.xlabel('Epoch')\n"
            "plt.ylabel('Loss')\n"
            "plt.title('Training and Validation Loss Curve')\n"
            "plt.legend()\n"
            "plt.grid(True)\n"
        ),
    },
    {
        "id": "eval-02",
        "category": "evaluation_diagnostics",
        "title": "Comprehensive Metric Reporting & Confusion Matrix",
        "keywords": [
            "confusion matrix",
            "precision",
            "recall",
            "f1 score",
            "classification report",
        ],
        "common_pitfall": "Only reporting overall accuracy and ignoring per-class performance and confusion patterns.",
        "recommended_practice": "Always compute the normalized Confusion Matrix and `classification_report` (Precision, Recall, F1 per class). This reveals which classes the model confuses most often.",
        "code_snippet": (
            "from sklearn.metrics import classification_report, confusion_matrix\n"
            "import seaborn as sns\n\n"
            "print(classification_report(y_true, y_pred))\n"
            "cm = confusion_matrix(y_true, y_pred, normalize='true')\n"
            "sns.heatmap(cm, annot=True, cmap='Blues')\n"
        ),
    },
    # ------------------ STEP 6: LAB REPORT GENERATION ------------------
    {
        "id": "rep-01",
        "category": "lab_report_guidance",
        "title": "Academic Lab Report Structure for Deep Learning",
        "keywords": [
            "lab report",
            "experiment documentation",
            "academic format",
            "conclusion",
            "methodology",
        ],
        "common_pitfall": "Writing unstructured reports missing the Aim, Dataset specs, loss formulations, or error analysis.",
        "recommended_practice": "Follow the standard IEEE/Academic 8-section format: 1. Title & Objective, 2. Theory & Formulations, 3. Dataset & Preprocessing, 4. Model Architecture Details, 5. Hyperparameters Setup, 6. Results & Plots, 7. Failure Case & Error Analysis, 8. Conclusion & Future Scope.",
        "code_snippet": (
            "# Standard Lab Report Sections:\n"
            "# 1. Aim / Objective\n"
            "# 2. Theoretical Background & Mathematical Formulation\n"
            "# 3. Dataset Description & Exploratory Data Analysis\n"
            "# 4. Preprocessing & Data Pipeline\n"
            "# 5. Proposed Deep Learning Architecture\n"
            "# 6. Experimental Setup & Hyperparameter Table\n"
            "# 7. Experimental Results & Performance Curves\n"
            "# 8. Error Analysis & Limitations\n"
            "# 9. Conclusion\n"
        ),
    },
]


def search_knowledge_base(
    query: str, category: str = "", top_k: int = 3
) -> List[Dict[str, Any]]:
    """Search knowledge base items matching query tokens and optional category."""
    query_tokens = set(query.lower().split())

    scored = []
    for item in ML_LAB_KNOWLEDGE:
        score = 0
        if category and item["category"] == category:
            score += 3

        # Match keywords
        for kw in item["keywords"]:
            for token in query_tokens:
                if token in kw or kw in token:
                    score += 2

        # Match title
        title_lower = item["title"].lower()
        for token in query_tokens:
            if token in title_lower:
                score += 1.5

        if score > 0:
            scored.append((score, item))

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        # Fallback to category items or general items
        if category:
            return [it for it in ML_LAB_KNOWLEDGE if it["category"] == category][
                :top_k
            ]
        return ML_LAB_KNOWLEDGE[:top_k]

    return [item for _, item in scored[:top_k]]
