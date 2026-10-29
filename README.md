# 🧪 Autonomous AI Lab Assistant Using Deep Learning

> **IDP Presentation Pitch:**
> *"An autonomous, interactive AI Lab Assistant that guides students step-by-step through their deep learning experiments—from dataset inspection and preprocessing to architecture selection, training, error debugging, and one-click academic lab report generation."*

---

## 🌟 Key Features

1. **Step-by-Step Experiment Companion**:
   - **Step 1: Dataset & Problem Setup**: Identify missing values, data leakage, and class imbalance.
   - **Step 2: Preprocessing & Splits**: Safe feature scaling, train/val/test splits, data augmentations.
   - **Step 3: Model Architecture & Loss**: Backbone recommendations (ResNet vs MLP vs ViT) and loss selection (preventing the common `nn.Softmax` + `nn.CrossEntropyLoss` pitfall).
   - **Step 4: Training & Hyperparameters**: Guidance on learning rates, AdamW, LR schedulers, early stopping, and gradient clipping to avoid NaN loss.
   - **Step 5: Evaluation & Diagnostics**: Confusion matrix analysis, macro F1 evaluation, and diagnosing overfitting vs underfitting from loss curves.
   - **Step 6: Automated Lab Report Generation**: One-click synthesis of a formal IEEE/Academic lab report with LaTeX equations and performance tables.

2. **RAG-Powered Domain Knowledge Base**:
   - Contains 12+ verified deep learning lab rules, common student pitfalls, and runnable PyTorch code patterns.
   - Works **offline** (retrieving rules and advice) or **online** with Google Gemini API (`gemini-3.8-flash`) for live feedback and debugging.

3. **Real-Time Code & Error Debugger**:
   - Students paste their buggy PyTorch code and runtime errors (e.g., shape mismatches, exploding gradients, CUDA OOM).
   - Assistant diagnoses the root cause, explains the concept simply, and provides corrected, verified code.

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit App
```bash
streamlit run app.py
```

### 3. Using with API Key
- Enter your **Google Gemini API Key** directly in the sidebar inside the app (or set `set GEMINI_API_KEY="your-api-key"` in your terminal).
- You can get a free API key at [Google AI Studio](https://aistudio.google.com).

---

## 📂 Project Architecture

```text
├── app.py                  # Main interactive Streamlit application
├── rag_engine.py           # RAG retrieval and Gemini AI assistant integration
├── knowledge_base.py       # Curated DL experiment rules, pitfalls, and code patterns
├── core/
│   ├── state.py            # Typed experiment state & data models
│   ├── config.py           # Configurations & hardware settings
│   └── sandbox.py          # Execution sandbox
├── agents/                 # Multi-agent architecture definitions
├── requirements.txt        # Project dependencies
└── README.md               # Documentation & quickstart
```
