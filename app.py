"""Autonomous AI Lab Assistant - Interactive Streamlit Application.

Provides dataset suggestions with expected accuracies, step-by-step guidance,
automatic evaluation metric generation, starter code generation, persistent storage,
and automated academic lab report generation for deep learning experiments.
"""

import os
import streamlit as st
import pandas as pd
from rag_engine import LabAssistantRAG, STEP_CATEGORY_MAP
from knowledge_base import ML_LAB_KNOWLEDGE, search_knowledge_base
import experiment_storage

# Page configuration
st.set_page_config(
    page_title="AI Deep Learning Lab Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished lab assistant styling
st.markdown(
    """
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .step-card {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        padding: 0.9rem 1.1rem;
        border-radius: 0.4rem;
        margin-bottom: 1.0rem;
    }
    .dataset-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-badge {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .accuracy-tag {
        background-color: #ECFDF5;
        color: #065F46;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .pitfall-box {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 0.75rem 1rem;
        border-radius: 0.3rem;
        margin: 0.8rem 0;
        font-size: 0.95rem;
    }
    .tip-box {
        background-color: #F0FDF4;
        border-left: 4px solid #22C55E;
        padding: 0.75rem 1rem;
        border-radius: 0.3rem;
        margin: 0.8rem 0;
        font-size: 0.95rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------- DEFAULT EXPERIMENT TEMPLATE -----------------
DEFAULT_EXPERIMENT = {
    "id": None,
    "student_name": "Student Researcher",
    "title": "Deep Learning Image Classification Experiment",
    "aim": "To construct, train, evaluate, and diagnose a Deep Learning model to classify images accurately.",
    "dataset_name": "CIFAR-10",
    "modality": "Vision (Images)",
    "task_type": "Multi-Class Classification",
    "preprocessing": "Resize 224x224, RandomHorizontalFlip, Normalization (mean=[0.485, 0.456, 0.406])",
    "model_arch": "ResNet-18 (Pretrained) with Custom FC Layer",
    "loss_opt": "CrossEntropyLoss, AdamW (lr=0.001, weight_decay=1e-4)",
    "hyperparams": "Batch Size: 32, Epochs: 20, Scheduler: CosineAnnealingLR",
    "metrics": "Train Accuracy: 96.2%, Val Accuracy: 91.5%, Macro F1: 0.90",
    "observations": "Model showed slight overfitting after epoch 12. Adding Dropout (p=0.3) stabilized validation loss.",
    "conclusion": "ResNet-18 converged smoothly and achieved >91% accuracy without vanishing gradients.",
}

# ----------------- SESSION STATE INITIALIZATION -----------------
if "experiment_data" not in st.session_state:
    st.session_state.experiment_data = dict(DEFAULT_EXPERIMENT)

if "suggested_datasets" not in st.session_state:
    st.session_state.suggested_datasets = []

if "auto_metrics" not in st.session_state:
    st.session_state.auto_metrics = None

if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""

if "generated_report" not in st.session_state:
    st.session_state.generated_report = ""

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/8649/8649607.png", width=64
    )
    st.title("AI Lab Assistant")
    st.caption("Deep Learning Experiment Companion")

    # API Key Input
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="Get a free key from Google AI Studio (aistudio.google.com)",
    )

    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        rag_assistant = LabAssistantRAG(api_key=api_key_input)
        st.success("🟢 AI Connected (gemini-3.8-flash)", icon="✅")
    else:
        rag_assistant = LabAssistantRAG()
        st.warning("🟡 Offline Mode (RAG Rules Only)", icon="⚠️")
        st.caption("Add an API key to enable live AI reasoning & report writing.")

    st.divider()

    # Step Selector
    steps = list(STEP_CATEGORY_MAP.keys())
    selected_step = st.radio(
        "📍 Current Lab Step",
        steps,
        index=0,
        help="Switch steps to get dedicated guidance, checks, and pitfalls.",
    )

    st.divider()

    # Persistence: Saved Experiments
    st.subheader("💾 Experiment Persistence")
    saved_list = experiment_storage.get_all_experiments()

    if saved_list:
        options = {
            f"{e.get('title', 'Untitled')} ({e.get('id')})": e.get("id")
            for e in saved_list
        }
        selected_exp_label = st.selectbox(
            "📂 Load Saved Experiment:",
            options=["-- Select an experiment --"] + list(options.keys()),
        )

        if (
            selected_exp_label != "-- Select an experiment --"
            and selected_exp_label in options
        ):
            chosen_id = options[selected_exp_label]
            if st.button("📥 Load Selected Experiment"):
                loaded = experiment_storage.get_experiment_by_id(chosen_id)
                if loaded:
                    st.session_state.experiment_data = loaded
                    st.success(f"Loaded '{loaded.get('title')}'!")
                    st.rerun()

    c_save, c_reset = st.columns(2)
    with c_save:
        if st.button("💾 Save Progress", type="primary"):
            exp_id = experiment_storage.save_or_update_experiment(
                st.session_state.experiment_data
            )
            st.session_state.experiment_data["id"] = exp_id
            st.success(f"Saved! ID: {exp_id}")
            st.rerun()

    with c_reset:
        if st.button("➕ New Blank"):
            st.session_state.experiment_data = dict(DEFAULT_EXPERIMENT)
            st.session_state.experiment_data["id"] = None
            st.session_state.experiment_data["title"] = "New DL Experiment"
            st.session_state.experiment_data["aim"] = ""
            st.session_state.generated_report = ""
            st.session_state.generated_code = ""
            st.rerun()

# ----------------- MAIN INTERFACE -----------------
st.markdown(
    '<div class="main-title">🧪 Autonomous AI Deep Learning Lab Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Automates dataset discovery with accuracy benchmarks, code generation, error corrections, and report generation.</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🧭 Step-by-Step Lab Guide",
        "💻 Starter Code Generator",
        "🛠️ Code & Error Debugger",
        "📝 Automated Lab Report",
        "📚 Knowledge Base Rules",
    ]
)

# -------------------------------------------------------------
# TAB 1: STEP-BY-STEP LAB GUIDE
# -------------------------------------------------------------
with tab1:
    st.markdown(
        f'<div class="step-card"><h3>{selected_step}</h3><p>Follow the lab instructions, verify your design, and record your experiment choices.</p></div>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.15, 0.85])

    with col_left:
        # Step-specific form inputs for the student's notebook
        st.subheader("✍️ Record Step Details")

        if "Step 1" in selected_step:
            st.session_state.experiment_data["student_name"] = st.text_input(
                "Student Name / Roll No:",
                st.session_state.experiment_data["student_name"],
            )
            st.session_state.experiment_data["title"] = st.text_input(
                "Experiment Title:",
                st.session_state.experiment_data["title"],
                placeholder="E.g., Automated Skin Lesion Classification Using Convolutional Networks",
            )
            st.session_state.experiment_data["aim"] = st.text_area(
                "Aim & Objective:",
                st.session_state.experiment_data["aim"],
                placeholder="E.g., To design, train and evaluate a deep learning model to accurately detect malignant skin lesions from dermoscopy images.",
                height=90,
            )

            c1, c2 = st.columns(2)
            st.session_state.experiment_data["modality"] = c1.selectbox(
                "Data Modality:",
                ["Vision (Images)", "Tabular (CSV)", "Text / NLP", "Multimodal"],
            )
            st.session_state.experiment_data["task_type"] = c2.selectbox(
                "Task Type:",
                ["Multi-Class Classification", "Binary Classification", "Regression", "Segmentation"],
            )

            # DATASET RECOMMENDATION SECTION
            st.markdown("---")
            st.markdown("#### 🔍 Intelligent Dataset Discovery & Benchmark Accuracies")
            st.caption(
                "Click below to let the AI analyze your Experiment Title and Aim, and suggest the most suitable datasets with their achievable accuracy ranges!"
            )

            if st.button("✨ Auto-Suggest Suitable Datasets with Expected Accuracies", type="primary"):
                with st.spinner("Analyzing problem statement and benchmarking candidate datasets..."):
                    suggestions = rag_assistant.suggest_datasets(
                        title=st.session_state.experiment_data["title"],
                        aim=st.session_state.experiment_data["aim"],
                        modality=st.session_state.experiment_data["modality"],
                    )
                    st.session_state.suggested_datasets = suggestions

            if st.session_state.suggested_datasets:
                st.markdown("##### 🎯 Recommended Datasets for Your Problem Statement:")
                for idx, ds in enumerate(st.session_state.suggested_datasets):
                    st.markdown(
                        f"""
                        <div class="dataset-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <strong style="font-size:1.1rem; color:#1E3A8A;">📦 {ds.get('name')}</strong>
                                <span class="accuracy-tag">🎯 Expected Accuracy: {ds.get('expected_accuracy')}</span>
                            </div>
                            <p style="margin:0.4rem 0 0.2rem 0; font-size:0.92rem;"><strong>Source:</strong> <code>{ds.get('source')}</code></p>
                            <p style="margin:0.2rem 0; font-size:0.92rem; color:#374151;">{ds.get('description')}</p>
                            <p style="margin:0.2rem 0; font-size:0.88rem; color:#1F2937;"><strong>💡 Why Recommended:</strong> {ds.get('why_recommended')}</p>
                            <p style="margin:0.2rem 0; font-size:0.88rem; color:#4B5563;"><strong>🧠 Suggested Architecture:</strong> {ds.get('recommended_architecture')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button(f"✅ Use '{ds.get('name')}' for this Experiment", key=f"sel_ds_{idx}"):
                        st.session_state.experiment_data["dataset_name"] = ds.get("name")
                        st.session_state.experiment_data["model_arch"] = ds.get("recommended_architecture")
                        st.success(f"Selected {ds.get('name')}! Dataset and architecture updated.")
                        st.rerun()

            st.session_state.experiment_data["dataset_name"] = st.text_input(
                "Chosen Dataset Name:", st.session_state.experiment_data["dataset_name"]
            )

        elif "Step 2" in selected_step:
            st.session_state.experiment_data["preprocessing"] = st.text_area(
                "Preprocessing & Augmentation Pipeline:",
                st.session_state.experiment_data["preprocessing"],
                help="E.g., StandardScaler, RandomHorizontalFlip, Train/Val/Test Split ratio",
                height=120,
            )
            # Optional sample CSV inspector
            uploaded_file = st.file_uploader(
                "Upload a sample CSV to inspect (optional)", type=["csv"]
            )
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("📊 Quick Dataset Preview:")
                    st.dataframe(df.head(4))
                    st.caption(
                        f"Rows: {df.shape[0]}, Columns: {df.shape[1]}, Missing values: {df.isna().sum().sum()}"
                    )
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

        elif "Step 3" in selected_step:
            st.session_state.experiment_data["model_arch"] = st.text_input(
                "Selected Model Architecture:",
                st.session_state.experiment_data["model_arch"],
                help="E.g., ResNet-18, 3-layer MLP, MobileNetV3, DistilBERT",
            )
            st.session_state.experiment_data["loss_opt"] = st.text_input(
                "Loss Function & Optimizer:",
                st.session_state.experiment_data["loss_opt"],
                help="E.g., nn.CrossEntropyLoss(), AdamW (lr=0.001)",
            )

        elif "Step 4" in selected_step:
            st.session_state.experiment_data["hyperparams"] = st.text_area(
                "Hyperparameters & Training Setup:",
                st.session_state.experiment_data["hyperparams"],
                help="Batch size, Epochs, Learning rate, LR Scheduler, Early stopping patience",
                height=100,
            )

        elif "Step 5" in selected_step:
            st.markdown("#### 📊 Evaluation Metrics Recommendation Engine")
            st.caption(
                "The assistant automatically generates the baseline evaluation metrics required for your experiment type:"
            )

            # Auto-generate evaluation metrics if not yet generated
            if st.button("⚡ Generate Recommended Evaluation Metrics", type="primary") or not st.session_state.auto_metrics:
                with st.spinner("Analyzing experiment requirements and calculating metric suite..."):
                    metrics_res = rag_assistant.recommend_evaluation_metrics(
                        title=st.session_state.experiment_data["title"],
                        aim=st.session_state.experiment_data["aim"],
                        task_type=st.session_state.experiment_data["task_type"],
                    )
                    st.session_state.auto_metrics = metrics_res

            if st.session_state.auto_metrics:
                m = st.session_state.auto_metrics
                st.markdown("**Essential Primary Metrics:**")
                st.markdown("".join([f'<span class="metric-badge">🎯 {pm}</span>' for pm in m.get("primary_metrics", [])]), unsafe_allow_html=True)
                st.markdown("**Secondary Diagnostic Metrics:**")
                st.markdown("".join([f'<span class="metric-badge">🔍 {sm}</span>' for sm in m.get("secondary_metrics", [])]), unsafe_allow_html=True)
                st.info(f"💡 **Why these metrics:** {m.get('explanation', '')}")
                with st.expander("🐍 Calculation Code (scikit-learn / PyTorch)"):
                    st.code(m.get("sklearn_code", ""), language="python")

            st.markdown("---")
            st.session_state.experiment_data["metrics"] = st.text_area(
                "Your Recorded Results & Metrics:",
                st.session_state.experiment_data["metrics"],
                help="Train/Val loss, Accuracy, Macro F1, Precision, Recall",
                height=90,
            )
            st.session_state.experiment_data["observations"] = st.text_area(
                "Error Analysis & Observations:",
                st.session_state.experiment_data["observations"],
                help="Did the model overfit? What classes were confused?",
                height=90,
            )

        elif "Step 6" in selected_step:
            st.session_state.experiment_data["conclusion"] = st.text_area(
                "Final Conclusion & Learnings:",
                st.session_state.experiment_data["conclusion"],
                height=100,
            )
            st.info("💡 Head to the **'Automated Lab Report'** tab to generate and download your full submission!")

        st.caption("✅ Auto-synced in session memory. Click '💾 Save Progress' in the sidebar to persist permanently.")

    with col_right:
        st.subheader("🤖 Step AI Consultant")

        quick_prompts = {
            "Step 1: Dataset & Problem Setup": [
                "How do I check for class imbalance and what metric should I use?",
                "What are the most common data leakage traps in this step?",
            ],
            "Step 2: Preprocessing & Leakage Checker": [
                "Should I scale features before or after splitting the dataset?",
                "Give me a PyTorch torchvision transform pipeline with data augmentations.",
            ],
            "Step 3: Model Architecture & Loss Selection": [
                "Do I need to add nn.Softmax() before nn.CrossEntropyLoss()?",
                "Which backbone should I choose: ResNet-18 vs Vision Transformer for a small dataset?",
            ],
            "Step 4: Training & Hyperparameter Tuning": [
                "My loss became NaN during training. What causes this and how do I fix it?",
                "How do I implement early stopping and a learning rate scheduler in PyTorch?",
            ],
            "Step 5: Evaluation & Error Analysis": [
                "My training loss is very low but validation loss is increasing. How do I fix overfitting?",
                "Explain the difference between Micro-F1 and Macro-F1.",
            ],
            "Step 6: Lab Report Generator": [
                "What are the required sections for a formal IEEE deep learning lab report?",
                "How should I structure the mathematical equations for loss and backpropagation?",
            ],
        }

        st.caption("Quick Inquiries:")
        current_prompts = quick_prompts.get(selected_step, [])
        user_query = ""
        for prompt in current_prompts:
            if st.button(f"👉 {prompt}", key=f"qp_{prompt}"):
                user_query = prompt

        custom_query = st.text_input(
            "Ask a specific question for this step:",
            value=user_query,
            placeholder="E.g., What metric should I use if positive cases are rare?",
        )

        code_sample = st.text_area(
            "Optional: Paste your code snippet for inspection",
            height=90,
            placeholder="# Paste your code here...",
        )

        if st.button("Get AI Guidance & Verification", type="primary"):
            if not custom_query and not code_sample:
                st.warning("Please type a question or paste code.")
            else:
                with st.spinner("Analyzing with RAG knowledge & Gemini AI..."):
                    response = rag_assistant.ask_step_assistant(
                        user_query=custom_query or "Review my code snippet.",
                        current_step=selected_step,
                        experiment_context=st.session_state.experiment_data,
                        code_context=code_sample if code_sample else None,
                    )
                    st.markdown("### 💡 Lab Assistant Guidance:")
                    st.markdown(response)

# -------------------------------------------------------------
# TAB 2: STARTER CODE GENERATOR
# -------------------------------------------------------------
with tab2:
    st.subheader("💻 Automatic Starter Code Generator")
    st.write(
        "Generate a complete, ready-to-run PyTorch script tailored specifically to your experiment title, aim, chosen dataset, and architecture."
    )

    exp = st.session_state.experiment_data
    col_cg1, col_cg2 = st.columns([2, 1])
    with col_cg1:
        st.markdown(f"**Experiment:** `{exp['title']}`")
        st.markdown(f"**Dataset:** `{exp['dataset_name']}` | **Model:** `{exp['model_arch']}`")

    with col_cg2:
        gen_code_btn = st.button("⚡ Generate Complete PyTorch Code", type="primary")

    if gen_code_btn or not st.session_state.generated_code:
        if gen_code_btn:
            with st.spinner("Generating end-to-end PyTorch code with data loading, model, and training loop..."):
                code = rag_assistant.generate_baseline_code(
                    title=exp["title"],
                    aim=exp["aim"],
                    dataset_name=exp["dataset_name"],
                    model_arch=exp["model_arch"],
                    modality=exp["modality"],
                    task_type=exp["task_type"],
                )
                st.session_state.generated_code = code

    if st.session_state.generated_code:
        st.download_button(
            label="📥 Download Python Script (experiment.py)",
            data=st.session_state.generated_code,
            file_name="experiment.py",
            mime="text/x-python",
        )
        st.code(st.session_state.generated_code, language="python")

# -------------------------------------------------------------
# TAB 3: CODE & ERROR DEBUGGER
# -------------------------------------------------------------
with tab3:
    st.subheader("🛠️ Deep Learning Code & Error Debugger")
    st.write(
        "Paste your buggy PyTorch / Python code along with the terminal traceback. "
        "The assistant identifies the root cause, explains the concept, and provides corrected code."
    )

    col_code, col_err = st.columns(2)
    with col_code:
        debug_code = st.text_area(
            "Your Code Snippet:",
            height=250,
            value=(
                "import torch\n"
                "import torch.nn as nn\n\n"
                "class SimpleModel(nn.Module):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                "        self.linear1 = nn.Linear(10, 50)\n"
                "        self.linear2 = nn.Linear(50, 2)\n"
                "        self.softmax = nn.Softmax(dim=1)\n\n"
                "    def forward(self, x):\n"
                "        x = torch.relu(self.linear1(x))\n"
                "        return self.softmax(self.linear2(x))\n\n"
                "criterion = nn.CrossEntropyLoss()\n"
                "# Why is the loss flattening and gradient updates unstable?"
            ),
        )

    with col_err:
        debug_err = st.text_area(
            "Error Message / Issue Description:",
            height=250,
            value="RuntimeError or strange training dynamics: Loss is not decreasing properly and seems flattened.",
        )

    if st.button("🔍 Diagnose & Fix Code", type="primary"):
        with st.spinner("Diagnosing root cause and verifying fix..."):
            diagnosis = rag_assistant.diagnose_code_and_errors(
                code_snippet=debug_code,
                error_traceback=debug_err,
                current_step=selected_step,
            )
            st.markdown(diagnosis)

# -------------------------------------------------------------
# TAB 4: AUTOMATED LAB REPORT GENERATOR
# -------------------------------------------------------------
with tab4:
    st.subheader("📝 Automated Academic Lab Report Generator")
    st.write(
        "Synthesizes your experiment inputs into a publication-ready, academic-style lab experiment report with mathematical equations, experimental tables, and critical analysis."
    )

    col_rep_info, col_rep_btn = st.columns([2, 1])
    with col_rep_info:
        st.markdown(
            f"**Experiment**: `{st.session_state.experiment_data['title']}` | "
            f"**Student**: `{st.session_state.experiment_data['student_name']}`"
        )
    with col_rep_btn:
        generate_btn = st.button("🚀 Generate Full Lab Report", type="primary")

    if generate_btn:
        with st.spinner("Synthesizing formal experiment report with Gemini..."):
            report = rag_assistant.generate_experiment_report(
                st.session_state.experiment_data
            )
            st.session_state.generated_report = report

    if st.session_state.generated_report:
        st.divider()
        st.download_button(
            label="📥 Download Report as Markdown (.md)",
            data=st.session_state.generated_report,
            file_name="Deep_Learning_Lab_Report.md",
            mime="text/markdown",
        )
        st.markdown(st.session_state.generated_report)

# -------------------------------------------------------------
# TAB 5: KNOWLEDGE BASE RULES EXPLORER
# -------------------------------------------------------------
with tab5:
    st.subheader("📚 Curated Deep Learning Lab Rules & Playbooks")
    st.write(
        "Explore the RAG rules engine embedded in the assistant. Each rule covers pitfalls, recommended solutions, and verified code patterns."
    )

    search_kw = st.text_input(
        "Search Knowledge Base:", placeholder="E.g., nan loss, softmax, leakage"
    )

    if search_kw:
        matched = search_knowledge_base(search_kw, top_k=5)
    else:
        matched = ML_LAB_KNOWLEDGE

    for item in matched:
        with st.expander(f"📌 [{item['category'].upper()}] {item['title']}"):
            st.markdown(
                f'<div class="pitfall-box"><strong>⚠️ Common Pitfall:</strong> {item["common_pitfall"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="tip-box"><strong>✅ Recommended Practice:</strong> {item["recommended_practice"]}</div>',
                unsafe_allow_html=True,
            )
            st.code(item["code_snippet"], language="python")
