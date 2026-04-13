"""
Fake Job Posting Detection — Streamlit Dashboard.
Provides prediction interface and model comparison visualizations.
"""

# CRITICAL: Import torch FIRST to prevent DLL conflict on Windows
import torch

import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import clean_text
from src.deep_model import CNN_LSTM_Hybrid, SimpleTokenizer

# ─────────────────────────── Page Config ───────────────────────────

st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ───────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a2e;
        --accent-purple: #6366f1;
        --accent-pink: #f43f5e;
        --accent-green: #10b981;
        --accent-amber: #f59e0b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border-subtle: rgba(99, 102, 241, 0.15);
    }

    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0a0f 100%);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 15, 26, 0.8) 0%, rgba(18, 18, 26, 0.95) 100%) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    [data-testid="stSidebarNav"] {
        display: none;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-primary) !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* ── Dynamic Sidebar Radio Navigation ── */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.03);
        padding: 0.8rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.05);
        cursor: pointer;
    }
    
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
        background: rgba(99, 102, 241, 0.1);
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateX(4px);
    }

    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(139, 92, 246, 0.15));
        border-color: rgba(99, 102, 241, 0.5);
        border-left: 4px solid #6366f1;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2);
    }

    [data-testid="stSidebar"] .stRadio div[data-testid="stMarkdownContainer"] p {
        font-weight: 600;
        font-size: 1.05rem;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        margin: 0;
    }
    
    [data-testid="stSidebar"] .stRadio span[data-baseweb="radio"] {
        display: none;
    }

    /* ── Sidebar Glass Info Card ── */
    .sidebar-glass-card {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.01));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    .sidebar-glass-card .info-title {
        color: var(--text-primary);
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .sidebar-glass-card .info-text {
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.5;
        margin-bottom: 1rem;
    }

    .sidebar-glass-card ul {
        list-style: none;
        padding-left: 0;
        margin: 0;
    }
    
    .sidebar-glass-card li {
        color: #a1a1aa;
        font-size: 0.85rem;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .sidebar-glass-card li::before {
        content: '•';
        color: #6366f1;
        font-size: 1.2rem;
    }

    /* ── Glowing Model Card ── */
    .model-glow-wrapper {
        position: relative;
        padding: 1px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.3), rgba(139, 92, 246, 0.3));
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .model-glow-wrapper::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 60%);
        animation: rotateGlow 8s linear infinite;
        pointer-events: none;
    }

    @keyframes rotateGlow {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .model-glow-card {
        background: var(--bg-card);
        border-radius: 19px;
        padding: 1.5rem;
        text-align: center;
        position: relative;
        z-index: 1;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
    }

    .model-glow-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #a5b4fc;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .model-glow-name {
        font-size: 1.15rem;
        font-weight: 800;
        color: white;
        background: linear-gradient(to right, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.8rem;
    }

    .model-glow-metric {
        display: inline-block;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
    }
    .model-glow-metric span:first-child {
        font-size: 0.75rem;
        color: var(--text-secondary);
        display: block;
        margin-bottom: 0.2rem;
    }
    .model-glow-metric span:last-child {
        font-size: 1.1rem;
        font-weight: 700;
        color: #10b981;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #f43f5e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        text-align: center;
        margin-top: 0.25rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.05));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s, border-color 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--accent-purple);
    }

    .metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }

    .prediction-real {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
        border: 2px solid rgba(16, 185, 129, 0.4);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }

    .prediction-fake {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.1), rgba(244, 63, 94, 0.05));
        border: 2px solid rgba(244, 63, 94, 0.4);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }

    .prediction-label {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .prediction-prob {
        font-size: 1.2rem;
        color: var(--text-secondary);
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-subtle);
    }

    .best-model-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.03em;
    }

    /* ── Verdict Hero Card ── */
    .verdict-card {
        border-radius: 24px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .verdict-card::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 24px;
        opacity: 0.12;
    }
    .verdict-card.safe {
        background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(16,185,129,0.04));
        border: 2px solid rgba(16,185,129,0.45);
    }
    .verdict-card.danger {
        background: linear-gradient(135deg, rgba(244,63,94,0.12), rgba(244,63,94,0.04));
        border: 2px solid rgba(244,63,94,0.45);
    }
    .verdict-icon { font-size: 3.5rem; margin-bottom: 0.3rem; }
    .verdict-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin-bottom: 0.2rem;
    }
    .verdict-sub {
        font-size: 1rem;
        color: var(--text-secondary);
        margin-bottom: 1rem;
    }

    /* ── Confidence Bar ── */
    .conf-bar-outer {
        width: 100%;
        height: 14px;
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        overflow: hidden;
        margin: 0.6rem 0;
    }
    .conf-bar-inner {
        height: 100%;
        border-radius: 10px;
        transition: width 0.6s ease;
    }

    /* ── Guidance Box ── */
    .guidance-box {
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        font-size: 1.05rem;
        line-height: 1.55;
    }
    .guidance-box.safe {
        background: rgba(16,185,129,0.08);
        border-left: 4px solid #10b981;
        color: #6ee7b7;
    }
    .guidance-box.danger {
        background: rgba(244,63,94,0.08);
        border-left: 4px solid #f43f5e;
        color: #fda4af;
    }

    /* ── Risk Chip ── */
    .risk-chip {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.9rem 1.1rem;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        margin-bottom: 0.6rem;
        transition: border-color 0.2s;
    }
    .risk-chip:hover {
        border-color: rgba(99,102,241,0.3);
    }
    .risk-chip .chip-icon { font-size: 1.25rem; flex-shrink: 0; margin-top: 2px; }
    .risk-chip .chip-body { flex: 1; }
    .risk-chip .chip-title {
        font-weight: 600;
        font-size: 0.92rem;
        color: var(--text-primary);
        margin-bottom: 2px;
    }
    .risk-chip .chip-desc {
        font-size: 0.82rem;
        color: var(--text-secondary);
        line-height: 1.35;
    }

    /* Hide Streamlit branding but keep sidebar toggle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.05));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(99, 102, 241, 0.1);
        border-radius: 10px;
        padding: 8px 20px;
        color: var(--text-secondary);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── Load Models ───────────────────────────

@st.cache_resource
def load_all_models():
    """Load all saved models and data."""
    models_data = {}

    # ML models
    if os.path.exists("ml_models.pkl"):
        with open("ml_models.pkl", "rb") as f:
            models_data["ml_models"] = pickle.load(f)

    # Deep model
    if os.path.exists("cnn_lstm_model.pth") and os.path.exists("tokenizer.pkl"):
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        model = CNN_LSTM_Hybrid(
            vocab_size=tokenizer.vocab_size,
            embed_dim=64, struct_dim=3,
        )
        model.load_state_dict(torch.load("cnn_lstm_model.pth", map_location="cpu"))
        model.eval()
        models_data["deep_model"] = model
        models_data["tokenizer"] = tokenizer

    # Deep model metrics
    if os.path.exists("deep_model_metrics.pkl"):
        with open("deep_model_metrics.pkl", "rb") as f:
            models_data["deep_metrics"] = pickle.load(f)

    # TF-IDF vectorizer
    if os.path.exists("tfidf_vectorizer.pkl"):
        with open("tfidf_vectorizer.pkl", "rb") as f:
            models_data["tfidf"] = pickle.load(f)

    # Dataset Stats (lightweight version of df)
    if os.path.exists("dataset_stats.pkl"):
        with open("dataset_stats.pkl", "rb") as f:
            models_data["df"] = pickle.load(f)["df"]

    # Comparison CSV
    if os.path.exists("model_comparison.csv"):
        models_data["comparison"] = pd.read_csv("model_comparison.csv")

    # Best model info
    if os.path.exists("best_model_info.pkl"):
        with open("best_model_info.pkl", "rb") as f:
            models_data["best_info"] = pickle.load(f)

    return models_data


# ─────────────────────────── Prediction Function ───────────────────────────

def predict_job_posting(text, telecommuting, has_logo, has_questions, models_data):
    """Make predictions using all available models."""
    cleaned = clean_text(text)
    results = {}

    # ML model predictions
    if "ml_models" in models_data and "tfidf" in models_data:
        tfidf = models_data["tfidf"]
        from scipy.sparse import hstack, csr_matrix
        text_features = tfidf.transform([cleaned])
        struct = csr_matrix([[telecommuting, has_logo, has_questions]], dtype=np.float32)
        combined = hstack([text_features, struct])

        for name, model_info in models_data["ml_models"].items():
            model = model_info["model"]
            pred = model.predict(combined)[0]
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(combined)[0][1]
            else:
                prob = float(pred)
            results[name] = {"prediction": int(pred), "probability": float(prob)}

    # Deep model prediction
    if "deep_model" in models_data and "tokenizer" in models_data:
        tokenizer = models_data["tokenizer"]
        model = models_data["deep_model"]
        text_ids = torch.LongTensor([tokenizer.encode(cleaned)])
        struct_tensor = torch.FloatTensor([[telecommuting, has_logo, has_questions]])

        with torch.no_grad():
            logit = model(text_ids, struct_tensor)
            prob = torch.sigmoid(logit).item()
            pred = 1 if prob >= 0.5 else 0

        results["CNN+LSTM Hybrid"] = {"prediction": pred, "probability": prob}

    return results


# ─────────────────────────── Plotly Charts ───────────────────────────

def create_accuracy_chart(comp_df):
    """Create an interactive accuracy comparison chart."""
    fig = go.Figure()

    colors = ["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd", "#ddd6fe"]
    fig.add_trace(go.Bar(
        x=comp_df["Model"],
        y=comp_df["Accuracy"],
        marker=dict(
            color=colors[:len(comp_df)],
            line=dict(color="rgba(255,255,255,0.3)", width=1),
        ),
        text=[f"{v:.4f}" for v in comp_df["Accuracy"]],
        textposition="outside",
        textfont=dict(size=13, color="white", family="Inter"),
    ))

    fig.update_layout(
        title=dict(text="Accuracy Comparison", font=dict(size=20, color="white", family="Inter")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(color="#94a3b8", size=12), gridcolor="rgba(99,102,241,0.1)"),
        yaxis=dict(
            range=[0, 1.08], tickfont=dict(color="#94a3b8", size=12),
            gridcolor="rgba(99,102,241,0.1)", title=dict(text="Accuracy", font=dict(color="#94a3b8")),
        ),
        margin=dict(t=60, b=40),
        height=420,
    )
    return fig


def create_f1_chart(comp_df):
    """Create an interactive F1-score comparison chart."""
    fig = go.Figure()

    colors = ["#f43f5e", "#fb7185", "#fda4af", "#fecdd3", "#ffe4e6"]
    fig.add_trace(go.Bar(
        x=comp_df["Model"],
        y=comp_df["F1-Score"],
        marker=dict(
            color=colors[:len(comp_df)],
            line=dict(color="rgba(255,255,255,0.3)", width=1),
        ),
        text=[f"{v:.4f}" for v in comp_df["F1-Score"]],
        textposition="outside",
        textfont=dict(size=13, color="white", family="Inter"),
    ))

    fig.update_layout(
        title=dict(text="F1-Score Comparison", font=dict(size=20, color="white", family="Inter")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(color="#94a3b8", size=12), gridcolor="rgba(99,102,241,0.1)"),
        yaxis=dict(
            range=[0, 1.08], tickfont=dict(color="#94a3b8", size=12),
            gridcolor="rgba(99,102,241,0.1)", title=dict(text="F1-Score", font=dict(color="#94a3b8")),
        ),
        margin=dict(t=60, b=40),
        height=420,
    )
    return fig


def create_radar_chart(comp_df):
    """Create a radar chart comparing all metrics per model."""
    fig = go.Figure()
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["#6366f1", "#10b981", "#f59e0b", "#f43f5e", "#8b5cf6"]

    for i, (_, row) in enumerate(comp_df.iterrows()):
        values = [row[m] for m in metrics] + [row[metrics[0]]]  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill="toself",
            fillcolor=f"rgba({int(colors[i % len(colors)][1:3], 16)}, "
                      f"{int(colors[i % len(colors)][3:5], 16)}, "
                      f"{int(colors[i % len(colors)][5:7], 16)}, 0.1)",
            line=dict(color=colors[i % len(colors)], width=2),
            name=row["Model"],
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor="rgba(99,102,241,0.15)",
                tickfont=dict(color="#94a3b8"),
            ),
            angularaxis=dict(
                gridcolor="rgba(99,102,241,0.15)",
                tickfont=dict(color="#94a3b8", size=13),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Model Performance Radar", font=dict(size=20, color="white", family="Inter")),
        legend=dict(font=dict(color="#94a3b8", size=12)),
        height=500,
        margin=dict(t=80),
    )
    return fig


def create_grouped_bar(comp_df):
    """Create a grouped bar chart with all metrics."""
    fig = go.Figure()
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["#6366f1", "#10b981", "#f59e0b", "#f43f5e"]

    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            x=comp_df["Model"], y=comp_df[metric],
            name=metric,
            marker_color=color,
            text=[f"{v:.3f}" for v in comp_df[metric]],
            textposition="outside",
            textfont=dict(size=10, color="white"),
        ))

    fig.update_layout(
        barmode="group",
        title=dict(text="All Metrics Comparison", font=dict(size=20, color="white", family="Inter")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(color="#94a3b8", size=12), gridcolor="rgba(99,102,241,0.1)"),
        yaxis=dict(
            range=[0, 1.12], tickfont=dict(color="#94a3b8", size=12),
            gridcolor="rgba(99,102,241,0.1)", title=dict(text="Score", font=dict(color="#94a3b8")),
        ),
        legend=dict(font=dict(color="#94a3b8"), orientation="h", y=1.12),
        margin=dict(t=80, b=40),
        height=480,
    )
    return fig


def create_heatmap(comp_df):
    """Create a heatmap of metrics across models."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    z = comp_df[metrics].values
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=metrics,
        y=comp_df["Model"],
        colorscale=[[0, "#0f0f1a"], [0.5, "#6366f1"], [1, "#f43f5e"]],
        text=[[f"{v:.4f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=13, color="white"),
        hovertemplate="Model: %{y}<br>Metric: %{x}<br>Score: %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Performance Heatmap", font=dict(size=20, color="white", family="Inter")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(color="#94a3b8", size=13)),
        yaxis=dict(tickfont=dict(color="#94a3b8", size=12), autorange="reversed"),
        height=350,
        margin=dict(t=60, b=40, l=160),
    )
    return fig


# ─────────────────────────── Main App ───────────────────────────

def main():
    # Load all models
    models_data = load_all_models()

    # ──── Header ────
    st.markdown('<h1 class="hero-title">🔍 Fake Job Posting Detector</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">AI-Powered Detection System using ML + CNN-LSTM Hybrid Architecture</p>',
        unsafe_allow_html=True,
    )

    # ──── Sidebar ────
    with st.sidebar:
        st.markdown("## 🎯 Navigation")
        # The CSS added earlier will automatically style these radio options
        # to look like modern glowing glass tabs.
        page = st.radio(
            "Select Page",
            ["🔮 Prediction", "📊 Model Comparison", "📈 Dataset Analysis"],
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-glass-card">
            <div class="info-title">ℹ️ System Info</div>
            <div class="info-text">
                Detects <strong>fraudulent job postings</strong> using an ensemble of ML algorithms and advanced deep learning.
            </div>
            <div style="font-size: 0.8rem; color: #a5b4fc; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; font-weight: 600;">Models Active:</div>
            <ul>
                <li>Logistic Regression</li>
                <li>Naive Bayes</li>
                <li>Random Forest</li>
                <li>XGBoost</li>
                <li>CNN+LSTM Hybrid</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if "best_info" in models_data:
            best = models_data["best_info"]
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🏆 Active Champion")
            st.markdown(
                f"""
                <div class="model-glow-wrapper">
                    <div class="model-glow-card">
                        <div class="model-glow-title">Best Performance</div>
                        <div class="model-glow-name">{best["name"]}</div>
                        <div class="model-glow-metric">
                            <span>F1-SCORE</span>
                            <span>{best['metrics']['f1']:.4f}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ──── Page: Prediction ────
    if "Prediction" in page:
        render_prediction_page(models_data)

    elif "Model Comparison" in page:
        render_comparison_page(models_data)

    elif "Dataset Analysis" in page:
        render_dataset_page(models_data)


def _get_best_prediction(results: dict, models_data: dict) -> tuple:
    """Pick the single best-model result. Returns (model_name, prediction_dict)."""
    best_name = None
    if "best_info" in models_data:
        best_name = models_data["best_info"]["name"]
    # Fall back if best_name not in results
    if best_name and best_name in results:
        return best_name, results[best_name]
    # Default to first available
    name = next(iter(results))
    return name, results[name]


def _build_risk_suggestions(job_title, job_description, requirements,
                            benefits, telecommuting, has_logo, has_questions):
    """Generate intelligent risk-analysis suggestions from the raw inputs."""
    suggestions = []

    # ── Missing / thin content checks ──
    if len(job_description.strip()) < 40:
        suggestions.append({
            "icon": "📋", "title": "Vague Job Description",
            "desc": "The description is unusually short. Legitimate employers typically provide detailed role information.",
        })

    if not requirements.strip():
        suggestions.append({
            "icon": "📌", "title": "No Requirements Listed",
            "desc": "Genuine job postings usually specify required skills or qualifications.",
        })

    if not benefits.strip():
        suggestions.append({
            "icon": "💰", "title": "No Benefits or Salary Information",
            "desc": "Verify compensation details directly with the employer before proceeding.",
        })

    if not has_logo:
        suggestions.append({
            "icon": "🏢", "title": "No Company Logo",
            "desc": "Be cautious of unverified employers. Check the company on LinkedIn or Glassdoor.",
        })

    if not has_questions:
        suggestions.append({
            "icon": "❓", "title": "No Screening Questions",
            "desc": "Legitimate postings often include screening questions to filter candidates.",
        })

    # ── Suspicious keyword patterns ──
    desc_lower = job_description.lower()
    suspicious_keywords = [
        ("work from home", "earn"),
        ("no experience",),
        ("send money",),
        ("wire transfer",),
        ("guaranteed income",),
        ("unlimited earning",),
        ("make money fast",),
        ("processing fee",),
        ("personal bank",),
    ]
    for kw_group in suspicious_keywords:
        if all(kw in desc_lower for kw in kw_group):
            suggestions.append({
                "icon": "🚩", "title": "Suspicious Language Detected",
                "desc": "The description contains phrases commonly found in fraudulent postings. Review the job description very carefully.",
            })
            break  # one flag is enough

    if telecommuting and len(job_description.strip()) < 100:
        suggestions.append({
            "icon": "🌐", "title": "Remote Job With Vague Details",
            "desc": "Confirm job legitimacy before applying. Request a video interview and verify the company domain.",
        })

    # Always provide at least one constructive tip
    if not suggestions:
        suggestions.append({
            "icon": "✅", "title": "No Major Red Flags Found",
            "desc": "The posting looks reasonable, but always research the company and never share financial details upfront.",
        })

    return suggestions


def render_prediction_page(models_data):
    """Render the decision-focused prediction interface."""
    st.markdown('<div class="section-header">📝 Enter Job Posting Details</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        job_title = st.text_input("Job Title", placeholder="e.g., Senior Data Scientist")
        job_description = st.text_area(
            "Job Description",
            height=180,
            placeholder="Paste the full job description here...",
        )
        requirements = st.text_area(
            "Requirements",
            height=100,
            placeholder="Required skills, experience, education...",
        )
        benefits = st.text_area(
            "Benefits (optional)",
            height=80,
            placeholder="Health insurance, PTO, salary info...",
        )

    with col2:
        st.markdown("### 🔧 Job Posting Features")
        telecommuting = st.toggle("Remote / Telecommuting", value=False)
        has_logo = st.toggle("Has Company Logo", value=True)
        has_questions = st.toggle("Has Screening Questions", value=True)

        # Show which model the system uses
        if "best_info" in models_data:
            best = models_data["best_info"]
            st.markdown("---")
            st.markdown("### 🤖 Detection Engine")
            st.markdown(
                f'<span class="best-model-badge">{best["name"]}</span>'
                f'<div style="color:#94a3b8;font-size:0.8rem;margin-top:6px;">'
                f'F1 {best["metrics"]["f1"]:.2%} · Acc {best["metrics"]["accuracy"]:.2%}</div>',
                unsafe_allow_html=True,
            )

    # ── Analyze button ──
    if st.button("🔍 Analyze Job Posting", type="primary", use_container_width=True):
        combined_text = f"{job_title} {job_description} {requirements} {benefits}"

        if len(combined_text.strip()) < 10:
            st.warning("⚠️ Please enter more details for an accurate analysis.")
            return

        with st.spinner("🧠 Analyzing with AI..."):
            results = predict_job_posting(
                combined_text,
                int(telecommuting), int(has_logo), int(has_questions),
                models_data,
            )

        if not results:
            st.error("❌ No trained models found. Run `python train.py` first.")
            return

        # Pick the best model result
        best_model_name, best_result = _get_best_prediction(results, models_data)
        is_fake = best_result["prediction"] == 1
        fraud_prob = best_result["probability"]
        confidence = fraud_prob if is_fake else (1 - fraud_prob)
        confidence_pct = confidence * 100

        # ── 1. Verdict Hero ──
        st.markdown('<div class="section-header">🎯 Analysis Result</div>', unsafe_allow_html=True)

        if is_fake:
            verdict_cls = "danger"
            verdict_icon = "🚨"
            verdict_title = "Fake Job Posting"
            verdict_sub = f"Fraud probability: {fraud_prob:.1%}"
            bar_color = "linear-gradient(90deg, #f43f5e, #e11d48)"
            title_color = "#f43f5e"
        else:
            verdict_cls = "safe"
            verdict_icon = "✅"
            verdict_title = "Real Job Posting"
            verdict_sub = f"Legitimacy confidence: {confidence:.1%}"
            bar_color = "linear-gradient(90deg, #10b981, #059669)"
            title_color = "#10b981"

        st.markdown(f"""
        <div class="verdict-card {verdict_cls}">
            <div class="verdict-icon">{verdict_icon}</div>
            <div class="verdict-title" style="color:{title_color};">{verdict_title}</div>
            <div class="verdict-sub">{verdict_sub}</div>
            <div style="max-width:420px;margin:0 auto;">
                <div class="conf-bar-outer">
                    <div class="conf-bar-inner" style="width:{confidence_pct:.0f}%;background:{bar_color};"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#94a3b8;">
                    <span>0%</span>
                    <span style="font-weight:600;color:{title_color};">{confidence_pct:.0f}% confidence</span>
                    <span>100%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 2. Decision Guidance ──
        if is_fake:
            guidance_msg = (
                "⚠️ <strong>This job posting is likely fraudulent.</strong> "
                "It is <strong>NOT recommended</strong> to apply. "
                "Do not share personal information or make any payments."
            )
            guidance_cls = "danger"
        else:
            guidance_msg = (
                "✅ <strong>This job posting appears legitimate.</strong> "
                "You may proceed with caution. Always verify the company "
                "through official channels before sharing sensitive data."
            )
            guidance_cls = "safe"

        st.markdown(
            f'<div class="guidance-box {guidance_cls}">{guidance_msg}</div>',
            unsafe_allow_html=True,
        )

        # ── 3. Risk Analysis Suggestions ──
        st.markdown(
            '<div class="section-header">🛡️ Risk Analysis & Suggestions</div>',
            unsafe_allow_html=True,
        )

        suggestions = _build_risk_suggestions(
            job_title, job_description, requirements,
            benefits, telecommuting, has_logo, has_questions,
        )

        for s in suggestions:
            st.markdown(f"""
            <div class="risk-chip">
                <div class="chip-icon">{s['icon']}</div>
                <div class="chip-body">
                    <div class="chip-title">{s['title']}</div>
                    <div class="chip-desc">{s['desc']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── 4. Confidence gauge (Plotly) ──
        st.markdown("", unsafe_allow_html=True)  # spacer

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob * 100,
            number={"suffix": "%", "font": {"size": 42, "color": "white"}},
            title={"text": "Fraud Probability", "font": {"size": 16, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#333",
                         "tickfont": {"color": "#94a3b8"}},
                "bar": {"color": "#f43f5e" if is_fake else "#10b981", "thickness": 0.35},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 30], "color": "rgba(16,185,129,0.12)"},
                    {"range": [30, 60], "color": "rgba(245,158,11,0.12)"},
                    {"range": [60, 100], "color": "rgba(244,63,94,0.12)"},
                ],
                "threshold": {
                    "line": {"color": "#f59e0b", "width": 3},
                    "thickness": 0.8, "value": 50,
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(t=40, b=0, l=40, r=40),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_comparison_page(models_data):
    """Render the model comparison page."""
    st.markdown('<div class="section-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)

    if "comparison" not in models_data:
        st.warning("⚠️ No comparison data found. Run `python train.py` first.")
        return

    comp_df = models_data["comparison"]

    # Best model highlight
    if "best_info" in models_data:
        best = models_data["best_info"]
        st.success(f"🏆 **Best Model: {best['name']}** — F1-Score: {best['metrics']['f1']:.4f} | "
                   f"Accuracy: {best['metrics']['accuracy']:.4f}")

    # Metrics cards
    st.markdown("### Performance Overview")
    cols = st.columns(len(comp_df))
    for i, (_, row) in enumerate(comp_df.iterrows()):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.3rem;">
                    {row['Model']}
                </div>
                <div class="metric-value">{row['F1-Score']:.3f}</div>
                <div class="metric-label">F1-Score</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    # Comparison Table
    st.markdown("### 📋 Detailed Metrics Table")
    styled_df = comp_df.style.format({
        "Accuracy": "{:.4f}", "Precision": "{:.4f}",
        "Recall": "{:.4f}", "F1-Score": "{:.4f}", "ROC-AUC": "{:.4f}",
    }).highlight_max(
        subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        color="rgba(99, 102, 241, 0.3)",
    )
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Charts
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Accuracy", "🎯 F1-Score", "🕸️ Radar Chart",
        "📈 All Metrics", "🔥 Heatmap",
    ])

    with tab1:
        st.plotly_chart(create_accuracy_chart(comp_df), use_container_width=True)

    with tab2:
        st.plotly_chart(create_f1_chart(comp_df), use_container_width=True)

    with tab3:
        st.plotly_chart(create_radar_chart(comp_df), use_container_width=True)

    with tab4:
        st.plotly_chart(create_grouped_bar(comp_df), use_container_width=True)

    with tab5:
        st.plotly_chart(create_heatmap(comp_df), use_container_width=True)

    # Static images if available
    st.markdown("---")
    st.markdown("### 📸 Training Artifacts")
    img_cols = st.columns(2)
    images = [
        ("confusion_matrices.png", "Confusion Matrices"),
        ("training_history.png", "CNN+LSTM Training History"),
    ]
    for i, (img_path, caption) in enumerate(images):
        if os.path.exists(img_path):
            with img_cols[i]:
                st.image(img_path, caption=caption, use_container_width=True)


def render_dataset_page(models_data):
    """Render the dataset analysis page."""
    st.markdown('<div class="section-header">📈 Dataset Analysis</div>', unsafe_allow_html=True)

    if "df" not in models_data:
        st.warning("⚠️ No preprocessed data found. Run `python train.py` first.")
        return

    df = models_data["df"]

    # Basic stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Postings", f"{len(df):,}")
    with col2:
        real_count = (df["fraudulent"] == 0).sum()
        st.metric("Real Jobs", f"{real_count:,}")
    with col3:
        fake_count = (df["fraudulent"] == 1).sum()
        st.metric("Fake Jobs", f"{fake_count:,}")
    with col4:
        st.metric("Fraud Rate", f"{fake_count / len(df):.1%}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        # Class distribution
        fig = go.Figure(data=[go.Pie(
            labels=["Real", "Fake"],
            values=[real_count, fake_count],
            marker=dict(colors=["#10b981", "#f43f5e"]),
            hole=0.55,
            textfont=dict(size=14, color="white"),
            textinfo="label+percent",
        )])
        fig.update_layout(
            title=dict(text="Class Distribution", font=dict(size=18, color="white")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(color="#94a3b8")),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Feature presence
        feature_data = pd.DataFrame({
            "Feature": ["Telecommuting", "Company Logo", "Screening Questions"],
            "Real Jobs": [
                df[df["fraudulent"] == 0]["telecommuting"].mean(),
                df[df["fraudulent"] == 0]["has_company_logo"].mean(),
                df[df["fraudulent"] == 0]["has_questions"].mean(),
            ],
            "Fake Jobs": [
                df[df["fraudulent"] == 1]["telecommuting"].mean(),
                df[df["fraudulent"] == 1]["has_company_logo"].mean(),
                df[df["fraudulent"] == 1]["has_questions"].mean(),
            ],
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=feature_data["Feature"], y=feature_data["Real Jobs"],
            name="Real", marker_color="#10b981",
            text=[f"{v:.1%}" for v in feature_data["Real Jobs"]],
            textposition="outside", textfont=dict(color="white"),
        ))
        fig.add_trace(go.Bar(
            x=feature_data["Feature"], y=feature_data["Fake Jobs"],
            name="Fake", marker_color="#f43f5e",
            text=[f"{v:.1%}" for v in feature_data["Fake Jobs"]],
            textposition="outside", textfont=dict(color="white"),
        ))
        fig.update_layout(
            barmode="group",
            title=dict(text="Feature Presence: Real vs Fake", font=dict(size=18, color="white")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(tickformat=".0%", gridcolor="rgba(99,102,241,0.1)",
                       tickfont=dict(color="#94a3b8")),
            xaxis=dict(tickfont=dict(color="#94a3b8")),
            legend=dict(font=dict(color="#94a3b8")),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Text length distribution
    st.markdown("### 📝 Text Length Analysis")
    df["text_len"] = df["clean_text"].str.len()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[df["fraudulent"] == 0]["text_len"],
        name="Real", marker_color="rgba(16, 185, 129, 0.6)",
        nbinsx=50,
    ))
    fig.add_trace(go.Histogram(
        x=df[df["fraudulent"] == 1]["text_len"],
        name="Fake", marker_color="rgba(244, 63, 94, 0.6)",
        nbinsx=50,
    ))
    fig.update_layout(
        barmode="overlay",
        title=dict(text="Combined Text Length Distribution", font=dict(size=18, color="white")),
        xaxis=dict(title="Character Count", tickfont=dict(color="#94a3b8"),
                   gridcolor="rgba(99,102,241,0.1)"),
        yaxis=dict(title="Count", tickfont=dict(color="#94a3b8"),
                   gridcolor="rgba(99,102,241,0.1)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#94a3b8")),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Missing values
    st.markdown("### 🔍 Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    fig = go.Figure(go.Bar(
        x=missing.values, y=missing.index,
        orientation="h",
        marker_color="#6366f1",
        text=missing.values,
        textposition="outside",
        textfont=dict(color="white"),
    ))
    fig.update_layout(
        title=dict(text="Missing Values by Column", font=dict(size=18, color="white")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(color="#94a3b8"), gridcolor="rgba(99,102,241,0.1)"),
        yaxis=dict(tickfont=dict(color="#94a3b8")),
        height=400, margin=dict(l=150),
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
