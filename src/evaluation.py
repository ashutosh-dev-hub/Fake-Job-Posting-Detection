"""
Evaluation & Visualization Module.
Generates comparison tables and plots.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ──────────────────── Comparison Table ────────────────────

def build_comparison_table(all_results: dict) -> pd.DataFrame:
    """
    Build a comparison DataFrame from all model results.

    Parameters
    ----------
    all_results : dict[str, dict]
        Each value must have a 'metrics' sub-dict with
        accuracy, precision, recall, f1 (and optionally roc_auc).
    """
    rows = []
    for name, res in all_results.items():
        m = res["metrics"]
        rows.append({
            "Model": name,
            "Accuracy":  round(m["accuracy"], 4),
            "Precision": round(m["precision"], 4),
            "Recall":    round(m["recall"], 4),
            "F1-Score":  round(m["f1"], 4),
            "ROC-AUC":   round(m.get("roc_auc", 0), 4),
        })
    df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    return df


# ──────────────────── Plots ────────────────────

def plot_accuracy_comparison(comparison_df: pd.DataFrame, save_path="accuracy_comparison.png"):
    """Bar chart comparing accuracy across models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd", "#ddd6fe"]
    bars = ax.bar(comparison_df["Model"], comparison_df["Accuracy"],
                  color=colors[:len(comparison_df)], edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, comparison_df["Accuracy"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Model Accuracy Comparison", fontsize=16, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved {save_path}")


def plot_f1_comparison(comparison_df: pd.DataFrame, save_path="f1_comparison.png"):
    """Bar chart comparing F1-score across models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#f43f5e", "#fb7185", "#fda4af", "#fecdd3", "#ffe4e6"]
    bars = ax.bar(comparison_df["Model"], comparison_df["F1-Score"],
                  color=colors[:len(comparison_df)], edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, comparison_df["F1-Score"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-Score", fontsize=13)
    ax.set_title("Model F1-Score Comparison", fontsize=16, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved {save_path}")


def plot_all_metrics_grouped(comparison_df: pd.DataFrame, save_path="all_metrics_comparison.png"):
    """Grouped bar chart with all metrics side by side."""
    fig, ax = plt.subplots(figsize=(14, 7))

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(comparison_df))
    width = 0.18
    colors = ["#6366f1", "#10b981", "#f59e0b", "#f43f5e"]

    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * width, comparison_df[metric], width,
                      label=metric, color=colors[i], edgecolor="white")
        for bar, val in zip(bars, comparison_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df["Model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("All Metrics Comparison", fontsize=16, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved {save_path}")


def plot_confusion_matrices(all_results: dict, y_test, save_path="confusion_matrices.png"):
    """Plot confusion matrix for each model in a grid."""
    n = len(all_results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, res) in enumerate(all_results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
        disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
        axes[idx].set_title(name, fontsize=12, fontweight="bold")

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Confusion Matrices", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved {save_path}")


def plot_training_history(history: dict, save_path="training_history.png"):
    """Plot CNN+LSTM training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "o-", color="#6366f1", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "s-", color="#f43f5e", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(epochs, history["val_acc"], "D-", color="#10b981", label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy", fontweight="bold")
    ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved {save_path}")


def identify_best_model(comparison_df: pd.DataFrame) -> str:
    """Return the name of the best model by F1-Score."""
    best = comparison_df.iloc[0]  # already sorted descending
    print(f"\n{'='*60}")
    print(f"  [BEST] Best Model: {best['Model']}")
    print(f"     F1-Score : {best['F1-Score']:.4f}")
    print(f"     Accuracy : {best['Accuracy']:.4f}")
    print(f"{'='*60}")
    return best["Model"]
