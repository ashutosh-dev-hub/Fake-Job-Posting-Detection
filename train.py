"""
Main Training Pipeline — Fake Job Posting Detection System.
Runs preprocessing, trains all models, evaluates, and saves results.
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_and_preprocess, save_preprocessed
from src.ml_models import train_all_models, save_models, evaluate_model
from src.deep_model import train_deep_model, save_deep_model
from src.evaluation import (
    build_comparison_table,
    plot_accuracy_comparison,
    plot_f1_comparison,
    plot_all_metrics_grouped,
    plot_confusion_matrices,
    plot_training_history,
    identify_best_model,
)


def main():
    CSV_PATH = "fake_job_postings.csv"

    # ── Step 1: Preprocessing ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 1: DATA PREPROCESSING")
    print("=" * 70)
    data = load_and_preprocess(CSV_PATH, max_tfidf_features=10000)
    save_preprocessed(data)

    # ── Step 2: Train Classical ML Models ──────────────────────
    print("\n" + "=" * 70)
    print("  STEP 2: TRAINING CLASSICAL ML MODELS")
    print("=" * 70)
    ml_results = train_all_models(
        data["X_train_tfidf"], data["y_train"],
        data["X_test_tfidf"],  data["y_test"],
    )
    save_models(ml_results)

    # ── Step 3: Train CNN + LSTM Hybrid ────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 3: TRAINING CNN + LSTM HYBRID MODEL")
    print("=" * 70)
    dl_result = train_deep_model(
        X_train_text=data["X_train_text"],
        X_train_struct=data["X_train_struct"],
        y_train=data["y_train"],
        X_test_text=data["X_test_text"],
        X_test_struct=data["X_test_struct"],
        y_test=data["y_test"],
        max_vocab=20000,
        max_len=300,
        embed_dim=128,
        epochs=10,
        batch_size=64,
        lr=0.001,
    )
    save_deep_model(dl_result)

    # ── Step 4: Compare All Models ─────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 4: MODEL COMPARISON")
    print("=" * 70)

    # Merge all results
    all_results = {}
    for name, res in ml_results.items():
        all_results[name] = res
    all_results["CNN+LSTM Hybrid"] = {
        "metrics": dl_result["metrics"],
        "y_pred":  dl_result["y_pred"],
        "y_prob":  dl_result["y_prob"],
    }

    # Comparison table
    comp_df = build_comparison_table(all_results)
    print("\n" + comp_df.to_string(index=False))
    comp_df.to_csv("model_comparison.csv", index=False)

    # Plots
    plot_accuracy_comparison(comp_df)
    plot_f1_comparison(comp_df)
    plot_all_metrics_grouped(comp_df)
    plot_confusion_matrices(all_results, data["y_test"])
    plot_training_history(dl_result["history"])

    # Best model
    best_name = identify_best_model(comp_df)

    # Save the best model info
    best_info = {"name": best_name, "metrics": all_results[best_name]["metrics"]}
    with open("best_model_info.pkl", "wb") as f:
        pickle.dump(best_info, f)

    print("\n[DONE] All models trained, compared, and saved!")
    print("  → model_comparison.csv")
    print("  → accuracy_comparison.png")
    print("  → f1_comparison.png")
    print("  → all_metrics_comparison.png")
    print("  → confusion_matrices.png")
    print("  → training_history.png")
    print("\n  Run the dashboard:  streamlit run dashboard.py")


if __name__ == "__main__":
    main()
