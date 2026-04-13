"""
Step 2: CNN+LSTM training + evaluation.
IMPORTANT: Import torch/deep_model BEFORE loading pickle data to avoid DLL conflict.
"""
import os, sys, pickle, warnings, traceback
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Starting step2...")
sys.stdout.flush()

try:
    # IMPORTANT: Import torch-based modules FIRST before loading pickle/scipy
    print("Importing deep model (torch)...")
    sys.stdout.flush()
    from src.deep_model import train_deep_model, save_deep_model

    print("Loading preprocessed data...")
    sys.stdout.flush()
    from src.preprocessing import load_preprocessed
    data = load_preprocessed()

    print("Starting CNN+LSTM training...")
    sys.stdout.flush()

    dl_result = train_deep_model(
        X_train_text=data["X_train_text"],
        X_train_struct=data["X_train_struct"],
        y_train=data["y_train"],
        X_test_text=data["X_test_text"],
        X_test_struct=data["X_test_struct"],
        y_test=data["y_test"],
        max_vocab=15000,
        max_len=150,
        embed_dim=64,
        epochs=8,
        batch_size=32,
        lr=0.001,
    )
    save_deep_model(dl_result)

    # Save full results
    with open("dl_result_full.pkl", "wb") as f:
        pickle.dump({
            "metrics": dl_result["metrics"],
            "y_pred": dl_result["y_pred"],
            "y_prob": dl_result["y_prob"],
            "history": dl_result["history"],
        }, f)

    print("CNN+LSTM training complete!")
    sys.stdout.flush()

    # ── Evaluation ──
    print("Generating comparison & plots...")
    sys.stdout.flush()

    from src.evaluation import (
        build_comparison_table,
        plot_accuracy_comparison,
        plot_f1_comparison,
        plot_all_metrics_grouped,
        plot_confusion_matrices,
        plot_training_history,
        identify_best_model,
    )

    # Load ML results
    with open("ml_results_full.pkl", "rb") as f:
        ml_results = pickle.load(f)

    all_results = dict(ml_results)
    all_results["CNN+LSTM Hybrid"] = {
        "metrics": dl_result["metrics"],
        "y_pred": dl_result["y_pred"],
        "y_prob": dl_result["y_prob"],
    }

    comp_df = build_comparison_table(all_results)
    print("\n" + comp_df.to_string(index=False))
    comp_df.to_csv("model_comparison.csv", index=False)

    plot_accuracy_comparison(comp_df)
    plot_f1_comparison(comp_df)
    plot_all_metrics_grouped(comp_df)
    plot_confusion_matrices(all_results, data["y_test"])
    plot_training_history(dl_result["history"])

    best_name = identify_best_model(comp_df)
    best_info = {"name": best_name, "metrics": all_results[best_name]["metrics"]}
    with open("best_model_info.pkl", "wb") as f:
        pickle.dump(best_info, f)

    print("\nAll done! Run: streamlit run dashboard.py")

except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
