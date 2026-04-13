"""
Classical ML Training Module.
Trains Logistic Regression, Naive Bayes, Random Forest, and XGBoost.
"""

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)


# ───────── model definitions ─────────

def get_models() -> dict:
    """Return a dict of name → untrained estimator."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0, random_state=42
        ),
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=30, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=17014 / 866,   # handle imbalance
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        ),
    }


# ───────── train & evaluate ─────────

def evaluate_model(y_true, y_pred, y_prob=None):
    """Compute standard classification metrics."""
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        from sklearn.metrics import roc_auc_score
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.0
    return metrics


def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train all classical ML models and return results.

    Returns
    -------
    results : dict[str, dict]
        Each entry has keys: model, metrics, y_pred, y_prob
    """
    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"  Training: {name}")
        print(f"{'='*60}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # probability for positive class
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = None

        metrics = evaluate_model(y_test, y_pred, y_prob)

        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1-score : {metrics['f1']:.4f}")
        if "roc_auc" in metrics:
            print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")

        results[name] = {
            "model":   model,
            "metrics": metrics,
            "y_pred":  y_pred,
            "y_prob":  y_prob,
        }

    return results


def save_models(results: dict, path: str = "ml_models.pkl"):
    """Persist trained models and their metrics."""
    save_data = {}
    for name, r in results.items():
        save_data[name] = {
            "model":   r["model"],
            "metrics": r["metrics"],
        }
    with open(path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"\n[INFO] Models saved to {path}")


def load_models(path: str = "ml_models.pkl") -> dict:
    """Load previously saved models."""
    with open(path, "rb") as f:
        return pickle.load(f)
