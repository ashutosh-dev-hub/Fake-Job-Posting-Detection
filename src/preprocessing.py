"""
Data Preprocessing Module for Fake Job Posting Detection.
Handles loading, cleaning, feature engineering, and splitting.
"""

import re
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix


# ───────────────────────── text cleaning ─────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, strip HTML, remove punctuation/digits, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)          # keep only letters
    text = re.sub(r"\s+", " ", text).strip()        # collapse spaces
    return text


# ─────────────────── main preprocessing pipeline ────────────────

def load_and_preprocess(csv_path: str,
                        max_tfidf_features: int = 10000,
                        test_size: float = 0.2,
                        random_state: int = 42):
    """
    End-to-end preprocessing:
      1. Load CSV
      2. Fill missing text fields
      3. Combine text columns into a single `combined_text`
      4. Clean text
      5. Build TF-IDF features  (for classical ML)
      6. Extract structured features (telecommuting, has_company_logo, has_questions)
      7. Train/test split (stratified)

    Returns
    -------
    dict with keys:
        X_train_tfidf, X_test_tfidf   – sparse matrices (TF-IDF + structured)
        X_train_text,  X_test_text     – raw cleaned text  (for deep learning tokenizer)
        X_train_struct, X_test_struct  – structured features only
        y_train, y_test                – labels
        tfidf_vectorizer               – fitted TfidfVectorizer
        df                             – processed DataFrame
    """
    # 1. Load
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[INFO] Class distribution:\n{df['fraudulent'].value_counts().to_string()}")

    # 2. Fill missing text
    text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
    for col in text_cols:
        df[col] = df[col].fillna("")

    # 3. Combine text fields
    df["combined_text"] = (
        df["title"] + " " +
        df["company_profile"] + " " +
        df["description"] + " " +
        df["requirements"] + " " +
        df["benefits"]
    )

    # 4. Clean
    df["clean_text"] = df["combined_text"].apply(clean_text)

    # 5. Structured binary features
    struct_cols = ["telecommuting", "has_company_logo", "has_questions"]
    X_struct = df[struct_cols].values.astype(np.float32)

    # 6. TF-IDF
    tfidf = TfidfVectorizer(
        max_features=max_tfidf_features,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(df["clean_text"])

    # Combine TF-IDF + structured
    X_combined = hstack([X_tfidf, csr_matrix(X_struct)])

    y = df["fraudulent"].values

    # 7. Stratified split
    (X_train_comb, X_test_comb,
     y_train, y_test,
     idx_train, idx_test) = train_test_split(
        X_combined, y, np.arange(len(y)),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train_text = df["clean_text"].iloc[idx_train].values
    X_test_text  = df["clean_text"].iloc[idx_test].values
    X_train_struct = X_struct[idx_train]
    X_test_struct  = X_struct[idx_test]

    print(f"[INFO] Train size: {len(y_train)}  |  Test size: {len(y_test)}")
    print(f"[INFO] TF-IDF features: {X_tfidf.shape[1]}  |  Structured features: {X_struct.shape[1]}")

    return {
        "X_train_tfidf": X_train_comb,
        "X_test_tfidf":  X_test_comb,
        "X_train_text":  X_train_text,
        "X_test_text":   X_test_text,
        "X_train_struct": X_train_struct,
        "X_test_struct":  X_test_struct,
        "y_train": y_train,
        "y_test":  y_test,
        "tfidf_vectorizer": tfidf,
        "df": df,
    }


def save_preprocessed(data: dict, path: str = "preprocessed_data.pkl"):
    """Persist the preprocessed data dict."""
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"[INFO] Preprocessed data saved to {path}")


def load_preprocessed(path: str = "preprocessed_data.pkl") -> dict:
    """Load previously saved preprocessed data."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[INFO] Preprocessed data loaded from {path}")
    return data
