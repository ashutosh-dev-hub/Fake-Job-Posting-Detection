"""
CNN + LSTM Deep Learning Model (PyTorch).
Hybrid architecture that combines text embeddings with structured features.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter


# ──────────────────── Tokenizer ────────────────────

class SimpleTokenizer:
    """Word-level tokenizer with vocab building."""

    def __init__(self, max_vocab: int = 20000, max_len: int = 300):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def fit(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        most_common = counter.most_common(self.max_vocab - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        return self

    def encode(self, text: str) -> list:
        tokens = text.split()
        ids = [self.word2idx.get(w, 1) for w in tokens]  # 1 = <UNK>
        # Pad or truncate
        if len(ids) < self.max_len:
            ids = ids + [0] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return ids

    def encode_batch(self, texts) -> np.ndarray:
        return np.array([self.encode(t) for t in texts])

    @property
    def vocab_size(self):
        return len(self.word2idx)


# ──────────────────── Dataset ────────────────────

class JobPostingDataset(Dataset):
    def __init__(self, text_ids, struct_features, labels):
        self.text_ids = torch.LongTensor(text_ids)
        self.struct   = torch.FloatTensor(struct_features)
        self.labels   = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text_ids[idx], self.struct[idx], self.labels[idx]


# ──────────────────── CNN + LSTM Model ────────────────────

class CNN_LSTM_Hybrid(nn.Module):
    """
    Hybrid architecture:
      - Embedding → Conv1D → MaxPool → LSTM  (text branch)
      - Structured features concatenated before final classifier
    """

    def __init__(self, vocab_size, embed_dim=128, num_filters=64,
                 filter_sizes=(3, 4, 5), lstm_hidden=64, struct_dim=3,
                 dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multi-kernel CNN
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, num_filters, kernel_size=fs, padding=fs // 2),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(50),  # reduce to fixed length
            )
            for fs in filter_sizes
        ])

        # LSTM on top of CNN features
        self.lstm = nn.LSTM(
            input_size=num_filters * len(filter_sizes),
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Classifier
        combined_dim = lstm_hidden * 2 + struct_dim  # bidirectional
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, text_ids, struct_features):
        # text_ids: (batch, seq_len)
        emb = self.embedding(text_ids)      # (batch, seq_len, embed_dim)
        emb = emb.permute(0, 2, 1)          # (batch, embed_dim, seq_len)

        # CNN: multiple filter sizes
        conv_outs = [conv(emb) for conv in self.convs]  # each: (batch, num_filters, 50)
        conv_cat = torch.cat(conv_outs, dim=1)          # (batch, num_filters*3, 50)
        conv_cat = conv_cat.permute(0, 2, 1)            # (batch, 50, num_filters*3)

        # LSTM
        lstm_out, (h_n, _) = self.lstm(conv_cat)
        # use last hidden state from both directions
        h_fwd = h_n[0]   # (batch, lstm_hidden)
        h_bwd = h_n[1]   # (batch, lstm_hidden)
        text_repr = torch.cat([h_fwd, h_bwd], dim=1)   # (batch, lstm_hidden*2)

        # Combine with structured features
        combined = torch.cat([text_repr, struct_features], dim=1)

        out = self.classifier(combined)
        return out.squeeze(-1)


# ──────────────────── Training ────────────────────

def train_deep_model(X_train_text, X_train_struct, y_train,
                     X_test_text, X_test_struct, y_test,
                     max_vocab=20000, max_len=300,
                     embed_dim=128, epochs=10, batch_size=64,
                     lr=0.001, device=None):
    """
    Full training pipeline for the CNN+LSTM hybrid model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Tokenize
    tokenizer = SimpleTokenizer(max_vocab=max_vocab, max_len=max_len)
    tokenizer.fit(X_train_text)

    train_ids = tokenizer.encode_batch(X_train_text)
    test_ids  = tokenizer.encode_batch(X_test_text)

    # Datasets
    train_ds = JobPostingDataset(train_ids, X_train_struct, y_train)
    test_ds  = JobPostingDataset(test_ids, X_test_struct, y_test)

    # Handle class imbalance with weighted sampler
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    weights = np.where(y_train == 1, neg_count / pos_count, 1.0)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.DoubleTensor(weights), num_samples=len(weights), replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # Model
    model = CNN_LSTM_Hybrid(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        struct_dim=X_train_struct.shape[1],
    ).to(device)

    # Loss with pos_weight for imbalance
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for text_batch, struct_batch, label_batch in train_loader:
            text_batch  = text_batch.to(device)
            struct_batch = struct_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            output = model(text_batch, struct_batch)
            loss = criterion(output, label_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for text_batch, struct_batch, label_batch in test_loader:
                text_batch  = text_batch.to(device)
                struct_batch = struct_batch.to(device)
                label_batch = label_batch.to(device)

                output = model(text_batch, struct_batch)
                loss = criterion(output, label_batch)
                val_loss += loss.item()

                probs = torch.sigmoid(output).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(label_batch.cpu().numpy())

        avg_val_loss = val_loss / len(test_loader)
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))

        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch+1}/{epochs}  |  "
              f"Train Loss: {avg_train_loss:.4f}  |  "
              f"Val Loss: {avg_val_loss:.4f}  |  "
              f"Val Acc: {val_acc:.4f}")

    # Final evaluation
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
    except ValueError:
        metrics["roc_auc"] = 0.0

    print(f"\n  [CNN+LSTM Hybrid] Final Metrics:")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-score : {metrics['f1']:.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "history": history,
        "device": device,
    }


def save_deep_model(result: dict, model_path="cnn_lstm_model.pth",
                    tokenizer_path="tokenizer.pkl"):
    """Save model weights and tokenizer."""
    torch.save(result["model"].state_dict(), model_path)
    with open(tokenizer_path, "wb") as f:
        pickle.dump(result["tokenizer"], f)
    # Save metrics too
    with open("deep_model_metrics.pkl", "wb") as f:
        pickle.dump(result["metrics"], f)
    print(f"[INFO] Deep model saved to {model_path}")


def load_deep_model(model_path="cnn_lstm_model.pth",
                    tokenizer_path="tokenizer.pkl",
                    vocab_size=20000, embed_dim=128, struct_dim=3):
    """Load saved model and tokenizer."""
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    model = CNN_LSTM_Hybrid(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim, struct_dim=struct_dim,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, tokenizer
