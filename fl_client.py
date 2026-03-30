# =============================================================================
# fl_client.py
# FedHealth-CN — Phase 3: Flower Federated Learning Client
# IEEE Conference — Privacy-Preserving ICU Mortality Prediction
# =============================================================================
#
# WHAT THIS FILE DOES:
#   Defines the Flower client class. Each client represents one hospital.
#   It loads its own hospital shard CSV, trains locally, and sends ONLY
#   model weight updates (never raw patient data) back to the server.
#
# HOW FLOWER WORKS (simple explanation):
#   Server says:  "Here are the current global weights. Train on your data."
#   Client does:  Local training → sends back updated weights + sample count
#   Server does:  Aggregates all client weights → new global model
#   Repeat for N rounds.
#
# DO NOT RUN THIS FILE DIRECTLY.
# Run fl_server.py — it starts the simulation automatically.
# =============================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
import flwr as fl
from typing import List, Tuple, Dict

# =============================================================================
# SHARED CONFIGURATION  (must match train_centralized.py exactly)
# =============================================================================

THRESHOLD    = 0.30
BATCH_SIZE   = 8        # Smaller than centralized — hospital shards are tiny
LOCAL_EPOCHS = 5        # Rounds of local training per federated round
LEARNING_RATE = 0.001
RANDOM_SEED  = 42

# =============================================================================
# NEURAL NETWORK  (identical architecture to centralized model)
# =============================================================================

class MortalityPredictor(nn.Module):
    """
    3-layer Feed-Forward Network for binary mortality prediction.
    Uses BatchNorm1d — needs at least 2 samples per batch.
    """
    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)           # Raw logit — sigmoid applied at inference
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class ICUDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels,   dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# =============================================================================
# HELPER: GET MODEL WEIGHTS AS LIST OF NUMPY ARRAYS
# (Flower requires weights in this format for transmission)
# =============================================================================

def get_weights(model: nn.Module) -> List[np.ndarray]:
    """Extract all model parameters as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    """Load a list of numpy arrays back into the model."""
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict  = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


# =============================================================================
# FLOWER CLIENT CLASS
# =============================================================================

class HospitalClient(fl.client.NumPyClient):
    """
    Each instance of HospitalClient represents one hospital.
    It loads a specific hospital shard CSV and trains locally.

    Flower calls three methods on each client per round:
      get_parameters() → send current weights to server
      fit()            → train locally, return updated weights
      evaluate()       → evaluate on local val data, return metrics
    """

    def __init__(self, hospital_id: int, data_path: str):
        self.hospital_id = hospital_id
        self.data_path   = data_path

        torch.manual_seed(RANDOM_SEED + hospital_id)  # Different seed per hospital

        # --- Load and split hospital shard ---
        df = pd.read_csv(data_path)

        if len(df) < 4:
            raise ValueError(f"Hospital {hospital_id} shard too small: {len(df)} rows")

        feature_cols = [c for c in df.columns if c != "MORTALITY_48H"]
        X = df[feature_cols].values.astype(np.float32)
        y = df["MORTALITY_48H"].values.astype(np.float32)

        self.input_size  = X.shape[1]
        self.n_samples   = len(X)

        # Split 80% train / 20% val (small shards need most data for training)
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
            )
        except ValueError:
            # If only one class present, skip stratify
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.20, random_state=RANDOM_SEED
            )

        self.y_train = y_train
        self.X_val   = X_val
        self.y_val   = y_val

        # --- Weighted sampler for class imbalance ---
        pos_count = max(int(y_train.sum()), 1)
        neg_count = max(len(y_train) - pos_count, 1)
        class_wts  = {0: 1.0 / neg_count, 1: 1.0 / pos_count}
        sample_wts = [class_wts[int(lbl)] for lbl in y_train]

        sampler = WeightedRandomSampler(
            weights=sample_wts,
            num_samples=len(sample_wts),
            replacement=True
        )

        self.train_loader = DataLoader(
            ICUDataset(X_train, y_train),
            batch_size=min(BATCH_SIZE, len(X_train) - 1),
            sampler=sampler,
            drop_last=True          # Drop last batch if size=1 (BatchNorm needs ≥2)
        )
        self.val_loader = DataLoader(
            ICUDataset(X_val, y_val),
            batch_size=max(min(BATCH_SIZE, len(X_val)), 2),
            shuffle=False,
            drop_last=False
        )

        # --- Model, loss, optimizer ---
        self.model     = MortalityPredictor(self.input_size)
        pos_weight     = torch.tensor([neg_count / pos_count], dtype=torch.float32)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
        )

        print(f"   [Hospital {hospital_id:02d}] Loaded {self.n_samples} samples | "
              f"Train: {len(X_train)} | Val: {len(X_val)} | "
              f"Positives: {int(y.sum())}")

    # -------------------------------------------------------------------------
    # METHOD 1: get_parameters
    # Called by Flower at start of each round to get current local weights.
    # -------------------------------------------------------------------------
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return get_weights(self.model)

    # -------------------------------------------------------------------------
    # METHOD 2: fit
    # Called by Flower with global weights. Train locally, return updates.
    # -------------------------------------------------------------------------
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:

        # Load global weights into local model
        set_weights(self.model, parameters)

        self.model.train()
        total_loss = 0.0
        batches    = 0

        for _ in range(LOCAL_EPOCHS):
            for X_batch, y_batch in self.train_loader:
                if len(X_batch) < 2:    # BatchNorm needs ≥2 samples
                    continue
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss   = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                batches    += 1

        avg_loss = total_loss / max(batches, 1)

        # Return: updated weights, number of training samples, metrics dict
        return (
            get_weights(self.model),
            len(self.y_train),
            {"train_loss": float(avg_loss)}
        )

    # -------------------------------------------------------------------------
    # METHOD 3: evaluate
    # Called by Flower to get local validation metrics.
    # -------------------------------------------------------------------------
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:

        set_weights(self.model, parameters)
        self.model.eval()

        total_loss  = 0.0
        all_logits  = []
        all_labels  = []

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                logits     = self.model(X_batch)
                loss       = self.criterion(logits, y_batch)
                total_loss += loss.item() * len(y_batch)
                all_logits.extend(logits.numpy())
                all_labels.extend(y_batch.numpy())

        avg_loss   = total_loss / max(len(self.y_val), 1)
        probs      = torch.sigmoid(torch.tensor(all_logits)).numpy()
        preds_bin  = (probs >= THRESHOLD).astype(int)
        labels_arr = np.array(all_labels)

        try:
            auc = float(roc_auc_score(labels_arr, probs))
        except ValueError:
            auc = 0.0

        f1  = float(f1_score(labels_arr, preds_bin, zero_division=0))
        rec = float(recall_score(labels_arr, preds_bin, zero_division=0))

        return (
            float(avg_loss),
            len(self.y_val),
            {"auc": auc, "f1": f1, "recall": rec,
             "hospital_id": self.hospital_id}
        )