# ============================================================
# ALL BANDS CNN — WINDOWS / VS CODE / DIRECTML VERSION
# ============================================================

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
# DEVICE SETUP
# ------------------------------------------------------------
USE_DIRECTML = False
try:
    import torch_directml
    DEVICE = torch_directml.device()
    USE_DIRECTML = True
    print("Using DirectML device")
except ImportError:
    DEVICE = torch.device("cpu")
    print("torch_directml not found. Falling back to CPU.")

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
SEED = 42
BATCH_SIZE = 16
EPOCHS = 10
PATIENCE = 3
NUM_WORKERS = 0   # safest for local npz loading

# CHANGE THIS ONLY IF YOUR DESKTOP PATH IS DIFFERENT
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

TRAIN_AUG_DIR = os.path.join(BASE_PATH, "train_augmented")
TEST_DIR = os.path.join(BASE_PATH, "test")
RESULTS_DIR = os.path.join(BASE_PATH, "all_bands_cnn")

TRAIN_MANIFEST = os.path.join(TRAIN_AUG_DIR, "manifest.csv")
TEST_MANIFEST = os.path.join(TEST_DIR, "manifest.csv")
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "all_bands_cnn_best.pt")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------
# REPRODUCIBILITY
# ------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("BASE_PATH:", BASE_PATH)
print("TRAIN_MANIFEST exists:", os.path.exists(TRAIN_MANIFEST))
print("TEST_MANIFEST exists :", os.path.exists(TEST_MANIFEST))
print("Results folder       :", RESULTS_DIR)

if not os.path.exists(TRAIN_MANIFEST):
    raise FileNotFoundError(f"Training manifest not found: {TRAIN_MANIFEST}")
if not os.path.exists(TEST_MANIFEST):
    raise FileNotFoundError(f"Test manifest not found: {TEST_MANIFEST}")

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def infer_file_col(df):
    for c in ["npz_path", "file_path", "filepath", "path", "filename", "file"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer file column from: {df.columns.tolist()}")


def infer_label_col(df):
    for c in ["label_num", "label", "y", "target", "class"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer label column from: {df.columns.tolist()}")


def resolve_path(base_dir, maybe_path):
    maybe_path = str(maybe_path)
    if os.path.isabs(maybe_path):
        return maybe_path
    return os.path.join(base_dir, maybe_path)


def parse_label(x):
    if isinstance(x, str):
        x = x.strip().lower()
        if x == "tumor":
            return 1
        elif x in ["nontumor", "non_tumor", "non-tumor"]:
            return 0
    return int(x)


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

# Custom PyTorch Dataset for loading hyperspectral image patches
#
# Each .npz file may contain:
# - a single hyperspectral patch
# - or multiple patches extracted from one histology sample
#
# The dataset expands each stored patch into an individual training sample.
# Patches are converted from:
# (Height, Width, Bands) -> (Channels, Height, Width)
# so they can be used by PyTorch Conv2D layers.

class HSINPZPatchDataset(Dataset):
    def __init__(self, df, base_dir, file_col, label_col):
        self.samples = []
        df = df.reset_index(drop=True).copy()

        for _, row in df.iterrows():
            patch_path = resolve_path(base_dir, row[file_col])
            npz = np.load(patch_path, allow_pickle=True)

            X = npz["X"] if "X" in npz.files else npz[npz.files[0]]
            label = parse_label(row[label_col])

            if X.ndim == 4:
                # shape: (N, H, W, Bands)
                for patch_idx in range(X.shape[0]):
                    self.samples.append((patch_path, patch_idx, label))
            elif X.ndim == 3:
                # shape: (H, W, Bands)
                self.samples.append((patch_path, None, label))
            else:
                raise ValueError(f"Unexpected array shape {X.shape} in file {patch_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch_path, patch_idx, label = self.samples[idx]

        npz = np.load(patch_path, allow_pickle=True)
        X = npz["X"] if "X" in npz.files else npz[npz.files[0]]

        patch = X[patch_idx] if X.ndim == 4 else X
        patch = patch.astype(np.float32)

        if patch.ndim != 3:
            raise ValueError(f"Unexpected patch shape {patch.shape} in file {patch_path}")

        # PyTorch Conv2D expects channel-first tensors:
        # (Channels, Height, Width)
        #
        # Original hyperspectral patches are stored as:
        # (Height, Width, Spectral Bands)
        #
        # Convert HWC -> CHW before training.
        patch = np.transpose(patch, (2, 0, 1))

        x = torch.tensor(patch, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
class FullBandsCNN(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super().__init__()

        # Feature extraction layers
        #
        # The CNN progressively learns:
        # - low-level spectral/spatial features
        # - intermediate texture patterns
        # - higher-level tumor representations
        #
        # MaxPooling reduces spatial dimensions.
        # Dropout helps reduce overfitting.
        # BatchNorm stabilizes training.
        self.features = nn.Sequential(
            
            # 1x1 convolution compresses the large spectral dimension
            # while preserving spatial information.
            #
            # This helps reduce computational complexity when using
            # all 826 spectral bands.
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),  # 32x32 -> 16 -> 8 -> 4
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ------------------------------------------------------------
# TRAIN / EVAL
# ------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    all_preds, all_probs, all_targets = [], [], []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(xb)
            loss = criterion(logits, yb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * xb.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        all_targets.extend(yb.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_prec = precision_score(all_targets, all_preds, zero_division=0)
    epoch_rec = recall_score(all_targets, all_preds, zero_division=0)
    epoch_f1 = f1_score(all_targets, all_preds, zero_division=0)

    if len(np.unique(all_targets)) > 1:
        epoch_auc = roc_auc_score(all_targets, all_probs)
    else:
        epoch_auc = np.nan

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "precision": epoch_prec,
        "recall": epoch_rec,
        "f1": epoch_f1,
        "roc_auc": epoch_auc,
        "targets": np.array(all_targets),
        "preds": np.array(all_preds),
        "probs": np.array(all_probs),
    }


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    # -------------------------
    # Load manifests
    # -------------------------
    train_aug_df = pd.read_csv(TRAIN_MANIFEST)
    test_df = pd.read_csv(TEST_MANIFEST)

    print("\nOriginal manifest sizes:")
    print("train_aug_df:", train_aug_df.shape)
    print("test_df     :", test_df.shape)

    print("\nTrain manifest columns:", train_aug_df.columns.tolist())
    print("Test manifest columns :", test_df.columns.tolist())

    FILE_COL = infer_file_col(train_aug_df)
    LABEL_COL = infer_label_col(train_aug_df)

    print("\nUsing file column :", FILE_COL)
    print("Using label column:", LABEL_COL)

    # normalize paths to filenames only
    train_aug_df[FILE_COL] = train_aug_df[FILE_COL].apply(lambda x: os.path.basename(str(x)))
    test_df[FILE_COL] = test_df[FILE_COL].apply(lambda x: os.path.basename(str(x)))

    # remove rows whose files do not exist
    train_aug_df = train_aug_df[
        train_aug_df[FILE_COL].apply(lambda f: os.path.exists(os.path.join(TRAIN_AUG_DIR, str(f))))
    ].reset_index(drop=True)

    test_df = test_df[
        test_df[FILE_COL].apply(lambda f: os.path.exists(os.path.join(TEST_DIR, str(f))))
    ].reset_index(drop=True)

    print("\nAfter filtering missing files:")
    print("train_aug_df:", train_aug_df.shape)
    print("test_df     :", test_df.shape)

    if len(train_aug_df) == 0:
        raise ValueError("No training files found after filtering.")
    if len(test_df) == 0:
        raise ValueError("No test files found after filtering.")

    # inspect one training file
    first_patch_path = resolve_path(TRAIN_AUG_DIR, train_aug_df.iloc[0][FILE_COL])
    sample_npz = np.load(first_patch_path, allow_pickle=True)
    sample_key = "X" if "X" in sample_npz.files else sample_npz.files[0]
    sample_array = sample_npz[sample_key]

    print("\nExample patch file:", first_patch_path)
    print("NPZ keys:", sample_npz.files)
    print("Stored array shape:", sample_array.shape)
    print("Stored dtype:", sample_array.dtype)

  # -------------------------
# Grouped train/validation split by original sample_id
# -------------------------

# Grouped train/validation split by original sample_id
#
# Important:
# Multiple augmented patches may originate from the same
# histology sample.
#
# Random splitting could place augmented versions of the
# same sample into both training and validation sets,
# causing validation leakage.
#
# Grouping by sample_id ensures all patches from the same
# original sample remain within a single split.
#
# This creates a more realistic evaluation of model
# generalization.

    if "sample_id" not in train_aug_df.columns:
        raise ValueError("Expected 'sample_id' column in train manifest for grouped splitting.")

    # Build one row per original sample_id for splitting
    group_df = (
        train_aug_df[["sample_id", LABEL_COL]]
        .drop_duplicates(subset=["sample_id"])
        .reset_index(drop=True)
    )

    group_df["label_for_split"] = group_df[LABEL_COL].apply(parse_label)

    train_groups, val_groups = train_test_split(
        group_df["sample_id"],
        test_size=0.15,
        random_state=SEED,
        stratify=group_df["label_for_split"]
    )

    train_df_split = train_aug_df[train_aug_df["sample_id"].isin(train_groups)].reset_index(drop=True)
    val_df_split = train_aug_df[train_aug_df["sample_id"].isin(val_groups)].reset_index(drop=True)

    print("\nGrouped split by sample_id:")
    print("Unique train sample_ids:", train_df_split["sample_id"].nunique())
    print("Unique val sample_ids  :", val_df_split["sample_id"].nunique())
    print("Train rows:", len(train_df_split))
    print("Val rows  :", len(val_df_split))
    print("Overlap in sample_id sets:", set(train_df_split["sample_id"]) & set(val_df_split["sample_id"]))

    print("\nLabel counts by grouped split:")
    print("Train label counts:")
    print(train_df_split[LABEL_COL].apply(parse_label).value_counts())

    print("\nVal label counts:")
    print(val_df_split[LABEL_COL].apply(parse_label).value_counts())

    print("\nSplit sizes:")
    print("Train split:", len(train_df_split))
    print("Val split  :", len(val_df_split))
    print("Test set   :", len(test_df))

    # -------------------------
    # Datasets / loaders
    # -------------------------
    train_dataset = HSINPZPatchDataset(train_df_split, TRAIN_AUG_DIR, FILE_COL, LABEL_COL)
    val_dataset = HSINPZPatchDataset(val_df_split, TRAIN_AUG_DIR, FILE_COL, LABEL_COL)
    test_dataset = HSINPZPatchDataset(test_df, TEST_DIR, FILE_COL, LABEL_COL)

    print("\nExpanded patch counts:")
    print("Train patches:", len(train_dataset))
    print("Val patches  :", len(val_dataset))
    print("Test patches :", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    xb, yb = next(iter(train_loader))
    print("\nBatch x shape:", xb.shape)   # expected: (B, 826, 32, 32)
    print("Batch y shape:", yb.shape)
    print("Unique labels in batch:", torch.unique(yb))

    # -------------------------
    # Model / optimizer
    # -------------------------
    model = FullBandsCNN(in_channels=xb.shape[1], num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    history = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": []}
    best_val_f1 = -1.0
    patience_counter = 0

    # -------------------------
    # Training loop
    # -------------------------
    print("\nStarting training...\n")

    total_start = time.time()

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            DEVICE,
            optimizer=optimizer
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        epoch_time = time.time() - epoch_start
        print(f"Epoch runtime: {epoch_time:.2f} seconds")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0

            torch.save(model.state_dict(), BEST_MODEL_PATH)

            print(f"  -> Saved new best model to {BEST_MODEL_PATH}")

        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience {patience_counter}/{PATIENCE}")
            
        # Early stopping prevents unnecessary training once
        # validation performance stops improving.
        #
        # This helps reduce overfitting and shortens training time.
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after epoch {epoch}.")
            break

        print("\nBest validation F1:", best_val_f1)
        print("Best model saved to:", BEST_MODEL_PATH)

        total_time = time.time() - total_start
        print(f"\nTotal training time: {total_time:.2f} seconds")
    
    # -------------------------
    # Save training curves
    # -------------------------
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_f1"], label="Train F1")
    plt.plot(history["val_f1"], label="Val F1")
    plt.title("F1 Curve")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()

    plt.tight_layout()
    curves_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=200)
    plt.close()

    print("Saved training curves to:", curves_path)

    # -------------------------
    # Final evaluation on completely held-out test data
    #
    # The test dataset is never used during training or
    # validation and provides the final estimate of
    # real-world model performance.
    # -------------------------
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    test_metrics = run_epoch(model, test_loader, criterion, DEVICE)


    # Evaluation metrics:
    # - Accuracy: overall correctness
    # - Precision: confidence in tumor predictions
    # - Recall: ability to detect tumors
    # - F1: balance between precision and recall
    # - ROC-AUC: class separation performance across thresholds
    print("\nTest Metrics:")
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")
    print(f"F1 Score : {test_metrics['f1']:.4f}")
    print(f"ROC-AUC  : {test_metrics['roc_auc']:.4f}")

    # -------------------------
    # Save confusion matrix
    # -------------------------
    cm = confusion_matrix(test_metrics["targets"], test_metrics["preds"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["NonTumor", "Tumor"])
    disp.plot(values_format="d")
    plt.title("All Bands CNN — Test Confusion Matrix")

    cm_path = os.path.join(RESULTS_DIR, "test_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    print("Saved confusion matrix to:", cm_path)


    # Save experiment configuration and performance metrics
    # for later comparison between CNN architectures and
    # band-selection methods.
    results = {
        "model": "AllBandsCNN",
        "input_type": "patches",
        "bands": "all_826",
        "patch_size": "32x32",
        "train_dataset": "train_augmented",
        "test_dataset": "test",
        "epochs_requested": EPOCHS,
        "batch_size": BATCH_SIZE,
        "accuracy": float(test_metrics["accuracy"]),
        "precision": float(test_metrics["precision"]),
        "recall": float(test_metrics["recall"]),
        "f1": float(test_metrics["f1"]),
        "roc_auc": float(test_metrics["roc_auc"]),
    }

    results_csv = os.path.join(RESULTS_DIR, "all_bands_cnn_results.csv")
    results_json = os.path.join(RESULTS_DIR, "all_bands_cnn_results.json")

    pd.DataFrame([results]).to_csv(results_csv, index=False)
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:")
    print("-", results_csv)
    print("-", results_json)
    print("-", BEST_MODEL_PATH)


if __name__ == "__main__":
    main()