from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "esm2_t12_35m_embeddings.npy"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "esm2_t12_35m_metadata.csv"
)

RANDOM_STATE = 42

CLASSES = [
    "alpha-amylase",
    "isoamylase",
    "pullulanase/limit dextrinase",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("ESM-2 + LOGISTIC REGRESSION")
print("=" * 60)

print("\nLoading embeddings...")
X = np.load(EMBEDDINGS_PATH)

print(f"Embeddings shape: {X.shape}")

print("\nLoading metadata...")
metadata = pd.read_csv(METADATA_PATH)

print(f"Metadata shape: {metadata.shape}")


# ============================================================
# BASIC VALIDATION
# ============================================================

assert len(X) == len(metadata), (
    "Number of embeddings does not match metadata rows"
)

required_columns = {
    "protein_id",
    "activity",
    "cluster_id",
    "fold",
}

missing_columns = required_columns - set(metadata.columns)

assert not missing_columns, (
    f"Missing metadata columns: {missing_columns}"
)

assert not np.isnan(X).any(), (
    "Embeddings contain NaN values"
)

assert not np.isinf(X).any(), (
    "Embeddings contain infinite values"
)


print("\n✓ Number of proteins:", len(X))
print("✓ Embedding dimension:", X.shape[1])
print("✓ No NaN values")
print("✓ No infinite values")


# ============================================================
# LABELS
# ============================================================

y = metadata["activity"].values

print("\nClass distribution:")

for cls in CLASSES:
    count = np.sum(y == cls)
    print(f"  {cls:30s} {count}")


# ============================================================
# CHECK FOLDS
# ============================================================

folds = metadata["fold"].values

unique_folds = sorted(np.unique(folds))

print("\nFolds:", unique_folds)

assert unique_folds == [0, 1, 2, 3, 4], (
    "Expected folds 0-4"
)


# ============================================================
# MODEL
# ============================================================

model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
    ]
)


# ============================================================
# CROSS-VALIDATION
# ============================================================

results = []

all_true = []
all_pred = []


print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)


for fold in unique_folds:

    train_mask = folds != fold
    valid_mask = folds == fold

    X_train = X[train_mask]
    X_valid = X[valid_mask]

    y_train = y[train_mask]
    y_valid = y[valid_mask]

    train_ids = metadata.loc[
        train_mask, "protein_id"
    ].tolist()

    valid_ids = metadata.loc[
        valid_mask, "protein_id"
    ].tolist()

    print(f"\n{'-' * 60}")
    print(f"FOLD {fold}")
    print(f"{'-' * 60}")

    print(f"Train proteins: {len(X_train)}")
    print(f"Valid proteins: {len(X_valid)}")

    print("\nValidation proteins:")

    for protein_id, label in zip(valid_ids, y_valid):
        print(f"  {protein_id:20s} {label}")

    print("\nTraining class distribution:")

    for cls in CLASSES:
        count = np.sum(y_train == cls)
        print(f"  {cls:30s} {count}")

    print("\nValidation class distribution:")

    for cls in CLASSES:
        count = np.sum(y_valid == cls)
        print(f"  {cls:30s} {count}")

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    y_pred = model.predict(X_valid)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_valid,
        y_pred,
    )

    macro_f1 = f1_score(
        y_valid,
        y_pred,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_valid,
        y_pred,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_valid,
        y_pred,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    print("\nMetrics:")
    print(f"  Accuracy:          {accuracy:.3f}")
    print(f"  Macro precision:   {macro_precision:.3f}")
    print(f"  Macro recall:      {macro_recall:.3f}")
    print(f"  Macro F1:          {macro_f1:.3f}")

    print("\nPredictions:")

    for protein_id, true_label, pred_label in zip(
        valid_ids,
        y_valid,
        y_pred,
    ):
        mark = "✓" if true_label == pred_label else "✗"

        print(
            f"  {mark} "
            f"{protein_id:20s} "
            f"true={true_label:30s} "
            f"pred={pred_label}"
        )

    results.append(
        {
            "fold": fold,
            "n_train": len(X_train),
            "n_valid": len(X_valid),
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }
    )

    all_true.extend(y_valid)
    all_pred.extend(y_pred)


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("CROSS-VALIDATION SUMMARY")
print("=" * 60)

print("\nPer-fold results:")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\nMean ± standard deviation:")

for metric in [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
]:

    mean = results_df[metric].mean()
    std = results_df[metric].std(ddof=1)

    print(
        f"  {metric:20s} "
        f"{mean:.3f} ± {std:.3f}"
    )


# ============================================================
# OUT-OF-FOLD CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("OUT-OF-FOLD CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=CLASSES,
)

cm_df = pd.DataFrame(
    cm,
    index=[f"true: {c}" for c in CLASSES],
    columns=[f"pred: {c}" for c in CLASSES],
)

print()
print(cm_df)


# ============================================================
# OUT-OF-FOLD CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("OUT-OF-FOLD CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_true,
        all_pred,
        labels=CLASSES,
        zero_division=0,
        digits=3,
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logistic_esm_cv_results.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)

print("\n✓ Saved fold results:")
print(f"  {output_path}")

print("\nDONE")