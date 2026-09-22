

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_EMBEDDINGS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "esm2_t12_35m_residue_only_embeddings.npy"
)

TRAIN_METADATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "esm2_t12_35m_residue_only_metadata.csv"
)

EXTERNAL_EMBEDDINGS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "esm2_t12_35m_external_residue_only_embeddings.npy"
)

EXTERNAL_METADATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "esm2_t12_35m_external_residue_only_metadata.csv"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "external_predictions.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("EXTERNAL TEST: ESM-2 + LINEAR CLASSIFIERS")
print("=" * 70)

X_train = np.load(TRAIN_EMBEDDINGS)
train_metadata = pd.read_csv(TRAIN_METADATA)

X_external = np.load(EXTERNAL_EMBEDDINGS)
external_metadata = pd.read_csv(EXTERNAL_METADATA)

print("\nTrain:")
print(f"  embeddings: {X_train.shape}")
print(f"  proteins:   {len(train_metadata)}")

print("\nExternal:")
print(f"  embeddings: {X_external.shape}")
print(f"  proteins:   {len(external_metadata)}")


# ============================================================
# VALIDATE
# ============================================================

assert X_train.shape == (31, 480)
assert X_external.shape == (6, 480)

assert len(train_metadata) == 31
assert len(external_metadata) == 6

assert not np.isnan(X_train).any()
assert not np.isnan(X_external).any()

assert not np.isinf(X_train).any()
assert not np.isinf(X_external).any()

assert "activity" in train_metadata.columns

print("\n✓ Train embeddings: 31 × 480")
print("✓ External embeddings: 6 × 480")
print("✓ No NaN values")
print("✓ No infinite values")


# ============================================================
# LABELS
# ============================================================

y_train = train_metadata["activity"].to_numpy()

print("\nTraining class distribution:")

for label, count in (
    train_metadata["activity"]
    .value_counts()
    .sort_index()
    .items()
):
    print(f"  {label}: {count}")


# ============================================================
# MODELS
# ============================================================

logistic = Pipeline(
    [
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)


svm = Pipeline(
    [
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LinearSVC(
                class_weight="balanced",
                random_state=42,
                max_iter=10000,
            ),
        ),
    ]
)


# ============================================================
# TRAIN + PREDICT
# ============================================================

print("\n" + "=" * 70)
print("TRAINING ON ALL 31 TRAINING PROTEINS")
print("=" * 70)

print("\nTraining Logistic Regression...")
logistic.fit(X_train, y_train)
logistic_pred = logistic.predict(X_external)

print("✓ Logistic Regression trained")

print("\nTraining Linear SVM...")
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_external)

print("✓ Linear SVM trained")


# ============================================================
# DECISION SCORES
# ============================================================

logistic_model = logistic.named_steps["classifier"]
svm_model = svm.named_steps["classifier"]

logistic_scores = logistic.decision_function(
    X_external
)

svm_scores = svm.decision_function(
    X_external
)


# ============================================================
# RESULT TABLE
# ============================================================

results = external_metadata[
    [
        "protein_id",
        "header",
        "length",
    ]
].copy()

# ------------------------------------------------------------
# Add known external ground-truth labels.
# These were defined BEFORE model prediction.
# ------------------------------------------------------------

TRUE_LABELS = {
    "Q7X9T1": "alpha-amylase",
    "Q94A41": "alpha-amylase",
    "A4PIS9": "isoamylase",
    "Q84UE6": "isoamylase",
    "XP_004975057.1": "pullulanase/limit dextrinase",
    "XP_018715184.2": "pullulanase/limit dextrinase",
}

results["true_activity"] = results[
    "protein_id"
].map(TRUE_LABELS)

assert results["true_activity"].notna().all()

results["logistic_prediction"] = logistic_pred
results["svm_prediction"] = svm_pred

results["models_agree"] = (
    results["logistic_prediction"]
    == results["svm_prediction"]
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT,
    index=False,
)

print("\n✓ Predictions saved:")
print(f"  {OUTPUT}")


# ============================================================
# PRINT PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("EXTERNAL PREDICTIONS")
print("=" * 70)

print(
    results[
        [
            "protein_id",
            "true_activity",
            "logistic_prediction",
            "svm_prediction",
            "models_agree",
        ]
    ].to_string(index=False)
)


# ============================================================
# METRICS
# ============================================================

labels = [
    "alpha-amylase",
    "isoamylase",
    "pullulanase/limit dextrinase",
]


def print_metrics(
    name,
    predictions,
):

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    y_true = results[
        "true_activity"
    ]

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    print(f"Accuracy:        {accuracy:.3f}")
    print(f"Macro precision:  {precision:.3f}")
    print(f"Macro recall:     {recall:.3f}")
    print(f"Macro F1:         {f1:.3f}")

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=labels,
    )

    print("\nConfusion matrix:")
    print(
        pd.DataFrame(
            cm,
            index=[
                f"true:{x}"
                for x in labels
            ],
            columns=[
                f"pred:{x}"
                for x in labels
            ],
        )
    )


print_metrics(
    "LOGISTIC REGRESSION",
    logistic_pred,
)

print_metrics(
    "LINEAR SVM",
    svm_pred,
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("EXTERNAL TEST COMPLETE")
print("=" * 70)

print(
    "\nThe six external proteins were kept fixed."
)
print(
    "Predictions were generated only after "
    "the external set was frozen."
)
