from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "esm2_pca.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("ESM-2 EMBEDDINGS — PCA VISUALIZATION")
print("=" * 60)

X = np.load(EMBEDDINGS_PATH)
metadata = pd.read_csv(METADATA_PATH)

print(f"\nEmbeddings shape: {X.shape}")
print(f"Metadata shape:   {metadata.shape}")


# ============================================================
# STANDARDIZATION
# ============================================================

print("\nStandardizing embeddings...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("✓ Standardization complete")


# ============================================================
# PCA
# ============================================================

print("\nRunning PCA...")

pca = PCA(
    n_components=2,
    random_state=42,
)

X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_

print(
    f"PC1 explained variance: "
    f"{explained_variance[0] * 100:.2f}%"
)

print(
    f"PC2 explained variance: "
    f"{explained_variance[1] * 100:.2f}%"
)

print(
    f"Total explained variance: "
    f"{explained_variance.sum() * 100:.2f}%"
)


# ============================================================
# SAVE PCA COORDINATES
# ============================================================

pca_df = pd.DataFrame(
    {
        "protein_id": metadata["protein_id"],
        "activity": metadata["activity"],
        "cluster_id": metadata["cluster_id"],
        "fold": metadata["fold"],
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
    }
)

pca_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(f"\n✓ Saved PCA coordinates:")
print(f"  {OUTPUT_PATH}")


# ============================================================
# PRINT COORDINATES
# ============================================================

print("\nPCA coordinates:")

print(
    pca_df[
        [
            "protein_id",
            "activity",
            "PC1",
            "PC2",
        ]
    ].to_string(index=False)
)


# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 8))

for activity in sorted(metadata["activity"].unique()):

    mask = metadata["activity"] == activity

    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=activity,
        s=80,
    )

    # Add protein IDs
    for i in np.where(mask)[0]:

        plt.annotate(
            metadata.iloc[i]["protein_id"],
            (
                X_pca[i, 0],
                X_pca[i, 1],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )


plt.xlabel(
    f"PC1 ({explained_variance[0] * 100:.1f}% variance)"
)

plt.ylabel(
    f"PC2 ({explained_variance[1] * 100:.1f}% variance)"
)

plt.title(
    "PCA of ESM-2 Protein Embeddings"
)

plt.legend()

plt.grid(
    alpha=0.2,
)

plt.tight_layout()

plt.show()


# ============================================================
# DONE
# ============================================================

print("\nDONE")