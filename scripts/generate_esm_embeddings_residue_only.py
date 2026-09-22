from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_dataset.csv"
)

FOLDS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "folds_stratified.csv"
)

OUTPUT_EMBEDDINGS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "esm2_t12_35m_residue_only_embeddings.npy"
)

OUTPUT_METADATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "esm2_t12_35m_residue_only_metadata.csv"
)

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
BATCH_SIZE = 4


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("ESM-2 RESIDUE-ONLY EMBEDDING GENERATION")
print("=" * 60)

dataset = pd.read_csv(DATASET_PATH)
folds = pd.read_csv(FOLDS_PATH)

print(f"\nProteins: {len(dataset)}")
print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE}")


# ============================================================
# MERGE METADATA
# ============================================================

metadata = dataset[
    [
        "protein_id",
        "activity",
        "length",
        "sequence",
    ]
].copy()

metadata = metadata.merge(
    folds[
        [
            "protein_id",
            "cluster_id",
            "fold",
        ]
    ],
    on="protein_id",
    how="left",
    validate="one_to_one",
)


assert len(metadata) == len(dataset)
assert metadata["fold"].notna().all()
assert metadata["cluster_id"].notna().all()


# ============================================================
# VALIDATE SEQUENCES
# ============================================================

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

for _, row in metadata.iterrows():

    sequence = row["sequence"]

    assert isinstance(sequence, str)
    assert len(sequence) > 0

    invalid = set(sequence) - VALID_AA

    assert not invalid, (
        f"Invalid amino-acid symbols in "
        f"{row['protein_id']}: {invalid}"
    )


print("✓ All sequences are non-empty")
print("✓ All sequences contain valid amino-acid symbols")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Loading ESM-2 model...")

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model = model.to(DEVICE)
model.eval()

print("✓ Model loaded")


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

all_embeddings = []

sequences = metadata["sequence"].tolist()
protein_ids = metadata["protein_id"].tolist()

print("\nGenerating residue-only embeddings...")


for start in range(0, len(sequences), BATCH_SIZE):

    end = min(
        start + BATCH_SIZE,
        len(sequences),
    )

    batch_sequences = sequences[start:end]
    batch_ids = protein_ids[start:end]

    print(
        f" {start + 1}-{end}: "
        + ", ".join(batch_ids)
    )

    encoded = tokenizer(
        batch_sequences,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    hidden = outputs.last_hidden_state

    # --------------------------------------------------------
    # RESIDUE-ONLY MASK
    # --------------------------------------------------------
    #
    # ESM-2 sequences are represented as:
    #
    #   <cls> AA AA AA ... AA <eos>
    #
    # We remove the first and last token and therefore
    # calculate the mean only over amino-acid residues.
    #

    residue_hidden = hidden[:, 1:-1, :]
    residue_mask = attention_mask[:, 1:-1]

    mask = residue_mask.unsqueeze(-1).to(
        residue_hidden.dtype
    )

    summed = (
        residue_hidden * mask
    ).sum(dim=1)

    counts = mask.sum(dim=1)

    pooled = summed / counts

    embeddings = (
        pooled
        .detach()
        .cpu()
        .numpy()
    )

    all_embeddings.append(embeddings)


# ============================================================
# COMBINE
# ============================================================

X = np.concatenate(
    all_embeddings,
    axis=0,
)

print("\nEmbedding matrix:")
print(f" shape = {X.shape}")


# ============================================================
# VALIDATE
# ============================================================

assert X.shape[0] == len(metadata)
assert X.shape[1] == 480

assert not np.isnan(X).any()
assert not np.isinf(X).any()

print("✓ Correct number of embeddings")
print("✓ Dimension = 480")
print("✓ No NaN values")
print("✓ No infinite values")


# ============================================================
# SAVE
# ============================================================

np.save(
    OUTPUT_EMBEDDINGS,
    X,
)

output_metadata = metadata[
    [
        "protein_id",
        "activity",
        "cluster_id",
        "fold",
        "length",
    ]
]

output_metadata.to_csv(
    OUTPUT_METADATA,
    index=False,
)


print("\n✓ Saved embeddings:")
print(f"  {OUTPUT_EMBEDDINGS}")

print("\n✓ Saved metadata:")
print(f"  {OUTPUT_METADATA}")


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print(f"Proteins:   {X.shape[0]}")
print(f"Dimensions: {X.shape[1]}")
print(f"Model:      {MODEL_NAME}")
print(f"Device:     {DEVICE}")

print("\n✓ Residue-only ESM-2 embeddings successfully generated")