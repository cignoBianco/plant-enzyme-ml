import csv
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel


DATASET_CSV = Path("data/processed/ml_dataset.csv")

OUTPUT_EMBEDDINGS = Path(
    "data/processed/esm2_t12_35m_embeddings.npy"
)

OUTPUT_METADATA = Path(
    "data/processed/esm2_t12_35m_metadata.csv"
)

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"

BATCH_SIZE = 4


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean pooling по аминокислотам.

    Специальные/padding tokens не учитываются.
    """

    mask = attention_mask.unsqueeze(-1).float()

    masked_embeddings = hidden_states * mask

    summed = masked_embeddings.sum(dim=1)

    counts = mask.sum(dim=1).clamp(min=1e-9)

    return summed / counts


def main():
    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

    with DATASET_CSV.open(
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    print("=" * 80)
    print("ESM-2 EMBEDDING GENERATION")
    print("=" * 80)

    print(f"\nDataset: {DATASET_CSV}")
    print(f"Proteins: {len(rows)}")
    print(f"Model: {MODEL_NAME}")

    # ---------------------------------------------------------
    # 2. Validate sequences
    # ---------------------------------------------------------

    valid_amino_acids = set(
        "ACDEFGHIKLMNPQRSTVWY"
    )

    for row in rows:
        protein_id = row["protein_id"]
        sequence = row["sequence"].strip()

        if not sequence:
            raise ValueError(
                f"Empty sequence: {protein_id}"
            )

        invalid = set(sequence) - valid_amino_acids

        if invalid:
            raise ValueError(
                f"Invalid amino acids in {protein_id}: "
                f"{sorted(invalid)}"
            )

    print("\n✓ All sequences are non-empty")
    print("✓ All sequences contain valid amino-acid symbols")

    # ---------------------------------------------------------
    # 3. Load model
    # ---------------------------------------------------------

    device = get_device()

    print(f"\nDevice: {device}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Loading ESM-2 model...")
    model = EsmModel.from_pretrained(
        MODEL_NAME
    )

    model = model.to(device)
    model.eval()

    print("✓ Model loaded")

    # ---------------------------------------------------------
    # 4. Generate embeddings
    # ---------------------------------------------------------

    all_embeddings = []

    sequences = [
        row["sequence"].strip()
        for row in rows
    ]

    protein_ids = [
        row["protein_id"]
        for row in rows
    ]

    print("\nGenerating embeddings...")

    with torch.no_grad():

        for start in range(
            0,
            len(sequences),
            BATCH_SIZE,
        ):
            batch_sequences = sequences[
                start:start + BATCH_SIZE
            ]

            batch_ids = protein_ids[
                start:start + BATCH_SIZE
            ]

            encoded = tokenizer(
                batch_sequences,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            outputs = model(
                **encoded
            )

            embeddings = mean_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )

            embeddings = embeddings.detach().cpu().numpy()

            all_embeddings.append(
                embeddings
            )

            print(
                f"  {start + 1:>2}-"
                f"{min(start + BATCH_SIZE, len(sequences)):>2}: "
                f"{', '.join(batch_ids)}"
            )

    embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    )

    # ---------------------------------------------------------
    # 5. Validate shape
    # ---------------------------------------------------------

    print("\nEmbedding matrix:")
    print(f"  shape = {embeddings.shape}")

    if embeddings.shape[0] != len(rows):
        raise RuntimeError(
            "Number of embeddings does not match "
            "number of proteins."
        )

    # ---------------------------------------------------------
    # 6. Save embeddings
    # ---------------------------------------------------------

    np.save(
        OUTPUT_EMBEDDINGS,
        embeddings,
    )

    print(
        f"\n✓ Saved embeddings:"
        f"\n  {OUTPUT_EMBEDDINGS}"
    )

    # ---------------------------------------------------------
    # 7. Save metadata
    # ---------------------------------------------------------

    metadata_fields = [
        "protein_id",
        "activity",
        "cluster_id",
        "fold",
        "length",
    ]

    # Load fixed folds
    folds_csv = Path(
        "data/processed/folds_stratified.csv"
    )

    with folds_csv.open(
        encoding="utf-8",
        newline="",
    ) as f:
        fold_rows = list(
            csv.DictReader(f)
        )

    fold_of = {
        row["protein_id"]: row["fold"]
        for row in fold_rows
    }

    cluster_of = {
        row["protein_id"]: row["cluster_id"]
        for row in fold_rows
    }

    with OUTPUT_METADATA.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=metadata_fields,
        )

        writer.writeheader()

        for row in rows:
            protein_id = row["protein_id"]

            writer.writerow(
                {
                    "protein_id": protein_id,
                    "activity": row["activity"],
                    "cluster_id": cluster_of[protein_id],
                    "fold": fold_of[protein_id],
                    "length": row["length"],
                }
            )

    print(
        f"✓ Saved metadata:"
        f"\n  {OUTPUT_METADATA}"
    )

    # ---------------------------------------------------------
    # 8. Final summary
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)

    print(f"Proteins:    {len(rows)}")
    print(f"Dimensions:  {embeddings.shape[1]}")
    print(f"Model:       {MODEL_NAME}")
    print(f"Device:      {device}")

    print("\n✓ ESM-2 embeddings successfully generated")


if __name__ == "__main__":
    main()