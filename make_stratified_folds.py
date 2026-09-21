import csv

from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold


DATASET_CSV = Path("data/processed/ml_dataset.csv")
FOLDS_CSV = Path("data/processed/folds.csv")
OUTPUT_CSV = Path("data/processed/folds_stratified.csv")

N_SPLITS = 5
RANDOM_STATE = 788

ACTIVITIES = [
    "alpha-amylase",
    "isoamylase",
    "pullulanase/limit dextrinase",
]


def main():
    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    with DATASET_CSV.open(
        encoding="utf-8",
        newline="",
    ) as f:
        dataset_rows = list(csv.DictReader(f))

    # ---------------------------------------------------------
    # Load homology clusters
    # ---------------------------------------------------------

    with FOLDS_CSV.open(
        encoding="utf-8",
        newline="",
    ) as f:
        fold_rows = list(csv.DictReader(f))

    cluster_of = {
        row["protein_id"]: row["cluster_id"]
        for row in fold_rows
    }

    # ---------------------------------------------------------
    # Prepare X / y / groups
    # ---------------------------------------------------------

    protein_ids = [
        row["protein_id"]
        for row in dataset_rows
    ]

    y = [
        row["activity"]
        for row in dataset_rows
    ]

    groups = [
        cluster_of[protein_id]
        for protein_id in protein_ids
    ]

    # ---------------------------------------------------------
    # Stratified Group K-Fold
    # ---------------------------------------------------------

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_of = {}

    print("=" * 80)
    print("FINAL STRATIFIED GROUP K-FOLD")
    print("=" * 80)

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(
            protein_ids,
            y,
            groups,
        )
    ):
        train_clusters = {
            groups[i]
            for i in train_idx
        }

        valid_clusters = {
            groups[i]
            for i in valid_idx
        }

        overlap = train_clusters & valid_clusters

        if overlap:
            raise RuntimeError(
                f"Homology leakage in fold {fold}: "
                f"{sorted(overlap)}"
            )

        print(f"\nFOLD {fold}")
        print("-" * 80)

        valid_class_proteins = defaultdict(list)
        valid_class_clusters = defaultdict(set)

        train_class_proteins = defaultdict(list)

        for i in train_idx:
            train_class_proteins[y[i]].append(
                protein_ids[i]
            )

        for i in valid_idx:
            activity = y[i]
            protein_id = protein_ids[i]
            cluster_id = groups[i]

            valid_class_proteins[activity].append(
                protein_id
            )

            valid_class_clusters[activity].add(
                cluster_id
            )

            fold_of[protein_id] = fold

        print("Validation:")

        for activity in ACTIVITIES:
            print(
                f"  {activity:<30}"
                f"proteins = "
                f"{len(valid_class_proteins[activity]):<3}"
                f"clusters = "
                f"{len(valid_class_clusters[activity])}"
            )

        print("Training:")

        for activity in ACTIVITIES:
            print(
                f"  {activity:<30}"
                f"proteins = "
                f"{len(train_class_proteins[activity])}"
            )

        print(
            f"\nTrain clusters: {len(train_clusters)}"
        )

        print(
            f"Valid clusters: {len(valid_clusters)}"
        )

        print(
            f"Cluster overlap: {len(overlap)}"
        )

        if not overlap:
            print("✓ No homology leakage")

        missing = [
            activity
            for activity in ACTIVITIES
            if not valid_class_proteins[activity]
        ]

        if missing:
            raise RuntimeError(
                f"Missing classes in fold {fold}: "
                f"{missing}"
            )

        print("✓ All classes present")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        fieldnames = [
            "protein_id",
            "activity",
            "cluster_id",
            "fold",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in dataset_rows:
            protein_id = row["protein_id"]

            writer.writerow(
                {
                    "protein_id": protein_id,
                    "activity": row["activity"],
                    "cluster_id": cluster_of[protein_id],
                    "fold": fold_of[protein_id],
                }
            )

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print(f"random_state = {RANDOM_STATE}")
    print(f"n_splits = {N_SPLITS}")
    print(f"saved = {OUTPUT_CSV}")

    print("\n✓ 5-fold StratifiedGroupKFold created")
    print("✓ All folds contain all 3 classes")
    print("✓ No homology cluster leakage")


if __name__ == "__main__":
    main()