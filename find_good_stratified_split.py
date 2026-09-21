import csv

from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold


DATASET_CSV = Path("data/processed/ml_dataset.csv")
FOLDS_CSV = Path("data/processed/folds.csv")

ACTIVITIES = [
    "alpha-amylase",
    "isoamylase",
    "pullulanase/limit dextrinase",
]

N_SPLITS = 5

MAX_SEED = 10000


def load_dataset():
    with DATASET_CSV.open(
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def load_clusters():
    with FOLDS_CSV.open(
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    return {
        row["protein_id"]: row["cluster_id"]
        for row in rows
    }


def evaluate_split(dataset_rows, cluster_of, seed):
    protein_ids = [
        row["protein_id"]
        for row in dataset_rows
    ]

    y = [
        row["activity"]
        for row in dataset_rows
    ]

    groups = [
        cluster_of[row["protein_id"]]
        for row in dataset_rows
    ]

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    fold_stats = []

    for fold, (_, valid_idx) in enumerate(
        splitter.split(
            protein_ids,
            y,
            groups,
        )
    ):
        class_proteins = defaultdict(int)
        class_clusters = defaultdict(set)

        for i in valid_idx:
            activity = y[i]
            cluster_id = groups[i]

            class_proteins[activity] += 1
            class_clusters[activity].add(cluster_id)

        fold_stats.append(
            {
                activity: (
                    class_proteins[activity],
                    len(class_clusters[activity]),
                )
                for activity in ACTIVITIES
            }
        )

    # ---------------------------------------------------------
    # Проверяем, что каждый класс есть в каждом fold
    # ---------------------------------------------------------

    valid = True

    for fold_stat in fold_stats:
        for activity in ACTIVITIES:
            proteins, clusters = fold_stat[activity]

            if proteins == 0 or clusters == 0:
                valid = False

    if not valid:
        return None

    # ---------------------------------------------------------
    # Считаем дисбаланс относительно идеального распределения
    # ---------------------------------------------------------

    score = 0

    for activity in ACTIVITIES:
        total = sum(
            fold_stat[activity][0]
            for fold_stat in fold_stats
        )

        ideal = total / N_SPLITS

        for fold_stat in fold_stats:
            actual = fold_stat[activity][0]

            score += abs(actual - ideal)

    return score, fold_stats


def main():
    dataset_rows = load_dataset()
    cluster_of = load_clusters()

    print("=" * 80)
    print("ПОИСК КОРРЕКТНОГО STRATIFIED GROUP 5-FOLD SPLIT")
    print("=" * 80)

    best_seed = None
    best_score = None
    best_stats = None

    valid_count = 0

    for seed in range(MAX_SEED):
        result = evaluate_split(
            dataset_rows,
            cluster_of,
            seed,
        )

        if result is None:
            continue

        valid_count += 1

        score, stats = result

        if best_score is None or score < best_score:
            best_score = score
            best_seed = seed
            best_stats = stats

    print()
    print(f"Проверено seeds: {MAX_SEED}")
    print(f"Корректных splits: {valid_count}")

    if best_seed is None:
        print()
        print("❌ Не найдено ни одного корректного split.")
        print()
        print(
            "Следующий шаг: построить собственный "
            "stratified group splitter."
        )
        return

    print()
    print("=" * 80)
    print("ЛУЧШИЙ НАЙДЕННЫЙ SPLIT")
    print("=" * 80)

    print(f"\nrandom_state = {best_seed}")
    print(f"score = {best_score}")

    for fold, fold_stat in enumerate(best_stats):
        print(f"\nFOLD {fold}")
        print("-" * 80)

        for activity in ACTIVITIES:
            proteins, clusters = fold_stat[activity]

            print(
                f"{activity:<30}"
                f"proteins = {proteins:<3}"
                f"clusters = {clusters}"
            )


if __name__ == "__main__":
    main()