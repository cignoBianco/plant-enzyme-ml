import csv
from collections import defaultdict, Counter
from pathlib import Path


DATASET_CSV = Path("data/processed/ml_dataset.csv")
FOLDS_CSV = Path("data/processed/folds.csv")

ACTIVITIES = [
    "alpha-amylase",
    "isoamylase",
    "pullulanase/limit dextrinase",
]


def main() -> None:
    # ---------------------------------------------------------
    # 1. Загружаем dataset
    # ---------------------------------------------------------

    with DATASET_CSV.open(encoding="utf-8", newline="") as f:
        dataset_rows = list(csv.DictReader(f))

    # ---------------------------------------------------------
    # 2. Загружаем информацию о кластерах
    # ---------------------------------------------------------

    with FOLDS_CSV.open(encoding="utf-8", newline="") as f:
        fold_rows = list(csv.DictReader(f))

    cluster_of = {
        row["protein_id"]: row["cluster_id"]
        for row in fold_rows
    }

    # ---------------------------------------------------------
    # 3. Собираем белки по классам и кластеры по классам
    # ---------------------------------------------------------

    class_proteins = defaultdict(list)
    class_clusters = defaultdict(lambda: defaultdict(list))

    for row in dataset_rows:
        protein_id = row["protein_id"]
        activity = row["activity"]
        cluster_id = cluster_of[protein_id]

        class_proteins[activity].append(protein_id)
        class_clusters[activity][cluster_id].append(protein_id)

    # ---------------------------------------------------------
    # 4. Основная статистика
    # ---------------------------------------------------------

    print("=" * 80)
    print("СТАТИСТИКА DATASET ДЛЯ 3-КЛАССОВОЙ КЛАССИФИКАЦИИ")
    print("=" * 80)

    print(
        f"\n{'Класс':<30}"
        f"{'Белков':>10}"
        f"{'Кластеров':>12}"
    )

    print("-" * 54)

    for activity in ACTIVITIES:
        n_proteins = len(class_proteins[activity])
        n_clusters = len(class_clusters[activity])

        print(
            f"{activity:<30}"
            f"{n_proteins:>10}"
            f"{n_clusters:>12}"
        )

    print("-" * 54)

    print(
        f"{'TOTAL':<30}"
        f"{len(dataset_rows):>10}"
        f"{len(set(cluster_of.values())):>12}"
    )

    # ---------------------------------------------------------
    # 5. Размеры кластеров
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("РАЗМЕРЫ КЛАСТЕРОВ ПО КЛАССАМ")
    print("=" * 80)

    for activity in ACTIVITIES:
        print(f"\n{activity}:")

        clusters = class_clusters[activity]

        for cluster_id, proteins in sorted(clusters.items()):
            print(
                f"  {cluster_id} -> {len(proteins)}"
            )

    # ---------------------------------------------------------
    # 6. Минимальный и максимальный размер кластера
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("ДИАПАЗОН РАЗМЕРОВ КЛАСТЕРОВ")
    print("=" * 80)

    min_cluster_counts = []
    max_cluster_counts = []

    for activity in ACTIVITIES:
        sizes = [
            len(proteins)
            for proteins in class_clusters[activity].values()
        ]

        min_size = min(sizes)
        max_size = max(sizes)

        min_cluster_counts.append(
            len(class_clusters[activity])
        )

        print(
            f"{activity:<30}"
            f"min = {min_size}, "
            f"max = {max_size}"
        )

    # ---------------------------------------------------------
    # 7. Дополнительная статистика по классам
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("СТАТИСТИКА НЕЗАВИСИМЫХ КЛАСТЕРОВ")
    print("=" * 80)

    for activity in ACTIVITIES:
        n_proteins = len(class_proteins[activity])
        n_clusters = len(class_clusters[activity])

        print(
            f"{activity:<30}"
            f"белков = {n_proteins}, "
            f"кластеров = {n_clusters}"
        )

    # ---------------------------------------------------------
    # 8. Определяем ограничение для stratified CV
    # ---------------------------------------------------------

    independent_cluster_counts = {
        activity: len(class_clusters[activity])
        for activity in ACTIVITIES
    }

    min_independent_clusters = min(
        independent_cluster_counts.values()
    )

    max_stratified_folds = min_independent_clusters

    print("\n")
    print("=" * 80)
    print("ОГРАНИЧЕНИЕ ДЛЯ STRATIFIED GROUP K-FOLD")
    print("=" * 80)

    print(
        f"\nМинимальное число независимых кластеров "
        f"среди классов: {min_independent_clusters}"
    )

    print(
        f"Максимально возможное число stratified folds: "
        f"{max_stratified_folds}"
    )

    # ---------------------------------------------------------
    # 9. Проверяем 5-fold CV
    # ---------------------------------------------------------

    print("\nПроверка 5-fold CV:")

    if min_independent_clusters >= 5:
        print(
            "✓ 5-fold CV возможно: каждый класс имеет "
            "как минимум 5 независимых кластеров."
        )
    else:
        print(
            "✗ 5-fold CV невозможно корректно выполнить "
            "с независимыми homology-группами: "
            "у одного из классов меньше 5 независимых кластеров."
        )

        limiting_classes = [
            activity
            for activity, count in independent_cluster_counts.items()
            if count == min_independent_clusters
        ]

        print(
            "Ограничивающий класс(ы): "
            f"{', '.join(limiting_classes)}"
        )

    # ---------------------------------------------------------
    # 10. Итог
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("ИТОГ")
    print("=" * 80)

    if max_stratified_folds >= 5:
        print("Dataset допускает 5-fold StratifiedGroupKFold.")
    else:
        print(
            f"Dataset ограничивает число stratified folds до "
            f"{max_stratified_folds}."
        )


if __name__ == "__main__":
    main()