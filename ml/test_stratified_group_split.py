import csv
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold


DATASET_CSV = Path("data/processed/ml_dataset.csv")
FOLDS_CSV = Path("data/processed/folds.csv")

ACTIVITIES = [
    "alpha-amylase",
    "isoamylase",
    "pullulanase/limit dextrinase",
]


def main() -> None:
    # ---------------------------------------------------------
    # 1. Загружаем ml_dataset.csv
    # ---------------------------------------------------------

    with DATASET_CSV.open(encoding="utf-8", newline="") as f:
        dataset_rows = list(csv.DictReader(f))

    # ---------------------------------------------------------
    # 2. Загружаем folds.csv
    # ---------------------------------------------------------

    with FOLDS_CSV.open(encoding="utf-8", newline="") as f:
        fold_rows = list(csv.DictReader(f))

    # protein_id -> cluster_id
    cluster_of = {
        row["protein_id"]: row["cluster_id"]
        for row in fold_rows
    }

    # ---------------------------------------------------------
    # 3. Собираем информацию по кластерам
    # ---------------------------------------------------------

    clusters = defaultdict(list)

    for row in dataset_rows:
        protein_id = row["protein_id"]
        activity = row["activity"]
        cluster_id = cluster_of[protein_id]

        clusters[cluster_id].append(
            {
                "protein_id": protein_id,
                "activity": activity,
            }
        )

    # ---------------------------------------------------------
    # 4. Подробная информация по каждому кластеру
    # ---------------------------------------------------------

    print("=" * 80)
    print("АНАЛИЗ КЛАСТЕРОВ")
    print("=" * 80)

    for cluster_id in sorted(clusters):
        members = clusters[cluster_id]

        activity_counts = Counter(
            member["activity"]
            for member in members
        )

        protein_ids = [
            member["protein_id"]
            for member in members
        ]

        activities = [
            member["activity"]
            for member in members
        ]

        print(f"\ncluster_id: {cluster_id}")
        print(f"Количество белков: {len(members)}")
        print(f"protein_id: {protein_ids}")
        print(f"activity: {activities}")

        print("Количество каждого класса:")

        for activity in ACTIVITIES:
            print(
                f"  {activity}: "
                f"{activity_counts[activity]}"
            )

    # ---------------------------------------------------------
    # 5. Отдельно выводим кластеры из нескольких белков
    # ---------------------------------------------------------

    multi_clusters = {
        cluster_id: members
        for cluster_id, members in clusters.items()
        if len(members) > 1
    }

    print("\n")
    print("=" * 80)
    print("КЛАСТЕРЫ ИЗ НЕСКОЛЬКИХ БЕЛКОВ")
    print("=" * 80)

    if multi_clusters:
        for cluster_id, members in sorted(multi_clusters.items()):
            activity_counts = Counter(
                member["activity"]
                for member in members
            )

            print(f"\ncluster_id: {cluster_id}")
            print(f"Количество белков: {len(members)}")

            for member in members:
                print(
                    f"  {member['protein_id']} "
                    f"-> {member['activity']}"
                )

            print(
                "Распределение классов:",
                dict(activity_counts)
            )
    else:
        print("Кластеров из нескольких белков нет.")

    # ---------------------------------------------------------
    # 6. Распределение pullulanase/limit dextrinase
    # ---------------------------------------------------------

    pullulanase_clusters = defaultdict(list)

    for cluster_id, members in clusters.items():
        for member in members:
            if member["activity"] == "pullulanase/limit dextrinase":
                pullulanase_clusters[cluster_id].append(
                    member["protein_id"]
                )

    print("\n")
    print("=" * 80)
    print("PULLULANASE / LIMIT DEXTRINASE ПО КЛАСТЕРАМ")
    print("=" * 80)

    for cluster_id in sorted(pullulanase_clusters):
        proteins = pullulanase_clusters[cluster_id]

        print(
            f"{cluster_id}: "
            f"{len(proteins)} белок(ов) -> {proteins}"
        )

    # ---------------------------------------------------------
    # 7. Количество кластеров по классам
    # ---------------------------------------------------------

    class_cluster_counts = Counter()

    for cluster_id, members in clusters.items():
        cluster_activities = {
            member["activity"]
            for member in members
        }

        for activity in cluster_activities:
            class_cluster_counts[activity] += 1

    # ---------------------------------------------------------
    # 8. Количество кластеров с 1 / 2 / 3 классами
    # ---------------------------------------------------------

    cluster_class_count = Counter()

    for cluster_id, members in clusters.items():
        cluster_activities = {
            member["activity"]
            for member in members
        }

        cluster_class_count[len(cluster_activities)] += 1

    # ---------------------------------------------------------
    # 9. Итоговая сводка
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("ИТОГОВАЯ СВОДКА ПО КЛАСТЕРАМ")
    print("=" * 80)

    print(f"\nКоличество кластеров: {len(clusters)}")

    print("\nКластеры, содержащие каждый класс:")

    for activity in ACTIVITIES:
        print(
            f"  {activity}: "
            f"{class_cluster_counts[activity]}"
        )

    print("\nКоличество кластеров по числу классов:")

    print(
        f"  Только один класс: "
        f"{cluster_class_count[1]}"
    )

    print(
        f"  Два класса: "
        f"{cluster_class_count[2]}"
    )

    print(
        f"  Три класса: "
        f"{cluster_class_count[3]}"
    )

def main_stratified_group() -> None:
    # ---------------------------------------------------------
    # 1. Загружаем dataset и folds
    # ---------------------------------------------------------

    with DATASET_CSV.open(encoding="utf-8", newline="") as f:
        dataset_rows = list(csv.DictReader(f))

    with FOLDS_CSV.open(encoding="utf-8", newline="") as f:
        fold_rows = list(csv.DictReader(f))

    cluster_of = {
        row["protein_id"]: row["cluster_id"]
        for row in fold_rows
    }

    # ---------------------------------------------------------
    # 2. Формируем X, y, groups
    # ---------------------------------------------------------

    X = dataset_rows
    y = [row["activity"] for row in dataset_rows]
    groups = [cluster_of[row["protein_id"]] for row in dataset_rows]

    n_folds = 5

    # ---------------------------------------------------------
    # 3. StratifiedGroupKFold
    # ---------------------------------------------------------

    sgkf = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=42,
    )

    print("\n")
    print("=" * 80)
    print("STRATIFIED GROUP K-FOLD")
    print("=" * 80)

    fold_of = {}

    for fold_idx, (_, valid_idx) in enumerate(
        sgkf.split(X, y, groups)
    ):
        valid_rows = [dataset_rows[i] for i in valid_idx]

        print(f"\nfold {fold_idx}:")

        counts = Counter(
            row["activity"]
            for row in valid_rows
        )

        for activity in ACTIVITIES:
            print(
                f"  {activity}: "
                f"{counts[activity]}"
            )

        for i in valid_idx:
            protein_id = dataset_rows[i]["protein_id"]
            fold_of[protein_id] = fold_idx

    # ---------------------------------------------------------
    # 4. Проверка protein_id
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("ПРОВЕРКА PROTEIN_ID")
    print("=" * 80)

    protein_counts = Counter(
        row["protein_id"]
        for row in dataset_rows
    )

    if (
        len(dataset_rows) == 31
        and len(protein_counts) == 31
        and all(count == 1 for count in protein_counts.values())
    ):
        print("✓ Все 31 protein_id уникальны.")
    else:
        print("✗ ПРОБЛЕМА С PROTEIN_ID")

    # ---------------------------------------------------------
    # 5. Проверка cluster_id
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("ПРОВЕРКА CLUSTER_ID")
    print("=" * 80)

    cluster_folds = defaultdict(set)

    for protein_id, fold in fold_of.items():
        cluster_id = cluster_of[protein_id]
        cluster_folds[cluster_id].add(fold)

    clusters_in_multiple_folds = {
        cluster_id: sorted(folds)
        for cluster_id, folds in cluster_folds.items()
        if len(folds) > 1
    }

    if not clusters_in_multiple_folds:
        print(
            "✓ Ни один cluster_id не встречается "
            "более чем в одном fold."
        )
    else:
        print(
            "✗ ПРОБЛЕМА: cluster_id оказался "
            "в нескольких fold:"
        )

        for cluster_id, folds in clusters_in_multiple_folds.items():
            print(
                f"  {cluster_id}: folds {folds}"
            )

    # ---------------------------------------------------------
    # 6. Проверяем multi-protein clusters
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("КЛАСТЕРЫ И ИХ FOLD")
    print("=" * 80)

    cluster_members = defaultdict(list)

    for row in dataset_rows:
        protein_id = row["protein_id"]
        cluster_id = cluster_of[protein_id]

        cluster_members[cluster_id].append(
            protein_id
        )

    for cluster_id, members in sorted(
        cluster_members.items()
    ):
        if len(members) > 1:
            fold = cluster_folds[cluster_id].pop()

            print(
                f"{cluster_id}: "
                f"{members} -> fold {fold}"
            )

            # возвращаем fold обратно в set,
            # чтобы не менять структуру проверки
            cluster_folds[cluster_id].add(fold)

    # ---------------------------------------------------------
    # 7. Итог
    # ---------------------------------------------------------

    proteins_ok = (
        len(dataset_rows) == 31
        and len(protein_counts) == 31
        and all(
            count == 1
            for count in protein_counts.values()
        )
    )

    clusters_ok = not clusters_in_multiple_folds

    print("\n")
    print("=" * 80)
    print("ИТОГ STRATIFIED GROUP K-FOLD")
    print("=" * 80)

    if proteins_ok:
        print("✓ protein_id: проверка пройдена.")
    else:
        print("✗ protein_id: проверка НЕ пройдена.")

    if clusters_ok:
        print("✓ cluster_id: проверка пройдена.")
    else:
        print("✗ cluster_id: проверка НЕ пройдена.")

    if proteins_ok and clusters_ok:
        print(
            "\n✓ StratifiedGroupKFold удовлетворяет "
            "обязательным ограничениям."
        )

if __name__ == "__main__":
    main()
    main_stratified_group()