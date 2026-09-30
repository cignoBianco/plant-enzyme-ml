import csv
from collections import Counter, defaultdict
from pathlib import Path


OUT_CSV = Path("data/processed/folds.csv")

ACTIVITIES = [
    "alpha-amylase",
    "isoamylase",
    "pullulanase/limit dextrinase",
]


def main() -> None:
    with OUT_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"Всего строк: {len(rows)}")

    # ---------------------------------------------------------
    # 1. Проверяем распределение активностей по фолдам
    # ---------------------------------------------------------

    folds = defaultdict(Counter)

    for row in rows:
        fold = row["fold"]
        activity = row["activity"]
        folds[fold][activity] += 1

    print("\nРаспределение по фолдам:")

    for fold in sorted(folds, key=int):
        print(f"\nfold {fold}:")

        for activity in ACTIVITIES:
            print(f"  {activity}: {folds[fold][activity]}")

    # ---------------------------------------------------------
    # 2. Проверяем, что все 31 белок встречаются ровно один раз
    # ---------------------------------------------------------

    protein_counts = Counter(row["protein_id"] for row in rows)

    duplicated_proteins = {
        protein_id: count
        for protein_id, count in protein_counts.items()
        if count != 1
    }

    print("\nПроверка белков:")

    print(f"  Всего строк: {len(rows)}")
    print(f"  Уникальных protein_id: {len(protein_counts)}")

    if len(rows) == 31 and len(protein_counts) == 31 and not duplicated_proteins:
        print("  ✓ Все 31 белок встречаются ровно один раз.")
    else:
        print("  ✗ ПРОБЛЕМА: есть повторения или неправильное количество белков.")
        print(f"  Повторы: {duplicated_proteins}")

    # ---------------------------------------------------------
    # 3. Проверяем, что cluster_id находится только в одном fold
    # ---------------------------------------------------------

    cluster_folds = defaultdict(set)

    for row in rows:
        cluster_folds[row["cluster_id"]].add(row["fold"])

    clusters_in_multiple_folds = {
        cluster_id: sorted(folds_for_cluster, key=int)
        for cluster_id, folds_for_cluster in cluster_folds.items()
        if len(folds_for_cluster) > 1
    }

    print("\nПроверка кластеров:")

    print(f"  Уникальных кластеров: {len(cluster_folds)}")

    if not clusters_in_multiple_folds:
        print("  ✓ Ни один cluster_id не встречается более чем в одном fold.")
    else:
        print("  ✗ ПРОБЛЕМА: обнаружены кластеры в нескольких фолдах:")

        for cluster_id, cluster_fold_list in clusters_in_multiple_folds.items():
            print(
                f"    {cluster_id}: folds {cluster_fold_list}"
            )

    # ---------------------------------------------------------
    # 4. Дополнительная проверка: один cluster_id может
    #    содержать несколько белков, но все они должны быть
    #    в одном fold
    # ---------------------------------------------------------

    print("\nКластеры с несколькими белками:")

    cluster_members = defaultdict(list)

    for row in rows:
        cluster_members[row["cluster_id"]].append(row["protein_id"])

    multi_clusters = {
        cluster_id: members
        for cluster_id, members in cluster_members.items()
        if len(members) > 1
    }

    if multi_clusters:
        for cluster_id, members in multi_clusters.items():
            fold = next(
                row["fold"]
                for row in rows
                if row["cluster_id"] == cluster_id
            )

            print(
                f"  {cluster_id}: {members} -> fold {fold}"
            )
    else:
        print("  Все кластеры одиночные.")

    # ---------------------------------------------------------
    # 5. Итоговая проверка
    # ---------------------------------------------------------

    proteins_ok = (
        len(rows) == 31
        and len(protein_counts) == 31
        and not duplicated_proteins
    )

    clusters_ok = not clusters_in_multiple_folds

    print("\nИТОГ:")

    if proteins_ok:
        print("✓ Условие 1 выполнено: 31 белок, каждый ровно один раз.")
    else:
        print("✗ Условие 1 НЕ выполнено.")

    if clusters_ok:
        print("✓ Условие 2 выполнено: каждый cluster_id только в одном fold.")
    else:
        print("✗ Условие 2 НЕ выполнено.")

    if proteins_ok and clusters_ok:
        print("\n✓ Обе проверки пройдены.")


if __name__ == "__main__":
    main()