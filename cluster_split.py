"""
строим homology-aware группы (кластеры MMseqs2) и
разбиваем 31 запись ml_dataset.csv на K фолдов через GroupKFold —
так, чтобы гомологичные белки из одного кластера никогда не оказывались
одновременно в train и valid.

Перед запуском:
    1. mmseqs easy-cluster data/sequences/ml_dataset.fasta \
           data/processed/mmseqs_out data/processed/mmseqs_tmp \
           --min-seq-id 0.7 -c 0.8 --cov-mode 1

       Это создаст data/processed/mmseqs_out_cluster.tsv (2 колонки:
       representative_id, member_id — по одной строке на белок).

    2. python cluster_split.py

Вход : data/processed/ml_dataset.csv
       data/processed/mmseqs_out_cluster.tsv
Выход: data/processed/folds.csv (protein_id, activity, cluster_id, fold)
       + печатает сводку по фолдам для проверки на глаз
"""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import GroupKFold

_ID_SPLIT = re.compile(r"[|\s]")


def _bare_id(token: str) -> str:
    """FASTA-заголовки вида 'P00693|alpha-amylase|Hordeum_vulgare' MMseqs2
    сохраняет в cluster.tsv целиком (обрезает только по пробелу/табу, не по
    '|'). Достаём отсюда только сам protein_id — то, что до первого '|' или
    пробела."""
    return _ID_SPLIT.split(token, maxsplit=1)[0]

DATASET_CSV = Path("data/processed/ml_dataset.csv")
CLUSTER_TSV = Path("data/processed/mmseqs_out_cluster.tsv")
OUT_CSV = Path("data/processed/folds.csv")

N_FOLDS = 5  # при 31 записи и минимальном классе 6 — не больше 5 фолдов


def load_clusters(path: Path) -> dict[str, str]:
    """protein_id -> id кластера (id представителя кластера)."""
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            rep, member = line.rstrip("\n").split("\t")
            mapping[_bare_id(member)] = _bare_id(rep)
    return mapping


def main() -> None:
    with DATASET_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cluster_of = load_clusters(CLUSTER_TSV)

    missing = [r["protein_id"] for r in rows if r["protein_id"] not in cluster_of]
    if missing:
        raise SystemExit(
            f"{len(missing)} protein_id из ml_dataset.csv не нашлись в "
            f"mmseqs_out_cluster.tsv (проверь, что кластеризовали именно "
            f"ml_dataset.fasta): {missing}"
        )

    for r in rows:
        r["cluster_id"] = cluster_of[r["protein_id"]]

    n_clusters = len(set(r["cluster_id"] for r in rows))
    print(f"Белков: {len(rows)}, кластеров при --min-seq-id 0.7: {n_clusters}")

    # Сводка по кластерам крупнее 1 белка — это и есть места, где
    # клёсеризация реально что-то объединила (стоит посмотреть глазами).
    cluster_members = defaultdict(list)
    for r in rows:
        cluster_members[r["cluster_id"]].append(r["protein_id"])
    multi = {k: v for k, v in cluster_members.items() if len(v) > 1}
    if multi:
        print("\nКластеры из нескольких белков:")
        for rep, members in multi.items():
            acts = {m: next(r["activity"] for r in rows if r["protein_id"] == m) for m in members}
            print(f"  {rep}: {members} -> активности: {acts}")
    else:
        print("\nВсе кластеры одиночные — после ручной чистки явных дублей не осталось.")

    n_folds = min(N_FOLDS, n_clusters)
    if n_folds < N_FOLDS:
        print(
            f"\nВНИМАНИЕ: кластеров меньше, чем запрошено фолдов "
            f"({n_clusters} < {N_FOLDS}) — использую {n_folds} фолдов."
        )

    groups = [r["cluster_id"] for r in rows]
    y = [r["activity"] for r in rows]
    gkf = GroupKFold(n_splits=n_folds)

    fold_of: dict[str, int] = {}
    print(f"\nРазбиение на {n_folds} фолдов (GroupKFold):")
    for fold_idx, (_, valid_idx) in enumerate(gkf.split(rows, y, groups)):
        valid_ids = [rows[i]["protein_id"] for i in valid_idx]
        valid_activities = Counter(rows[i]["activity"] for i in valid_idx)
        for i in valid_idx:
            fold_of[rows[i]["protein_id"]] = fold_idx
        print(f"  fold {fold_idx}: {len(valid_idx)} белков, состав: {dict(valid_activities)}")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["protein_id", "activity", "cluster_id", "fold"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "protein_id": r["protein_id"],
                    "activity": r["activity"],
                    "cluster_id": r["cluster_id"],
                    "fold": fold_of[r["protein_id"]],
                }
            )

    print(f"\nЗаписано: {OUT_CSV}")


if __name__ == "__main__":
    main()