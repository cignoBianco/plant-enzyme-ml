"""
из полного gh13_dataset.csv (все 56 записей,
включая транспозиции/дубликаты/фрагменты) собираем финальный ML-датасет -
только записи, прошедшие ручную курацию (include == 'yes').

Вход : data/processed/gh13_dataset.csv
Выход: data/processed/ml_dataset.csv  
       data/sequences/ml_dataset.fasta
"""

import csv
from pathlib import Path

IN_CSV = Path("data/processed/gh13_dataset.csv")
OUT_CSV = Path("data/processed/ml_dataset.csv")
OUT_FASTA = Path("data/sequences/ml_dataset.fasta")

KEEP_FIELDS = [
    "protein_id",
    "organism",
    "gh_family",
    "subfamily",
    "ec_number",
    "activity",
    "length",
    "sequence",
]


def main() -> None:
    with IN_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    kept = [r for r in rows if r["include"].strip().lower() == "yes"]

    # Финальная защита: даже если include=yes, отбрасываем мультиактивные
    # записи (у них ';' в activity) — они не подходят для single-label классификации.
    kept = [r for r in kept if ";" not in r["activity"]]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FASTA.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KEEP_FIELDS)
        w.writeheader()
        for r in kept:
            w.writerow({k: r[k] for k in KEEP_FIELDS})

    with OUT_FASTA.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(f">{r['protein_id']}|{r['activity']}|{r['organism'].replace(' ', '_')}\n")
            seq = r["sequence"]
            for i in range(0, len(seq), 60):
                f.write(seq[i : i + 60] + "\n")

    by_activity: dict[str, int] = {}
    for r in kept:
        by_activity[r["activity"]] = by_activity.get(r["activity"], 0) + 1

    print(f"Всего записей в ML-датасете: {len(kept)}")
    for act, n in sorted(by_activity.items()):
        print(f"  {act}: {n}")


if __name__ == "__main__":
    main()
