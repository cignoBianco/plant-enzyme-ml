"""
Автоматически убирает из external_candidates утечку — кандидатов,
которые слишком похожи на что-то из ml_dataset (train).

Раньше это решалось "на глаз" по таблице mmseqs (external_vs_training_mmseqs.csv),
и именно поэтому Q7X9T1 (99.1% identity к P17859) проскользнул в тест.
Теперь порог применяется программно и без исключений.

Вход : data/processed/external_vs_training_mmseqs.csv  (результат mmseqs_external_vs_training.py)
       data/sequences/external_candidates.fasta
       data/processed/external_candidates.csv
Выход: data/sequences/external_candidates_clean.fasta
       data/processed/external_candidates_clean.csv
       печатает, кого и почему выкинули
"""

import csv
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HITS_CSV = PROJECT_ROOT / "data/processed/external_vs_training_mmseqs.csv"
CANDIDATES_FASTA = PROJECT_ROOT / "data/sequences/external_candidates.fasta"
CANDIDATES_CSV = PROJECT_ROOT / "data/processed/external_candidates.csv"

OUT_FASTA = PROJECT_ROOT / "data/sequences/external_candidates_clean.fasta"
OUT_CSV = PROJECT_ROOT / "data/processed/external_candidates_clean.csv"

# Порог отсечения. 70% — тот же уровень, на котором мы кластеризовали
# тренировочный набор для CV, поэтому логично применить его и здесь:
# если кандидат гомологичен train на уровне, при котором мы бы объединили
# их в один кластер, он не может считаться независимым.
MAX_IDENTITY = 70.0
MIN_COVERAGE_FOR_CHECK = 0.5  # ниже этого coverage сравнение ненадёжно


def read_fasta(path: Path) -> dict[str, str]:
    seqs, cur_id, cur_seq = {}, None, []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(cur_seq)
                cur_id = line[1:].split("|")[0].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            seqs[cur_id] = "".join(cur_seq)
    return seqs


def main() -> None:
    hits = pd.read_csv(HITS_CSV)
    seqs = read_fasta(CANDIDATES_FASTA)

    with CANDIDATES_CSV.open(encoding="utf-8") as f:
        meta = {r["accession"]: r for r in csv.DictReader(f)}

    rejected, kept = [], []
    for _, row in hits.iterrows():
        qid = row["query"]
        pident = row["pident"]
        qcov = row.get("query_coverage")
        tcov = row.get("target_coverage")
        cov = max([c for c in (qcov, tcov) if pd.notna(c)], default=0.0)

        is_leak = pident >= MAX_IDENTITY and cov >= MIN_COVERAGE_FOR_CHECK
        (rejected if is_leak else kept).append(
            {"protein_id": qid, "pident": pident, "coverage": cov, "target": row["target"]}
        )

    print(f"Кандидатов всего: {len(hits)}")
    print(f"Отклонено (identity >= {MAX_IDENTITY}%, coverage >= {MIN_COVERAGE_FOR_CHECK}): {len(rejected)}")
    for r in rejected:
        print(f"  ✗ {r['protein_id']}: {r['pident']:.1f}% identity, cov={r['coverage']:.2f}, к {r['target']}")
    print(f"Оставлено как честный внешний тест: {len(kept)}")
    for r in kept:
        print(f"  ✓ {r['protein_id']}: {r['pident']:.1f}% identity, cov={r['coverage']:.2f}")

    kept_ids = {r["protein_id"] for r in kept}

    with OUT_FASTA.open("w", encoding="utf-8") as f:
        for pid in kept_ids:
            seq = seqs.get(pid)
            if seq is None:
                print(f"  ВНИМАНИЕ: {pid} не нашёлся в FASTA, пропущен")
                continue
            m = meta.get(pid, {})
            organism = (m.get("organism", "unknown") or "unknown").replace(" ", "_")
            activity = m.get("activity", "unknown")
            f.write(f">{pid}|{activity}|{organism}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i : i + 60] + "\n")

    if meta:
        fieldnames = list(next(iter(meta.values())).keys())
        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for pid in kept_ids:
                if pid in meta:
                    w.writerow(meta[pid])

    print(f"\nЗаписано: {OUT_FASTA}")
    print(f"Записано: {OUT_CSV}")


if __name__ == "__main__":
    main()
