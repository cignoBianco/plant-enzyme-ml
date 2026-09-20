"""
PlantEnzyme AI — сборка датасета из списка кандидатов CAZy.

Вход : data/raw/gh13_plant_candidates.csv
Выход: data/processed/gh13_dataset.csv
       data/sequences/gh13.fasta
       data/processed/fetch_log.csv

Запускать локально (нужен интернет):
    python build_dataset.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import requests

RAW = Path("data/raw/gh13_plant_candidates.csv")
OUT_DIR = Path("data/processed")
SEQ_DIR = Path("data/sequences")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEQ_DIR.mkdir(parents=True, exist_ok=True)

UNIPROT = "https://rest.uniprot.org/uniprotkb/{acc}.json"
NCBI_EFETCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=protein&id={acc}&rettype=fasta&retmode=text"
)
FIELDS = (
    "accession,id,reviewed,protein_name,organism_name,length,sequence,"
    "cc_subcellular_location,ft_signal,protein_existence"
)


def fetch_uniprot(acc: str) -> dict | None:
    """Забираем одну запись UniProt. None -> записи нет (или она устарела/слита)."""
    r = requests.get(UNIPROT.format(acc=acc), timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


class InactiveEntry(Exception):
    """UniProt-запись устарела/слита с другой и не содержит sequence."""


def fetch_genbank_fasta(acc: str) -> dict | None:
    """Fallback: тянем последовательность напрямую из NCBI по GenBank accession.

    Используется только когда ВСЕ UniProt ID для записи не сработали
    (DELETED без merge-цели и т.п.). Метаданных здесь меньше — reviewed,
    protein_existence и organism из UniProt взять неоткуда, они не заполняются.
    """
    r = requests.get(NCBI_EFETCH.format(acc=acc), timeout=30)
    if r.status_code != 200 or not r.text.strip().startswith(">"):
        return None
    lines = r.text.strip().splitlines()
    seq = "".join(lines[1:]).replace(" ", "")
    if not seq:
        return None
    return {
        "uniprot_id": acc,               # тут это GenBank-акцешен, не UniProt
        "uniprot_entry_name": None,
        "reviewed": False,               # GenBank-запись не эквивалент Swiss-Prot review
        "uniprot_protein_name": None,
        "uniprot_organism": None,
        "taxon_id": None,
        "protein_existence": None,
        "is_fragment": False,            # неизвестно — не проверяем на этом пути
        "length": len(seq),
        "sequence": seq,
        "md5": hashlib.md5(seq.encode()).hexdigest().upper(),
    }


def parse(entry: dict) -> dict:
    if "sequence" not in entry:
        reason = entry.get("inactiveReason", {})
        merged_into = reason.get("mergeDemergeTo", [])
        raise InactiveEntry(
            f"inactive entry, reason={reason.get('inactiveReasonType')}, "
            f"merged_into={merged_into}"
        )
    seq = entry["sequence"]["value"]
    name = (
        entry.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value")
    )
    flags = entry.get("proteinDescription", {}).get("flag", "")
    return {
        "uniprot_id": entry["primaryAccession"],
        "uniprot_entry_name": entry.get("uniProtkbId"),
        "reviewed": entry.get("entryType", "").startswith("UniProtKB reviewed"),
        "uniprot_protein_name": name,
        "uniprot_organism": entry.get("organism", {}).get("scientificName"),
        "taxon_id": entry.get("organism", {}).get("taxonId"),
        "protein_existence": entry.get("proteinExistence"),
        "is_fragment": "Fragment" in str(flags),
        "length": len(seq),
        "sequence": seq,
        "md5": hashlib.md5(seq.encode()).hexdigest().upper(),
    }


def main() -> None:
    rows, log = [], []

    with RAW.open(encoding="utf-8") as f:
        candidates = list(csv.DictReader(f))

    for c in candidates:
        # Пробуем все известные ID для этой записи, а не только "suggested":
        # у многозначных строк (гомеологи, слитые записи и т.п.) предложенный
        # аккешен может быть DELETED или MERGED — тогда берём следующий по списку.
        suggested = (c["uniprot_suggested"] or "").strip()
        all_ids = [a.strip() for a in (c["uniprot_all"] or "").split(";") if a.strip()]
        candidates_order = ([suggested] if suggested else []) + [
            a for a in all_ids if a != suggested
        ]

        entry = None
        attempts = []
        used_acc = None

        if not candidates_order:
            attempts.append("no uniprot_suggested/uniprot_all — skipping to GenBank")

        # Пробуем ВСЕ ID и складываем все успешные разборы, а не берём первый
        # попавшийся: иначе можно остановиться на коротком фрагменте (is_fragment)
        # или неревьюченной записи, хотя следующий ID в списке может быть полной
        # Swiss-Prot последовательностью. Отбор лучшего — после полного перебора.
        successes: list[tuple[str, dict]] = []

        for acc in candidates_order:
            try:
                fetched = fetch_uniprot(acc)
            except Exception as exc:                  # noqa: BLE001
                attempts.append(f"{acc}: error {exc}")
                continue
            finally:
                time.sleep(0.3)                       # вежливость к API

            if fetched is None:
                attempts.append(f"{acc}: not_found (404)")
                continue

            try:
                parsed = parse(fetched)
            except InactiveEntry as exc:
                attempts.append(f"{acc}: {exc}")
                continue

            attempts.append(f"{acc}: ok (length={parsed['length']}, fragment={parsed['is_fragment']}, reviewed={parsed['reviewed']})")
            successes.append((acc, parsed))

        if successes:
            # Скор: не-фрагмент лучше фрагмента, reviewed лучше не-reviewed,
            # длиннее лучше короче. Так мы не застрянем на первом же ID,
            # если он оказался обрезанным куском белка.
            def score(item):
                acc, p = item
                return (not p["is_fragment"], p["reviewed"], p["length"])

            used_acc, p = max(successes, key=score)
            entry = True  # entry уже распарсен в p, дальше используем p напрямую
            if len(successes) > 1 and used_acc != successes[0][0]:
                attempts.append(
                    f"chosen {used_acc} over {successes[0][0]} "
                    f"(better: non-fragment/reviewed/longer)"
                )

        used_genbank_fallback = False

        if entry is None:
            gb_acc = (c["genbank_first"] or "").strip()
            gb_result = None
            if gb_acc:
                try:
                    gb_result = fetch_genbank_fasta(gb_acc)
                except Exception as exc:                  # noqa: BLE001
                    attempts.append(f"genbank {gb_acc}: error {exc}")
                finally:
                    time.sleep(0.3)

            if gb_result is None:
                log.append(
                    {**c, "status": "all_ids_failed | " + " || ".join(attempts)}
                )
                continue

            p = gb_result
            used_acc = gb_acc
            used_genbank_fallback = True
            attempts.append(f"genbank {gb_acc}: ok (fallback)")

        if used_genbank_fallback:
            note_extra = f"UniProt IDs all failed, used GenBank {used_acc} instead"
            c = {**c, "note": (c["note"] + "; " if c["note"] else "") + note_extra}
        elif used_acc != suggested:
            note_extra = f"used fallback ID {used_acc} (suggested {suggested} failed)"
            c = {**c, "note": (c["note"] + "; " if c["note"] else "") + note_extra}
        rows.append(
            {
                "protein_id": p["uniprot_id"],
                "protein_name": c["cazy_protein_name"],
                "organism": c["organism"],
                "taxon_id": p["taxon_id"],
                "gh_family": c["gh_family"],
                "subfamily": c["subfamily"],
                "ec_number": c["ec_number"],
                "activity": c["activity"],
                "class_group": c["class_group"],
                "length": p["length"],
                "sequence": p["sequence"],
                "md5": p["md5"],
                "reviewed": p["reviewed"],
                "is_fragment": p["is_fragment"],
                "protein_existence": p["protein_existence"],
                "uniprot_organism": p["uniprot_organism"],
                "include": c["include"],
                "source": "CAZy;GenBank" if used_genbank_fallback else "CAZy;UniProt",
                "note": c["note"],
            }
        )
        log.append({**c, "status": "ok"})

    # --- дедупликация по последовательности -------------------------------
    seen: dict[str, str] = {}
    for r in rows:
        if r["md5"] in seen:
            r["note"] = (r["note"] + "; " if r["note"] else "") + \
                        f"duplicate sequence of {seen[r['md5']]}"
            r["include"] = "no"
        else:
            seen[r["md5"]] = r["protein_id"]

    # --- запись -----------------------------------------------------------
    with (OUT_DIR / "gh13_dataset.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with (SEQ_DIR / "gh13.fasta").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(f">{r['protein_id']}|{r['activity']}|{r['organism'].replace(' ', '_')}\n")
            for i in range(0, len(r["sequence"]), 60):
                f.write(r["sequence"][i : i + 60] + "\n")

    with (OUT_DIR / "fetch_log.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        w.writeheader()
        w.writerows(log)

    summary = {
        "candidates": len(candidates),
        "fetched": len(rows),
        "unique_sequences": len(seen),
        "by_activity": {},
    }
    for r in rows:
        if r["include"] == "yes":
            summary["by_activity"][r["activity"]] = \
                summary["by_activity"].get(r["activity"], 0) + 1
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()