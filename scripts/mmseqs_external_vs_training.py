
from pathlib import Path
import subprocess
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXTERNAL_FASTA = PROJECT_ROOT / "data/sequences/external_candidates.fasta"
TRAINING_FASTA = PROJECT_ROOT / "data/sequences/ml_dataset.fasta"

OUT_DIR = PROJECT_ROOT / "data/processed/mmseqs_external"
RESULT_CSV = PROJECT_ROOT / "data/processed/external_vs_training_mmseqs.csv"

QUERY_DB = OUT_DIR / "external_db"
TARGET_DB = OUT_DIR / "training_db"
RESULT_DB = OUT_DIR / "external_vs_training"


# ============================================================
# MMSEQS PARAMETERS
# ============================================================

# We are NOT filtering candidates yet.
# The goal of this step is to inspect the nearest training
# sequence for every external candidate.
#
# These parameters provide a sensitive homology search.
#
# Important:
# - no class labels are used
# - no model predictions are used
# - no candidate is removed at this stage
#
MIN_SEQ_ID = 0.0
COV_MODE = 0
COV = 0.0
EVALUE = 1e-3


def run_command(command):
    print("\n$ " + " ".join(map(str, command)))

    subprocess.run(
        command,
        check=True,
    )


def main():
    print("=" * 70)
    print("MMSEQS2: EXTERNAL CANDIDATES VS TRAINING SET")
    print("=" * 70)

    if not EXTERNAL_FASTA.exists():
        raise FileNotFoundError(
            f"External FASTA not found: {EXTERNAL_FASTA}"
        )

    if not TRAINING_FASTA.exists():
        raise FileNotFoundError(
            f"Training FASTA not found: {TRAINING_FASTA}"
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Create MMseqs databases
    # --------------------------------------------------------

    print("\n[1/4] Creating MMseqs2 databases...")

    run_command([
        "mmseqs",
        "createdb",
        str(EXTERNAL_FASTA),
        str(QUERY_DB),
    ])

    run_command([
        "mmseqs",
        "createdb",
        str(TRAINING_FASTA),
        str(TARGET_DB),
    ])

    # --------------------------------------------------------
    # 2. Search external candidates against training proteins
    # --------------------------------------------------------

    print("\n[2/4] Searching external candidates against training set...")

    run_command([
        "mmseqs",
        "search",
        str(QUERY_DB),
        str(TARGET_DB),
        str(RESULT_DB),
        str(OUT_DIR / "tmp"),
        "--min-seq-id",
        str(MIN_SEQ_ID),
        "-c",
        str(COV),
        "--cov-mode",
        str(COV_MODE),
        "-e",
        str(EVALUE),
        "--threads",
        "4",
    ])

    # --------------------------------------------------------
    # 3. Convert results to TSV
    # --------------------------------------------------------

    print("\n[3/4] Exporting search results...")

    result_tsv = OUT_DIR / "external_vs_training.tsv"

    run_command([
        "mmseqs",
        "convertalis",
        str(QUERY_DB),
        str(TARGET_DB),
        str(RESULT_DB),
        str(result_tsv),
        "--format-output",
        "query,target,pident,alnlen,qstart,qend,tstart,tend,evalue,bits",
    ])

    # --------------------------------------------------------
    # 4. Load and keep best hit per external candidate
    # --------------------------------------------------------

    print("\n[4/4] Building best-hit table...")

    columns = [
        "query",
        "target",
        "pident",
        "alnlen",
        "qstart",
        "qend",
        "tstart",
        "tend",
        "evalue",
        "bits",
    ]

    df = pd.read_csv(
        result_tsv,
        sep="\t",
        names=columns,
    )

    if df.empty:
        print("\nNo MMseqs2 hits found.")
        return

    # Keep the strongest hit for each external candidate.
    #
    # Bitscore is used as the primary ranking criterion.
    # E-value is used as a secondary criterion.
    best_hits = (
        df.sort_values(
            ["query", "bits", "evalue"],
            ascending=[True, False, True],
        )
        .groupby("query", as_index=False)
        .first()
    )

    # --------------------------------------------------------
    # Calculate alignment coverage
    # --------------------------------------------------------

    # Read sequence lengths from FASTA.
    def fasta_lengths(path):
        lengths = {}

        current_id = None
        current_seq = []

        with open(path) as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                if line.startswith(">"):
                    if current_id is not None:
                        lengths[current_id] = len(
                            "".join(current_seq)
                        )

                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)

            if current_id is not None:
                lengths[current_id] = len(
                    "".join(current_seq)
                )

        return lengths

    query_lengths = fasta_lengths(EXTERNAL_FASTA)
    target_lengths = fasta_lengths(TRAINING_FASTA)

    best_hits["query_length"] = best_hits["query"].map(
        query_lengths
    )

    best_hits["target_length"] = best_hits["target"].map(
        target_lengths
    )

    # Alignment coverage of the external/query protein.
    best_hits["query_coverage"] = (
        best_hits["alnlen"]
        / best_hits["query_length"]
    )

    # Alignment coverage of the training/target protein.
    best_hits["target_coverage"] = (
        best_hits["alnlen"]
        / best_hits["target_length"]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    best_hits.to_csv(
        RESULT_CSV,
        index=False,
    )

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print(f"External candidates: {len(query_lengths)}")
    print(f"Candidates with hits: {len(best_hits)}")

    print(f"\nSaved:")
    print(f"  {RESULT_CSV}")

    print("\nBest training hit per external candidate:\n")

    display_columns = [
        "query",
        "target",
        "pident",
        "query_coverage",
        "target_coverage",
        "evalue",
        "bits",
    ]

    print(
        best_hits[display_columns]
        .to_string(index=False)
    )

    print("\nIMPORTANT:")
    print(
        "No candidates were removed. "
        "This table is for homology inspection only."
    )


if __name__ == "__main__":
    main()
