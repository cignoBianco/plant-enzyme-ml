from pathlib import Path
import csv
import time
import requests


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"

OUTPUT_FASTA = SEQUENCES_DIR / "external_candidates.fasta"
OUTPUT_METADATA = OUTPUT_DIR / "external_candidates.csv"


# Candidate pool.
#
# IMPORTANT:
# This is NOT yet the final external test set.
# We will later compare these sequences against the 31 training
# proteins using MMseqs2 and remove close homologues.
#
# accession, activity, organism, evidence_note
CANDIDATES = [
    # Existing alpha-amylase candidates
    {
        "accession": "Q8GUR0",
        "activity": "alpha-amylase",
        "organism": "unknown",
        "evidence_note": "candidate external alpha-amylase",
    },
    {
        "accession": "M1ADS0",
        "activity": "alpha-amylase",
        "organism": "unknown",
        "evidence_note": "candidate external alpha-amylase",
    },
    {
        "accession": "A0A3B6KJH0",
        "activity": "alpha-amylase",
        "organism": "unknown",
        "evidence_note": "candidate external alpha-amylase",
    },
    {
        "accession": "A0A1D6I3N3",
        "activity": "alpha-amylase",
        "organism": "unknown",
        "evidence_note": "candidate external alpha-amylase",
    },

    # Existing isoamylase candidates
    {
        "accession": "Q84UE6",
        "activity": "isoamylase",
        "organism": "unknown",
        "evidence_note": "candidate external isoamylase",
    },
    {
        "accession": "A4PIS8",
        "activity": "isoamylase",
        "organism": "unknown",
        "evidence_note": "candidate external isoamylase",
    },
    {
        "accession": "A4PIS9",
        "activity": "isoamylase",
        "organism": "unknown",
        "evidence_note": "candidate external isoamylase",
    },
    {
        "accession": "A4PIT0",
        "activity": "isoamylase",
        "organism": "unknown",
        "evidence_note": "candidate external isoamylase",
    },

    # Existing pullulanase / limit dextrinase
    {
        "accession": "O48541",
        "activity": "pullulanase/limit dextrinase",
        "organism": "Hordeum vulgare",
        "evidence_note": "existing external limit-dextrinase candidate",
    },

    # New pullulanase / limit dextrinase candidates
    {
        "accession": "A0A368RRA7",
        "activity": "pullulanase/limit dextrinase",
        "organism": "Setaria italica",
        "evidence_note": "Pullulanase 1, chloroplastic; RefSeq XP_004975057.1",
    },
    {
        "accession": "A0A059BGW3",
        "activity": "pullulanase/limit dextrinase",
        "organism": "Eucalyptus grandis",
        "evidence_note": "Pullulanase 1, chloroplastic; RefSeq XP_018715184.2",
    },
    {
        "accession": "Q8GTR4",
        "activity": "pullulanase/limit dextrinase",
        "organism": "Arabidopsis thaliana",
        "evidence_note": "Limit dextrinase / Pullulanase 1; EC 3.2.1.142; ambiguous dual annotation",
    },
]


# UniProt REST API.
# Requesting FASTA directly by accession.
UNIPROT_FASTA_URL = (
    "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
)


# Request timeout in seconds.
TIMEOUT = 30

# Small delay between requests.
REQUEST_DELAY = 0.2


# ============================================================
# HELPERS
# ============================================================

def fetch_fasta(accession: str) -> str:
    """
    Download one protein sequence from UniProt.

    Returns:
        Raw FASTA text.
    """

    url = UNIPROT_FASTA_URL.format(accession=accession)

    response = requests.get(
        url,
        timeout=TIMEOUT,
        headers={
            "User-Agent": "plant-enzyme-ml/1.0"
        },
    )

    response.raise_for_status()

    fasta = response.text.strip()

    if not fasta.startswith(">"):
        raise ValueError(
            f"UniProt response for {accession} "
            f"does not look like FASTA."
        )

    return fasta


def parse_fasta(fasta: str):
    """
    Parse a single-entry FASTA.

    Returns:
        header, sequence
    """

    lines = fasta.splitlines()

    header = lines[0].strip()

    sequence = "".join(
        line.strip()
        for line in lines[1:]
        if line.strip()
    )

    if not header.startswith(">"):
        raise ValueError("Invalid FASTA header.")

    if not sequence:
        raise ValueError("Empty protein sequence.")

    return header, sequence


def normalize_sequence(sequence: str) -> str:
    """
    Remove whitespace and convert sequence to uppercase.
    """

    return "".join(sequence.split()).upper()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DOWNLOAD EXTERNAL CANDIDATE PROTEINS")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SEQUENCES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"\nCandidates: {len(CANDIDATES)}")
    print(f"FASTA output: {OUTPUT_FASTA}")
    print(f"Metadata output: {OUTPUT_METADATA}")

    downloaded = []
    failed = []

    fasta_records = []

    # --------------------------------------------------------
    # Download sequences
    # --------------------------------------------------------

    for i, candidate in enumerate(CANDIDATES, start=1):

        accession = candidate["accession"]

        print(
            f"\n[{i}/{len(CANDIDATES)}] "
            f"{accession} "
            f"({candidate['activity']})"
        )

        try:
            fasta = fetch_fasta(accession)

            header, sequence = parse_fasta(fasta)

            sequence = normalize_sequence(sequence)

            # Validate protein sequence.
            valid_amino_acids = set(
                "ACDEFGHIKLMNPQRSTVWY"
            )

            invalid = sorted(
                set(sequence) - valid_amino_acids
            )

            if invalid:
                raise ValueError(
                    f"Invalid amino-acid symbols: {invalid}"
                )

            # Create our own stable FASTA header.
            #
            # Keeping accession + class + organism makes the
            # file easy to inspect manually.
            organism_slug = (
                candidate["organism"]
                .replace(" ", "_")
            )

            fasta_header = (
                f">{accession}|"
                f"{candidate['activity']}|"
                f"{organism_slug}"
            )

            fasta_records.append(
                fasta_header + "\n" + sequence
            )

            downloaded.append(
                {
                    **candidate,
                    "accession": accession,
                    "uniprot_header": header,
                    "length": len(sequence),
                    "sequence": sequence,
                    "download_status": "success",
                }
            )

            print(
                f"  ✓ downloaded"
                f"  length={len(sequence)} aa"
            )

        except Exception as exc:

            print(
                f"  ✗ FAILED: {exc}"
            )

            failed.append(
                {
                    **candidate,
                    "download_status": "failed",
                    "error": str(exc),
                }
            )

        time.sleep(REQUEST_DELAY)

    # --------------------------------------------------------
    # Write FASTA
    # --------------------------------------------------------

    with open(
        OUTPUT_FASTA,
        "w",
        encoding="utf-8",
    ) as f:

        for record in fasta_records:
            f.write(record + "\n")

    # --------------------------------------------------------
    # Write metadata
    # --------------------------------------------------------

    metadata_fields = [
        "accession",
        "activity",
        "organism",
        "evidence_note",
        "uniprot_header",
        "length",
        "download_status",
        "error",
    ]

    with open(
        OUTPUT_METADATA,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=metadata_fields,
        )

        writer.writeheader()

        for record in downloaded:

            writer.writerow(
                {
                    "accession": record["accession"],
                    "activity": record["activity"],
                    "organism": record["organism"],
                    "evidence_note": record["evidence_note"],
                    "uniprot_header": record["uniprot_header"],
                    "length": record["length"],
                    "download_status": record["download_status"],
                    "error": "",
                }
            )

        for record in failed:

            writer.writerow(
                {
                    "accession": record["accession"],
                    "activity": record["activity"],
                    "organism": record["organism"],
                    "evidence_note": record["evidence_note"],
                    "uniprot_header": "",
                    "length": "",
                    "download_status": "failed",
                    "error": record["error"],
                }
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print(f"Requested:  {len(CANDIDATES)}")
    print(f"Downloaded: {len(downloaded)}")
    print(f"Failed:     {len(failed)}")

    if downloaded:

        print("\nDownloaded candidates by class:")

        class_counts = {}

        for record in downloaded:
            activity = record["activity"]
            class_counts[activity] = (
                class_counts.get(activity, 0) + 1
            )

        for activity, count in class_counts.items():
            print(f"  {activity}: {count}")

    print("\n✓ FASTA:")
    print(f"  {OUTPUT_FASTA}")

    print("\n✓ Metadata:")
    print(f"  {OUTPUT_METADATA}")

    if failed:

        print("\n⚠ Failed accessions:")

        for record in failed:
            print(
                f"  {record['accession']}: "
                f"{record['error']}"
            )

    print("\nIMPORTANT:")
    print(
        "These proteins are still CANDIDATES. "
        "They must pass the homology filtering step "
        "against the 31 training proteins before "
        "being used as the independent test set."
    )


if __name__ == "__main__":
    main()