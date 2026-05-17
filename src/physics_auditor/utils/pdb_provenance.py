"""PDB provenance verification.

Confirms that a PDB file's SOURCE record reports the organism the caller
expects, BEFORE any downstream parsing or analysis runs.

History
-------
In May 2026 a selectivity-attribution run was committed under the
framing "SmDHODH vs HsDHODH" while one of the two inputs (1MVS) was in
fact HsDHFR. The mislabeling was caught only after the run shipped, and
the dossier had to be corrected in commit a19a2a2. This module exists
so that class of mistake cannot recur: every PDB ingestion path must
call ``verify_pdb_source(path, expected_organism)`` before parsing.

PDB SOURCE record format
------------------------
Lines beginning ``SOURCE`` carry semicolon-separated key/value pairs.
The organism is on a line of the form::

    SOURCE   2 ORGANISM_SCIENTIFIC: PLASMODIUM FALCIPARUM;

Multi-MOL_ID structures repeat this block; the first
ORGANISM_SCIENTIFIC encountered is taken as the structure's primary
organism for verification.
"""

from __future__ import annotations

from pathlib import Path


def verify_pdb_source(path: Path, expected_organism: str) -> None:
    """Verify the SOURCE record's ORGANISM_SCIENTIFIC matches expectation.

    Parameters
    ----------
    path : Path
        Path to a PDB file on disk.
    expected_organism : str
        The organism the caller asserts the file should report, e.g.
        ``"PLASMODIUM FALCIPARUM"`` or ``"HOMO SAPIENS"``. Matched
        case-insensitively against the ORGANISM_SCIENTIFIC value, with
        surrounding whitespace and trailing semicolons stripped.

    Raises
    ------
    ValueError
        If the file does not exist, has no SOURCE records, has SOURCE
        records but no ORGANISM_SCIENTIFIC field, or reports an
        ORGANISM_SCIENTIFIC value that does not match
        ``expected_organism``. The error message names the file and
        the actual value found.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(
            f"PDB file not found: {path}. "
            f"Cannot verify ORGANISM_SCIENTIFIC before parse."
        )

    text = path.read_text()
    source_lines = [
        line for line in text.splitlines() if line.startswith("SOURCE")
    ]
    if not source_lines:
        raise ValueError(
            f"No SOURCE records in {path}. "
            f"Cannot verify ORGANISM_SCIENTIFIC before parse."
        )

    organism: str | None = None
    for line in source_lines:
        if "ORGANISM_SCIENTIFIC" not in line:
            continue
        _, _, rhs = line.partition("ORGANISM_SCIENTIFIC:")
        candidate = rhs.strip().rstrip(";").strip()
        if candidate:
            organism = candidate
            break

    if organism is None:
        raise ValueError(
            f"SOURCE records in {path} contain no ORGANISM_SCIENTIFIC "
            f"field (or the field is empty). Cannot verify provenance."
        )

    if organism.upper() != expected_organism.upper():
        raise ValueError(
            f"PDB provenance mismatch for {path}: "
            f"SOURCE ORGANISM_SCIENTIFIC is '{organism}', "
            f"expected '{expected_organism}'. Aborting before parse."
        )
