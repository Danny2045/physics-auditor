"""Tests for the always-on, ligand-independent structure completeness gate.

This gate is the always-on half of the three-state readiness design: it runs
at the front door of every audit regardless of whether a ligand is supplied,
so a bare PDB audit still gets a SEQRES-vs-ATOM gap signal. INCOMPLETE iff
at least one INTERNAL gap exists anywhere in the structure (resolved
neighbors on BOTH sequence sides); terminal tails are reported informationally
and never trigger INCOMPLETE.

These tests reuse the synthetic PDB builders from ``test_pocket_completeness``
so the inputs are deterministic.
"""

from __future__ import annotations

from pathlib import Path

from physics_auditor.causality.pocket_completeness import (
    pocket_completeness_gate,
    structure_completeness_gate,
)
from physics_auditor.core.parser import parse_pdb_string

REPO_ROOT = Path(__file__).resolve().parent.parent


def _atom_line(
    serial: int,
    atom_name: str,
    res_name: str,
    chain: str,
    res_seq: int,
    x: float,
    y: float,
    z: float,
    element: str = "C",
    record: str = "ATOM",
) -> str:
    """Format a fixed-width PDB ATOM/HETATM record matching the parser slices."""
    if len(atom_name) >= 4:
        name_field = atom_name[:4]
    elif len(atom_name) == 3:
        name_field = " " + atom_name
    elif len(atom_name) == 2:
        name_field = " " + atom_name + " "
    else:
        name_field = " " + atom_name + "  "
    record_field = record.ljust(6)
    return (
        f"{record_field}{serial:5d} {name_field} {res_name:>3} {chain:1s}"
        f"{res_seq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n"
    )


def _seqres_lines(chain: str, codes: list[str]) -> str:
    """Format SEQRES records for one chain (13 codes per line)."""
    out = []
    total = len(codes)
    per_line = 13
    serial = 1
    for start in range(0, total, per_line):
        chunk = codes[start : start + per_line]
        codes_str = " ".join(f"{c:>3}" for c in chunk)
        out.append(f"SEQRES{serial:4d} {chain:1s}{total:5d}  {codes_str}\n")
        serial += 1
    return "".join(out)


def _build_residue(
    serial_start: int, chain: str, res_seq: int, res_name: str, ca_xyz: tuple
) -> tuple[str, int]:
    """Emit four backbone atoms (N, CA, C, O) at small offsets from `ca_xyz`."""
    cx, cy, cz = ca_xyz
    lines = []
    lines.append(
        _atom_line(serial_start, "N", res_name, chain, res_seq, cx, cy - 1.0, cz, "N")
    )
    lines.append(
        _atom_line(serial_start + 1, "CA", res_name, chain, res_seq, cx, cy, cz, "C")
    )
    lines.append(
        _atom_line(
            serial_start + 2, "C", res_name, chain, res_seq,
            cx + 0.5, cy + 0.5, cz, "C",
        )
    )
    lines.append(
        _atom_line(
            serial_start + 3, "O", res_name, chain, res_seq,
            cx + 0.5, cy + 1.5, cz, "O",
        )
    )
    return "".join(lines), serial_start + 4


def _build_ligand(serial_start: int, xyz: tuple) -> tuple[str, int]:
    """Single-atom HETATM ligand for pocket definition."""
    x, y, z = xyz
    return (
        _atom_line(
            serial_start, "C1", "LIG", "X", 1, x, y, z, "C", record="HETATM"
        ),
        serial_start + 1,
    )


def _assemble_pdb(seqres_block: str, atom_lines: str) -> str:
    return seqres_block + atom_lines + "END\n"


def test_structure_with_no_gaps_is_complete():
    """A fully resolved 5-residue chain has no gaps → COMPLETE."""
    chain = "A"
    codes = ["ALA"] * 5
    atom_block = ""
    serial = 1
    for i, ca in enumerate(
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0),
         (6.0, 0.0, 0.0), (8.0, 0.0, 0.0)],
        start=1,
    ):
        block, serial = _build_residue(serial, chain, i, "ALA", ca)
        atom_block += block
    pdb = _assemble_pdb(_seqres_lines(chain, codes), atom_block)

    structure = parse_pdb_string(pdb, name="synthetic_complete")
    result = structure_completeness_gate(structure)

    assert result.verdict == "COMPLETE", result.summary()
    assert result.is_complete is True
    assert result.missing_residues == []
    assert result.internal_gaps == []
    assert result.unresolved_termini == []


def test_internal_gap_flips_to_incomplete_without_ligand():
    """A missing residue with resolved neighbors on both sides → INCOMPLETE,
    no ligand needed.

    This is the core of the always-on check: structure_completeness_gate
    catches internal gaps without any ligand designation.
    """
    chain = "A"
    codes = ["ALA"] * 5
    coords = {
        1: (0.0, 0.0, 0.0),
        2: (2.0, 0.0, 0.0),
        4: (6.0, 0.0, 0.0),  # residue 3 missing internally
        5: (8.0, 0.0, 0.0),
    }
    atom_block = ""
    serial = 1
    for res_seq, ca in coords.items():
        block, serial = _build_residue(serial, chain, res_seq, "ALA", ca)
        atom_block += block
    pdb = _assemble_pdb(_seqres_lines(chain, codes), atom_block)

    structure = parse_pdb_string(pdb, name="synthetic_internal_gap")
    result = structure_completeness_gate(structure)

    assert result.verdict == "INCOMPLETE", result.summary()
    assert len(result.internal_gaps) == 1
    gap = result.internal_gaps[0]
    assert gap.chain_id == "A"
    assert gap.res_seq == 3
    assert gap.res_name == "ALA"
    assert gap.is_terminal is False
    # The structure gate never looks at a ligand, so in_pocket_region must
    # stay at its enumeration default.
    assert gap.in_pocket_region is False
    assert result.unresolved_termini == []


def test_terminal_tail_does_not_trigger_incomplete():
    """A trailing missing tail has only one resolved sequence neighbor → terminal,
    reported informationally, verdict stays COMPLETE.
    """
    chain = "A"
    codes = ["ALA"] * 7
    resolved_coords = {
        1: (0.0, 0.0, 0.0),
        2: (2.0, 0.0, 0.0),
        3: (4.0, 0.0, 0.0),
    }
    atom_block = ""
    serial = 1
    for rs, ca in resolved_coords.items():
        block, serial = _build_residue(serial, chain, rs, "ALA", ca)
        atom_block += block
    pdb = _assemble_pdb(_seqres_lines(chain, codes), atom_block)

    structure = parse_pdb_string(pdb, name="synthetic_c_term_tail")
    result = structure_completeness_gate(structure)

    assert result.verdict == "COMPLETE", result.summary()
    assert result.internal_gaps == []
    term_seqs = sorted(m.res_seq for m in result.unresolved_termini)
    assert term_seqs == [4, 5, 6, 7], (
        f"all 4 trailing missing residues must be reported as terminal; "
        f"got {term_seqs}"
    )
    for m in result.unresolved_termini:
        assert m.is_terminal is True


def test_n_terminal_truncation_is_terminal_not_internal():
    """Missing N-terminal residues (only resolved neighbors AFTER) are terminal.

    Uses distinct residue names so SEQRES-to-ATOM offset is unambiguous.
    """
    chain = "A"
    codes = ["MET", "ALA", "GLY", "VAL", "LEU"]
    resolved = {
        3: (codes[2], (4.0, 0.0, 0.0)),
        4: (codes[3], (6.0, 0.0, 0.0)),
        5: (codes[4], (8.0, 0.0, 0.0)),
    }
    atom_block = ""
    serial = 1
    for rs, (name, ca) in resolved.items():
        block, serial = _build_residue(serial, chain, rs, name, ca)
        atom_block += block
    pdb = _assemble_pdb(_seqres_lines(chain, codes), atom_block)

    structure = parse_pdb_string(pdb, name="synthetic_n_term_trunc")
    result = structure_completeness_gate(structure)

    assert result.verdict == "COMPLETE", result.summary()
    assert result.internal_gaps == []
    term_seqs = sorted(m.res_seq for m in result.unresolved_termini)
    assert term_seqs == [1, 2], (
        f"SEQRES positions 1-2 (MET, ALA) are unresolved at the N-terminus "
        f"and must be classified as terminal; got res_seqs {term_seqs}"
    )


def test_multi_chain_internal_gap_in_any_chain_triggers_incomplete():
    """An internal gap in ANY chain flips the structure verdict."""
    codes_a = ["ALA"] * 3
    codes_b = ["GLY", "VAL", "LEU", "SER"]
    # Chain A: fully resolved (no gaps).
    atom = ""
    serial = 1
    for rs, ca in enumerate(
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0)], start=1
    ):
        block, serial = _build_residue(serial, "A", rs, "ALA", ca)
        atom += block
    # Chain B: residue 3 is missing internally.
    b_coords = {
        1: (10.0, 0.0, 0.0),
        2: (12.0, 0.0, 0.0),
        4: (16.0, 0.0, 0.0),
    }
    b_names = {1: "GLY", 2: "VAL", 4: "SER"}
    for rs, ca in b_coords.items():
        block, serial = _build_residue(serial, "B", rs, b_names[rs], ca)
        atom += block
    pdb = (
        _seqres_lines("A", codes_a)
        + _seqres_lines("B", codes_b)
        + atom
        + "END\n"
    )

    structure = parse_pdb_string(pdb, name="synthetic_multichain")
    result = structure_completeness_gate(structure)

    assert result.verdict == "INCOMPLETE", result.summary()
    gap_ids = [(g.chain_id, g.res_seq) for g in result.internal_gaps]
    assert ("B", 3) in gap_ids
    # Chain A has no gaps.
    assert all(cid == "B" for cid, _ in gap_ids)


def test_pocket_and_structure_gates_share_enumeration():
    """Both gates must agree on the set of missing residues and their
    terminal/internal classification — this is the single-source-of-truth
    invariant that justifies the refactor.

    Build a structure with one internal gap inside the pocket and one
    terminal tail away from the pocket; the missing-residue lists and the
    terminal classifications must match exactly between the structure gate
    and the pocket gate.
    """
    chain = "A"
    codes = ["ALA"] * 7
    coords = {
        1: (0.0, 0.0, 0.0),
        2: (2.0, 0.0, 0.0),
        4: (6.0, 0.0, 0.0),  # residue 3 missing internally, near pocket
        5: (8.0, 0.0, 0.0),
        # residues 6, 7 unresolved C-terminal tail
    }
    atom = ""
    serial = 1
    for rs, ca in coords.items():
        block, serial = _build_residue(serial, chain, rs, "ALA", ca)
        atom += block
    lig_block, _ = _build_ligand(serial, (4.0, 2.0, 0.0))
    pdb = _assemble_pdb(_seqres_lines(chain, codes), atom + lig_block)
    structure = parse_pdb_string(pdb, name="synthetic_shared")

    struct_result = structure_completeness_gate(structure)
    pocket_result = pocket_completeness_gate(
        structure, ligand_resname="LIG", cutoff=5.0
    )

    # Same set of missing residues (chain_id, res_seq), same is_terminal.
    struct_set = {
        (m.chain_id, m.res_seq, m.is_terminal)
        for m in struct_result.missing_residues
    }
    pocket_set = {
        (m.chain_id, m.res_seq, m.is_terminal)
        for m in pocket_result.missing_residues
    }
    assert struct_set == pocket_set, (
        f"the gates' missing-residue enumerations disagree:\n"
        f"  structure: {struct_set}\n"
        f"  pocket   : {pocket_set}"
    )

    # Same chain offsets.
    assert struct_result.chain_offsets == pocket_result.chain_offsets

    # Internal gaps drive structure verdict.
    assert struct_result.verdict == "INCOMPLETE"
    assert sorted(m.res_seq for m in struct_result.internal_gaps) == [3]
    assert sorted(m.res_seq for m in struct_result.unresolved_termini) == [6, 7]


def test_4plz_real_structure_is_incomplete_without_ligand():
    """End-to-end check on the real 4PLZ deposit: the always-on gate must
    flag the W107f failure mode (disordered substrate-specificity loop,
    residues 82-93) on a bare audit with no ligand supplied.
    """
    from physics_auditor.core.parser import parse_pdb

    pdb_path = (
        REPO_ROOT / "benchmark" / "structures" / "experimental" / "4PLZ.pdb"
    )
    structure = parse_pdb(pdb_path)
    result = structure_completeness_gate(structure)

    assert result.verdict == "INCOMPLETE", (
        "4PLZ has a disordered substrate-specificity loop (residues 82-93); "
        "the always-on structure-level gate must catch this with no ligand."
    )
    gap_seqs = sorted(m.res_seq for m in result.internal_gaps if m.chain_id == "A")
    assert gap_seqs == list(range(82, 94)), (
        f"expected internal gaps at residues 82-93 (substrate-specificity loop); "
        f"got {gap_seqs}"
    )
