"""Synthetic known-answer tests for the pocket-completeness gate.

These cases validate the gate on hand-built structures with deterministic
expected outcomes BEFORE it is ever applied to real deposits. Each case
exercises one branch of the gate's logic:

1. complete pocket  -> COMPLETE
2. pocket-region residue missing -> INCOMPLETE, names the missing residue
3. residue missing OUTSIDE the pocket region -> COMPLETE for the pocket
"""

from __future__ import annotations

from physics_auditor.causality.pocket_completeness import (
    pocket_completeness_gate,
)
from physics_auditor.core.parser import parse_pdb_string


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
    """Emit four backbone atoms (N, CA, C, O) at small offsets from `ca_xyz`.

    Returns the concatenated ATOM lines and the next available serial number.
    """
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
            serial_start + 2,
            "C",
            res_name,
            chain,
            res_seq,
            cx + 0.5,
            cy + 0.5,
            cz,
            "C",
        )
    )
    lines.append(
        _atom_line(
            serial_start + 3,
            "O",
            res_name,
            chain,
            res_seq,
            cx + 0.5,
            cy + 1.5,
            cz,
            "O",
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


def test_complete_pocket_returns_complete():
    """All SEQRES residues are present in ATOM and all sit within cutoff of ligand."""
    chain = "A"
    codes = ["ALA"] * 5
    coords = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0),
              (6.0, 0.0, 0.0), (8.0, 0.0, 0.0)]
    atom_block = ""
    serial = 1
    for i, ca in enumerate(coords, start=1):
        block, serial = _build_residue(serial, chain, i, "ALA", ca)
        atom_block += block
    lig_block, _ = _build_ligand(serial, (4.0, 2.0, 0.0))
    pdb = _assemble_pdb(_seqres_lines(chain, codes), atom_block + lig_block)

    structure = parse_pdb_string(pdb, name="synthetic_complete")
    result = pocket_completeness_gate(structure, ligand_resname="LIG", cutoff=5.0)

    assert result.verdict == "COMPLETE", result.summary()
    assert result.n_resolved_pocket_residues == 5
    assert result.missing_residues == []
    assert result.missing_in_pocket == []


def test_incomplete_pocket_names_missing_residue():
    """Residue 3 is in SEQRES but absent from ATOM; its flanks (2, 4) are pocket residues."""
    chain = "A"
    codes = ["ALA"] * 5
    coords = {
        1: (0.0, 0.0, 0.0),
        2: (2.0, 0.0, 0.0),
        4: (6.0, 0.0, 0.0),
        5: (8.0, 0.0, 0.0),
    }
    atom_block = ""
    serial = 1
    for res_seq, ca in coords.items():
        block, serial = _build_residue(serial, chain, res_seq, "ALA", ca)
        atom_block += block
    lig_block, _ = _build_ligand(serial, (4.0, 2.0, 0.0))
    pdb = _assemble_pdb(_seqres_lines(chain, codes), atom_block + lig_block)

    structure = parse_pdb_string(pdb, name="synthetic_incomplete")
    result = pocket_completeness_gate(structure, ligand_resname="LIG", cutoff=5.0)

    assert result.verdict == "INCOMPLETE", result.summary()
    assert len(result.missing_in_pocket) == 1
    miss = result.missing_in_pocket[0]
    assert miss.res_seq == 3
    assert miss.res_name == "ALA"
    assert miss.chain_id == "A"
    assert miss.in_pocket_region is True
    assert result.n_resolved_pocket_residues == 4


def test_missing_residue_outside_pocket_returns_complete():
    """Residue 6 is missing but both flanks (5, 7) are far from the ligand."""
    chain = "A"
    codes = ["ALA"] * 7
    pocket_coords = {
        1: (0.0, 0.0, 0.0),
        2: (2.0, 0.0, 0.0),
        3: (4.0, 0.0, 0.0),
    }
    distal_coords = {
        4: (20.0, 0.0, 0.0),
        5: (22.0, 0.0, 0.0),
        7: (26.0, 0.0, 0.0),
    }
    atom_block = ""
    serial = 1
    for res_seq, ca in {**pocket_coords, **distal_coords}.items():
        block, serial = _build_residue(serial, chain, res_seq, "ALA", ca)
        atom_block += block
    lig_block, _ = _build_ligand(serial, (2.0, 2.0, 0.0))
    pdb = _assemble_pdb(_seqres_lines(chain, codes), atom_block + lig_block)

    structure = parse_pdb_string(pdb, name="synthetic_distal_gap")
    result = pocket_completeness_gate(structure, ligand_resname="LIG", cutoff=5.0)

    assert result.verdict == "COMPLETE", result.summary()
    assert result.n_resolved_pocket_residues == 3
    missing_seqs = sorted(m.res_seq for m in result.missing_residues)
    assert 6 in missing_seqs, (
        f"residue 6 should be detected as missing; got {missing_seqs}"
    )
    assert result.missing_in_pocket == [], (
        "the missing residue is far from the ligand, so it must NOT be flagged "
        "as in-pocket"
    )


def test_reach_criterion_flags_long_missing_loop_anchored_off_pocket():
    """A long disordered run anchored further than `cutoff` from the ligand
    is still flagged in_pocket if its peptide reach can plausibly span the
    gap. This is the 4PLZ substrate-specificity loop case: anchors sit
    ~12 Å from the ligand but a 12-residue loop can physically reach in.
    """
    chain = "A"
    # 14-residue chain. Residues 1-2 and 13-14 are resolved; 3-12 (10 residues)
    # are missing. The resolved flanks sit ~10 Å from the ligand — well
    # outside the 5 Å pocket cutoff, so the strict-flank rule would say
    # COMPLETE — but a 10-residue gap with max reach 10*3.8=38 Å easily
    # spans 10 Å, so the reach-aware gate must flag INCOMPLETE.
    codes = ["ALA"] * 14
    resolved_coords = {
        1: (0.0, 0.0, 0.0),
        2: (2.0, 0.0, 0.0),
        13: (26.0, 0.0, 0.0),
        14: (28.0, 0.0, 0.0),
    }
    atom_block = ""
    serial = 1
    for rs, ca in resolved_coords.items():
        block, serial = _build_residue(serial, chain, rs, "ALA", ca)
        atom_block += block
    # Place the ligand inside the gap, ~10 Å from the closest flank (res 2).
    lig_block, _ = _build_ligand(serial, (12.0, 0.0, 0.0))
    pdb = _assemble_pdb(_seqres_lines(chain, codes), atom_block + lig_block)

    structure = parse_pdb_string(pdb, name="synthetic_long_loop")
    result = pocket_completeness_gate(structure, ligand_resname="LIG", cutoff=5.0)

    assert result.verdict == "INCOMPLETE", result.summary()
    flagged = {(m.chain_id, m.res_seq) for m in result.missing_in_pocket}
    # Every residue in the missing run can be reached by some flank.
    for rs in range(3, 13):
        assert ("A", rs) in flagged, (
            f"residue {rs} (in the long missing gap) should be flagged "
            f"in-pocket by the reach criterion; flagged set = {flagged}"
        )


def test_seqres_offset_alignment_is_inferred():
    """SEQRES positions can differ from author res_seq by an integer offset.

    Build a 5-residue chain where ATOM starts at res_seq=10 (offset = -9), and
    confirm the gate aligns SEQRES to ATOM by content rather than assuming
    position 1 == res_seq 1.
    """
    chain = "A"
    codes = ["ALA", "GLY", "SER", "VAL", "LEU"]
    atom_block = ""
    serial = 1
    coords_by_rs = {
        10: (0.0, 0.0, 0.0),
        11: (2.0, 0.0, 0.0),
        12: (4.0, 0.0, 0.0),
        13: (6.0, 0.0, 0.0),
        14: (8.0, 0.0, 0.0),
    }
    for rs, ca in coords_by_rs.items():
        idx = rs - 10
        block, serial = _build_residue(serial, chain, rs, codes[idx], ca)
        atom_block += block
    lig_block, _ = _build_ligand(serial, (4.0, 2.0, 0.0))
    pdb = _assemble_pdb(_seqres_lines(chain, codes), atom_block + lig_block)

    structure = parse_pdb_string(pdb, name="synthetic_offset")
    result = pocket_completeness_gate(structure, ligand_resname="LIG", cutoff=5.0)

    assert result.verdict == "COMPLETE", result.summary()
    assert result.chain_offsets["A"] == -9, (
        f"offset should map SEQRES position 1 to res_seq 10 (offset -9); "
        f"got {result.chain_offsets}"
    )
    assert result.missing_residues == []
