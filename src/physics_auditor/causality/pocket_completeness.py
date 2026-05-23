"""Pocket-completeness gate.

Determines whether the binding-pocket region of a structure is fully resolved
in the deposited coordinates. The gate compares SEQRES (the expected sequence
per chain) against resolved ATOM residues to detect gaps, then reports
whether any gap falls within the pocket-defining distance of the ligand.

Built to catch the failure mode discovered in the B-3 LDH blind test
(commit 3ee581e): the 4PLZ deposit (a W107fA mutant) had the substrate-
specificity loop disordered in the crystal, so the selectivity engine could
not flag W107f as a divergence target because the residue was simply absent
from the coordinates. A pocket with missing residues in the functionally
relevant region cannot be used for selectivity computation; the gate flags
this BEFORE downstream analysis runs on a silently incomplete pocket.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from physics_auditor.causality.binding_site import extract_binding_site
from physics_auditor.core.parser import STANDARD_RESIDUES, Structure

# Max virtual Cα–Cα step length in a polypeptide (extended/β-strand
# conformation). Used to derive a sequence-distance-aware lower bound on how
# close a missing residue can possibly come to the ligand given the position
# and distance of its nearest resolved sequence neighbor.
PEPTIDE_STEP_ANGSTROMS = 3.8


@dataclass
class MissingResidue:
    """A residue expected by SEQRES but absent from the resolved ATOM records.

    Attributes
    ----------
    chain_id : str
        Chain to which the missing residue belongs.
    res_seq : int
        Implied author residue number, derived from the SEQRES position via
        the per-chain SEQRES-to-ATOM offset.
    res_name : str
        3-letter residue code from SEQRES (or "???" if SEQRES was unavailable
        and the gap was inferred from res_seq numbering alone).
    in_pocket_region : bool
        True if the missing residue can plausibly lie within `cutoff` of the
        ligand given the position and ligand-distance of its nearest resolved
        sequence neighbor. Formally: with the nearest resolved residue at
        distance D from the ligand and k peptide steps away in sequence, the
        residue's distance to the ligand is bounded below by
        max(0, D - k * 3.8 Å). The flag is set when that lower bound is
        within `cutoff`. This catches disordered loops anchored near the
        pocket — including the 4PLZ substrate-specificity loop, whose
        anchor residues sit ~12 Å from the ligand but whose 12-residue span
        physically reaches into the active site when ordered.
    """

    chain_id: str
    res_seq: int
    res_name: str
    in_pocket_region: bool


@dataclass
class PocketCompletenessResult:
    """Verdict on whether the pocket region is fully resolved in the deposit.

    Attributes
    ----------
    verdict : "COMPLETE" or "INCOMPLETE"
        COMPLETE iff no pocket-region residue is missing from the coordinates.
    n_resolved_pocket_residues : int
        Count of protein residues within `cutoff` Å of the ligand that have
        coordinates in the deposit.
    resolved_pocket_residues : list of (chain_id, res_seq, res_name)
        Identifiers of the resolved pocket residues.
    missing_residues : list of MissingResidue
        ALL residues expected by SEQRES but absent from ATOM records (across
        all chains, irrespective of pocket region).
    missing_in_pocket : list of MissingResidue
        Subset of `missing_residues` deemed to lie in the pocket region.
    chain_offsets : dict
        Per-chain integer offset such that SEQRES position (1-indexed) =
        res_seq + offset. Reported for transparency.
    """

    verdict: Literal["COMPLETE", "INCOMPLETE"]
    n_resolved_pocket_residues: int
    resolved_pocket_residues: list[tuple[str, int, str]]
    missing_residues: list[MissingResidue]
    missing_in_pocket: list[MissingResidue]
    chain_offsets: dict[str, int] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return self.verdict == "COMPLETE"

    def summary(self) -> str:
        """One-line human-readable verdict."""
        if self.verdict == "COMPLETE":
            return (
                f"COMPLETE: pocket fully resolved "
                f"({self.n_resolved_pocket_residues} residues, "
                f"no gaps in pocket region)"
            )
        ids = ", ".join(
            f"{m.chain_id}/{m.res_name}{m.res_seq}" for m in self.missing_in_pocket
        )
        return (
            f"INCOMPLETE: {len(self.missing_in_pocket)} pocket-region "
            f"residue(s) missing: {ids}"
        )


def _align_seqres_to_resolved(
    seqres_codes: list[str], resolved: list[tuple[int, str]]
) -> int:
    """Find offset such that SEQRES position (1-indexed) = res_seq + offset.

    The PDB SEQRES record uses construct-internal numbering (1..N), while
    ATOM records use author residue numbering (res_seq) that may differ by
    an offset (e.g. when the construct has a cloning tag or starts from a
    UniProt-canonical position). This finds the integer offset that aligns
    the two by trying every 1-indexed SEQRES position whose residue name
    matches the first resolved residue, and picking the offset that
    maximises the count of matching residues across all resolved positions.

    Parameters
    ----------
    seqres_codes : list of str
        Ordered SEQRES 3-letter codes for the chain.
    resolved : list of (res_seq, res_name)
        Resolved protein residues for the chain, sorted ascending by res_seq.

    Returns
    -------
    int
        Best offset. Returns 0 if alignment cannot be determined (e.g.
        empty inputs or no name matches).
    """
    if not seqres_codes or not resolved:
        return 0

    first_seq, first_name = resolved[0]
    candidates = [i + 1 for i, c in enumerate(seqres_codes) if c == first_name]
    if not candidates:
        return 0

    best_offset = candidates[0] - first_seq
    best_count = -1
    for pos in candidates:
        offset = pos - first_seq
        count = 0
        for rs, nm in resolved:
            idx0 = rs + offset - 1
            if 0 <= idx0 < len(seqres_codes) and seqres_codes[idx0] == nm:
                count += 1
        if count > best_count:
            best_count = count
            best_offset = offset
    return best_offset


def _resolve_ligand_coords(
    structure: Structure,
    ligand_coords: np.ndarray | None,
    ligand_resname: str | None,
) -> np.ndarray:
    """Determine ligand heavy-atom coordinates from explicit coords or a HETATM resname."""
    if ligand_coords is not None:
        coords = np.asarray(ligand_coords, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                "ligand_coords must have shape (L, 3); got " f"{coords.shape}"
            )
        return coords

    if ligand_resname is None:
        raise ValueError(
            "Must supply either ligand_coords or ligand_resname to identify "
            "the ligand defining the pocket."
        )

    ligand_resname = ligand_resname.upper().strip()
    selected = [
        i
        for i, atom in enumerate(structure.atoms)
        if atom.res_name.upper() == ligand_resname and not atom.is_hydrogen
    ]
    if not selected:
        raise ValueError(
            f"No HETATM records found for ligand resname '{ligand_resname}'"
        )
    return structure.coords[selected]


def _min_distance_to_ligand(
    structure: Structure, chain_id: str, res_seq: int, lig_coords: np.ndarray
) -> float:
    """Min distance from any heavy atom of the (chain, res_seq) residue to any ligand atom.

    Returns +inf if the residue is not present in the structure.
    """
    # Find a residue object with matching chain and res_seq (insertion code "")
    for rid, residue in structure.residues.items():
        if rid[0] == chain_id and rid[1] == res_seq and rid[2] == "":
            heavy_coords = np.array(
                [
                    [a.x, a.y, a.z]
                    for a in residue.atoms.values()
                    if not a.is_hydrogen
                ],
                dtype=np.float32,
            )
            if heavy_coords.size == 0:
                return float("inf")
            dists = np.linalg.norm(
                lig_coords[None, :, :] - heavy_coords[:, None, :], axis=-1
            )
            return float(dists.min())
    return float("inf")


def _reach_bound(flank_distance: float, peptide_steps: int) -> float:
    """Lower bound on a missing residue's distance to the ligand.

    Given a resolved sequence-neighbor at `flank_distance` Å from the ligand
    and `peptide_steps` peptide bonds away, the missing residue's distance
    to the ligand cannot be below this value because each peptide step
    advances at most `PEPTIDE_STEP_ANGSTROMS`.
    """
    return max(0.0, flank_distance - peptide_steps * PEPTIDE_STEP_ANGSTROMS)


def pocket_completeness_gate(
    structure: Structure,
    ligand_coords: np.ndarray | None = None,
    cutoff: float = 5.0,
    ligand_resname: str | None = None,
) -> PocketCompletenessResult:
    """Decide whether the pocket region is fully resolved in the deposit.

    A residue expected by SEQRES but absent from ATOM records is classified
    as "in the pocket region" by a sequence-distance-aware lower bound: with
    the nearest resolved sequence neighbor at distance D from the ligand and
    k peptide steps away, the missing residue cannot lie closer than
    max(0, D - k * 3.8 Å). If that lower bound is at or below `cutoff`, the
    residue can plausibly contact the ligand and the gate flags it.

    This criterion correctly handles both narrow cases (a single missing
    residue between two pocket-contacting flanks) and wide ones (a long
    disordered loop anchored several angstroms from the pocket but
    physically able to swing into it — e.g. the PfLDH substrate-
    specificity loop in 4PLZ).

    The gate does NOT silently proceed when INCOMPLETE; the caller decides
    whether to refuse, retry on a different deposit, or merely flag.

    Parameters
    ----------
    structure : Structure
        Parsed structure. Must contain SEQRES records for rigorous gating;
        if SEQRES is absent the gate falls back to detecting res_seq numeric
        gaps within the resolved range (weaker — cannot detect terminal
        truncations).
    ligand_coords : np.ndarray, optional
        (L, 3) heavy-atom coordinates of the bound ligand. Either this or
        `ligand_resname` must be supplied.
    cutoff : float
        Pocket-defining distance in Å (default 5.0). Identical to the
        cutoff used by `extract_binding_site`.
    ligand_resname : str, optional
        3-letter HETATM residue name; if given (and `ligand_coords` is None),
        the gate auto-extracts the ligand's heavy-atom coordinates.

    Returns
    -------
    PocketCompletenessResult
        Verdict, identifiers of resolved pocket residues, and lists of all
        missing residues plus the subset deemed to lie in the pocket region.
    """
    lig_coords = _resolve_ligand_coords(structure, ligand_coords, ligand_resname)

    pocket = extract_binding_site(
        structure, lig_coords, cutoff=cutoff, protein_only=True
    )
    resolved_pocket_list = [
        (r.chain_id, r.res_seq, r.res_name) for r in pocket.residues
    ]

    missing_all: list[MissingResidue] = []
    chain_offsets: dict[str, int] = {}

    def _reach_in_pocket(chain_id: str, missing_rs: int, resolved_seqs: list[int]) -> bool:
        """True iff the missing residue's reach-bound to the ligand is <= cutoff.

        The nearest resolved residue before and after `missing_rs` in sequence
        define two reach bounds; the residue is reachable if either is at or
        below `cutoff`.
        """
        before = [s for s in resolved_seqs if s < missing_rs]
        after = [s for s in resolved_seqs if s > missing_rs]
        bounds: list[float] = []
        if before:
            flank = max(before)
            d = _min_distance_to_ligand(structure, chain_id, flank, lig_coords)
            bounds.append(_reach_bound(d, missing_rs - flank))
        if after:
            flank = min(after)
            d = _min_distance_to_ligand(structure, chain_id, flank, lig_coords)
            bounds.append(_reach_bound(d, flank - missing_rs))
        return any(b <= cutoff for b in bounds)

    if structure.seqres:
        for chain_id, seqres_codes in structure.seqres.items():
            chain = structure.chains.get(chain_id)
            if chain is None:
                continue
            resolved = sorted(
                [(r.res_seq, r.res_name) for r in chain.residues if r.is_protein],
                key=lambda x: x[0],
            )
            if not resolved:
                continue
            offset = _align_seqres_to_resolved(seqres_codes, resolved)
            chain_offsets[chain_id] = offset
            resolved_rs_set = {rs for rs, _ in resolved}
            resolved_seqs_sorted = sorted(resolved_rs_set)
            for pos1 in range(1, len(seqres_codes) + 1):
                expected_name = seqres_codes[pos1 - 1]
                if expected_name not in STANDARD_RESIDUES:
                    continue
                implied_rs = pos1 - offset
                if implied_rs in resolved_rs_set:
                    continue
                in_pocket = _reach_in_pocket(
                    chain_id, implied_rs, resolved_seqs_sorted
                )
                missing_all.append(
                    MissingResidue(
                        chain_id=chain_id,
                        res_seq=implied_rs,
                        res_name=expected_name,
                        in_pocket_region=in_pocket,
                    )
                )
    else:
        for chain_id, chain in structure.chains.items():
            resolved = sorted(
                [(r.res_seq, r.res_name) for r in chain.residues if r.is_protein],
                key=lambda x: x[0],
            )
            if not resolved:
                continue
            resolved_set = {rs for rs, _ in resolved}
            resolved_seqs_sorted = sorted(resolved_set)
            min_rs = resolved[0][0]
            max_rs = resolved[-1][0]
            for rs in range(min_rs, max_rs + 1):
                if rs in resolved_set:
                    continue
                in_pocket = _reach_in_pocket(chain_id, rs, resolved_seqs_sorted)
                missing_all.append(
                    MissingResidue(
                        chain_id=chain_id,
                        res_seq=rs,
                        res_name="???",
                        in_pocket_region=in_pocket,
                    )
                )

    missing_in_pocket = [m for m in missing_all if m.in_pocket_region]
    verdict: Literal["COMPLETE", "INCOMPLETE"] = (
        "INCOMPLETE" if missing_in_pocket else "COMPLETE"
    )

    return PocketCompletenessResult(
        verdict=verdict,
        n_resolved_pocket_residues=len(resolved_pocket_list),
        resolved_pocket_residues=resolved_pocket_list,
        missing_residues=missing_all,
        missing_in_pocket=missing_in_pocket,
        chain_offsets=chain_offsets,
    )
