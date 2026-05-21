"""Kabsch structural superposition + cofactor-anchored residue correspondence.

This module is the structure-side peer of correspondence.py. Where
correspondence.py builds residue pairings from a SEQUENCE alignment,
this module builds them from a rigid-body STRUCTURAL alignment seeded
by a shared cofactor whose pose is independent of the protein
correspondence. The two methods are designed to be composable and
diagnostic: agreement between sequence-based pairings (v1) and
cofactor-anchored structural pairings (this module) is the convergent
validation v2 was built to perform.

LOCKED DESIGN CHOICES
---------------------
- Kabsch via numpy.linalg.svd. Closed-form optimum, deterministic.
  Body is pure numpy; signature shape is backend-agnostic so a future
  JAX swap is a body reimplementation behind the same interface (no
  abstraction layer built now).
- weights: np.ndarray | None parameter from the first commit. Uniform
  in v2 (None = uniform). The parameter is the SOCKET for future
  robust / OT-RMSD / sequence-prior variants; weighting logic beyond
  uniform is intentionally NOT implemented here.
- The reflection correction (det(R)<0 -> flip last singular vector
  sign) is the standard Kabsch chiral fix. Without it, near-reflected
  point sets superpose as mirror images, which is geometrically
  optimal in raw RMSD but biologically wrong.
- Diagnostics first-class. PocketSuperposition mirrors the
  PocketAlignment / alignment_provenance convention so the dossier
  block layout is parallel across v1 and v2.

NON-CLAIMS
----------
- Rigid-body only. Domain motion, hinge bending, or loop
  rearrangements between target and ortholog will inflate
  anchor_rmsd and mean_match_distance. A high anchor_rmsd is the
  falsifiable signal that the rigid assumption breaks for this pair
  and the structural correspondence cannot be trusted as-is.
- Spatial nearest-CA correspondence is NOT a full structural
  alignment in the TM-align sense. We pair by post-superposition CA
  proximity within a hard cutoff; we do not optimize a structural
  alignment score, do not iterate, do not weight by SSE membership.
- The shared-cofactor anchor assumes the cofactor occupies the same
  relative pose in both structures (a foundational property of the
  enzyme family, but falsifiable — the assumption is checked via
  anchor_rmsd).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics_auditor.causality.binding_site import BindingSite
from physics_auditor.core.parser import Structure


def kabsch_superpose(
    P: np.ndarray,
    Q: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Optimal rigid-body superposition of P onto Q (SVD Kabsch).

    Finds the rotation ``R`` and translation ``t`` that minimise the
    (weighted) RMSD of ``P @ R.T + t`` vs ``Q``. The reflection case
    (``det(R) < 0``) is corrected by flipping the sign of the last
    singular vector, the standard Kabsch chiral fix.

    Parameters
    ----------
    P, Q : np.ndarray
        Each ``(N, 3)``. Paired point sets — row ``i`` of P is the
        partner of row ``i`` of Q.
    weights : np.ndarray | None
        ``(N,)`` per-point weights. ``None`` (the only value
        supported in v2) means uniform weighting. The parameter is
        the socket for future robust / OT / sequence-prior variants;
        passing a non-None value raises ``NotImplementedError``.

    Returns
    -------
    R : np.ndarray
        ``(3, 3)`` proper rotation (det ≈ +1).
    t : np.ndarray
        ``(3,)`` translation.
    rmsd : float
        Post-superposition root-mean-square deviation between the
        transformed P and Q.
    """
    if weights is not None:
        raise NotImplementedError(
            "kabsch_superpose: non-uniform weights are a v3 socket; "
            "pass weights=None for the v2 uniform-weight Kabsch."
        )

    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(
            f"kabsch_superpose: P and Q must both be (N, 3) and same shape; "
            f"got P.shape={P.shape}, Q.shape={Q.shape}"
        )
    n = P.shape[0]
    if n < 3:
        raise ValueError(
            f"kabsch_superpose: need at least 3 paired points for a "
            f"non-degenerate rotation; got N={n}"
        )

    c_P = P.mean(axis=0)
    c_Q = Q.mean(axis=0)
    P0 = P - c_P
    Q0 = Q - c_Q

    H = P0.T @ Q0
    U, _, Vt = np.linalg.svd(H)

    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    if d == 0.0:
        d = 1.0
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = c_Q - R @ c_P

    transformed = P @ R.T + t
    diff = transformed - Q
    rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

    return R, t, rmsd


def extract_cofactor_atoms(structure: Structure, resname: str) -> dict[str, np.ndarray]:
    """Return ``{atom_name: (3,) coord}`` for HETATM atoms of the named cofactor.

    Iterates ``structure.atoms``, keeps HETATM records whose
    ``res_name`` matches ``resname``. Returns a dict keyed by atom
    name so a future call to ``build_anchor_pairing`` can pair atoms
    across two structures by canonical name (PDB CCD convention).

    Raises
    ------
    ValueError
        If two atoms of the same name are encountered (signals
        multiple cofactor copies in the structure, which would
        require disambiguation before pairing).
    """
    out: dict[str, np.ndarray] = {}
    for atom in structure.atoms:
        if not atom.is_hetatm:
            continue
        if atom.res_name != resname:
            continue
        if atom.name in out:
            raise ValueError(
                f"extract_cofactor_atoms: multiple atoms named "
                f"{atom.name!r} in cofactor {resname!r} of structure "
                f"{structure.name!r}; multiple copies present — "
                f"disambiguate before pairing."
            )
        out[atom.name] = np.array([atom.x, atom.y, atom.z], dtype=np.float64)
    return out


def build_anchor_pairing(
    target_structure: Structure,
    ortholog_structure: Structure,
    anchor_resname: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pair shared-cofactor atoms by canonical atom name.

    This is the sequence-INDEPENDENT seed for the convergent test:
    the pairing is determined entirely by atom-name equality on a
    shared HETATM molecule (whose chemistry is the same in both
    structures by virtue of the PDB Chemical Component Dictionary),
    with no reference to the protein sequence alignment we are
    testing. If the atom-name sets differ, the unmatched atoms are
    reported and only the shared subset is paired.
    """
    target_atoms = extract_cofactor_atoms(target_structure, anchor_resname)
    ortholog_atoms = extract_cofactor_atoms(ortholog_structure, anchor_resname)

    target_names = set(target_atoms.keys())
    ortholog_names = set(ortholog_atoms.keys())
    shared = sorted(target_names & ortholog_names)
    only_target = sorted(target_names - ortholog_names)
    only_ortholog = sorted(ortholog_names - target_names)

    if only_target or only_ortholog:
        print(
            f"[build_anchor_pairing] atom-name mismatch on {anchor_resname!r}: "
            f"only in target ({target_structure.name}): {only_target} | "
            f"only in ortholog ({ortholog_structure.name}): {only_ortholog} | "
            f"pairing {len(shared)} shared atoms"
        )

    if len(shared) < 3:
        raise ValueError(
            f"build_anchor_pairing: need at least 3 shared atoms on "
            f"{anchor_resname!r} for a non-degenerate Kabsch; got "
            f"{len(shared)} (shared={shared})"
        )

    P = np.stack([target_atoms[name] for name in shared], axis=0)
    Q = np.stack([ortholog_atoms[name] for name in shared], axis=0)
    return P, Q, shared


@dataclass
class PocketMatchRecord:
    """Per-pocket-residue diagnostics for the structural correspondence.

    Attributes
    ----------
    target_engine_idx : int
        Engine index of the target-pocket residue (its position in
        ``list(structure.residues.keys())``).
    ortholog_engine_idx : int | None
        Engine index of the matched ortholog residue, or None if the
        nearest CA was further than ``max_match_distance`` (or if no
        ortholog CA exists). ``None`` means the pair was rejected and
        is NOT in ``PocketSuperposition.correspondence``.
    nearest_distance : float
        Post-superposition CA-CA distance to the closest ortholog
        residue, regardless of cutoff. Always populated.
    runner_up_distance : float
        Post-superposition CA-CA distance to the SECOND-closest
        ortholog residue. ``inf`` if only one ortholog residue
        exists. Diagnostic for match unambiguity — if nearest is
        0.9 Å and runner-up is 8 Å the assignment is unambiguous; if
        the two are within a fraction of an Ångström the assignment
        is cutoff-sensitive.
    accepted : bool
        Whether ``nearest_distance <= max_match_distance``. Mirrors
        whether the pair appears in the parent's ``correspondence``.
    """

    target_engine_idx: int
    ortholog_engine_idx: int | None
    nearest_distance: float
    runner_up_distance: float
    accepted: bool


@dataclass
class PocketSuperposition:
    """Result of cofactor-anchored structural correspondence.

    Diagnostics layout mirrors PocketAlignment / alignment_provenance
    so a dossier emits a parallel ``superposition_provenance`` block.

    Attributes
    ----------
    method : str
        Identifier of the superposition method
        (here, ``"kabsch_cofactor_anchored"``).
    anchor_resname : str
        PDB resname of the shared cofactor used to seed the
        superposition (e.g. ``"FMN"``, ``"ORO"``).
    n_anchor_atoms : int
        Number of atom-name pairs used in the Kabsch fit.
    anchor_rmsd : float
        RMSD of the cofactor atoms themselves after superposition.
        This is the seed-quality diagnostic — a low value confirms
        the rigid-body assumption holds at the cofactor; a high
        value falsifies the assumption for this pair and the
        downstream pocket correspondences should not be trusted.
    correspondence : list[tuple[int, int]]
        ``(target_engine_idx, ortholog_engine_idx)`` pairs derived
        from spatial nearest-CA matching after the cofactor-anchored
        transform. Ready to pass as the ``alignment`` argument of
        ``compute_selectivity_map`` — same convention as
        ``PocketAlignment.correspondence``.
    n_pocket_target, n_pocket_ortholog : int
        Pocket sizes on each side.
    n_spatially_matched : int
        Target-pocket residues paired to an ortholog residue within
        ``max_match_distance``.
    max_match_distance : float
        Cutoff (Angstrom) used to accept a nearest-CA match.
    mean_match_distance : float
        Mean post-superposition CA-to-CA distance over the
        spatially-matched pairs (NaN if none matched). Tightness
        diagnostic — close to zero means the pocket-CA frames
        coincide; large values mean even after the cofactor
        superposition the pocket-CA positions diverge.
    n_unmatched_target : int
        Target-pocket residues whose nearest ortholog CA was further
        than ``max_match_distance`` (or which had no CA at all).
    match_details : list[PocketMatchRecord]
        One record per target-pocket residue: nearest- and runner-up
        CA distances, the matched ortholog engine index (or None),
        and an acceptance flag. The same target residue order as
        ``target_pocket.residue_indices``. Provides the per-residue
        unambiguity diagnostic the convergent-validation reviewer
        needs to distinguish rigid-core matches from cutoff-edge
        matches.
    """

    method: str
    anchor_resname: str
    n_anchor_atoms: int
    anchor_rmsd: float
    correspondence: list[tuple[int, int]]
    n_pocket_target: int
    n_pocket_ortholog: int
    n_spatially_matched: int
    max_match_distance: float
    mean_match_distance: float
    n_unmatched_target: int
    match_details: list[PocketMatchRecord]


def _pick_pocket_chain(structure: Structure, pocket: BindingSite) -> str:
    """Pick the chain holding the majority of pocket residues (alpha-tie-break)."""
    counts: dict[str, int] = {}
    for r in pocket.residues:
        if not r.is_protein:
            continue
        counts[r.chain_id] = counts.get(r.chain_id, 0) + 1
    if counts:
        best = max(counts.items(), key=lambda kv: (kv[1], -ord(kv[0][0])))
        return best[0]
    for cid, chain in structure.chains.items():
        if chain.is_protein:
            return cid
    raise ValueError(
        f"structure {structure.name!r} has no protein chain for "
        f"spatial correspondence"
    )


def _collect_protein_ca(
    structure: Structure, chain_id: str
) -> tuple[np.ndarray, list[int]]:
    """Collect CA coords for protein residues on the given chain.

    Returns
    -------
    ca_coords : np.ndarray
        ``(M, 3)`` array of CA coordinates (in structure's own
        frame). Residues whose CA is missing in the parsed PDB are
        skipped.
    engine_indices : list[int]
        For each row of ``ca_coords``, the residue's position in
        ``list(structure.residues.keys())`` — the engine index
        compute_selectivity_map and PocketAlignment use.
    """
    rid_list = list(structure.residues.keys())
    coords_out: list[np.ndarray] = []
    engine_out: list[int] = []
    for engine_idx, rid in enumerate(rid_list):
        residue = structure.residues[rid]
        if residue.chain_id != chain_id:
            continue
        if not residue.is_protein:
            continue
        ca = residue.get_coord("CA")
        if ca is None:
            continue
        coords_out.append(np.asarray(ca, dtype=np.float64))
        engine_out.append(engine_idx)
    if not coords_out:
        return np.empty((0, 3), dtype=np.float64), []
    return np.stack(coords_out, axis=0), engine_out


def superpose_and_correspond(
    target_structure: Structure,
    ortholog_structure: Structure,
    target_pocket: BindingSite,
    ortholog_pocket: BindingSite,
    anchor_resname: str = "FMN",
    match_cutoff: float = 4.0,
    target_chain: str | None = None,
    ortholog_chain: str | None = None,
) -> PocketSuperposition:
    """Build a structural residue correspondence by cofactor-anchored Kabsch.

    Workflow
    --------
    1. Pair the shared-cofactor heavy atoms by canonical atom name
       (sequence-independent). Build (P, Q).
    2. Solve Kabsch on (P, Q) -> (R, t, anchor_rmsd).
    3. Apply (R, t) to every target-chain CA coordinate.
    4. For each TARGET POCKET residue's transformed CA, find the
       nearest ortholog-chain CA. If within ``match_cutoff``, record
       a ``(target_engine_idx, ortholog_engine_idx)`` pair; else
       count as unmatched. (This nearest-CA correspondence is
       independent of the v1 sequence alignment.)
    5. Populate and return ``PocketSuperposition``.
    """
    t_chain = target_chain or _pick_pocket_chain(target_structure, target_pocket)
    o_chain = ortholog_chain or _pick_pocket_chain(ortholog_structure, ortholog_pocket)

    P, Q, shared_atom_names = build_anchor_pairing(
        target_structure, ortholog_structure, anchor_resname
    )
    R, t, anchor_rmsd = kabsch_superpose(P, Q)

    target_ca, target_engine = _collect_protein_ca(target_structure, t_chain)
    ortholog_ca, ortholog_engine = _collect_protein_ca(ortholog_structure, o_chain)
    if target_ca.shape[0] == 0 or ortholog_ca.shape[0] == 0:
        raise ValueError(
            "superpose_and_correspond: empty protein CA set on at least one "
            "side; cannot build a spatial correspondence."
        )

    target_ca_transformed = target_ca @ R.T + t

    target_engine_to_row: dict[int, int] = {
        eng: i for i, eng in enumerate(target_engine)
    }

    correspondence: list[tuple[int, int]] = []
    matched_distances: list[float] = []
    n_unmatched = 0
    n_pocket_t = len(target_pocket.residue_indices)
    match_details: list[PocketMatchRecord] = []

    for t_ri in target_pocket.residue_indices:
        row = target_engine_to_row.get(t_ri)
        if row is None:
            n_unmatched += 1
            match_details.append(PocketMatchRecord(
                target_engine_idx=t_ri,
                ortholog_engine_idx=None,
                nearest_distance=float("inf"),
                runner_up_distance=float("inf"),
                accepted=False,
            ))
            continue
        t_pos = target_ca_transformed[row]
        diffs = ortholog_ca - t_pos
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        order = np.argsort(dists)
        j = int(order[0])
        d_nearest = float(dists[j])
        d_runner = float(dists[order[1]]) if len(order) > 1 else float("inf")
        accepted = d_nearest <= match_cutoff
        if accepted:
            o_ri = ortholog_engine[j]
            correspondence.append((t_ri, o_ri))
            matched_distances.append(d_nearest)
        else:
            o_ri = None
            n_unmatched += 1
        match_details.append(PocketMatchRecord(
            target_engine_idx=t_ri,
            ortholog_engine_idx=o_ri,
            nearest_distance=d_nearest,
            runner_up_distance=d_runner,
            accepted=accepted,
        ))

    mean_dist = float(np.mean(matched_distances)) if matched_distances else float("nan")

    return PocketSuperposition(
        method="kabsch_cofactor_anchored",
        anchor_resname=anchor_resname,
        n_anchor_atoms=len(shared_atom_names),
        anchor_rmsd=anchor_rmsd,
        correspondence=correspondence,
        n_pocket_target=n_pocket_t,
        n_pocket_ortholog=len(ortholog_pocket.residue_indices),
        n_spatially_matched=len(correspondence),
        max_match_distance=match_cutoff,
        mean_match_distance=mean_dist,
        n_unmatched_target=n_unmatched,
        match_details=match_details,
    )
