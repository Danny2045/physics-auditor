"""Library-level structure audit.

This module is the bridge promised by the consolidation brief: typed input
(``StructureAuditRequest``), typed output (``StructureAuditResult``), and a
single ``audit()`` function that runs the entire pipeline. The CLI is now a
thin wrapper that builds a request, calls ``audit()``, and renders the
result.

Readiness gate semantics
------------------------
The three-state readiness design is enforced at the front door:

1. **Structure-level (always-on, ligand-independent)** — every audit runs
   ``structure_completeness_gate`` to detect internal SEQRES-vs-ATOM gaps
   anywhere in the structure. A bare-PDB audit still gets this signal.

2. **Pocket-level (ligand-gated)** — when a ligand is supplied (resname or
   coordinates), ``pocket_completeness_gate`` additionally runs to detect
   internal gaps in the pocket region. Otherwise ``pocket_level`` is
   ``NOT_ASSESSED``.

3. **Composite gating** — the audit is ``ready`` iff every *applicable*
   completeness check is COMPLETE. When unready, the raw per-check results
   (clashes, LJ energy, dihedrals) are still computed and reported — they
   are facts about what *is* resolved — but the composite, the
   ``global_score``, and the recommendation are SET TO ``None``. The gate
   gates the confident verdict, not the raw numbers.

This prevents the W107f failure mode: a confident composite emitted on a
silently incomplete deposit (the 4PLZ substrate-specificity loop was
disordered in the crystal, so the selectivity engine had no way to flag
W107f as a divergence target).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np

from physics_auditor.causality.pocket_completeness import (
    MissingResidue,
    PocketCompletenessResult,
    StructureCompletenessResult,
    pocket_completeness_gate,
    structure_completeness_gate,
)
from physics_auditor.checks.clashes import check_clashes
from physics_auditor.composite import CompositeProvenance, compute_composite
from physics_auditor.config import AuditorConfig
from physics_auditor.core.energy import run_lj_analysis
from physics_auditor.core.geometry import (
    compute_distance_matrix,
    extract_backbone_dihedrals,
)
from physics_auditor.core.parser import parse_pdb
from physics_auditor.core.topology import (
    build_bonded_mask,
    infer_bonds_from_topology,
)


@dataclass
class StructureAuditRequest:
    """Typed input to ``audit()``.

    Attributes
    ----------
    pdb_path : Path
        The structure to audit. Parsed inside ``audit()``.
    config : AuditorConfig
        All thresholds and weights. Defaults to ``AuditorConfig()``.
    ligand_resname : str, optional
        3-letter HETATM residue name. Supplying either this or
        ``ligand_coords`` activates the pocket-completeness check in
        addition to the always-on structure-level check.
    ligand_coords : np.ndarray, optional
        (L, 3) ligand heavy-atom coordinates. Mutually exclusive
        alternative to ``ligand_resname``.
    pocket_cutoff : float
        Pocket-defining distance in Å (default 5.0).
    """

    pdb_path: Path
    config: AuditorConfig = field(default_factory=AuditorConfig)
    ligand_resname: str | None = None
    ligand_coords: np.ndarray | None = None
    pocket_cutoff: float = 5.0


@dataclass
class ReadinessVerdict:
    """Three-state readiness verdict carried in every audit result.

    Attributes
    ----------
    ready : bool
        True iff every *applicable* completeness check is COMPLETE.
        Always-on structure-level check is always applicable; pocket-level
        check is applicable only when a ligand was supplied.
    structure_level : "COMPLETE" | "INCOMPLETE"
        Always assessed.
    pocket_level : "COMPLETE" | "INCOMPLETE" | "NOT_ASSESSED"
        ``NOT_ASSESSED`` when no ligand was supplied.
    structure_internal_gaps : list of MissingResidue
        Internal gaps anywhere in the structure (drives structure-level
        verdict). Each entry's ``in_pocket_region`` is False here — pocket-
        region membership is reported separately under ``pocket_missing``.
    structure_unresolved_termini : list of MissingResidue
        Terminal tails reported informationally; never flip the verdict.
    pocket_missing : list of MissingResidue
        Internal gaps within the pocket region. Empty when no ligand was
        supplied or when pocket_level is COMPLETE.
    reason : str
        One-line human-readable explanation.
    """

    ready: bool
    structure_level: Literal["COMPLETE", "INCOMPLETE"]
    pocket_level: Literal["COMPLETE", "INCOMPLETE", "NOT_ASSESSED"]
    structure_internal_gaps: list[MissingResidue]
    structure_unresolved_termini: list[MissingResidue]
    pocket_missing: list[MissingResidue]
    reason: str

    def to_dict(self) -> dict:
        """Serializable readiness block for the JSON report."""
        return {
            "ready": self.ready,
            "structure_level": self.structure_level,
            "pocket_level": self.pocket_level,
            "structure_internal_gaps": [
                _missing_to_dict(m) for m in self.structure_internal_gaps
            ],
            "structure_unresolved_termini": [
                _missing_to_dict(m) for m in self.structure_unresolved_termini
            ],
            "pocket_missing": [
                _missing_to_dict(m) for m in self.pocket_missing
            ],
            "reason": self.reason,
        }


@dataclass
class StructureAuditResult:
    """Typed output of ``audit()``.

    When ``readiness.ready`` is False the ``composite``, ``global_score``,
    and ``recommendation`` fields are all ``None`` — the gate withholds
    the confident verdict. The raw per-check results in ``checks`` are
    always populated.
    """

    file: str
    readiness: ReadinessVerdict
    composite: CompositeProvenance | None
    global_score: float | None
    recommendation: str | None
    checks: dict
    metadata: dict

    def to_dict(self) -> dict:
        """JSON-ready audit report. Mirrors the CLI's prior dict layout
        with one additive ``readiness`` block; composite / global_score /
        recommendation are emitted as JSON ``null`` when unready.
        """
        return {
            "file": self.file,
            "global_score": (
                None if self.global_score is None else round(self.global_score, 3)
            ),
            "recommendation": self.recommendation,
            "readiness": self.readiness.to_dict(),
            "composite": (
                None if self.composite is None else self.composite.to_dict()
            ),
            "checks": self.checks,
            "metadata": self.metadata,
        }


def audit(request: StructureAuditRequest) -> StructureAuditResult:
    """Run the full structure audit: readiness gate → physics checks → composite.

    Pipeline order (structural, not just report order):

    1. Parse the PDB.
    2. Run ``structure_completeness_gate`` (always).
    3. Run ``pocket_completeness_gate`` iff a ligand is supplied.
    4. Run the raw physics checks (clashes, LJ, dihedrals) — always, even
       when unready, because they are facts about what *is* resolved.
    5. If ready, build the composite + recommendation; otherwise set all
       three to ``None``.

    Returns
    -------
    StructureAuditResult
    """
    t0 = time.time()
    structure = parse_pdb(request.pdb_path)
    cfg = request.config

    # --- Readiness: always-on structure-level check ---
    struct_completeness = structure_completeness_gate(structure)

    # --- Readiness: ligand-gated pocket-completeness check ---
    pocket_completeness: PocketCompletenessResult | None = None
    if request.ligand_coords is not None or request.ligand_resname is not None:
        pocket_completeness = pocket_completeness_gate(
            structure,
            ligand_coords=request.ligand_coords,
            cutoff=request.pocket_cutoff,
            ligand_resname=request.ligand_resname,
        )

    readiness = _build_readiness(struct_completeness, pocket_completeness)

    # --- Physics checks always run (transparency on raw facts) ---
    bonds = infer_bonds_from_topology(structure)
    mask = build_bonded_mask(structure.n_atoms, bonds)
    coords_jnp = jnp.array(structure.coords)
    dist_matrix = compute_distance_matrix(coords_jnp)
    mask_jnp = jnp.array(mask)

    clash_result = check_clashes(
        dist_matrix,
        structure.elements,
        mask_jnp,
        structure.res_indices,
        structure.n_residues,
        cfg.clashes,
    )
    lj_result = run_lj_analysis(
        dist_matrix,
        structure.elements,
        mask_jnp,
        structure.res_indices,
        structure.n_residues,
        cfg.lennard_jones.energy_cap,
    )
    dihedrals = extract_backbone_dihedrals(
        structure.coords,
        structure.atom_names,
        structure.res_indices,
        structure.is_protein_mask,
        structure.chain_ids_array,
    )

    runtime = time.time() - t0

    # --- Composite + recommendation: only when ready ---
    composite_prov: CompositeProvenance | None
    global_score: float | None
    recommendation: str | None
    if readiness.ready:
        composite_prov = compute_composite(
            {"steric_clashes": clash_result.subscore}, cfg.composite
        )
        global_score = composite_prov.score
        if global_score >= cfg.composite.accept_threshold:
            recommendation = "accept"
        elif global_score >= cfg.composite.short_md_threshold:
            recommendation = "short_md"
        else:
            recommendation = "discard"
    else:
        composite_prov = None
        global_score = None
        recommendation = None

    checks = {
        "steric_clashes": {
            "scored": True,
            "n_clashes": clash_result.n_clashes,
            "n_severe": clash_result.n_severe_clashes,
            "clashscore": round(clash_result.clashscore, 2),
            "worst_overlap_angstrom": round(clash_result.worst_overlap, 3),
            "subscore": round(clash_result.subscore, 3),
        },
        "lennard_jones": {
            "scored": False,
            "total_energy_kcal": round(lj_result["total_energy"], 2),
            "n_hot_pairs": lj_result["n_hot_pairs"],
        },
        "backbone_dihedrals": {
            "scored": False,
            "n_phi": len(dihedrals["phi"]["res_indices"]),
            "n_psi": len(dihedrals["psi"]["res_indices"]),
            "n_omega": len(dihedrals["omega"]["res_indices"]),
        },
    }

    metadata = {
        "n_atoms": structure.n_atoms,
        "n_residues": structure.n_residues,
        "n_chains": structure.n_chains,
        "n_bonds_inferred": len(bonds),
        "protein_chains": len(structure.protein_chains),
        "runtime_seconds": round(runtime, 3),
    }

    return StructureAuditResult(
        file=str(request.pdb_path),
        readiness=readiness,
        composite=composite_prov,
        global_score=global_score,
        recommendation=recommendation,
        checks=checks,
        metadata=metadata,
    )


def _build_readiness(
    struct_result: StructureCompletenessResult,
    pocket_result: PocketCompletenessResult | None,
) -> ReadinessVerdict:
    """Build a ReadinessVerdict from the two completeness-gate results."""
    structure_level = struct_result.verdict
    if pocket_result is not None:
        pocket_level: Literal["COMPLETE", "INCOMPLETE", "NOT_ASSESSED"] = (
            pocket_result.verdict
        )
        pocket_missing = list(pocket_result.missing_in_pocket)
    else:
        pocket_level = "NOT_ASSESSED"
        pocket_missing = []

    structure_ok = structure_level == "COMPLETE"
    pocket_ok = pocket_level != "INCOMPLETE"
    ready = structure_ok and pocket_ok

    if ready:
        if pocket_level == "NOT_ASSESSED":
            reason = (
                "structure-level COMPLETE; pocket-level NOT_ASSESSED "
                "(no ligand supplied)"
            )
        else:
            reason = "structure-level COMPLETE; pocket-level COMPLETE"
    else:
        parts: list[str] = []
        if not structure_ok:
            parts.append(
                f"structure-level INCOMPLETE: "
                f"{len(struct_result.internal_gaps)} internal-gap residue(s)"
            )
        if not pocket_ok:
            parts.append(
                f"pocket-level INCOMPLETE: "
                f"{len(pocket_missing)} pocket-region residue(s) missing"
            )
        reason = "; ".join(parts)

    return ReadinessVerdict(
        ready=ready,
        structure_level=structure_level,
        pocket_level=pocket_level,
        structure_internal_gaps=list(struct_result.internal_gaps),
        structure_unresolved_termini=list(struct_result.unresolved_termini),
        pocket_missing=pocket_missing,
        reason=reason,
    )


def _missing_to_dict(m: MissingResidue) -> dict:
    """Serialize a MissingResidue for the JSON report."""
    return {
        "chain_id": m.chain_id,
        "res_seq": m.res_seq,
        "res_name": m.res_name,
        "is_terminal": m.is_terminal,
        "in_pocket_region": m.in_pocket_region,
    }
