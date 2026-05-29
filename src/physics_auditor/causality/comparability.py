"""Composable opt-in comparability gate for cross-structure benchmarks.

A pair-level guard: when comparing a target protein to an ortholog protein
(typically parasite vs human), before treating the per-residue selectivity
numbers as meaningful, this gate asks whether the two structures are
*comparable enough* to be put side by side.

Three sub-criteria, each independently opt-in:

  - anchor RMSD ≤ cutoff          (after Kabsch on a shared cofactor)
  - |Δn_pocket_residues| ≤ tol    (pocket-size parity)
  - len(shared_cofactor_atoms) ≥ N (cofactor identity by atom-name overlap)

The gate is **composable by data**: a criterion is enabled iff the caller
provides the data and threshold for it. Passing partial criterion args
(e.g. a threshold without the observed value) is a programming error and
raises ``ValueError``. With no criteria enabled the verdict is
``NOT_ASSESSED`` — the gate refuses to mint a fake PASS.

Design intent
-------------
This module makes the comparability rule explicit and observable as data
rather than only as control flow. The canonical three-criterion form is
``run_w107f_selectivity.py``; the DHODH benchmark scripts have *different*
guard sets (organism provenance + sequence identity) and
``run_selectivity_map.py`` has none. **Unifying the standard — deciding
which comparisons should be gated and to what threshold — is a deferred,
deliberate scientific decision.** This refactor canonicalizes one
extant gate; it does not impose it elsewhere.

Note on the ≥3-shared-atoms rule
--------------------------------
``physics_auditor.causality.superposition.build_anchor_pairing`` raises
``ValueError`` if fewer than 3 atoms can be paired — this is a *geometric*
lower bound for Kabsch (the rotation fit is degenerate below 3 points),
not a comparability threshold. The script-level ≥3-or-more criterion in
this module is **observable as data** even when the library raise never
fires (in w107f's NAI case, 44 atoms pair, so the gate's count is the
operative check).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ComparabilityCriterion:
    """One enabled sub-criterion's threshold, observed value, and verdict.

    Recorded for every criterion the caller enabled, in the order this
    module defines (anchor_rmsd → pocket_parity → shared_atom_names).
    """

    name: Literal["anchor_rmsd", "pocket_parity", "shared_atom_names"]
    threshold: float | int
    observed: float | int
    passed: bool


@dataclass
class ComparabilityResult:
    """Pair-level comparability verdict + per-criterion provenance.

    Attributes
    ----------
    verdict : "PASS" | "FAIL" | "NOT_ASSESSED"
        ``NOT_ASSESSED`` when no criterion was enabled; otherwise PASS iff
        every enabled criterion passed.
    passed : bool
        Mirrors ``verdict == "PASS"``. Convenience for ``if not …`` callers.
    criteria : list of ComparabilityCriterion
        Each enabled criterion's threshold, observed value, and pass flag.
        Disabled criteria are absent (we record only what we measured).
    reason : str
        One-line explanation for ``verdict``.
    shared_atom_names : list of str
        The shared atom names supplied by the caller (recorded for the JSON
        dossier so a downstream reader can see *which* atoms paired, not
        only the count). Empty if the criterion was disabled.
    """

    verdict: Literal["PASS", "FAIL", "NOT_ASSESSED"]
    passed: bool
    criteria: list[ComparabilityCriterion]
    reason: str
    shared_atom_names: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-line human-readable verdict."""
        if self.verdict == "NOT_ASSESSED":
            return "comparability NOT_ASSESSED (no criteria enabled)"
        bits = []
        for c in self.criteria:
            mark = "✓" if c.passed else "✗"
            bits.append(f"{mark} {c.name} (observed={c.observed}, threshold={c.threshold})")
        return f"comparability {self.verdict}: " + "; ".join(bits)

    def to_dict(self) -> dict:
        """JSON-ready dict — mirrors the legacy w107f dossier shape."""
        return {
            "verdict": self.verdict,
            "passed": self.passed,
            "reason": self.reason,
            "criteria": [
                {
                    "name": c.name,
                    "threshold": c.threshold,
                    "observed": c.observed,
                    "passed": c.passed,
                }
                for c in self.criteria
            ],
            "shared_atom_names": list(self.shared_atom_names),
        }


def comparability_gate(
    *,
    anchor_rmsd: float | None = None,
    anchor_rmsd_cutoff: float | None = None,
    target_pocket_size: int | None = None,
    ortholog_pocket_size: int | None = None,
    pocket_parity_tolerance: int | None = None,
    shared_atom_names: list[str] | None = None,
    min_shared_atoms: int | None = None,
) -> ComparabilityResult:
    """Run the comparability gate over the criteria the caller enables.

    A criterion is **enabled** iff the caller passes ALL of its enabling
    args. Mixing partial args (e.g. ``anchor_rmsd`` without
    ``anchor_rmsd_cutoff``) raises ``ValueError`` — a programming error,
    not a user-data error. Passing zero criteria returns ``NOT_ASSESSED``.

    Parameters
    ----------
    anchor_rmsd, anchor_rmsd_cutoff
        Observed Kabsch anchor RMSD (Å) and pass cutoff. Enabled iff both
        non-None; passes iff ``anchor_rmsd <= anchor_rmsd_cutoff``.
    target_pocket_size, ortholog_pocket_size, pocket_parity_tolerance
        Number of pocket residues on each side and the allowed absolute
        difference. Enabled iff all three non-None; passes iff
        ``abs(target_pocket_size - ortholog_pocket_size) <= tolerance``.
    shared_atom_names, min_shared_atoms
        Atom names paired by ``build_anchor_pairing`` and the minimum
        count required. Enabled iff both non-None; passes iff
        ``len(shared_atom_names) >= min_shared_atoms``.

    Returns
    -------
    ComparabilityResult
        Verdict + per-criterion provenance.
    """
    criteria: list[ComparabilityCriterion] = []

    # Criterion 1: anchor RMSD
    _validate_pair("anchor_rmsd", anchor_rmsd, "anchor_rmsd_cutoff", anchor_rmsd_cutoff)
    if anchor_rmsd is not None:
        criteria.append(
            ComparabilityCriterion(
                name="anchor_rmsd",
                threshold=float(anchor_rmsd_cutoff),
                observed=float(anchor_rmsd),
                passed=float(anchor_rmsd) <= float(anchor_rmsd_cutoff),
            )
        )

    # Criterion 2: pocket-size parity
    parity_args = (
        target_pocket_size,
        ortholog_pocket_size,
        pocket_parity_tolerance,
    )
    provided = sum(a is not None for a in parity_args)
    if provided not in (0, 3):
        raise ValueError(
            "Pocket-parity criterion requires ALL of target_pocket_size, "
            "ortholog_pocket_size, pocket_parity_tolerance, or none. "
            f"Got {provided}/3 provided."
        )
    if provided == 3:
        delta = abs(int(target_pocket_size) - int(ortholog_pocket_size))
        criteria.append(
            ComparabilityCriterion(
                name="pocket_parity",
                threshold=int(pocket_parity_tolerance),
                observed=delta,
                passed=delta <= int(pocket_parity_tolerance),
            )
        )

    # Criterion 3: shared atom-name count
    _validate_pair(
        "shared_atom_names", shared_atom_names, "min_shared_atoms", min_shared_atoms
    )
    if shared_atom_names is not None:
        n = len(shared_atom_names)
        criteria.append(
            ComparabilityCriterion(
                name="shared_atom_names",
                threshold=int(min_shared_atoms),
                observed=n,
                passed=n >= int(min_shared_atoms),
            )
        )

    if not criteria:
        return ComparabilityResult(
            verdict="NOT_ASSESSED",
            passed=False,
            criteria=[],
            reason="no comparability criteria were enabled",
            shared_atom_names=list(shared_atom_names) if shared_atom_names else [],
        )

    all_pass = all(c.passed for c in criteria)
    verdict: Literal["PASS", "FAIL"] = "PASS" if all_pass else "FAIL"
    if all_pass:
        reason = (
            f"all {len(criteria)} enabled criteria passed "
            f"({', '.join(c.name for c in criteria)})"
        )
    else:
        failures = [c.name for c in criteria if not c.passed]
        reason = f"failed criteria: {', '.join(failures)}"

    return ComparabilityResult(
        verdict=verdict,
        passed=all_pass,
        criteria=criteria,
        reason=reason,
        shared_atom_names=list(shared_atom_names) if shared_atom_names is not None else [],
    )


def _validate_pair(name_a: str, val_a, name_b: str, val_b) -> None:
    """Raise if exactly one of a paired (observed, threshold) is None.

    Either both must be supplied to enable the criterion, or neither.
    Mixing them is a programming error.
    """
    if (val_a is None) != (val_b is None):
        raise ValueError(
            f"{name_a} and {name_b} must be supplied together (both or neither); "
            f"got {name_a}={val_a!r}, {name_b}={val_b!r}."
        )
