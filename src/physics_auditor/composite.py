"""Honest partial composite scoring.

The composite trust score is computed over ONLY the checks that are currently
implemented AND scored — today that is steric clashes alone. Checks that are
*computed but not scored* (Lennard-Jones energy, Ramachandran dihedrals) and
checks that are *not implemented at all* (the five config-only scaffolds) are
recorded in the provenance so a consumer can never mistake a partial composite
for a full one.

This module is the single source of truth for "what does the composite cover".
The audit/CLI path builds the composite by handing this module the subscores of
the checks that are scored; the module re-derives the partial-ness, the
normalized weights, and the unscored / not-implemented breakdown from the
registry below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from physics_auditor.config import CompositeConfig


class CheckStatus(str, Enum):
    """Implementation status of a check relative to the composite."""

    SCORED = "scored"  # implemented, normalized into the composite
    COMPUTED_UNSCORED = "computed_but_unscored"  # runs, but no subscore exists
    NOT_IMPLEMENTED = "not_implemented"  # config/weights scaffold only


# Canonical registry of the eight checks the composite was *designed* to cover,
# each with its current implementation status and the CompositeConfig weight
# attribute that would govern it once scored. Order is the intended report
# order. This is the authoritative answer to "8 checks?" — exactly one is
# scored today.
CHECK_REGISTRY: dict[str, tuple[CheckStatus, str]] = {
    "steric_clashes": (CheckStatus.SCORED, "weight_clashes"),
    "lennard_jones": (CheckStatus.COMPUTED_UNSCORED, "weight_lj_energy"),
    "ramachandran": (CheckStatus.COMPUTED_UNSCORED, "weight_ramachandran"),
    "bond_geometry": (CheckStatus.NOT_IMPLEMENTED, "weight_bond_geometry"),
    "peptide_planarity": (CheckStatus.NOT_IMPLEMENTED, "weight_peptide_planarity"),
    "chirality": (CheckStatus.NOT_IMPLEMENTED, "weight_chirality"),
    "rotamers": (CheckStatus.NOT_IMPLEMENTED, "weight_rotamers"),
    "disulfides": (CheckStatus.NOT_IMPLEMENTED, "weight_disulfides"),
}

N_TOTAL_CHECKS = len(CHECK_REGISTRY)


def _names_with_status(status: CheckStatus) -> list[str]:
    return [n for n, (s, _) in CHECK_REGISTRY.items() if s is status]


@dataclass
class CheckContribution:
    """One scored check's contribution to the composite.

    Attributes
    ----------
    check : str
        Registry name of the check.
    normalized_weight : float
        The check's CompositeConfig weight renormalized so the weights of all
        *scored* checks sum to 1.0. With a single scored check this is 1.0.
    subscore : float
        The check's own [0, 1] subscore.
    """

    check: str
    normalized_weight: float
    subscore: float


@dataclass
class CompositeProvenance:
    """The composite score together with the truth about what built it.

    A consumer reading this can see "composite over N of 8 checks" and exactly
    which checks were scored, which were computed but not scored, and which are
    not implemented — so a bare float is never mistaken for a full audit.
    """

    score: float | None
    is_partial: bool
    n_scored_checks: int
    n_total_checks: int
    contributing_checks: list[CheckContribution]
    computed_but_unscored: list[str] = field(default_factory=list)
    not_implemented: list[str] = field(default_factory=list)
    note: str = ""

    def summary(self) -> str:
        """One-line human-readable provenance."""
        scored = ", ".join(c.check for c in self.contributing_checks) or "none"
        score_str = "n/a" if self.score is None else f"{self.score:.3f}"
        return (
            f"composite={score_str} over {self.n_scored_checks} of "
            f"{self.n_total_checks} checks ({scored}); "
            f"{len(self.computed_but_unscored)} computed-but-unscored, "
            f"{len(self.not_implemented)} not-implemented"
        )

    def to_dict(self) -> dict:
        """Serializable provenance block for the JSON report."""
        return {
            "score": None if self.score is None else round(self.score, 3),
            "is_partial": self.is_partial,
            "n_scored_checks": self.n_scored_checks,
            "n_total_checks": self.n_total_checks,
            "contributing_checks": [
                {
                    "check": c.check,
                    "normalized_weight": round(c.normalized_weight, 3),
                    "subscore": round(c.subscore, 3),
                }
                for c in self.contributing_checks
            ],
            "computed_but_unscored": list(self.computed_but_unscored),
            "not_implemented": list(self.not_implemented),
            "note": self.note,
        }


def compute_composite(
    subscores: dict[str, float],
    config: CompositeConfig | None = None,
) -> CompositeProvenance:
    """Compute the honest partial composite over the currently-scored checks.

    Parameters
    ----------
    subscores : dict[str, float]
        Maps the name of each *scored* check (status ``SCORED`` in the
        registry) to its [0, 1] subscore. Must contain exactly the scored
        checks — passing a check that is not ``SCORED`` is a programming error
        and raises, because that is precisely the "silently weighting an
        unbuilt check" failure this module exists to prevent.
    config : CompositeConfig, optional
        Source of the per-check weights. Defaults to ``CompositeConfig()``.

    Returns
    -------
    CompositeProvenance
        The composite score (weighted mean of the scored subscores using their
        renormalized config weights) plus the full provenance: partial-ness,
        contributing checks with normalized weights, and the
        computed-but-unscored / not-implemented breakdown.

    Notes
    -----
    With a single scored check (clashes today), the normalized weight is 1.0
    and the composite equals that check's subscore exactly.
    """
    if config is None:
        config = CompositeConfig()

    scored_names = _names_with_status(CheckStatus.SCORED)
    expected = set(scored_names)
    provided = set(subscores)
    if provided != expected:
        raise ValueError(
            "subscores must contain exactly the scored checks "
            f"{sorted(expected)}; got {sorted(provided)}. Only checks with "
            "status SCORED may contribute to the composite."
        )

    raw_weights = {n: getattr(config, CHECK_REGISTRY[n][1]) for n in scored_names}
    total_weight = sum(raw_weights.values())

    contributing: list[CheckContribution] = []
    score = 0.0
    for name in scored_names:
        norm_w = raw_weights[name] / total_weight if total_weight > 0 else 0.0
        ss = float(subscores[name])
        score += norm_w * ss
        contributing.append(
            CheckContribution(check=name, normalized_weight=norm_w, subscore=ss)
        )

    computed_unscored = _names_with_status(CheckStatus.COMPUTED_UNSCORED)
    not_impl = _names_with_status(CheckStatus.NOT_IMPLEMENTED)
    is_partial = len(scored_names) < N_TOTAL_CHECKS

    note = (
        f"Composite computed over {len(scored_names)} of {N_TOTAL_CHECKS} "
        f"intended checks ({', '.join(scored_names)}). "
        f"Computed but not scored: {', '.join(computed_unscored)}. "
        f"Not implemented: {', '.join(not_impl)}. "
        "The score reflects only the scored check(s); it is NOT a full "
        "eight-check audit."
    )

    return CompositeProvenance(
        score=score,
        is_partial=is_partial,
        n_scored_checks=len(scored_names),
        n_total_checks=N_TOTAL_CHECKS,
        contributing_checks=contributing,
        computed_but_unscored=computed_unscored,
        not_implemented=not_impl,
        note=note,
    )
