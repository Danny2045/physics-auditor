"""Causality module — the differentiator.

Binding site extraction, per-residue energy decomposition,
selectivity attribution, and local vs global divergence analysis.
"""

from physics_auditor.causality.binding_site import (
    BindingSite,
    PocketComparison,
    compare_binding_sites,
    extract_binding_site,
)
from physics_auditor.causality.energy_decomp import (
    DecompositionDifference,
    ResidueDifference,
    ResidueEnergy,
    StructureDecomposition,
    decomposition_to_dict,
    difference_to_dict,
    per_residue_decomposition,
    per_residue_difference,
)

__all__ = [
    # Binding site
    "BindingSite",
    "PocketComparison",
    "compare_binding_sites",
    "extract_binding_site",
    # Energy decomposition
    "DecompositionDifference",
    "ResidueDifference",
    "ResidueEnergy",
    "StructureDecomposition",
    "decomposition_to_dict",
    "difference_to_dict",
    "per_residue_decomposition",
    "per_residue_difference",
]
