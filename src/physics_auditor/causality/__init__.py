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
from physics_auditor.causality.selectivity_map import (
    SelectivityMap,
    SelectivityResiduePair,
    compute_selectivity_map,
    find_ligand_atoms_by_resname,
    selectivity_map_to_dict,
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
    # Selectivity map
    "SelectivityMap",
    "SelectivityResiduePair",
    "compute_selectivity_map",
    "find_ligand_atoms_by_resname",
    "selectivity_map_to_dict",
]
