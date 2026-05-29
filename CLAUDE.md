# Physics Auditor

Physics validation for AI-generated protein structures. JAX-based, CLI. One
physics check is scored today (steric clashes); the causality engine is the
differentiator. See "Score honesty" below before trusting the composite.

## Key commands
- `pip install -e ".[dev]"` to install
- `pytest tests/ -v` to test (~155 tests, all must pass)
- `ruff check src/ tests/ --select E,F,I,W --ignore E501` must pass before commit
- `physics-auditor validate <file.pdb>` to validate a structure

## Architecture
- `src/physics_auditor/core/` — parser, geometry, topology, energy (JAX kernels)
- `src/physics_auditor/checks/` — clash detection
- `src/physics_auditor/causality/` — binding site extraction, pocket comparison,
  pocket + structure completeness gates
- `benchmark/` — experimental vs AI-predicted structure comparison
- `src/physics_auditor/composite.py` — single source of truth for what the
  composite covers (the check registry + honest provenance)
- `src/physics_auditor/audit.py` — library-level entry point: typed
  `StructureAuditRequest` / `StructureAuditResult`, single `audit()` function,
  three-state readiness gate that withholds the composite when unready
- `src/physics_auditor/cli.py` — thin wrapper over `audit()`

## Score honesty
The composite `global_score` is **partial**: it is computed over the one check
that is currently scored — steric clashes — and equals that check's subscore.
The audit report carries a `composite` provenance block stating this (`is_partial`,
`n_scored_checks` of `n_total_checks`, contributing checks, and the breakdown
below). `compute_composite` in `composite.py` raises if anything but a scored
check is fed in, so no unbuilt check is ever silently weighted.

The eight-check registry's current status:
- **Scored (1):** steric clashes.
- **Computed but NOT scored (2):** Lennard-Jones energy and Ramachandran
  backbone dihedrals — both run and are reported, but neither has a subscore,
  so neither is folded into the composite (wiring LJ in would require inventing
  a normalization rubric; deferred, not faked).
- **Not implemented (5):** bond_geometry, peptide_planarity, chirality,
  rotamers, disulfides — config weights only, no implementing code. Do not
  implement these as a side effect of other work.

## Readiness gate
Every audit runs a readiness gate at the front door, BEFORE the composite is
built. Two checks, three states:
- **Structure-level (always-on, ligand-independent):**
  `structure_completeness_gate` walks SEQRES vs ATOM in every chain and flags
  any INTERNAL gap (resolved neighbors on both sequence sides). Terminal tails
  are reported informationally and never trigger INCOMPLETE.
- **Pocket-level (ligand-gated):** `pocket_completeness_gate` runs only when
  a ligand is supplied (resname or coords). State is `NOT_ASSESSED` otherwise.

The audit is `ready` iff every *applicable* check is COMPLETE. When unready,
`composite`, `global_score`, and `recommendation` are all set to `None` —
the gate gates the confident verdict, not the raw facts. Raw per-check results
(clashes, LJ energy, dihedrals) are still computed and reported.

Both gates share `_enumerate_missing_residues` as their single source of truth
for SEQRES-vs-ATOM gap detection; the pocket gate layers a ligand-dependent
reach-bound (peptide step length 3.8 Å) on top to decide pocket-region
membership. This caught the W107f failure mode (4PLZ's disordered substrate-
specificity loop, residues 82-93) on a bare-PDB audit with no ligand supplied.

## Current benchmark finding
AlphaFold predictions score 4-11x better than experimental crystal structures on
whole-protein clashscore, but the advantage shrinks 29-46% at the binding site.
This means standard physics metrics cannot serve as quality filters for drug design.

## Active work: binding-site clashscore benchmark
- benchmark/benchmark_pocket_clashscore.py runs the comparison
- Known issue: 4HJO (HsHDAC8) uses PDB numbering starting at residue 679, AF uses UniProt numbering starting at 1. Offset is 678. Fix needed in extract_pocket_by_residue_numbers to auto-detect or manually apply offset.
- Next checks to add: rotamer validation, per-residue RMSD comparison

## Code style
- Type hints everywhere
- Docstrings on public functions (NumPy style)
- No magic numbers — all thresholds in config.py
- JAX arrays for computation, NumPy for I/O boundaries
- Prefer functional style; classes only for data containers

## Rules
- Every commit must pass ruff and pytest
- Never overclaim — state what the evidence supports
- PDB files use fixed-width columns (not space-delimited)
- Parser stops at first ENDMDL (NMR structures have multiple models)
