# Physics Auditor

Physics validation for AI-generated protein structures. JAX-based, 8 checks, CLI.

## Key commands
- `pip install -e ".[dev]"` to install
- `pytest tests/ -v` to test (45 tests, all must pass)
- `ruff check src/ tests/ --select E,F,I,W --ignore E501` must pass before commit
- `physics-auditor validate <file.pdb>` to validate a structure

## Architecture
- `src/physics_auditor/core/` — parser, geometry, topology, energy (JAX kernels)
- `src/physics_auditor/checks/` — clash detection
- `src/physics_auditor/causality/` — binding site extraction, pocket comparison
- `benchmark/` — experimental vs AI-predicted structure comparison

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
