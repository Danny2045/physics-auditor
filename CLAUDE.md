# Physics Auditor — Project Context

## What This Is
An open-source physics validation and mechanistic explanation tool for AI-generated protein structures. Takes PDB/mmCIF files from Boltz-2, AlphaFold 3, Chai-1, RFdiffusion3, Protenix, or any structure predictor and produces a Trust Report with physics checks + causality analysis.

## Strategic Position
IsoDDE (Isomorphic Labs, Feb 2026) has internal physics violation filtering. The open ecosystem (Boltz-2, Chai-1, Protenix, OpenFold3) does not. This tool fills that gap AND adds a mechanistic causality layer that no one — including IsoDDE — provides: per-residue energy decomposition, binding-site divergence analysis, and selectivity attribution maps.

## Architecture
- `src/physics_auditor/core/` — PDB parser, topology builder, JAX geometry/energy kernels
- `src/physics_auditor/checks/` — 8 physics checks (clashes, bond geometry, ramachandran, peptide planarity, chirality, rotamers, LJ energy, disulfides) + composite scorer
- `src/physics_auditor/causality/` — THE DIFFERENTIATOR. Binding site extraction, per-residue energy decomposition, selectivity attribution maps, local vs global divergence
- `src/physics_auditor/report/` — JSON + HTML Trust Report generation
- `src/physics_auditor/reference/` — Engh-Huber bond params, Ramachandran distributions, rotamer library, LJ parameters

## Key Design Decisions
- All geometry/energy computations in JAX with @jax.jit for batch throughput
- Distance matrix computed once, reused across all checks
- Bonded pair masks precomputed from topology (static per structure)
- Atom type parameters from AMBER ff14SB
- For proteins up to ~1000 residues, full N×N distance matrix; neighbor lists for larger
- CLI via typer, rich for terminal output

## Testing
- `tests/fixtures/` contains known-good (high-res X-ray) and known-bad (perturbed) PDB files
- Every check module has unit tests against these fixtures
- Benchmark suite in `benchmarks/` for systematic evaluation

## Dependencies
JAX (core compute), NumPy, Typer (CLI), Rich (terminal), Jinja2 (HTML reports), PyYAML (config)

## Build Order
1. PDB parser → internal Atoms representation
2. Topology → bond graph, atom typing, mask generation
3. JAX geometry kernels → distances, dihedrals, angles
4. LJ energy kernel
5. Individual physics checks (one at a time, with tests)
6. Causality module — binding site extraction, energy decomposition, selectivity maps
7. Composite score + Trust Report
8. CLI
9. Benchmark on calibration set
10. Polish, README, package

## Code Style
- Type hints everywhere
- Docstrings on public functions (NumPy style)
- No magic numbers — all thresholds in config.py with clear names
- JAX arrays for computation, NumPy for I/O boundaries
- Prefer functional style; classes only for data containers
