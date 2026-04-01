# Physics Auditor — Complete Setup Guide

Every command, in order, from opening your terminal to a working repo on GitHub.

---

## Step 1: Download and extract

Download `physics-auditor.tar.gz` from Claude chat (click the file above).
It will go to your Downloads folder.

```bash
# Move to your projects area (next to ~/kira)
cd ~
tar xzf ~/Downloads/physics-auditor.tar.gz
cd physics-auditor
```

Verify the structure exists:

```bash
ls -la src/physics_auditor/core/
# Should show: __init__.py  energy.py  geometry.py  parser.py  topology.py
```

If the tar was empty or corrupt, see **Appendix A** at the bottom to build from scratch.

---

## Step 2: Create and activate the conda environment

You have two options: use your existing `bio-builder` env (it already has JAX) or create a fresh one.

### Option A: Use bio-builder (recommended — JAX already installed)

```bash
conda activate bio-builder
```

### Option B: Fresh environment

```bash
conda create -n physics-auditor python=3.11 -y
conda activate physics-auditor
```

---

## Step 3: Install the package

From inside the `physics-auditor` directory:

```bash
cd ~/physics-auditor
pip install -e ".[dev]"
```

This installs:
- `jax`, `jaxlib` (computation engine)
- `numpy` (array I/O)
- `typer`, `rich` (CLI)
- `jinja2` (HTML reports — used later)
- `pyyaml` (config overrides)
- `pytest`, `pytest-cov`, `ruff` (dev tools)

If you get any JAX errors on M4 Pro, you may need the Metal backend:

```bash
# Only if jax install fails or you want GPU acceleration on Apple Silicon
pip install jax-metal
```

---

## Step 4: Run the tests

```bash
cd ~/physics-auditor
pytest tests/ -v
```

Expected output — **45 tests, all pass**:

```
tests/test_causality.py::TestBindingSiteExtraction::test_extract_near_origin PASSED
tests/test_causality.py::TestBindingSiteExtraction::test_large_cutoff_gets_all PASSED
tests/test_causality.py::TestBindingSiteExtraction::test_zero_cutoff_gets_none PASSED
tests/test_causality.py::TestBindingSiteExtraction::test_sequence_property PASSED
tests/test_causality.py::TestPocketComparison::test_identical_pockets PASSED
tests/test_causality.py::TestPocketComparison::test_comparison_fields PASSED
tests/test_energy.py::TestLJEnergy::test_lj_params_shape PASSED
tests/test_energy.py::TestLJEnergy::test_lj_energy_finite PASSED
tests/test_energy.py::TestLJEnergy::test_per_residue_sums_to_total PASSED
tests/test_energy.py::TestLJEnergy::test_per_atom_energy_shape PASSED
tests/test_energy.py::TestClashDetection::test_no_clashes_in_good_structure PASSED
tests/test_energy.py::TestClashDetection::test_clashes_in_bad_structure PASSED
tests/test_energy.py::TestClashDetection::test_clash_result_fields PASSED
tests/test_geometry.py::TestDistanceMatrix::test_simple_distances PASSED
... (32 more tests)

============================== 45 passed ==============================
```

If any test fails, paste the full output and we'll fix it.

---

## Step 5: Test the CLI

### Validate the good test structure:

```bash
physics-auditor validate tests/fixtures/tri_ala.pdb
```

Expected:
```
╭─── Physics Auditor ───╮
│ tri_ala  |  Score: 1.000  |  Recommendation: ACCEPT
╰───────────────────────╯
  Steric Clashes: 0 clashes (0 severe), clashscore=0.0  →  1.000
  Lennard-Jones:  E=-1.9 kcal/mol, 0 hot pairs
```

### Validate the bad test structure:

```bash
physics-auditor validate tests/fixtures/clashing.pdb
```

Expected:
```
╭─── Physics Auditor ───╮
│ clashing  |  Score: 0.000  |  Recommendation: DISCARD
╰───────────────────────╯
  Steric Clashes: 4 clashes (4 severe), clashscore=400.0  →  0.000
  Lennard-Jones:  E=3999.3 kcal/mol, 4 hot pairs
```

### JSON output (for pipelines):

```bash
physics-auditor validate tests/fixtures/tri_ala.pdb --json
```

### Batch mode with saved reports:

```bash
physics-auditor validate tests/fixtures/*.pdb -o reports/
ls reports/
# Should show: tri_ala_report.json  clashing_report.json
```

### Structure info:

```bash
physics-auditor info tests/fixtures/tri_ala.pdb
```

---

## Step 6: Test on a real PDB structure

Download a real structure to validate:

```bash
# Download a high-resolution crystal structure (lysozyme)
curl -o /tmp/1aki.pdb "https://files.rcsb.org/download/1AKI.pdb"
physics-auditor validate /tmp/1aki.pdb
```

Or use a structure from your Kira data:

```bash
# If you have PDB files from docking
physics-auditor validate ~/kira/data/docking/2X99.pdb
```

---

## Step 7: Initialize Git and push to GitHub

```bash
cd ~/physics-auditor

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.egg-info/
.pytest_cache/
dist/
build/
*.pyc
.ruff_cache/
reports/
EOF

# Initialize and commit
git init
git add .
git commit -m "Initial commit: Physics Auditor v0.1.0

- PDB parser with flat numpy arrays for JAX computation
- Topology builder with template + distance bond inference
- JIT-compiled JAX geometry kernels (distances, dihedrals, angles)
- Lennard-Jones energy with per-residue decomposition
- Steric clash detection with severity grading
- Binding site extraction and pairwise comparison (causality layer)
- CLI with rich output, JSON mode, batch processing
- 45 tests passing
- YAML-overridable configuration with frozen dataclasses"

# Create GitHub repo and push
# (gh CLI is already authenticated as Danny2045)
gh repo create physics-auditor --public --source=. --push --description "Physics validation and mechanistic explanation for AI-generated protein structures"
```

After this, your repo is live at: `https://github.com/Danny2045/physics-auditor`

---

## Step 8: Configure Claude Code for the project

```bash
cd ~/physics-auditor

# CLAUDE.md is already in the repo — verify it's there
cat CLAUDE.md
```

Now you can use Claude Code from this directory:

```bash
claude  # Opens Claude Code in the physics-auditor context
```

---

## What's built (working now)

| Module | Status | What it does |
|--------|--------|--------------|
| `core/parser.py` | ✅ Done | PDB → Structure with coords, masks, residues |
| `core/topology.py` | ✅ Done | Bond inference, bonded mask, vdW radii |
| `core/geometry.py` | ✅ Done | JIT distance matrix, dihedrals, bond angles |
| `core/energy.py` | ✅ Done | LJ kernel, per-atom + per-residue decomposition |
| `checks/clashes.py` | ✅ Done | Steric clash detection + scoring |
| `causality/binding_site.py` | ✅ Done | Pocket extraction + pairwise comparison |
| `cli.py` | ✅ Done | Full CLI with validate, info commands |
| `config.py` | ✅ Done | All thresholds, YAML override |

## What's next (build in upcoming sessions)

| Module | What it does |
|--------|--------------|
| `checks/ramachandran.py` | Backbone φ/ψ validation |
| `checks/peptide_planarity.py` | ω angle deviation from trans |
| `checks/chirality.py` | L-amino acid verification |
| `checks/rotamers.py` | Sidechain χ angle validation |
| `checks/bond_geometry.py` | Bond length/angle vs Engh-Huber |
| `checks/disulfides.py` | S-S bond geometry |
| `checks/composite.py` | Weighted combination → global score |
| `causality/selectivity_map.py` | Per-residue selectivity attribution |
| `causality/energy_decomp.py` | Binding-site energy decomposition |
| `causality/divergence.py` | Local vs global similarity metrics |
| `report/html_report.py` | Visual Trust Report dashboard |
| `benchmarks/` | Calibration on real structures |

---

## Appendix A: Build from scratch (if tar didn't work)

If the tar file was empty or corrupt, clone the structure manually:

```bash
cd ~
mkdir -p physics-auditor/{src/physics_auditor/{core,checks,causality,report,reference},tests/fixtures,benchmarks/scripts,docs}
cd physics-auditor
```

Then paste each file from the Claude chat conversation into the right location.
The files are (in order of creation):

1. `pyproject.toml`
2. `CLAUDE.md`
3. `README.md`
4. `src/physics_auditor/__init__.py`
5. `src/physics_auditor/config.py`
6. `src/physics_auditor/core/__init__.py`
7. `src/physics_auditor/core/parser.py`
8. `src/physics_auditor/core/topology.py`
9. `src/physics_auditor/core/geometry.py`
10. `src/physics_auditor/core/energy.py`
11. `src/physics_auditor/checks/__init__.py`
12. `src/physics_auditor/checks/clashes.py`
13. `src/physics_auditor/causality/__init__.py`
14. `src/physics_auditor/causality/binding_site.py`
15. `src/physics_auditor/causality/selectivity_map.py` (stub)
16. `src/physics_auditor/causality/energy_decomp.py` (stub)
17. `src/physics_auditor/causality/divergence.py` (stub)
18. `src/physics_auditor/report/__init__.py`
19. `src/physics_auditor/reference/__init__.py`
20. `src/physics_auditor/cli.py`
21. `tests/fixtures/tri_ala.pdb`
22. `tests/fixtures/clashing.pdb`
23. `tests/test_parser.py`
24. `tests/test_topology.py`
25. `tests/test_geometry.py`
26. `tests/test_energy.py`
27. `tests/test_causality.py`

Then continue from **Step 3** above.
