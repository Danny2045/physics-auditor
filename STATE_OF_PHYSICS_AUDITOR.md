# State of Physics Auditor

**Snapshot anchor commit:** `96eac03` — *manuscript: update author name to Daniel Ngabonziza*
**Snapshot date:** 2026-05-17
**Repository:** [github.com/Danny2045/physics-auditor](https://github.com/Danny2045/physics-auditor)

---

## How to use this document

This file is the canonical state-of-repository reference for Physics Auditor. It exists so that any reader — human collaborator, future Claude window, peer reviewer — can come up to speed on what the repository currently is, what is reproducible from committed state, what claims are supported by the code as it stands at the snapshot commit, and what work is outstanding, **without reconstructing that picture from chat history, PR descriptions, or speculative readings of the code**.

Specific guidance:

- **Read this before starting any substantial work** on Physics Auditor. The Outstanding Work Register (§9) is the authoritative list of open findings; if you are about to open a PR, check there first to confirm your work is scoped to a real open finding and is not already partially done on an existing branch.
- **This document is descriptive, not normative.** The [README](README.md) is the marketing-and-onboarding surface. [`CLAUDE.md`](CLAUDE.md) carries the four-line "Rules" block that is the closest thing this repository currently has to governance. The PfDHODH dossier's inline `claims` field (`benchmark/results/selectivity_maps/PfDHODH_vs_HsDHODH_selectivity.json` on `feat/pfdhodh-selectivity`) and the prose document next to it (`PFDHODH_SELECTIVITY_FINDINGS.md`) are the closest analogs to per-artifact provenance discipline. This file is descriptive: it tells you what *is*, not what *ought to be*.
- **This document is updated periodically, not continuously.** Every line carries an implicit "as of commit `96eac03`". Branch lists, missing-file tables, finding-status grids, and per-artifact reproducibility lines can drift between snapshots. When you make a substantive structural change to the repository, update this file as part of the same PR; when you only fix code without changing the structure of what's reproducible, you may leave it for the next snapshot. The "Update protocol" section at the end (§11) describes the cadence in more detail.
- **Do not introduce new scientific claims or new metrics in this document.** If a number is not already documented in committed state (a result artifact, a findings doc, the manuscript), it does not belong here. This file is a faithful summary; it is not a place to publish.
- **The Outstanding Work Register pairs every open finding with a "Natural next-PR scope" note.** Use those notes as starting points; refine them when you actually open the PR.
- **Paired with Kira.** Physics Auditor is the structure-validation half of the platform Kira anchors. The corresponding state document lives at [`STATE_OF_KIRA.md`](https://github.com/Danny2045/Kira/blob/main/STATE_OF_KIRA.md) in the Kira repository. See §12.

---

## 1. Repository overview

Physics Auditor is a JAX-based structure-validation toolkit for AI-generated protein-ligand complexes. It performs steric-clash detection and Lennard-Jones energy decomposition on parsed PDB structures, and on the headline branch it additionally performs per-residue selectivity attribution between a target protein and its ortholog.

The repository contains, at the snapshot commit on `master`:

- an **active engine** under `src/physics_auditor/` (17 Python files across 6 sub-packages, three of which are stubs at the snapshot commit),
- a **CLI** (`physics-auditor validate <file.pdb>` and `physics-auditor info <file.pdb>`) exposed via `pyproject.toml` `[project.scripts]`,
- a **benchmark suite** under `benchmark/` (35 files, ~20.9 MB) — 33 PDB structures (18 experimental + 15 AlphaFold v6 predictions), one benchmark script (`benchmark_pocket_clashscore.py`), one result JSON (`pocket_clashscore_comparison.json`),
- a **test suite** of 45 collected tests under `tests/` (all 45 pass; ruff clean),
- a **manuscript** under `manuscript/` (`preprint.md` + `preprint.pdf`) reporting the 14-pair binding-site clashscore benchmark,
- and a small set of **root files** (README, SETUP.md, CLAUDE.md, pyproject.toml, .gitignore, .github/workflows/ci.yml).

There is **no `docs/` directory** at the snapshot commit. Governance documents that exist in the sister repository (Kira's Scientific Constitution, Claim Inflation Audit, Disposition Plan) have no Physics Auditor counterpart yet — see Finding 5.

The **headline scientific result of the project — per-residue Lennard-Jones selectivity attribution recovering the published PfDHODH selectivity cluster (HIS185, PHE188, ARG265) blind — lives on `feat/pfdhodh-selectivity` (PR #1, open since 2026-05-12), not on master.** §13 catalogs that work in full. This state document anchors to master and notes throughout where the unmerged branch carries additional content.

The full module-level map is in §5 (Methodology map).

---

## 2. Quick-reference dashboard

| Property | Value |
|---|---|
| Snapshot commit | `96eac03` |
| Snapshot date | 2026-05-17 |
| Main branch | `master` |
| Tests collected (master) | 45 |
| Tests passing (master) | 45 |
| Tests skipped (master) | 0 |
| Tests failing (master) | 0 |
| Tests on `feat/pfdhodh-selectivity` (PR #1) | 92 passed, 1 skipped |
| Committed files (tracked) on master | 67 |
| Committed size on master | ~21.0 MB (overwhelmingly committed PDB structures under `benchmark/`) |
| Active engine modules under `src/physics_auditor/` | 17 (`.py`) — of which 3 (`causality/divergence.py`, `causality/energy_decomp.py`, `causality/selectivity_map.py`) are 10-line stubs |
| Implemented IsoDDE-style validation checks | 2 of 8 (steric clashes, Lennard-Jones energy) |
| Benchmark PDB structures (master) | 33 (18 experimental + 15 AlphaFold v6 predicted) |
| Provenance sidecars (Kira-style `*_provenance.json`) | 0 |
| Inline-`claims` artifacts | 2 on `feat/pfdhodh-selectivity` (the PfDHODH selectivity dossier and `divergence/summary.json`); 0 on master |
| Merged PRs on master | 0 |
| Open PRs | 1 (PR #1, `feat/pfdhodh-selectivity`) |
| Merge commits on master | 0 |
| Open findings registered by this document | 7 (all new; this is the first audit) |
| Governance documents (Scientific Constitution / Claim Inflation Audit / Disposition Plan) | 0 |
| CI matrix | Python 3.11, 3.13 on Ubuntu (`.github/workflows/ci.yml`) |

---

## 3. Git history and branch state

### 3.1 Remote configuration

- `origin` → `https://github.com/Danny2045/physics-auditor.git` (fetch + push)
- Refspec: `+refs/heads/*:refs/remotes/origin/*` (standard)

### 3.2 Full commit history on `master`

Master has 15 commits, all linear, **zero merge commits**. The full history is small enough to list:

```
96eac03 manuscript: update author name to Daniel Ngabonziza
f669792 manuscript: generate PDF from preprint markdown via pandoc
c5d3173 manuscript: fix version references and verify data consistency
cf392f8 manuscript: add Physics Auditor benchmark preprint draft v0.1
c752aed benchmark: expand to 14 pairs across major drug target families
4b0923a benchmark: expand to 9 pairs across diverse enzyme families
a574218 benchmark: expand to 5 pairs with ubiquitin and LmPTR1
81d644a benchmark: fix HDAC8 residue-numbering offset for AF pocket extraction
e89486a docs: update CLAUDE.md with current benchmark context and active work
fd9d2a2 benchmark: add experimental and AlphaFold structures with initial validation results
a65d454 fix: stop parsing after first MODEL/ENDMDL to handle NMR ensembles
2f26c9b docs: fix install command to use git+https (not on PyPI yet)
46aeb70 chore: fix ruff lint (sort imports, remove unused)
680e1d8 ci: add GitHub Actions workflow for lint and tests
88c2ea9 Initial commit: Physics Auditor v0.1.0
```

### 3.3 Merged PRs on master

**Zero merged PRs.** Every commit on master is a direct push, not a merge. The repository has never used a PR workflow on master at the snapshot commit.

### 3.4 Outstanding branches

The following branches exist at the snapshot commit. **No branch is deleted by the PR that introduces this document.** Cleanup is left to deliberate follow-up PRs so that a human can confirm each deletion.

#### Stack of unmerged feature branches (PR #1)

The four `feat/*` branches form a **strict linear stack ending at `feat/pfdhodh-selectivity`**. The longer branches subsume the shorter ones; the shortest contains the foundation of the causality layer, and each successive branch adds another increment of scope on top.

| Branch | Head commit | Ahead of master | Purpose | Pushed to origin? |
|---|---|---:|---|---|
| `feat/causality-energy-decomp` | `1e2d7a6` | 4 | First end-to-end implementation of the per-residue LJ energy decomposition. Also restores `tests/test_regressions.py` and fixes a parser ENDMDL handling bug in `parse_pdb_string` (parity with the master-side `parse_pdb` fix at `a65d454`). | Yes |
| `feat/selectivity-map` | `e2d23f6` | 6 | Adds the selectivity-attribution map (per-residue compound preference); adds `.claude/` to `.gitignore`. | Yes |
| `feat/divergence` | `a19a2a2` | 10 | Adds `causality/divergence.py` (ESM-2 full-sequence vs pocket cosine via fair-esm), the `run_divergence.py` benchmark, and the **mislabel-correction commit `a19a2a2`** that relabeled the original 1D3H/1MVS selectivity run as HsDHODH vs HsDHFR (two different human enzymes), not Sm/Hs. | Yes |
| `feat/pfdhodh-selectivity` | `73bddb1` | 13 | **PR #1 head.** Adds `utils/pdb_provenance.py` (the SOURCE-record gate that exists in response to the mislabel incident), commits the PfDHODH structure `4ORM.pdb`, and runs the chemotype-matched PfDHODH-vs-HsDHODH selectivity attribution end-to-end. | Yes |

PR #1 is **+68,135 / −14 lines, 36 files**, opened 2026-05-12. It is the only PR on the repository. Mergeability: **MERGEABLE** at the snapshot commit (the audit branch was created from master after this PR was opened; nothing on master has touched any file in the PR since).

When PR #1 merges, the three shorter `feat/*` branches above (`feat/causality-energy-decomp`, `feat/selectivity-map`, `feat/divergence`) **become deletable** — they are strict prefixes of `feat/pfdhodh-selectivity` and carry no unique work. Cleanup belongs in a follow-up PR.

#### Pushed bug-fix branch (Finding 4)

| Branch | Head commit | Ahead of master | Purpose | Pushed to origin? |
|---|---|---:|---|---|
| `fixes/technical-pass-1` | `dd9d085` | 2 | Clash subscore recalibration (linear `1 − clashscore/40` → sigmoidal `1 / (1 + (clashscore/50)^2)` with `accept=0.80`, `short_md=0.50`), O(N) rewrite of `extract_binding_site`, `parse_pdb_string` ENDMDL fix (independently reapplied on the feat stack), LJ subscore wired into the CLI composite via renormalized weights, `core/energy.py` docstring fix (`n_hot_pairs` not `n_clashing_pairs`), and a 135-line regression test suite. | **Yes — pushed by Phase 2 of the audit producing this document.** Was local-only when the audit began; pushing eliminated the data-loss risk. |

This branch needs to be rebased against current master and opened as a PR — likely conflicting in `binding_site.py` against the causality-layer changes on the feat stack, and in `cli.py` against the composite wiring. See Finding 4.

#### Branch introduced by this document

| Branch | Head commit | Ahead of master | Purpose |
|---|---|---:|---|
| `audit/state-of-physics-auditor-document` | (this document's commit) | 1 | Commits this state-of-repository reference. |

### 3.5 Submodules

None. The repository is self-contained.

---

## 4. File tree inventory

The repository has **67 committed files** totaling **~21.0 MB** at the snapshot commit on `master`. The size is dominated by committed PDB structures (~20.5 MB across 33 files under `benchmark/structures/`); the active source tree is small.

### 4.1 Top-level directory breakdown

| Top-level dir | Files | Committed bytes |
|---|---:|---:|
| `benchmark/` | 35 | 20,862,086 |
| `src/` | 17 | 68,303 |
| `tests/` | 7 | 25,344 |
| `manuscript/` | 2 | 76,432 |
| `.github/` | 1 | 505 |
| Root files (`README.md`, `SETUP.md`, `CLAUDE.md`, `pyproject.toml`, `.gitignore`) | 5 | 16,132 |

### 4.2 `benchmark/` sub-breakdown

| Sub-directory | Files | Purpose |
|---|---:|---|
| `benchmark/` (root) | 1 | `benchmark_pocket_clashscore.py` (14-pair clashscore script) |
| `benchmark/results/` | 1 | `pocket_clashscore_comparison.json` (14 entries) |
| `benchmark/structures/experimental/` | 18 | Crystal PDBs: 1D3H, 1HSG, 1HWL, 1M17, 1MVS, 1UBQ, 1X70, 2BPR, 2HYY, 2QWF, 2X99, 3EQM, 3LN1, 3NJW, 3OG7, 4EY7, 4HJO, 5R82 |
| `benchmark/structures/predicted/` | 15 | AlphaFold Database v6 predictions, one per benchmark pair |
| `benchmark/structures/predicted/boltz2/` | 0 | Empty placeholder for future Boltz-2 predictions |
| `benchmark/structures/predicted/chai1/` | 0 | Empty placeholder for future Chai-1 predictions |

Three of the 18 experimental PDBs (1HSG, 2QWF, 3NJW) are committed but not referenced by the 14-pair benchmark script. They are present for ad-hoc use; no committed code reads them.

The 14-pair benchmark pairs the following experimental ↔ AlphaFold structures (in benchmark order):

```
HsDHODH      1MVS  ↔ AF-Q02127  (Q02127)
SmDHODH      1D3H  ↔ AF-G4VFD7  (G4VFD7)   ⚠ mislabeled — see Finding 1
HsHDAC8      4HJO  ↔ AF-Q9BY41  (Q9BY41)   offset +678
Ubiquitin    1UBQ  ↔ AF-P0CG48  (P0CG48)
LmPTR1       2BPR  ↔ AF-Q01782  (Q01782)   offset +380
HsABL1       2HYY  ↔ AF-P00519  (P00519)
HsAromatase  3EQM  ↔ AF-P11511  (P11511)
HsBRAF       3OG7  ↔ AF-P15056  (P15056)
CoV2Mpro     5R82  ↔ AF-P0DTD1  (P0DTD1)   offset +3
HsAChE       4EY7  ↔ AF-P22303  (P22303)   offset -31
MmCOX2       3LN1  ↔ AF-Q05769  (Q05769)
HsEGFR       1M17  ↔ AF-P00533  (P00533)   offset -24
HsHMGCR      1HWL  ↔ AF-P04035  (P04035)
HsDPP4       1X70  ↔ AF-P27487  (P27487)
```

### 4.3 `src/physics_auditor/` map

```
src/physics_auditor/
├── __init__.py           (version marker only)
├── cli.py                (typer CLI — validate, info; composite uses clash subscore only)
├── config.py             (9 frozen-dataclass configs; YAML override loader)
├── core/
│   ├── __init__.py
│   ├── parser.py         (PDB parser → flat numpy arrays; stops at first ENDMDL)
│   ├── topology.py       (bond inference + bonded-pair mask + vdW radii)
│   ├── geometry.py       (JAX distance matrix, dihedrals, bond angles, backbone φ/ψ/ω)
│   └── energy.py         (Lennard-Jones kernel: total, per-atom, per-residue)
├── checks/
│   ├── __init__.py
│   └── clashes.py        (steric clash detection + ClashResult dataclass + clashscore)
├── causality/
│   ├── __init__.py
│   ├── binding_site.py   (pocket extraction + pairwise sequence-identity comparison)
│   ├── divergence.py     ← STUB on master (10 lines)
│   ├── energy_decomp.py  ← STUB on master (10 lines)
│   └── selectivity_map.py← STUB on master (10 lines)
├── reference/
│   └── __init__.py       (1-line placeholder; no reference data committed)
└── report/
    └── __init__.py       (1-line placeholder; no HTML/JSON reporter beyond CLI)
```

The three stub modules under `causality/` are implemented on `feat/pfdhodh-selectivity` — see §13. `reference/` and `report/` are empty docstring-only packages on both master and the feat branch.

### 4.4 `tests/` map (master)

```
tests/
├── fixtures/
│   ├── clashing.pdb   (test fixture, 975 B)
│   └── tri_ala.pdb    (test fixture, 1.3 KB)
├── test_parser.py     (TestParsePDB)
├── test_topology.py   (TestBondInference, TestBondedMask, TestVdWRadii)
├── test_geometry.py   (TestDistanceMatrix, TestDihedralAngles, TestBondAngles, TestBackboneDihedrals)
├── test_energy.py     (TestLJEnergy, TestClashDetection)
└── test_causality.py  (TestBindingSiteExtraction, TestPocketComparison)
```

45 tests. `pytest tests/ -q` returns `45 passed in 0.81s`; `ruff check src/ tests/ --select E,F,I,W --ignore E501` returns `All checks passed!`.

On `feat/pfdhodh-selectivity` the test count rises to 92 passed + 1 skipped, via six additional test files: `test_disulfide.py`, `test_divergence.py`, `test_energy_decomp.py`, `test_pdb_provenance.py`, `test_regressions.py`, `test_selectivity_map.py`.

---

## 5. Methodology map

Module-level map at master commit `96eac03`. Functions and classes listed are those exposed at the module surface; private helpers are omitted unless load-bearing.

### `src/physics_auditor/core/parser.py`

- **Purpose:** PDB file parser → flat numpy arrays suitable for vectorized JAX computation. Stops at the first `ENDMDL` so NMR ensembles do not concatenate all models.
- **Public surface:** `Atom`, `Residue`, `Chain`, `Structure` dataclasses; `parse_pdb(path, keep_hydrogens, keep_altloc)`, `parse_pdb_string(pdb_string, name, ...)`.
- **Reads:** PDB files.
- **Writes:** none.
- **Test coverage:** `tests/test_parser.py`.

### `src/physics_auditor/core/topology.py`

- **Purpose:** Infers covalent bonds (residue templates + peptide bonds + distance-based fallback for ligands), builds the non-bonded mask (default excludes 1-2 and 1-3 pairs), provides vdW radii lookups.
- **Public surface:** `infer_bonds_from_topology(structure)`, `build_bonded_mask(n_atoms, bonds, exclude_neighbors=3)`, `build_1_4_mask(n_atoms, bonds)`, `get_vdw_radius(element)`, `get_vdw_radii_array(structure)`; module-level constants `COVALENT_BOND_CUTOFFS`, `BACKBONE_BONDS`, `SIDECHAIN_BONDS` (templates for the 20 standard amino acids plus MSE), `VDW_RADII` (Bondi 1964 / AMBER values).
- **Reads:** none.
- **Writes:** none.
- **Test coverage:** `tests/test_topology.py`.

### `src/physics_auditor/core/geometry.py`

- **Purpose:** JIT-compiled JAX geometry kernels and a NumPy helper that extracts per-residue backbone φ/ψ/ω quads.
- **Public surface:** `@jax.jit compute_distance_matrix(coords)`, `compute_dihedral_angles(p0, p1, p2, p3)`, `compute_bond_angles(p0, p1, p2)`, `compute_distances(p0, p1)`; `extract_backbone_dihedrals(coords, atom_names, res_indices, is_protein, chain_ids)` (returns a dict with `phi`, `psi`, `omega` keys, each with `angles` and `res_indices`).
- **Reads:** none.
- **Writes:** none.
- **Test coverage:** `tests/test_geometry.py`.

### `src/physics_auditor/core/energy.py`

- **Purpose:** Lennard-Jones kernel using Lorentz-Berthelot combining rules and element-level (not atom-type-level) σ/ε parameters.
- **Public surface:** `get_lj_params_arrays(elements)`, `@jax.jit compute_lj_energy_matrix(...)`, `compute_total_lj_energy()`, `compute_per_atom_lj_energy()`, `compute_per_residue_lj_energy()`, `run_lj_analysis(...)`. The last returns a dict with `total_energy`, `per_atom_energy`, `per_residue_energy`, `energy_matrix`, `n_hot_pairs` (pairs with E > 10 kcal/mol — the docstring on master says `n_clashing_pairs`; the key was renamed but the docstring was not, fixed on `fixes/technical-pass-1`).
- **Reads:** none.
- **Writes:** none.
- **Test coverage:** `tests/test_energy.py::TestLJEnergy`.

### `src/physics_auditor/checks/clashes.py`

- **Purpose:** Steric-clash detection. A clash is defined as `(vdw_i + vdw_j) − tolerance > distance` on a non-bonded pair; severity gating via `severe_clash_threshold` (default 0.2 Å).
- **Public surface:** `ClashResult` dataclass (`n_clashes`, `n_severe_clashes`, `clashscore`, `worst_overlap`, `clashing_pairs`, `per_residue_clashes`, `subscore`); `check_clashes(dist_matrix, elements, nonbonded_mask, res_indices, n_residues, config)`; `get_vdw_radii_array_from_elements(elements)`.
- **Clashscore:** clashes per 1000 atoms; subscore on master decays linearly `max(0.0, 1.0 − clashscore / 40.0)`. `fixes/technical-pass-1` replaces this with a sigmoidal decay.
- **Reads:** none.
- **Writes:** none.
- **Test coverage:** `tests/test_energy.py::TestClashDetection`.

### `src/physics_auditor/causality/binding_site.py`

- **Purpose:** Extracts binding-pocket residues by distance to a ligand atom set; provides pairwise pocket comparison.
- **Public surface:** `BindingSite` dataclass; `extract_binding_site(structure, ligand_coords, cutoff=5.0, protein_only=True)`; `PocketComparison` dataclass; `compare_binding_sites(site1, site2, alignment=None)` (sequential default alignment with the explicit caveat *"should come from a proper structural alignment"*).
- **Reads:** none (operates on a parsed `Structure`).
- **Writes:** none.
- **Test coverage:** `tests/test_causality.py`.

### `src/physics_auditor/causality/divergence.py`, `energy_decomp.py`, `selectivity_map.py` (master)

- **Status at snapshot commit:** All three are **10-line TODO stubs**.
  - `divergence.py`: docstring + `from __future__ import annotations`. Marker text: *"TODO: Implement after binding_site and energy_decomp are tested."*
  - `energy_decomp.py`: same shape. Marker text: *"TODO: Implement after LJ kernel integration tests pass."*
  - `selectivity_map.py`: same shape. Marker text: *"TODO: Implement after energy decomposition and binding site modules are tested."*
- **Status on `feat/pfdhodh-selectivity`:** fully implemented; see §13.

### `src/physics_auditor/cli.py`

- **Purpose:** Typer CLI with two subcommands: `validate` and `info`.
- **Implemented composite:** `composite = clash_result.subscore` (line 97) with the inline comment *"Full composite will include all 8 checks — for now use clashes + LJ. Simplified for v0.1."* The LJ subscore is computed via `run_lj_analysis` but **not folded into the composite** at master. `fixes/technical-pass-1` wires LJ into the composite.
- **Recommendation thresholds:** `accept >= 0.85`, `short_md >= 0.60`, else `discard` (master defaults from `CompositeConfig`).
- **Outputs:** JSON to stdout (via `--json`), JSON files to disk (via `--output-dir`), rich-text panel to terminal otherwise. **No HTML.**
- **Test coverage:** none. CLI behavior is not exercised by `pytest`.

### `src/physics_auditor/config.py`

- **Purpose:** Frozen-dataclass configuration for every documented check, plus a YAML-override loader.
- **Public surface:** 9 dataclasses (`ClashConfig`, `BondGeometryConfig`, `RamachandranConfig`, `PeptideConfig`, `ChiralityConfig`, `RotamerConfig`, `LennardJonesConfig`, `DisulfideConfig`, `CompositeConfig`) and `AuditorConfig` aggregating them; `load_config(yaml_path)` for YAML overrides.
- **Note:** All eight check configs exist; **only `ClashConfig` and `LennardJonesConfig` are read by code at the snapshot commit.** The other six are dataclass definitions whose fields are referenced only by the `CompositeConfig` weights table.
- **Test coverage:** none.

### `src/physics_auditor/reference/__init__.py`, `report/__init__.py`

- Both are 1-line docstring placeholders. No code, no committed reference data, no HTML/JSON-reporter beyond what CLI emits.

### IsoDDE-style 8-check status

| README label | Module | Status at snapshot commit |
|---|---|---|
| Steric clashes | `checks/clashes.py` | **Implemented** and wired into the CLI composite |
| Bond geometry (length + angle vs Engh-Huber) | (no module) | **Absent.** `BondGeometryConfig` exists; no consumer code. |
| Ramachandran (backbone φ/ψ) | (no module) | **Absent.** `RamachandranConfig` exists; backbone φ/ψ are *extracted* in `core/geometry.py::extract_backbone_dihedrals` but not scored. |
| Peptide planarity (ω) | (no module) | **Absent.** `PeptideConfig` exists; ω extracted, not scored. |
| Chirality (L-AA verification) | (no module) | **Absent.** `ChiralityConfig` exists. |
| Rotamer outliers | (no module) | **Absent.** `RotamerConfig` exists. No rotamer library committed. |
| Lennard-Jones energy | `core/energy.py` | **Implemented** (not in `checks/` namespace, but functionally available; not folded into the CLI composite at master) |
| Disulfide geometry | (no module) | **Absent.** `DisulfideConfig` exists. `tests/test_disulfide.py` on `feat/pfdhodh-selectivity` references behavior in `topology.py` (the `S-S 2.6 Å` covalent cutoff), not a dedicated check module. |

**2 of 8 checks implemented.** The other six are scaffolding only. See Finding 3.

---

## 6. Expected-but-uncommitted file registry

A grep across `src/`, `tests/`, and `benchmark/` for hardcoded paths surfaces every file the active code expects to find. **At master commit `96eac03`, every referenced path is committed.** The repository commits its PDB inputs directly into the tree (~21 MB across `benchmark/structures/`) instead of relying on a download step, which eliminates the entire class of expected-but-missing-data failures.

| Reference site | Path | Status |
|---|---|---|
| `benchmark/benchmark_pocket_clashscore.py` `BENCHMARK_PAIRS` | 14× `benchmark/structures/experimental/*.pdb` | All committed |
| same | 14× `benchmark/structures/predicted/AF-*-F1-model_v6.pdb` | All committed |
| same | `benchmark/results/pocket_clashscore_comparison.json` | Committed |
| `tests/test_parser.py:112` | `/nonexistent/path.pdb` | Intentional negative-test sentinel; not a real path |
| `benchmark/structures/predicted/{boltz2,chai1}/` | (empty placeholder dirs) | No code references them; reserved for future predictors |

Three experimental PDBs (1HSG, 2QWF, 3NJW) are committed but not referenced by any committed code — they appear to be reserved for ad-hoc use.

`reference/` advertises (in its package docstring) "ideal bond parameters, Ramachandran distributions, rotamer libraries, LJ params" but **commits none** of those reference datasets. Implementing the six missing checks (Finding 3) requires populating this directory; the current state is descriptive of an intent, not of any data.

---

## 7. Per-artifact reproducibility

This section documents which committed artifacts can be regenerated from committed inputs by committed code, and which cannot.

### 7.1 PfDHODH selectivity attribution dossier (on `feat/pfdhodh-selectivity`, **not** on master)

- **Artifact:** `benchmark/results/selectivity_maps/PfDHODH_vs_HsDHODH_selectivity.json`
- **Producing script:** `benchmark/run_pfdhodh_vs_hsdhodh_selectivity.py`
- **Inputs:** `benchmark/structures/experimental/4ORM.pdb` (committed in `02c15aa`) and `benchmark/structures/experimental/1D3H.pdb` (committed earlier).
- **Provenance gate:** Yes. The script calls `verify_pdb_source(PF_DHODH_PDB, "PLASMODIUM FALCIPARUM")` and `verify_pdb_source(HS_DHODH_PDB, "HOMO SAPIENS")` before parsing. Both SOURCE records were confirmed independently during the audit (4ORM → `PLASMODIUM FALCIPARUM` strain 3D7; 1D3H → `HOMO SAPIENS`).
- **Reproducibility check (executed during the audit):** The script was re-run from the committed inputs. The regenerated JSON matched the committed dossier **byte-for-byte** (`git diff --stat` returned empty). The headline result — pocket Δ +12.78 kcal/mol; top-ranked target-selective residues HIS185 (+4.34), PHE188 (+3.32), VAL532 (+3.01), PHE227 (+2.07), LEU531 (+1.69); ARG265 at rank 6 (+1.50) — is fully reproducible from committed state.
- **Claims discipline:** The dossier carries an inline top-level `claims` field with four keys (`what_this_is`, `what_this_is_not`, `supported_finding`, `pending_followup`). The prose findings document next to it (`PFDHODH_SELECTIVITY_FINDINGS.md`) restates the same framing with a literature cross-check section explicitly naming the F188/R265/H185 cluster from the Phillips/Rathod DSM-series antimalarial literature.

### 7.2 14-pair binding-site clashscore benchmark (on master)

- **Artifact:** `benchmark/results/pocket_clashscore_comparison.json` (14 pair entries)
- **Producing script:** `benchmark/benchmark_pocket_clashscore.py`
- **Inputs:** 28 PDBs (14 experimental + 14 AlphaFold v6 predictions), all committed under `benchmark/structures/`.
- **Provenance gate:** **None.** The script reads PDBs via `parse_pdb()` directly; `verify_pdb_source` is not called. See Finding 2.
- **Reproducibility check:** **Not re-executed in this audit.** The script does not require any external downloads and has no random sampling, so byte-identical regeneration is expected but not verified here. A future PR closing Finding 1 will need to re-run this benchmark with corrected labels and confirm bit-identity (or document any drift).
- **Mislabel issue:** Pair 2 is labeled `SmDHODH` but uses `1D3H.pdb` whose SOURCE record reads `HOMO SAPIENS`. The structure is the catalytic construct of HsDHODH (UniProt Q02127), not the parasite enzyme. The AlphaFold side (`AF-G4VFD7-F1-model_v6.pdb`) is the true SmDHODH; the experimental side is mislabeled. The mislabel propagates into `manuscript/preprint.md` Table 1. See Finding 1.

### 7.3 14-pair benchmark manuscript

- **Artifact:** `manuscript/preprint.md` + `manuscript/preprint.pdf`
- **Inputs:** the 14-pair clashscore JSON above, plus narrative prose.
- **Reproducibility:** The PDF is generated from the markdown via `pandoc` per commit `f669792`. The numeric content depends on the upstream clashscore JSON; correcting Finding 1 will require regenerating both the JSON and the manuscript.

### 7.4 Causality decomposition (on `feat/pfdhodh-selectivity`)

- **Artifacts:** `benchmark/results/causality/{HsDHODH,SmDHODH,HsHDAC8,LmPTR1,HsAromatase,CoV2Mpro}_causality.json` (6 of the 14 benchmark pairs), `causality_summary.csv`, `CAUSALITY_FINDINGS.md`, `TOPOLOGY_FIXES.md`.
- **Producing script:** `benchmark/run_causality_decomposition.py`.
- **Coverage gap:** **6 of 14 pairs.** `CAUSALITY_FINDINGS.md` records that 8 pairs were skipped due to a 4500-atom memory budget on the JAX distance matrix. Closing this gap requires either pocket-restricted decomposition, chunked computation, or raising the budget on a larger machine — see Finding 7.

### 7.5 ESM-2 divergence (on `feat/pfdhodh-selectivity`)

- **Artifact:** `benchmark/results/divergence/summary.json`.
- **Producing script:** `benchmark/run_divergence.py`.
- **Headline number:** `full_sequence_cosine = 0.989732` for the true SmDHODH (G4VFD7) vs HsDHODH (Q02127) UniProt sequences — matches the Kira-cited "0.9897" slogan to four decimals.
- **Pocket-level leg:** Only computed for the 1D3H↔1MVS pair (HsDHODH catalytic construct vs HsDHFR — a functional-paralog pair, not an ortholog pair). The true ortholog pocket-divergence number is **not yet computable from in-repo structures** because no parasite-side DHODH co-crystal is committed.

### 7.6 Test fixtures and CLI golden outputs

- **Fixtures:** `tests/fixtures/{tri_ala,clashing}.pdb` are tiny hand-crafted PDBs used by 45 of 45 tests. They are not regenerated; they are the inputs.
- **CLI golden outputs:** `SETUP.md` documents expected CLI output for `physics-auditor validate tests/fixtures/{tri_ala,clashing}.pdb` (score 1.000 / 0.000, clashscore 0.0 / 400.0). These are not encoded as tests.

---

## 8. Constitution and governance audit

### 8.1 What exists

The full governance surface at the snapshot commit is the four-line **"Rules" block in `CLAUDE.md`**:

> - Every commit must pass ruff and pytest
> - Never overclaim — state what the evidence supports
> - PDB files use fixed-width columns (not space-delimited)
> - Parser stops at first ENDMDL (NMR structures have multiple models)

There is **no `docs/` directory** at the snapshot commit. There is no Scientific Constitution. There is no Claim Inflation Audit. There is no Disposition Plan.

The closest existing analog to per-artifact claim discipline is the inline `claims` field used by two artifacts on `feat/pfdhodh-selectivity`:

1. `benchmark/results/selectivity_maps/PfDHODH_vs_HsDHODH_selectivity.json` carries a top-level `claims` object with four keys (`what_this_is`, `what_this_is_not`, `supported_finding`, `pending_followup`). The accompanying `PFDHODH_SELECTIVITY_FINDINGS.md` document includes explicit "Non-claims" and "Honesty note on the literature cross-check" sections.
2. `benchmark/results/divergence/summary.json` carries per-pair `claims` lists and a `notes` field documenting known caveats.

There is no schema, no validator, no test that checks for `claims` presence on new result artifacts. The pattern is informal and applied per-artifact.

### 8.2 README accuracy review

The README makes a number of claims; verdict against the code as it stands at the snapshot commit:

| Claim in README | Verdict |
|---|---|
| *"Physics Validation (8 checks)"* listing clashes, bond geometry, Ramachandran, peptide planarity, chirality, rotamer outliers, Lennard-Jones, disulfide geometry | **Overclaims.** 2 of 8 implemented. `SETUP.md`'s "What's next" table acknowledges 6 are not built. README and SETUP.md contradict each other. See Finding 3. |
| *"Mechanistic Causality"* — per-residue energy decomposition, binding-site comparison, selectivity attribution | **Master = absent or stub.** Pocket extraction + sequence-identity comparison are implemented; the other three are stubs on master. Implemented on `feat/pfdhodh-selectivity`. |
| *"Trust Report — JSON (for pipelines) + HTML (for humans)"* | **HTML not implemented.** `report/__init__.py` is a 1-line placeholder; CLI emits JSON and rich-text only. See Finding 7. |
| *"Per-residue heatmaps and binding-site annotations"* | **Not implemented.** No plotting code in the repository. See Finding 7. |
| *"Composite physics score (0–1)"* + *"accept / run short MD / discard"* | Implemented, but the composite is `clash_result.subscore` only — the CLI's own comment marks this as "simplified for v0.1." `fixes/technical-pass-1` wires in the LJ subscore. |
| `physics-auditor selectivity --target ... --ortholog ... --ligand ...` (Quick Start example) | **Subcommand does not exist.** `cli.py` registers only `validate` and `info`. See Finding 7. |
| *"Why Not Molprobity? 1. Calibrated for AI structures."* | **Unverified.** No calibration data is committed. The current thresholds (clashscore decay to zero by 40 clashes/1000 atoms; `accept=0.85`, `short_md=0.60`) appear to be defaults. `fixes/technical-pass-1` recalibrates these against the 14-pair benchmark. |
| `pip install git+https://github.com/Danny2045/physics-auditor.git` | Works. |

### 8.3 CLAUDE.md accuracy

`CLAUDE.md` opens with `"JAX-based, 8 checks, CLI"` — same overclaim as the README. The architecture section accurately reflects master. The "Active work" section documents the HDAC8 residue-numbering offset issue as if it were open; it was in fact closed by commit `81d644a` and the offset is correctly applied (`pocket_n_residues=46` in the JSON confirms this). The note is stale but harmless.

### 8.4 SETUP.md accuracy

`SETUP.md` is **the most accurate top-level document.** Its "What's next" table (lines 252–267) cleanly separates implemented modules from planned ones and is consistent with the code at master. Its expected pytest output ("45 tests, all pass") matches reality.

---

## 9. Outstanding work register

The repository carries **seven** findings at the snapshot commit. **All seven are new** — this is the first state-of-repository audit of Physics Auditor, so the numbering starts at 1 and is independent of the Kira finding numbers.

Each finding pairs current status with a "Natural next-PR scope" line. The scope lines are suggestions for the future PR that closes the finding; they are not commitments and the engineer who picks up the work is expected to refine them.

### Finding 1 — Mislabel propagation in master-published benchmark and manuscript

**Status:** Open. Scientific-integrity issue. **First cleanup PR to land after this document.**

`benchmark/results/pocket_clashscore_comparison.json` (pair 2) labels `1D3H.pdb` as "SmDHODH" and pairs it with the AlphaFold model `AF-G4VFD7-F1-model_v6.pdb` (which *is* SmDHODH). But `1D3H.pdb`'s SOURCE record reads `ORGANISM_SCIENTIFIC: HOMO SAPIENS`; the structure is the catalytic construct of *human* DHODH (UniProt Q02127), not the parasite. The numerical clashscore values are valid for the underlying experimental structure (1D3H); the *label* "SmDHODH" attached to it is wrong.

The mislabel propagates into `manuscript/preprint.md` Table 1, where the "SmDHODH" row reads `1D3H | G4VFD7 | …` — pairing a human PDB with a parasite UniProt ID. The same conflation is in the producing script's `BENCHMARK_PAIRS` description string: *"Parasite DHODH — 30.8x selectivity target."*

The mislabel was originally caught and recorded during the `feat/divergence` review on 2026-05-11 (commit `a19a2a2`: *"relabel 1D3H/1MVS run — HsDHODH vs HsDHFR, not Sm/Hs"*) for a *different* result artifact (the early HsDHODH-vs-HsDHFR selectivity-map run). It was not propagated back to the master-side benchmark or to the manuscript, and the master-side `parse_pdb()` function still has no provenance gate (see Finding 2).

**Natural next-PR scope:** Relabel the affected entries — pair 2 in `pocket_clashscore_comparison.json` becomes either the true HsDHODH-vs-HsDHODH pair (if `AF-Q02127` is the right AF model) or is split into two pairs (one HsDHODH with 1MVS/Q02127, one true SmDHODH using an AF-only "experimental" stand-in or a different parasite-DHODH crystal if one exists in the PDB). Regenerate the JSON (which should be bit-identical for every entry except pair 2 once labels are fixed). Update `manuscript/preprint.md` Table 1 and any narrative references. Add a call to `verify_pdb_source` to `benchmark_pocket_clashscore.py`'s pair iteration so the mislabel cannot recur. Consider whether the manuscript needs to be re-bumped to a new draft version with an acknowledgments note about the relabel.

### Finding 2 — `verify_pdb_source` built but not enforced

**Status:** Open. Policy/enforcement gap.

`src/physics_auditor/utils/pdb_provenance.py` (on `feat/pfdhodh-selectivity`) implements `verify_pdb_source(path, expected_organism)`, which reads the PDB file's `SOURCE` records, extracts `ORGANISM_SCIENTIFIC`, and raises `ValueError` on any mismatch. The docstring asserts the policy explicitly:

> *"every PDB ingestion path must call `verify_pdb_source(path, expected_organism)` before parsing."*

In practice, the gate is called by exactly one consumer: `benchmark/run_pfdhodh_vs_hsdhodh_selectivity.py`. **`src/physics_auditor/core/parser.py:parse_pdb()` does not call it**, and every other benchmark script (`benchmark_pocket_clashscore.py`, `run_causality_decomposition.py`, `run_divergence.py`, `run_selectivity_map.py`) reads PDBs via `parse_pdb()` without the provenance check. The mislabel that surfaced as Finding 1 is precisely the class of mistake the gate was built to prevent — but the gate is not enforced where it would have caught it.

**Natural next-PR scope:** Decide between three policies and implement consistently:

1. **Parser-level enforcement.** Add `expected_organism` as a required kwarg to `parse_pdb()`. Every caller must declare an organism; mismatch raises `ValueError`. Strongest guarantee, breaks every existing caller including the 14-pair benchmark.
2. **Benchmark-script enforcement.** Add `verify_pdb_source` calls to every benchmark script's PDB iteration. Leaves `parse_pdb()` free to be used in ad-hoc / exploratory contexts; the script is the gate.
3. **Soft warning.** `parse_pdb()` calls `verify_pdb_source` if an `expected_organism` is passed, logs a warning otherwise. Preserves backward compatibility; weaker guarantee.

Option (2) is the most consistent with how the gate is already used and is the natural target for the same PR that closes Finding 1.

### Finding 3 — Eight-check suite is two checks

**Status:** Open. Documentation contradiction; substantial implementation work to close.

The README advertises "Physics Validation (8 checks)" listing steric clashes, bond geometry, Ramachandran, peptide planarity, chirality, rotamer outliers, Lennard-Jones, disulfide geometry. At the snapshot commit, **only steric clashes and Lennard-Jones energy are implemented.** The other six are config-and-weight scaffolding:

- `BondGeometryConfig`, `RamachandranConfig`, `PeptideConfig`, `ChiralityConfig`, `RotamerConfig`, `DisulfideConfig` exist in `config.py`.
- `CompositeConfig.weight_bond_geometry`, `weight_ramachandran`, `weight_peptide_planarity`, `weight_chirality`, `weight_rotamers`, `weight_disulfides` all exist.
- No `checks/{bond_geometry,ramachandran,peptide_planarity,chirality,rotamers,disulfides}.py` files exist at the snapshot commit.
- `reference/__init__.py` advertises "ideal bond parameters, Ramachandran distributions, rotamer libraries, LJ params" but **commits no reference data** beyond the LJ element-level σ/ε constants embedded directly in `core/energy.py`.

`SETUP.md`'s "What's next" table (lines 252–267) is honest about this. The README contradicts SETUP.md.

The CLI itself flags the gap inline at `cli.py:96`:

> *"Full composite will include all 8 checks — for now use clashes + LJ. Simplified for v0.1."*

**Natural next-PR scope (split):**

1. **README rewrite PR.** Mirror SETUP.md's framing. Use an "implemented vs. planned" table; remove the unqualified "8 checks" claim from prose; pair every planned check with its config scaffold so the surface is honest. This is doc-only and should land before any implementation work.
2. **Per-check implementation PRs.** Bond geometry is the lowest-hanging fruit: a single Engh-Huber bond-length and bond-angle reference table can drive a z-score check against `core/geometry.py::compute_bond_angles` and the residue-template bond list in `topology.py`. Land each check as its own PR with a dedicated test fixture, dedicated reference data committed under `reference/`, and a CLI extension that surfaces the new subscore in the rich-text panel and the JSON report.

### Finding 4 — `fixes/technical-pass-1` was local-only

**Status:** Operationally mitigated by this document's PR (branch pushed to origin); implementation work still open.

The branch was created on 2026-05-06 and remained local-only through the snapshot date. **Phase 2 of the audit producing this document pushed it to `origin/fixes/technical-pass-1`** to eliminate the data-loss risk. The branch is now durable; it still needs to be rebased, conflict-resolved, and merged.

The branch carries (per commit `0b3e16e`):

- **Clash subscore recalibration.** Linear `1 − clashscore/40` (which made every experimental crystal in the 14-pair benchmark — clashscore 11–56 — fall in DISCARD or SHORT_MD, contradicting the tool's own positioning) is replaced with sigmoidal `1 / (1 + (clashscore/50)^2)` and thresholds shift to `accept=0.80`, `short_md=0.50`.
- **`extract_binding_site` rewrite.** O(N) via vectorized `np.isin` instead of the original O(N·P) inner scan. Material for large structures (e.g. MmCOX2 at 33k atoms).
- **`parse_pdb_string` ENDMDL fix.** The string-API counterpart to the master commit `a65d454` fix on `parse_pdb`. **Already independently reapplied on the feat stack at commit `174c543`** — this part of the technical-pass branch is redundant against `feat/pfdhodh-selectivity` and will not conflict.
- **LJ subscore wiring.** `n_hot_pairs_per_1000_atoms` becomes a real subscore with the same sigmoidal decay; it is folded into the composite via renormalized `CompositeConfig` weights and surfaced in both JSON and the rich-text panel. The CLI now reports `roadmap_checks` (the planned but unimplemented checks) alongside the active subscores.
- **`core/energy.py` docstring fix.** The `n_hot_pairs` / `n_clashing_pairs` key-name mismatch is corrected.
- **`tests/test_regressions.py`.** 135-line regression test suite for the calibration and wiring above. Note: a file by the same name was independently created on the feat stack at `e685f7d` (*"test: restore test_regressions.py (lost in causality patch base)"*) — the two files are not the same, and a rebase will need to merge them carefully.

**Natural next-PR scope:** Rebase `fixes/technical-pass-1` against current master (after PR #1 merges; conflicts will be heavier against the post-merge tree because of the causality-layer changes). Likely conflict points:

- `src/physics_auditor/causality/binding_site.py` — O(N) rewrite vs. the causality-layer's modifications to the same file (commit `e4fabb9`).
- `src/physics_auditor/cli.py` — composite wiring vs. (presently unchanged) CLI on the feat stack.
- `tests/test_regressions.py` — two independent versions must be reconciled.

Open as a PR titled along the lines of *"fix: technical pass — clash subscore calibration, O(N) pocket, LJ composite wiring"*, request review, merge.

### Finding 5 — No governance documents

**Status:** Open.

Unlike Kira, Physics Auditor has no Scientific Constitution, no Claim Inflation Audit, no Disposition Plan, no `docs/` directory. The only governance content is the four-line "Rules" block in `CLAUDE.md`. The PfDHODH dossier's inline `claims` field and the `PFDHODH_SELECTIVITY_FINDINGS.md` "Non-claims" section are the only per-artifact claim discipline currently practiced.

**Natural next-PR scope:** Decide between three options and implement:

1. **Adopt the Kira Constitution by reference.** Physics Auditor operates under Kira's `docs/SCIENTIFIC_CONSTITUTION.md`. A small `docs/GOVERNANCE.md` in Physics Auditor cites the Kira document as authoritative, lists which provisions apply (forbidden-term lists; evidence-tier framing), and notes any Physics-Auditor-specific exceptions.
2. **Write a Physics-Auditor-specific governance document.** Narrower scope, focused on structure-validation claims: distinguishing "the code computed X" from "X is a property of the underlying structure" from "X has biological meaning." Generalize the PfDHODH dossier's `claims` schema (`what_this_is`, `what_this_is_not`, `supported_finding`, `pending_followup`) into a schema every new result artifact must follow; add a pytest gate that rejects new `benchmark/results/**/*.json` files lacking a `claims` field with the required keys.
3. **Defer.** Document the dependency on Kira's governance in this state document (already done in §8.1 and §12) and revisit when the bridge interface (Finding 6) is implemented and demands shared discipline.

Option (2) is the most consistent with the discipline already in place on `feat/pfdhodh-selectivity`. The schema is small enough to write in one sitting; the gate is small enough to implement as a single pytest.

### Finding 6 — Bridge interface to Kira absent on both sides

**Status:** Open.

The intended interface contract between Kira and Physics Auditor — `StructureAuditRequest` (Kira → Physics Auditor) and `StructureAuditResult` (Physics Auditor → Kira) — is **not implemented in either repository.** A grep for either type name returns no hits across all branches of Physics Auditor. The Kira-side document referenced in the audit brief (`docs/PHYSICS_AUDITOR_TICKET_GATE.md` in Kira) describes the intended interface; no Physics-Auditor-side counterpart exists, and no code on either side implements it.

The PfDHODH dossier JSON format (with the `claims` field) is the closest existing artifact to a `StructureAuditResult` payload, but it is not typed, not validated, and not declared as a public contract.

**Natural next-PR scope:** Design `StructureAuditRequest` and `StructureAuditResult` as frozen dataclasses in `src/physics_auditor/bridge/` (or `src/physics_auditor/contracts/`). `StructureAuditRequest` carries: target PDB path or content, optional ortholog PDB, ligand identification (HETATM resname or coords), the expected organism per structure (for the provenance gate), and the audit type (`validate`, `selectivity`, `causality_decomposition`). `StructureAuditResult` carries: the existing dossier fields, the `claims` schema from Finding 5, a `code_git_sha` field, and a `computed_at_utc` field. Implement a `handle_request(req: StructureAuditRequest) -> StructureAuditResult` entry point that dispatches to existing machinery. Add a Kira-side glue PR that issues `StructureAuditRequest`s and persists `StructureAuditResult`s as Kira's existing provenance-sidecar pattern. The two PRs should land within a release of each other.

### Finding 7 — README accuracy gaps beyond the eight-check claim

**Status:** Open.

The README claims three additional features that do not exist in committed code at master:

- **`physics-auditor selectivity` CLI subcommand.** The Quick Start block shows `physics-auditor selectivity --target parasite.pdb --ortholog human.pdb --ligand compound.sdf`. The subcommand does not exist; `cli.py` registers only `validate` and `info`. The selectivity work on `feat/pfdhodh-selectivity` is exposed as a *script* (`benchmark/run_pfdhodh_vs_hsdhodh_selectivity.py`), not a CLI subcommand.
- **HTML reports.** The README lists "JSON (for pipelines) + HTML (for humans)" under Trust Report. `report/__init__.py` is a 1-line docstring placeholder; no HTML generator exists.
- **Per-residue heatmaps and binding-site annotations.** No plotting code exists in the repository.

**Natural next-PR scope:** Rewrite the README to match what the code does at master. Either:

1. **Demote the unimplemented features** to a "Planned" or "Roadmap" section parallel to `SETUP.md`'s "What's next" table, or
2. **Implement them as part of the README rewrite PR** — `physics-auditor selectivity` is a small wrapper around the existing `compute_selectivity_map` machinery on the feat stack; HTML reports are a Jinja2 template against the existing JSON dossier (jinja2 is already declared in `pyproject.toml`); heatmaps need a plotting dependency to be added.

Option (1) is the doc-only, immediately-actionable form and is the natural pair with the doc-only half of Finding 3.

### Status summary

| Finding | Status | Origin |
|---|---|---|
| 1. Mislabel propagation (1D3H labeled "SmDHODH" in master benchmark + manuscript) | Open | **New — surfaced by this audit** |
| 2. `verify_pdb_source` built but not enforced | Open | **New — surfaced by this audit** |
| 3. Eight-check suite is two checks (README contradicts SETUP.md) | Open | **New — surfaced by this audit** |
| 4. `fixes/technical-pass-1` was local-only | Branch now pushed; rebase + PR pending | **New — surfaced by this audit; operationally mitigated by this PR** |
| 5. No governance documents | Open | **New — surfaced by this audit** |
| 6. Bridge interface (`StructureAuditRequest` / `StructureAuditResult`) absent on both sides | Open | **New — surfaced by this audit** |
| 7. README accuracy gaps beyond the eight-check claim | Open | **New — surfaced by this audit** |

---

## 10. Open questions and known unknowns

These are the things the audit producing this document surfaced that do not have an answer in committed state. They are decisions for human review, not actions.

1. **PR #1 size and review strategy.** The PfDHODH selectivity PR is +68,135 / −14 lines across 36 files. The bulk is the committed `4ORM.pdb` plus six new test files plus three new module implementations plus four new benchmark scripts plus eight new result artifacts. Splitting for review is defensible but would fragment the artifact-to-code coupling that the PR currently preserves. Decision: is one PR acceptable, or should the merge be staged?
2. **CLAUDE.md test-count claim post-merge.** `CLAUDE.md` line 6 says *"45 tests, all must pass"*. After PR #1 merges, this rises to 92 + 1 skipped. The CLAUDE.md update was not part of the PR scope; it needs a follow-up.
3. **What does `test_disulfide.py` on the feat stack actually test?** No `checks/disulfides.py` exists on any branch. The test either exercises the `S-S 2.6 Å` covalent cutoff in `topology.py::COVALENT_BOND_CUTOFFS` (i.e. it's an integration test of bond inference, not of a missing module), or it is an unfinished stub. Worth a careful read in the PR that closes Finding 3.
4. **CI matrix coverage.** `.github/workflows/ci.yml` runs on Python 3.11 and 3.13 only. `pyproject.toml` declares `requires-python = ">=3.10"`. Either 3.10 and 3.12 should be added to the matrix, or `requires-python` should be tightened.
5. **Boltz-2 and Chai-1 placeholder directories.** `benchmark/structures/predicted/{boltz2,chai1}/` are empty. The manuscript's "Limitations" section says extending the benchmark to these predictors is a natural next step. No code references them. Should the directories stay (as a visible roadmap signal) or be removed (until there is something to commit)?
6. **Causality decomposition coverage on the feat stack.** Only 6 of the 14 benchmark pairs ran to completion; 8 were skipped at the 4500-atom memory budget per `CAUSALITY_FINDINGS.md`. Is the right path forward (a) pocket-restricted decomposition (only compute LJ for `pocket × rest_of_protein` pairs), (b) chunked computation, (c) raising the budget on a larger machine? Each has different scientific implications for what "per-residue decomposition" means on a 33k-atom structure.
7. **True-ortholog pocket-divergence number.** The divergence runner produces `full_sequence_cosine = 0.989732` for true SmDHODH vs HsDHODH UniProt sequences, but the pocket leg is unmeasured because no parasite-side DHODH co-crystal is committed. Is the right path to commit `6FMD` (SmDHODH + inhibitor co-crystal, per the divergence module's own note) and rerun?
8. **Should the 14-pair benchmark re-run as part of Finding 1, or should the relabel be doc-only?** The clashscore numbers are valid for the underlying structures; only the *labels* are wrong. A purely-doc fix (relabel the JSON entries and the manuscript table) is defensible. A full re-run with the provenance gate enforced is more thorough but may produce drift on the JSON if any other unrelated changes have crept into the codebase between the original benchmark and now.
9. **Disposition of the `feat/*` branch stack post-merge.** Once PR #1 merges, the three shorter `feat/*` branches (`feat/causality-energy-decomp`, `feat/selectivity-map`, `feat/divergence`) become strict prefixes of merged master and carry no unique work. Should they be deleted in a single follow-up PR, or left as historical pointers?
10. **`reference/` data sourcing.** Closing Finding 3 requires committing Engh-Huber bond ideals (small, well-known), a Ramachandran distribution (small, multiple published sources, license-permissive options exist), and a rotamer library (Dunbrack or similar — license review needed). The disulfide-geometry data is small enough to hardcode. What is the right balance between vendoring these and depending on an external package?

---

## 11. Update protocol

This document is intended to be refreshed periodically, not continuously.

- **When to update inline (same PR):** when your PR changes the structure of what is committed or what is reproducible — adding a new active-code dependency on a data file, adding a new provenance pattern, changing the test count by more than a handful, opening or closing a numbered finding, merging or deleting a branch listed in §3.4. Notably: when PR #1 merges, this document must be updated to (a) reanchor to the new merge-commit SHA, (b) shift §13's "headline scientific work on the feat branch" content into the main body where appropriate, (c) update the test-count dashboard, (d) update Finding 4's expected conflicts now that the feat-stack changes are on master.
- **When to defer to the next snapshot:** small documentation tweaks, bug fixes in already-tested code paths, formatting and lint cleanups.
- **When to produce a new snapshot:** at least once per major milestone (e.g. after closing two or more findings), or whenever the divergence between this document and reality becomes uncomfortable enough to slow down a new contributor. Re-run the per-file inventories (§4), the per-artifact reproducibility checks (§7), and the eight-check status table (§5). Bump the snapshot commit hash and date at the top.
- **What this document does *not* track:** ephemeral task lists, in-progress branches that have not yet been pushed to origin, chat-level discussion. Those belong in the PR description, in branch-level commit messages, or in scratchpad notes outside the repository. Once a branch is pushed to origin, it becomes visible to this document and may be listed in §3.4 at the next snapshot.

---

## 12. Paired with Kira

Physics Auditor is the structure-validation half of a two-repository platform. The other half is **Kira**, which performs selectivity analysis, compound-activity benchmarking, ESM-2-based parasite-vs-human ortholog comparison, and experimental ticket generation. The two repositories share scientific lineage and are intended to communicate via the `StructureAuditRequest` / `StructureAuditResult` interface described in Finding 6 — currently unimplemented on both sides.

The Kira repository's canonical state document is [`STATE_OF_KIRA.md`](https://github.com/Danny2045/Kira/blob/main/STATE_OF_KIRA.md). It anchors to Kira commit `f12fe3d` (PR #19 merge), dated 2026-05-17 — the same day this document was first written, ensuring the two snapshots are coherent with each other.

Together, the two state documents form the **operational ground truth for the platform**. A reader who needs to understand both halves should read them as a pair: Kira's `STATE_OF_KIRA.md` for the selectivity/inverse-biology surface, this document for the structure-validation surface. The shared scientific narrative — the SmDHODH-vs-HsDHODH 0.9897 ESM-2 cosine slogan, the 30× selectivity figure, the DSM-series antimalarial selectivity literature — appears in both repositories at different levels of resolution: Kira at the sequence/embedding level, Physics Auditor at the per-residue structure level.

When the bridge interface lands (Finding 6), this section should be expanded with the concrete entry-point names on each side, the shared dataclass schemas, and any cross-repository test fixtures that verify round-trip behavior.

---

## 13. Headline scientific work on `feat/pfdhodh-selectivity` (PR #1, unmerged)

This section catalogs the contents of PR #1 (head: `feat/pfdhodh-selectivity` @ `73bddb1`, opened 2026-05-12, mergeable at the snapshot commit). The work is **not on master at the snapshot commit**; this state document anchors to master, where the work has not yet landed. When PR #1 merges, the contents of this section should be folded into the main body of the document and §13 itself removed in a follow-up PR.

### 13.1 The PfDHODH selectivity result

The headline scientific result of the project is the per-residue Lennard-Jones selectivity attribution between *Plasmodium falciparum* DHODH (PDB `4ORM`, bound to the triazolopyrimidine antimalarial **2V6 / DSM338** from the DSM compound series) and *Homo sapiens* DHODH (PDB `1D3H`, bound to **A26 / teriflunomide**). Both ligands compete at the same quinone-tunnel subsite of DHODH; the chemotypes differ, the binding subsite is shared, which is the correct substrate for an ortholog selectivity-attribution run.

Headline numbers from `benchmark/results/selectivity_maps/PfDHODH_vs_HsDHODH_selectivity.json`:

- Aligned pocket residues: 19.
- Total PfDHODH (4ORM) ligand LJ interaction: −44.73 kcal/mol.
- Total HsDHODH (1D3H) ligand LJ interaction: −29.57 kcal/mol.
- Pocket-level delta (HsDHODH − PfDHODH): **+12.78 kcal/mol** favoring the parasite.

Top-ranked target-selective (parasite-preferred) residues:

| Rank | 4ORM residue | 1D3H residue | Δ kcal/mol |
|--:|---|---|--:|
| 1 | **HIS185** | ALA59 | **+4.339** |
| 2 | **PHE188** | THR63 | **+3.321** |
| 3 | VAL532 | PRO364 | +3.006 |
| 4 | PHE227 | VAL134 | +2.067 |
| 5 | LEU531 | GLY363 | +1.688 |
| 6 | **ARG265** | TYR356 | **+1.497** |

The three positions in **bold** — **HIS185, PHE188, ARG265** — are the published F188 / R265 / H185 cluster cited in the antimalarial DHODH literature (Phillips & Rathod and successors; the DSM1 → DSM74 → DSM265 → DSM338 lineage) as the residues that drive PfDHODH-vs-HsDHODH selectivity for the triazolopyrimidine series. All three are recovered as target-selective by a static LJ-only attribution, with HIS185 at rank 1, PHE188 at rank 2, and ARG265 at rank 6.

The producing script (`benchmark/run_pfdhodh_vs_hsdhodh_selectivity.py`) was re-executed during the audit producing this document. The regenerated JSON matched the committed dossier **byte-for-byte**; the headline result is fully reproducible from committed inputs on the branch.

### 13.2 Causality module implementations (move-stubs-to-real-code on the branch)

PR #1 implements all three causality stubs that are 10-line TODOs on master:

- **`src/physics_auditor/causality/energy_decomp.py`** (332 lines). `per_residue_decomposition()` (decomposes an LJ analysis into per-residue contributions with binding-site flags), `per_residue_difference()` (between two homologous structures at aligned pocket positions), `decomposition_to_dict()`, `difference_to_dict()`.
- **`src/physics_auditor/causality/selectivity_map.py`** (334 lines). `compute_selectivity_map(target_structure, target_ligand_resname, ortholog_structure, ortholog_ligand_resname, pocket_cutoff=5.0)` returns a `SelectivityMap` dataclass with `SelectivityResiduePair` entries; `selectivity_map_to_dict()` serializes the dossier; convenience methods `top_n_target_selective(n)` and `top_n_ortholog_selective(n)`.
- **`src/physics_auditor/causality/divergence.py`** (306 lines). `compute_divergence()` runs ESM-2 (default model `esm2_t33_650M_UR50D`) and produces three cosine numbers per pair: `full_sequence_cosine` (mean-pooled over the whole sequence), `pocket_meanpool_cosine` (mean restricted to pocket positions of the same forward pass), `pocket_subseq_cosine` (separate forward pass on the pocket residues as a standalone subsequence). The reported "divergence amplification" is `full − pocket_meanpool`.

Each module's docstring carries a `NON-CLAIMS` block stating what the module does *not* claim (LJ-only attribution; no electrostatics, hydrogen bonds, solvation, entropy; per-residue energies are not free energies; sequential pocket alignment is the default and structurally-divergent pockets require an explicit alignment; etc.).

`core/energy.py` is extended with `per_residue_ligand_interaction_energy()`, used by `selectivity_map.py`.

### 13.3 PDB provenance gate (the safety scaffold)

`src/physics_auditor/utils/pdb_provenance.py` (92 lines) implements `verify_pdb_source(path, expected_organism) -> None`. The module's docstring records the May 2026 mislabeling incident (commit `a19a2a2`) as the reason the helper exists:

> *"In May 2026 a selectivity-attribution run was committed under the framing 'SmDHODH vs HsDHODH' while one of the two inputs (1MVS) was in fact HsDHFR. The mislabeling was caught only after the run shipped, and the dossier had to be corrected in commit a19a2a2. This module exists so that class of mistake cannot recur."*

The gate is called by `benchmark/run_pfdhodh_vs_hsdhodh_selectivity.py` before any PDB parse. It is **not enforced** at the `parse_pdb()` level — this is Finding 2.

### 13.4 New benchmark scripts and result artifacts

Beyond the master-side `benchmark_pocket_clashscore.py`, the branch adds four producing scripts and 15 result artifacts:

| Script | Output |
|---|---|
| `benchmark/run_causality_decomposition.py` | `benchmark/results/causality/{HsDHODH,SmDHODH,HsHDAC8,LmPTR1,HsAromatase,CoV2Mpro}_causality.json` (6 of 14 pairs — 8 skipped at memory budget), `causality_summary.csv`, `CAUSALITY_FINDINGS.md`, `TOPOLOGY_FIXES.md` |
| `benchmark/run_divergence.py` | `benchmark/results/divergence/summary.json` (full-sequence cosine reproduces the Kira-cited 0.9897 slogan for true SmDHODH/HsDHODH UniProt sequences) |
| `benchmark/run_selectivity_map.py` | `benchmark/results/selectivity_maps/HsDHODH_vs_HsDHFR_pocket_attribution.json` and `SELECTIVITY_FINDINGS.md` (the corrected functional-paralog stand-in run) |
| `benchmark/run_pfdhodh_vs_hsdhodh_selectivity.py` | `benchmark/results/selectivity_maps/PfDHODH_vs_HsDHODH_selectivity.json` and `PFDHODH_SELECTIVITY_FINDINGS.md` (the headline result, §13.1) |

### 13.5 Test coverage on the branch

Tests rise from 45 (master) to **92 passed + 1 skipped** on `feat/pfdhodh-selectivity`. New test files:

- `tests/test_disulfide.py`
- `tests/test_divergence.py`
- `tests/test_energy_decomp.py`
- `tests/test_pdb_provenance.py` (real-file match, real-file mismatch, missing file, malformed `SOURCE`; uses 1D3H and 4ORM as real fixtures)
- `tests/test_regressions.py` (note: a name collision with `fixes/technical-pass-1`'s `test_regressions.py` — see Finding 4)
- `tests/test_selectivity_map.py`

### 13.6 Committed structures and inputs added by the branch

- `benchmark/structures/experimental/4ORM.pdb` (PfDHODH bound to DSM338; 556 KB; SOURCE record `PLASMODIUM FALCIPARUM` strain 3D7) — committed in `02c15aa`.

### 13.7 What this work means for the state document

When PR #1 merges:

- The §13 stub modules in §5's methodology map become real entries with documented public surfaces.
- The dashboard test count updates from 45 to ~92.
- Finding 4's expected conflicts (against the post-merge tree) become heavier; the rebase plan needs to account for the now-merged causality-layer changes.
- Findings 1, 2, and 7 remain open exactly as written — they are master-side issues that the feat branch does not address.
- Finding 5 (governance) and Finding 6 (bridge interface) remain unaffected.
- §13 of this document is folded into the main body and removed in the same PR that re-anchors the snapshot commit.
