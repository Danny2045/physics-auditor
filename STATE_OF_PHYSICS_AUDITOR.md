# State of Physics Auditor

**Snapshot anchor commit:** `e7064de` — *Merge pull request #1 from Danny2045/feat/pfdhodh-selectivity*
**Snapshot date:** 2026-05-17
**Repository:** [github.com/Danny2045/physics-auditor](https://github.com/Danny2045/physics-auditor)

---

## How to use this document

This file is the canonical state-of-repository reference for Physics Auditor. It exists so that any reader — human collaborator, future Claude window, peer reviewer — can come up to speed on what the repository currently is, what is reproducible from committed state, what claims are supported by the code as it stands at the snapshot commit, and what work is outstanding, **without reconstructing that picture from chat history, PR descriptions, or speculative readings of the code**.

Specific guidance:

- **Read this before starting any substantial work** on Physics Auditor. The Outstanding Work Register (§9) is the authoritative list of open findings; if you are about to open a PR, check there first to confirm your work is scoped to a real open finding and is not already partially done on an existing branch.
- **This document is descriptive, not normative.** The [README](README.md) is the marketing-and-onboarding surface. [`CLAUDE.md`](CLAUDE.md) carries the four-line "Rules" block that is the closest thing this repository currently has to governance. The PfDHODH dossier's inline `claims` field (`benchmark/results/selectivity_maps/PfDHODH_vs_HsDHODH_selectivity.json`, on master at the snapshot commit) and the prose document next to it (`PFDHODH_SELECTIVITY_FINDINGS.md`) are the closest analogs to per-artifact provenance discipline. This file is descriptive: it tells you what *is*, not what *ought to be*.
- **This document is updated periodically, not continuously.** Every line carries an implicit "as of commit `e7064de`". Branch lists, missing-file tables, finding-status grids, and per-artifact reproducibility lines can drift between snapshots. When you make a substantive structural change to the repository, update this file as part of the same PR; when you only fix code without changing the structure of what's reproducible, you may leave it for the next snapshot. The "Update protocol" section at the end (§11) describes the cadence in more detail.
- **Do not introduce new scientific claims or new metrics in this document.** If a number is not already documented in committed state (a result artifact, a findings doc, the manuscript), it does not belong here. This file is a faithful summary; it is not a place to publish.
- **The Outstanding Work Register pairs every open finding with a "Natural next-PR scope" note.** Use those notes as starting points; refine them when you actually open the PR.
- **Paired with Kira.** Physics Auditor is the structure-validation half of the platform Kira anchors. The corresponding state document lives at [`STATE_OF_KIRA.md`](https://github.com/Danny2045/Kira/blob/main/STATE_OF_KIRA.md) in the Kira repository. See §12.

---

## 1. Repository overview

Physics Auditor is a JAX-based structure-validation toolkit for AI-generated protein-ligand complexes. It performs steric-clash detection, Lennard-Jones energy decomposition (including per-residue attribution), binding-pocket extraction and pairwise comparison, ESM-2 full-sequence-vs-pocket divergence amplification, and per-residue selectivity attribution between a target protein and its ortholog. The headline scientific result of the project — recovery of the published F188 / R265 / H185 antimalarial DHODH selectivity cluster blind, from a static LJ-only attribution between PfDHODH (4ORM, bound to DSM338) and HsDHODH (1D3H, bound to teriflunomide) — is on master.

The repository contains, at the snapshot commit on `master`:

- an **active engine** under `src/physics_auditor/` (19 Python files across 7 sub-packages: `core/`, `checks/`, `causality/`, `utils/`, `reference/`, `report/`, plus the package root),
- a **CLI** (`physics-auditor validate <file.pdb>` and `physics-auditor info <file.pdb>`) exposed via `pyproject.toml` `[project.scripts]`,
- a **benchmark suite** under `benchmark/` (54 files, ~22.8 MB) — 34 PDB structures (19 experimental including 4ORM + 15 AlphaFold v6 predictions), five benchmark scripts, and 16 result artifacts (the 14-pair clashscore JSON, six per-pair causality JSONs + a summary CSV + two findings documents, the ESM-2 divergence summary, the PfDHODH selectivity dossier + its findings doc, the HsDHODH-vs-HsDHFR functional-paralog stand-in + its findings doc),
- a **test suite** of 92 collected tests + 1 skip under `tests/` (all pass; ruff clean),
- a **manuscript** under `manuscript/` (`preprint.md` + `preprint.pdf`) reporting the 14-pair binding-site clashscore benchmark,
- and a small set of **root files** (README, SETUP.md, CLAUDE.md, pyproject.toml, .gitignore, .github/workflows/ci.yml, this document).

There is **no `docs/` directory** at the snapshot commit. Governance documents that exist in the sister repository (Kira's Scientific Constitution, Claim Inflation Audit, Disposition Plan) have no Physics Auditor counterpart yet — see Finding 5.

The full module-level map is in §5 (Methodology map).

---

## 2. Quick-reference dashboard

| Property | Value |
|---|---|
| Snapshot commit | `e7064de` |
| Snapshot date | 2026-05-17 |
| Main branch | `master` |
| Tests collected | 92 + 1 skipped |
| Tests passing | 92 |
| Tests skipped | 1 |
| Tests failing | 0 |
| Committed files (tracked) | 95 |
| Committed size | ~22.8 MB (overwhelmingly committed PDB structures under `benchmark/`) |
| Active engine modules under `src/physics_auditor/` | 19 (`.py`) |
| Implemented IsoDDE-style validation checks | 2 of 8 (steric clashes, Lennard-Jones energy) |
| Benchmark PDB structures | 34 (19 experimental + 15 AlphaFold v6 predicted) |
| Benchmark producing scripts | 5 (one clashscore, four under the causality/selectivity/divergence umbrella) |
| Benchmark result artifacts | 16 |
| Provenance sidecars (Kira-style `*_provenance.json`) | 0 |
| Inline-`claims` result artifacts | 2 (`PfDHODH_vs_HsDHODH_selectivity.json`, `divergence/summary.json`) |
| Merged PRs on master | 1 (PR #1) |
| Open PRs | 1 (PR #2, this document) |
| Merge commits on master | 1 (PR #1 merge at `e7064de`) |
| Open findings registered by this document | 7 |
| Governance documents (Scientific Constitution / Claim Inflation Audit / Disposition Plan) | 0 |
| CI matrix | Python 3.11, 3.13 on Ubuntu (`.github/workflows/ci.yml`) |

---

## 3. Git history and branch state

### 3.1 Remote configuration

- `origin` → `https://github.com/Danny2045/physics-auditor.git` (fetch + push)
- Refspec: `+refs/heads/*:refs/remotes/origin/*` (standard)

### 3.2 Last 20 commits on `master`

Master at the snapshot commit has 28 commits total. The 13 commits before `e7064de` are the contents of PR #1 (`feat/pfdhodh-selectivity`); everything before that was direct pushes during initial development.

```
e7064de Merge pull request #1 from Danny2045/feat/pfdhodh-selectivity
73bddb1 benchmark: PfDHODH-vs-HsDHODH selectivity attribution
02c15aa benchmark: add 4ORM (PfDHODH bound to DSM338)
d30be39 feat(utils): pdb_provenance — verify SOURCE before parse
a19a2a2 fix(selectivity-map): relabel 1D3H/1MVS run — HsDHODH vs HsDHFR, not Sm/Hs
e6ec67d benchmark: run_divergence.py — quantify Kira ESM-2 slogan with claims field
51c961d test(causality): cover divergence.py — identity, synthetic, validation, real data
91c5372 feat(causality): divergence.py — ESM-2 full vs pocket cosine
e2d23f6 chore: ignore .claude/ local agents directory
c55528e feat: selectivity-attribution map (per-residue compound preference)
1e2d7a6 fix: topology gaps surfaced by causality layer (disulfide + C-term OXT)
174c543 fix: parse_pdb_string also stops at first ENDMDL
e685f7d test: restore test_regressions.py (lost in causality patch base)
e4fabb9 feat: causality layer — per-residue LJ energy decomposition
96eac03 manuscript: update author name to Daniel Ngabonziza
f669792 manuscript: generate PDF from preprint markdown via pandoc
c5d3173 manuscript: fix version references and verify data consistency
cf392f8 manuscript: add Physics Auditor benchmark preprint draft v0.1
c752aed benchmark: expand to 14 pairs across major drug target families
4b0923a benchmark: expand to 9 pairs across diverse enzyme families
```

The full history before that (8 more commits) carries the initial benchmark expansion from 5 pairs to 9, the HDAC8 residue-numbering offset fix, the ENDMDL parser fix, the CLAUDE.md context update, the initial benchmark commit, install-command fix, ruff lint cleanup, CI workflow setup, and the initial commit (`88c2ea9`).

### 3.3 Merged PRs on master

- **PR #1** (`feat/pfdhodh-selectivity` → `master`, merged 2026-05-17 at `e7064de`). +68,135 / −14 lines, 36 files. Brings the headline PfDHODH selectivity attribution work, the three causality module implementations (`energy_decomp.py`, `divergence.py`, `selectivity_map.py`), `utils/pdb_provenance.py`, four new benchmark scripts, the 4ORM PDB structure, six new test files, and 15 new result artifacts. Merged with a merge commit, not squash. See §13 for the post-merge update record.

This is the only PR merged into master at the snapshot commit.

### 3.4 Outstanding branches

The following branches exist at the snapshot commit. **No branch is deleted by the PR that introduces this document.** Cleanup is left to deliberate follow-up PRs so that a human can confirm each deletion.

#### Branches that are strict prefixes of the merged PR #1 stack (deletable)

Three local + remote branches formed the linear stack that ended at `feat/pfdhodh-selectivity`. Every commit on these branches is now reachable from `master` via the PR #1 merge. They carry no unique work and are candidates for deletion both locally and on `origin`.

| Branch | Local + origin? | Subsumed by |
|---|---|---|
| `feat/causality-energy-decomp` (head `1e2d7a6`) | Both | PR #1 merge at `e7064de` |
| `feat/selectivity-map` (head `e2d23f6`) | Both | PR #1 merge at `e7064de` |
| `feat/divergence` (head `a19a2a2`) | Both | PR #1 merge at `e7064de` |

The PR #1 head branch `feat/pfdhodh-selectivity` itself was **already deleted from `origin` at merge time** (standard GitHub post-merge cleanup) and remains locally only — also a candidate for deletion.

#### Pushed bug-fix branch (Finding 4)

| Branch | Head commit | Ahead of master | Purpose | Pushed to origin? |
|---|---|---:|---|---|
| `fixes/technical-pass-1` | `dd9d085` | 2 | Clash subscore recalibration (linear `1 − clashscore/40` → sigmoidal `1 / (1 + (clashscore/50)^2)` with `accept=0.80`, `short_md=0.50`), O(N) rewrite of `extract_binding_site`, `parse_pdb_string` ENDMDL fix (already independently applied as commit `174c543` on the merged feat stack), LJ subscore wired into the CLI composite via renormalized weights, `core/energy.py` docstring fix (`n_hot_pairs` not `n_clashing_pairs`), and a 135-line regression test suite. | **Yes — pushed by Phase 2 of the audit producing this document.** Was local-only when the audit began; pushing eliminated the data-loss risk. |

This branch needs to be rebased against the post-merge master and opened as a PR. With PR #1 now merged, the rebase will conflict against the causality-layer changes in `binding_site.py` (commit `e4fabb9`) and against the `test_regressions.py` independently restored on the feat stack at `e685f7d`. See Finding 4.

#### Branch introduced by this document

| Branch | Head commit | Ahead of master | Purpose |
|---|---|---:|---|
| `audit/state-of-physics-auditor-document` | (this document's commit) | 1+ | Commits and maintains this state-of-repository reference. PR #2. |

### 3.5 Submodules

None. The repository is self-contained.

---

## 4. File tree inventory

The repository has **95 committed files** totaling **~22.8 MB** at the snapshot commit on `master`. The size is dominated by committed PDB structures (~21.0 MB across 34 files under `benchmark/structures/`); the active source tree is small.

### 4.1 Top-level directory breakdown

| Top-level dir | Files | Committed bytes |
|---|---:|---:|
| `benchmark/` | 54 | 22,842,520 |
| `src/` | 19 | 116,967 |
| `tests/` | 13 | 78,828 |
| `manuscript/` | 2 | 76,432 |
| `STATE_OF_PHYSICS_AUDITOR.md` | 1 | (this file) |
| `.github/` | 1 | 505 |
| Root files (`README.md`, `SETUP.md`, `CLAUDE.md`, `pyproject.toml`, `.gitignore`) | 5 | 16,401 |

### 4.2 `benchmark/` sub-breakdown

| Sub-directory | Files | Purpose |
|---|---:|---|
| `benchmark/` (root) | 5 | `benchmark_pocket_clashscore.py` + four `run_*.py` scripts (causality decomposition, divergence, selectivity map, PfDHODH selectivity) |
| `benchmark/results/` | 1 | `pocket_clashscore_comparison.json` (14 entries) |
| `benchmark/results/causality/` | 9 | Six per-pair `*_causality.json` (HsDHODH, SmDHODH, HsHDAC8, LmPTR1, HsAromatase, CoV2Mpro), `causality_summary.csv`, `CAUSALITY_FINDINGS.md`, `TOPOLOGY_FIXES.md` |
| `benchmark/results/divergence/` | 1 | `summary.json` (ESM-2 full-sequence-vs-pocket cosines, with inline `claims` per pair) |
| `benchmark/results/selectivity_maps/` | 4 | `PfDHODH_vs_HsDHODH_selectivity.json` (headline result, with inline `claims`), `PFDHODH_SELECTIVITY_FINDINGS.md`, `HsDHODH_vs_HsDHFR_pocket_attribution.json` (the corrected functional-paralog stand-in), `SELECTIVITY_FINDINGS.md` |
| `benchmark/structures/experimental/` | 19 | Crystal PDBs: 1D3H, 1HSG, 1HWL, 1M17, 1MVS, 1UBQ, 1X70, 2BPR, 2HYY, 2QWF, 2X99, 3EQM, 3LN1, 3NJW, 3OG7, 4EY7, 4HJO, 4ORM, 5R82 |
| `benchmark/structures/predicted/` | 15 | AlphaFold Database v6 predictions, one per 14-pair-benchmark pair |
| `benchmark/structures/predicted/boltz2/` | 0 | Empty placeholder for future Boltz-2 predictions |
| `benchmark/structures/predicted/chai1/` | 0 | Empty placeholder for future Chai-1 predictions |

Three of the 19 experimental PDBs (1HSG, 2QWF, 3NJW) are committed but not referenced by any benchmark script. They are present for ad-hoc use.

The 14-pair clashscore benchmark pairs the following experimental ↔ AlphaFold structures (in benchmark order):

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
│   └── energy.py         (Lennard-Jones kernel: total, per-atom, per-residue, per-residue-vs-ligand)
├── checks/
│   ├── __init__.py
│   └── clashes.py        (steric clash detection + ClashResult dataclass + clashscore)
├── causality/
│   ├── __init__.py
│   ├── binding_site.py   (pocket extraction + pairwise sequence-identity comparison)
│   ├── divergence.py     (ESM-2 full-sequence vs pocket cosine via fair-esm)
│   ├── energy_decomp.py  (per-residue LJ decomposition + per-residue difference between structures)
│   └── selectivity_map.py(per-residue ligand-LJ attribution between target and ortholog)
├── utils/
│   ├── __init__.py
│   └── pdb_provenance.py (verify_pdb_source — the SOURCE-record gate)
├── reference/
│   └── __init__.py       (1-line placeholder; no reference data committed)
└── report/
    └── __init__.py       (1-line placeholder; no HTML/JSON reporter beyond CLI)
```

`reference/` and `report/` are still empty docstring-only packages.

### 4.4 `tests/` map

```
tests/
├── fixtures/
│   ├── clashing.pdb        (test fixture, 975 B)
│   └── tri_ala.pdb         (test fixture, 1.3 KB)
├── test_parser.py          (TestParsePDB)
├── test_topology.py        (TestBondInference, TestBondedMask, TestVdWRadii)
├── test_geometry.py        (TestDistanceMatrix, TestDihedralAngles, TestBondAngles, TestBackboneDihedrals)
├── test_energy.py          (TestLJEnergy, TestClashDetection)
├── test_causality.py       (TestBindingSiteExtraction, TestPocketComparison)
├── test_disulfide.py       (S-S bond inference via the topology cutoff)
├── test_divergence.py      (identity, synthetic, validation, real-data ESM-2 cosines; one skipped test gated on a heavy model download)
├── test_energy_decomp.py   (per-residue decomposition + per-residue difference)
├── test_pdb_provenance.py  (real-file match, real-file mismatch, missing file, malformed SOURCE — uses 1D3H and 4ORM as real fixtures)
├── test_regressions.py     (regressions surfaced during the causality patch base)
└── test_selectivity_map.py (compute_selectivity_map end-to-end + ranking helpers)
```

92 passing tests, 1 skipped. `pytest tests/ -q` returns `92 passed, 1 skipped in 1.58s`; `ruff check src/ tests/ --select E,F,I,W --ignore E501` returns `All checks passed!`. CLAUDE.md still says *"45 tests, all must pass"* — that line is out of date and is tracked in §10.

---

## 5. Methodology map

Module-level map at master commit `e7064de`. Functions and classes listed are those exposed at the module surface; private helpers are omitted unless load-bearing.

### `src/physics_auditor/core/parser.py`

- **Purpose:** PDB file parser → flat numpy arrays suitable for vectorized JAX computation. Stops at the first `ENDMDL` so NMR ensembles do not concatenate all models. `parse_pdb_string` (the string-API counterpart) carries the same ENDMDL behavior since commit `174c543`.
- **Public surface:** `Atom`, `Residue`, `Chain`, `Structure` dataclasses; `parse_pdb(path, keep_hydrogens, keep_altloc)`, `parse_pdb_string(pdb_string, name, ...)`.
- **Reads:** PDB files.
- **Writes:** none.
- **Test coverage:** `tests/test_parser.py`.

### `src/physics_auditor/core/topology.py`

- **Purpose:** Infers covalent bonds (residue templates + peptide bonds + distance-based fallback for ligands), builds the non-bonded mask (default excludes 1-2 and 1-3 pairs), provides vdW radii lookups. Disulfide bonds are inferred via the `S-S 2.6 Å` cutoff in `COVALENT_BOND_CUTOFFS`.
- **Public surface:** `infer_bonds_from_topology(structure)`, `build_bonded_mask(n_atoms, bonds, exclude_neighbors=3)`, `build_1_4_mask(n_atoms, bonds)`, `get_vdw_radius(element)`, `get_vdw_radii_array(structure)`; module-level constants `COVALENT_BOND_CUTOFFS`, `BACKBONE_BONDS`, `SIDECHAIN_BONDS` (templates for the 20 standard amino acids plus MSE), `VDW_RADII` (Bondi 1964 / AMBER values).
- **Reads:** none.
- **Writes:** none.
- **Test coverage:** `tests/test_topology.py`, plus `tests/test_disulfide.py` for the S-S inference path.

### `src/physics_auditor/core/geometry.py`

- **Purpose:** JIT-compiled JAX geometry kernels and a NumPy helper that extracts per-residue backbone φ/ψ/ω quads.
- **Public surface:** `@jax.jit compute_distance_matrix(coords)`, `compute_dihedral_angles(p0, p1, p2, p3)`, `compute_bond_angles(p0, p1, p2)`, `compute_distances(p0, p1)`; `extract_backbone_dihedrals(coords, atom_names, res_indices, is_protein, chain_ids)` (returns a dict with `phi`, `psi`, `omega` keys, each with `angles` and `res_indices`).
- **Reads:** none.
- **Writes:** none.
- **Test coverage:** `tests/test_geometry.py`.

### `src/physics_auditor/core/energy.py`

- **Purpose:** Lennard-Jones kernel using Lorentz-Berthelot combining rules and element-level (not atom-type-level) σ/ε parameters.
- **Public surface:** `get_lj_params_arrays(elements)`, `@jax.jit compute_lj_energy_matrix(...)`, `compute_total_lj_energy()`, `compute_per_atom_lj_energy()`, `compute_per_residue_lj_energy()`, `run_lj_analysis(...)`, `per_residue_ligand_interaction_energy(...)` (used by the selectivity-map module).
- **Note:** `run_lj_analysis` returns a dict with `total_energy`, `per_atom_energy`, `per_residue_energy`, `energy_matrix`, `n_hot_pairs` (pairs with E > 10 kcal/mol — the docstring on master says `n_clashing_pairs`; the key was renamed but the docstring was not, fixed on `fixes/technical-pass-1`).
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

### `src/physics_auditor/causality/energy_decomp.py`

- **Purpose:** Per-residue Lennard-Jones decomposition and per-residue difference between two homologous structures at aligned pocket positions. Powers the causality benchmark (`run_causality_decomposition.py`).
- **Public surface:** `per_residue_decomposition(structure, pocket_residue_indices=None)`, `per_residue_difference(target, ortholog, alignment=None)`, `decomposition_to_dict(...)`, `difference_to_dict(...)`.
- **Reads:** none.
- **Writes:** none.
- **Test coverage:** `tests/test_energy_decomp.py`.

### `src/physics_auditor/causality/selectivity_map.py`

- **Purpose:** End-to-end per-residue selectivity attribution between a target and its ortholog, each with bound ligand. Produces the PfDHODH headline result.
- **Public surface:** `compute_selectivity_map(target_structure, target_ligand_resname, ortholog_structure, ortholog_ligand_resname, pocket_cutoff=5.0) -> SelectivityMap`; the `SelectivityMap` dataclass with `SelectivityResiduePair` entries; convenience methods `top_n_target_selective(n)`, `top_n_ortholog_selective(n)`; `selectivity_map_to_dict(smap, top_n=15)` for dossier serialization.
- **Module docstring carries a `NON-CLAIMS` block.** It states explicitly that the module computes LJ-only attribution (no electrostatics, hydrogen bonds, solvation, or entropy), that two compared structures must have ligands in comparable poses (the module does not align ligand poses), that the default sequential pocket alignment is appropriate only for closely homologous pockets, that different compounds on each side conflate residue chemistry with compound geometry, and that the per-residue numbers are not free energies.
- **Reads:** none.
- **Writes:** none.
- **Test coverage:** `tests/test_selectivity_map.py`.

### `src/physics_auditor/causality/divergence.py`

- **Purpose:** ESM-2 full-sequence-vs-pocket cosine divergence, operationalizing the slogan *"SmDHODH and HsDHODH share full-sequence ESM-2 cosine 0.9897 yet admit ~30× experimental compound selectivity for the parasite."*
- **Public surface:** `compute_divergence(target_sequence, ortholog_sequence, target_pocket_indices, ortholog_pocket_indices, model_name="esm2_t33_650M_UR50D") -> DivergenceResult`. Returns three cosine numbers per pair: `full_sequence_cosine`, `pocket_meanpool_cosine`, `pocket_subseq_cosine`; the reported "divergence amplification" is `full − pocket_meanpool`.
- **Module docstring carries a `NON-CLAIMS` block.** Cosines are not free energies; pocket indices are caller-supplied 0-based positions and the caller is responsible for ensuring they reference the supplied sequence; results depend on the ESM-2 layer used and the model name is pinned in the output.
- **Reads:** ESM-2 model weights (downloaded by `fair-esm` on first use; not committed).
- **Writes:** none.
- **Test coverage:** `tests/test_divergence.py` (one test skipped pending the heavy model download).

### `src/physics_auditor/utils/pdb_provenance.py`

- **Purpose:** SOURCE-record provenance gate. Reads a PDB file's `SOURCE` records, extracts `ORGANISM_SCIENTIFIC`, and raises `ValueError` on any mismatch with the caller's expected organism. The module docstring records the May 2026 mislabeling incident (commit `a19a2a2`) as the reason the helper exists.
- **Public surface:** `verify_pdb_source(path: Path, expected_organism: str) -> None`.
- **Enforcement scope:** The gate is called by `benchmark/run_pfdhodh_vs_hsdhodh_selectivity.py` before any PDB parse. **`parse_pdb()` does not call it.** Most other benchmark scripts do not call it either. See Finding 2.
- **Reads:** PDB files (as text, for the SOURCE block; `parse_pdb` does the actual ATOM parsing).
- **Writes:** none.
- **Test coverage:** `tests/test_pdb_provenance.py` (real-file match, real-file mismatch, missing file, malformed SOURCE — uses 1D3H and 4ORM as real fixtures).

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
| Disulfide geometry | (no module) | **Absent.** `DisulfideConfig` exists. `tests/test_disulfide.py` exercises the `S-S 2.6 Å` covalent cutoff in `topology.py`, not a dedicated check module. |

**2 of 8 checks implemented.** The other six are scaffolding only. See Finding 3.

---

## 6. Expected-but-uncommitted file registry

A grep across `src/`, `tests/`, and `benchmark/` for hardcoded paths surfaces every file the active code expects to find. **At master commit `e7064de`, every referenced path is committed.** The repository commits its PDB inputs directly into the tree (~21 MB across `benchmark/structures/`) instead of relying on a download step, which eliminates the entire class of expected-but-missing-data failures.

| Reference site | Path | Status |
|---|---|---|
| `benchmark/benchmark_pocket_clashscore.py` `BENCHMARK_PAIRS` | 14× `benchmark/structures/experimental/*.pdb` | All committed |
| same | 14× `benchmark/structures/predicted/AF-*-F1-model_v6.pdb` | All committed |
| same | `benchmark/results/pocket_clashscore_comparison.json` | Committed |
| `benchmark/run_pfdhodh_vs_hsdhodh_selectivity.py` | `benchmark/structures/experimental/{4ORM,1D3H}.pdb` | Both committed |
| `benchmark/run_causality_decomposition.py`, `run_divergence.py`, `run_selectivity_map.py` | various PDBs under `benchmark/structures/` plus their result JSONs | All committed; six pairs covered for causality (8 skipped at memory budget — see §7.4) |
| `tests/test_parser.py:112` | `/nonexistent/path.pdb` | Intentional negative-test sentinel; not a real path |
| `tests/test_pdb_provenance.py` | `1D3H.pdb`, `4ORM.pdb` (real fixtures) | Both committed |
| `benchmark/structures/predicted/{boltz2,chai1}/` | (empty placeholder dirs) | No code references them; reserved for future predictors |

Three experimental PDBs (1HSG, 2QWF, 3NJW) are committed but not referenced by any committed code — they appear to be reserved for ad-hoc use.

`reference/` advertises (in its package docstring) "ideal bond parameters, Ramachandran distributions, rotamer libraries, LJ params" but **commits none** of those reference datasets. Implementing the six missing checks (Finding 3) requires populating this directory; the current state is descriptive of an intent, not of any data.

ESM-2 model weights (`esm2_t33_650M_UR50D`) used by `causality/divergence.py` are not committed; `fair-esm` downloads them on first use. This is the only external dependency the repository pulls at runtime.

---

## 7. Per-artifact reproducibility

This section documents which committed artifacts can be regenerated from committed inputs by committed code, and which cannot.

### 7.1 PfDHODH selectivity attribution dossier

- **Artifact:** `benchmark/results/selectivity_maps/PfDHODH_vs_HsDHODH_selectivity.json`
- **Producing script:** `benchmark/run_pfdhodh_vs_hsdhodh_selectivity.py`
- **Inputs:** `benchmark/structures/experimental/4ORM.pdb` (committed in `02c15aa`) and `benchmark/structures/experimental/1D3H.pdb` (committed earlier).
- **Provenance gate:** Yes. The script calls `verify_pdb_source(PF_DHODH_PDB, "PLASMODIUM FALCIPARUM")` and `verify_pdb_source(HS_DHODH_PDB, "HOMO SAPIENS")` before parsing. Both SOURCE records have been confirmed independently (4ORM → `PLASMODIUM FALCIPARUM` strain 3D7; 1D3H → `HOMO SAPIENS`).
- **Reproducibility check (executed on master at `e7064de`):** The script was re-run from the committed inputs against current master. The regenerated JSON matched the committed dossier **byte-for-byte** (`git diff --stat HEAD -- benchmark/results/selectivity_maps/PfDHODH_vs_HsDHODH_selectivity.json` returned empty). The headline result — pocket Δ +12.78 kcal/mol; top-ranked target-selective residues HIS185 (+4.34), PHE188 (+3.32), VAL532 (+3.01), PHE227 (+2.07), LEU531 (+1.69); ARG265 at rank 6 (+1.50) — is fully reproducible from committed state on master.
- **Claims discipline:** The dossier carries an inline top-level `claims` field with four keys (`what_this_is`, `what_this_is_not`, `supported_finding`, `pending_followup`). The prose findings document next to it (`PFDHODH_SELECTIVITY_FINDINGS.md`) restates the same framing with a literature cross-check section explicitly naming the F188/R265/H185 cluster from the Phillips/Rathod DSM-series antimalarial literature.

### 7.2 14-pair binding-site clashscore benchmark

- **Artifact:** `benchmark/results/pocket_clashscore_comparison.json` (14 pair entries)
- **Producing script:** `benchmark/benchmark_pocket_clashscore.py`
- **Inputs:** 28 PDBs (14 experimental + 14 AlphaFold v6 predictions), all committed under `benchmark/structures/`.
- **Provenance gate:** **None.** The script reads PDBs via `parse_pdb()` directly; `verify_pdb_source` is not called. See Finding 2.
- **Reproducibility check:** Not re-executed at the snapshot. The script does not require any external downloads and has no random sampling, so byte-identical regeneration is expected but not verified here. A future PR closing Finding 1 will re-run this benchmark with corrected labels and confirm bit-identity (or document any drift).
- **Mislabel issue:** Pair 2 is labeled `SmDHODH` but uses `1D3H.pdb` whose SOURCE record reads `HOMO SAPIENS`. The structure is the catalytic construct of HsDHODH (UniProt Q02127), not the parasite enzyme. The AlphaFold side (`AF-G4VFD7-F1-model_v6.pdb`) is the true SmDHODH; the experimental side is mislabeled. The mislabel propagates into `manuscript/preprint.md` Table 1. See Finding 1.

### 7.3 14-pair benchmark manuscript

- **Artifact:** `manuscript/preprint.md` + `manuscript/preprint.pdf`
- **Inputs:** the 14-pair clashscore JSON above, plus narrative prose.
- **Reproducibility:** The PDF is generated from the markdown via `pandoc` per commit `f669792`. The numeric content depends on the upstream clashscore JSON; correcting Finding 1 will require regenerating both the JSON and the manuscript.

### 7.4 Causality decomposition

- **Artifacts:** `benchmark/results/causality/{HsDHODH,SmDHODH,HsHDAC8,LmPTR1,HsAromatase,CoV2Mpro}_causality.json` (6 of the 14 benchmark pairs), `causality_summary.csv`, `CAUSALITY_FINDINGS.md`, `TOPOLOGY_FIXES.md`.
- **Producing script:** `benchmark/run_causality_decomposition.py`.
- **Coverage gap:** **6 of 14 pairs.** `CAUSALITY_FINDINGS.md` records that 8 pairs were skipped due to a 4500-atom memory budget on the JAX distance matrix. Closing this gap requires either pocket-restricted decomposition, chunked computation, or raising the budget on a larger machine — see §10.
- **Provenance gate:** None. The script reads PDBs via `parse_pdb()`. See Finding 2.

### 7.5 ESM-2 divergence

- **Artifact:** `benchmark/results/divergence/summary.json`.
- **Producing script:** `benchmark/run_divergence.py`.
- **Headline number:** `full_sequence_cosine = 0.989732` for the true SmDHODH (G4VFD7) vs HsDHODH (Q02127) UniProt sequences — matches the Kira-cited "0.9897" slogan to four decimals.
- **Pocket-level leg:** Only computed for the 1D3H↔1MVS pair (HsDHODH catalytic construct vs HsDHFR — a functional-paralog pair, not an ortholog pair). The true ortholog pocket-divergence number is **not yet computable from in-repo structures** because no parasite-side DHODH co-crystal is committed.
- **External dependency:** ESM-2 model weights are downloaded by `fair-esm` on first use; one test (in `tests/test_divergence.py`) is skipped pending this download.

### 7.6 HsDHODH-vs-HsDHFR functional-paralog selectivity stand-in

- **Artifacts:** `benchmark/results/selectivity_maps/HsDHODH_vs_HsDHFR_pocket_attribution.json`, `SELECTIVITY_FINDINGS.md`.
- **Producing script:** `benchmark/run_selectivity_map.py`.
- **Framing:** This is the **corrected** version of the run that was originally written up as "SmDHODH vs HsDHODH" before the 2026-05-11 mislabel review (commit `a19a2a2`). The numerical content is valid; the framing is now honest: 1D3H is the HsDHODH catalytic construct (UniProt Q02127) and 1MVS is HsDHFR (a different human enzyme). The pair is a functional paralog comparison, not an ortholog comparison.
- **Provenance gate:** Not yet wired into this script. See Finding 2.

### 7.7 Test fixtures and CLI golden outputs

- **Fixtures:** `tests/fixtures/{tri_ala,clashing}.pdb` are tiny hand-crafted PDBs. They are not regenerated; they are the inputs.
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

The closest existing analog to per-artifact claim discipline is the inline `claims` field used by two artifacts on master:

1. `benchmark/results/selectivity_maps/PfDHODH_vs_HsDHODH_selectivity.json` carries a top-level `claims` object with four keys (`what_this_is`, `what_this_is_not`, `supported_finding`, `pending_followup`). The accompanying `PFDHODH_SELECTIVITY_FINDINGS.md` document includes explicit "Non-claims" and "Honesty note on the literature cross-check" sections.
2. `benchmark/results/divergence/summary.json` carries per-pair `claims` lists and a `notes` field documenting known caveats.

There is no schema, no validator, no test that checks for `claims` presence on new result artifacts. The pattern is informal and applied per-artifact.

### 8.2 README accuracy review

The README makes a number of claims; verdict against the code as it stands at the snapshot commit:

| Claim in README | Verdict |
|---|---|
| *"Physics Validation (8 checks)"* listing clashes, bond geometry, Ramachandran, peptide planarity, chirality, rotamer outliers, Lennard-Jones, disulfide geometry | **Overclaims.** 2 of 8 implemented. `SETUP.md`'s "What's next" table acknowledges 6 are not built. README and SETUP.md contradict each other. See Finding 3. |
| *"Mechanistic Causality"* — per-residue energy decomposition, binding-site comparison, selectivity attribution | **Accurate.** All three are implemented at master: `causality/energy_decomp.py`, `causality/binding_site.py`, `causality/selectivity_map.py`. |
| *"Trust Report — JSON (for pipelines) + HTML (for humans)"* | **HTML not implemented.** `report/__init__.py` is a 1-line placeholder; CLI emits JSON and rich-text only. See Finding 7. |
| *"Per-residue heatmaps and binding-site annotations"* | **Not implemented.** No plotting code in the repository. See Finding 7. |
| *"Composite physics score (0–1)"* + *"accept / run short MD / discard"* | Implemented, but the composite is `clash_result.subscore` only — the CLI's own comment marks this as "simplified for v0.1." `fixes/technical-pass-1` wires in the LJ subscore. |
| `physics-auditor selectivity --target ... --ortholog ... --ligand ...` (Quick Start example) | **Subcommand does not exist.** `cli.py` registers only `validate` and `info`. The selectivity machinery exists as a script (`benchmark/run_pfdhodh_vs_hsdhodh_selectivity.py`), not a CLI subcommand. See Finding 7. |
| *"Why Not Molprobity? 1. Calibrated for AI structures."* | **Unverified.** No calibration data is committed. The current thresholds (clashscore decay to zero by 40 clashes/1000 atoms; `accept=0.85`, `short_md=0.60`) appear to be defaults. `fixes/technical-pass-1` recalibrates these against the 14-pair benchmark. |
| `pip install git+https://github.com/Danny2045/physics-auditor.git` | Works. |

### 8.3 CLAUDE.md accuracy

`CLAUDE.md` opens with `"JAX-based, 8 checks, CLI"` — same overclaim as the README. The architecture section still names only the master-pre-PR-1 surface (`parser`, `geometry`, `topology`, `energy`, `checks/clashes`, `causality/binding-site`, `benchmark/`) and is missing the post-merge additions (`utils/pdb_provenance`, `causality/{energy_decomp,divergence,selectivity_map}`). The "Key commands" line still says *"45 tests, all must pass"* — wrong post-merge; the actual count is 92 + 1 skipped. The "Active work" section also documents the HDAC8 residue-numbering offset issue as if it were open; it was in fact closed by commit `81d644a`. These are routine post-merge maintenance items tracked in §10.

### 8.4 SETUP.md accuracy

`SETUP.md` is **the most accurate top-level document.** Its "What's next" table cleanly separates implemented modules from planned ones. Some entries that were "planned" when SETUP.md was written are now implemented (the causality stack, `selectivity_map`, `energy_decomp`, `divergence`); SETUP.md has not been updated to reflect this. Its expected pytest output ("45 tests, all pass") matches the pre-merge state, not the current 92 + 1 skipped.

---

## 9. Outstanding work register

The repository carries **eleven** findings. The first seven were surfaced by the original audit (Findings 1–7). The last four (Findings 8–11) were surfaced by PR-A1's `verify_pdb_source` enforcement work and planned benchmark rerun: Findings 8 and 9 are wrong-protein errors in the benchmark inputs (2BPR is DnaK, 4HJO is EGFR), Finding 10 is a construct-artifact annotation (3OG7 is an AKAP9-BRAF fusion), and Finding 11 is the cross-environment numerical drift of the pocket-clashscore pipeline that prevents PR-A1 from regenerating the benchmark JSON. Each finding pairs current status with a "Natural next-PR scope" line. The scope lines are suggestions for the future PR that closes the finding; they are not commitments and the engineer who picks up the work is expected to refine them.

### Finding 1 — Mislabel propagation in master-published benchmark and manuscript

**Status:** Partially closed by PR `fix/benchmark-mislabel-pair1-and-causality-artifacts` (PR-A1). BENCHMARK_PAIRS schema corrected: pair 1 mapped to 1D3H ↔ AF-Q02127 (true HsDHODH-vs-HsDHODH); pairs 2, 3, 5 removed pending substitutions (PR-B, PR-D, PR-C); pair 8 annotated for AKAP9-BRAF construct. Causality artifact filenames and internal labels corrected. The `manuscript/preprint.md` edits and the `benchmark/results/pocket_clashscore_comparison.json` regeneration are deferred to PR-A2 pending Finding 11 resolution.

`benchmark/results/pocket_clashscore_comparison.json` (pair 2) labels `1D3H.pdb` as "SmDHODH" and pairs it with the AlphaFold model `AF-G4VFD7-F1-model_v6.pdb` (which *is* SmDHODH). But `1D3H.pdb`'s SOURCE record reads `ORGANISM_SCIENTIFIC: HOMO SAPIENS`; the structure is the catalytic construct of *human* DHODH (UniProt Q02127), not the parasite. The numerical clashscore values are valid for the underlying experimental structure (1D3H); the *label* "SmDHODH" attached to it is wrong.

The mislabel propagates into `manuscript/preprint.md` Table 1, where the "SmDHODH" row reads `1D3H | G4VFD7 | …` — pairing a human PDB with a parasite UniProt ID. The same conflation is in the producing script's `BENCHMARK_PAIRS` description string: *"Parasite DHODH — 30.8x selectivity target."*

The mislabel was originally caught and recorded during the `feat/divergence` review on 2026-05-11 (commit `a19a2a2`: *"relabel 1D3H/1MVS run — HsDHODH vs HsDHFR, not Sm/Hs"*) for a *different* result artifact (the early HsDHODH-vs-HsDHFR selectivity-map run, now `benchmark/results/selectivity_maps/HsDHODH_vs_HsDHFR_pocket_attribution.json`). The 14-pair benchmark and the manuscript were not updated; PR #1 added the provenance machinery and used it correctly for the PfDHODH run, but did not retro-fix the master-side artifacts.

**Natural next-PR scope:** Relabel the affected entries — pair 2 in `pocket_clashscore_comparison.json` becomes either the true HsDHODH-vs-HsDHODH pair (if `AF-Q02127` is the right AF model) or is split into two pairs (one HsDHODH with 1MVS/Q02127, one true SmDHODH using an AF-only "experimental" stand-in or a different parasite-DHODH crystal if one exists in the PDB). Regenerate the JSON (which should be bit-identical for every entry except pair 2 once labels are fixed). Update `manuscript/preprint.md` Table 1 and any narrative references. Add a call to `verify_pdb_source` to `benchmark_pocket_clashscore.py`'s pair iteration so the mislabel cannot recur. Consider whether the manuscript needs to be re-bumped to a new draft version with an acknowledgments note about the relabel.

### Finding 2 — `verify_pdb_source` built but not enforced

**Status:** Closed by PR `fix/benchmark-mislabel-pair1-and-causality-artifacts` (PR-A1) for `benchmark_pocket_clashscore.py`. `verify_pdb_source` is now enforced inside the pair-iteration loop with `expected_organism_experimental` and `expected_organism_predicted` fields on every entry in `BENCHMARK_PAIRS`. The gate's enforcement surfaced both pair 3 (4HJO = EGFR, not HsHDAC8) and pair 5 (2BPR = DnaK, not LmPTR1) wrong-protein errors during PR-A1 — a worked example of why provenance verification is load-bearing. Three other benchmark scripts (`run_causality_decomposition.py`, `run_divergence.py`, `run_selectivity_map.py`) still bypass the gate; their enforcement is tracked separately.

`src/physics_auditor/utils/pdb_provenance.py` implements `verify_pdb_source(path, expected_organism)`, which reads the PDB file's `SOURCE` records, extracts `ORGANISM_SCIENTIFIC`, and raises `ValueError` on any mismatch. The docstring asserts the policy explicitly:

> *"every PDB ingestion path must call `verify_pdb_source(path, expected_organism)` before parsing."*

In practice, the gate is called by exactly one consumer at the snapshot commit: `benchmark/run_pfdhodh_vs_hsdhodh_selectivity.py`. **`src/physics_auditor/core/parser.py:parse_pdb()` does not call it**, and every other benchmark script (`benchmark_pocket_clashscore.py`, `run_causality_decomposition.py`, `run_divergence.py`, `run_selectivity_map.py`) reads PDBs via `parse_pdb()` without the provenance check. The mislabel that surfaced as Finding 1 is precisely the class of mistake the gate was built to prevent — but the gate is not enforced where it would have caught it.

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

`SETUP.md`'s "What's next" table is honest about this. The README contradicts SETUP.md.

The CLI itself flags the gap inline at `cli.py:96`:

> *"Full composite will include all 8 checks — for now use clashes + LJ. Simplified for v0.1."*

PR #1's contribution — the causality stack and the selectivity-attribution machinery — closes the README's claims about *causality* but does not touch the eight-validation-check claim.

**Natural next-PR scope (split):**

1. **README rewrite PR.** Mirror SETUP.md's framing. Use an "implemented vs. planned" table; remove the unqualified "8 checks" claim from prose; pair every planned check with its config scaffold so the surface is honest. This is doc-only and should land before any implementation work.
2. **Per-check implementation PRs.** Bond geometry is the lowest-hanging fruit: a single Engh-Huber bond-length and bond-angle reference table can drive a z-score check against `core/geometry.py::compute_bond_angles` and the residue-template bond list in `topology.py`. Land each check as its own PR with a dedicated test fixture, dedicated reference data committed under `reference/`, and a CLI extension that surfaces the new subscore in the rich-text panel and the JSON report.

### Finding 4 — `fixes/technical-pass-1` was local-only

**Status:** Operationally mitigated (branch pushed to origin); rebase + PR still pending against the post-merge tree.

The branch was created on 2026-05-06 and remained local-only until the audit producing this document pushed it to `origin/fixes/technical-pass-1`. The branch is now durable; it still needs to be rebased, conflict-resolved, and merged.

The branch carries (per commit `0b3e16e`):

- **Clash subscore recalibration.** Linear `1 − clashscore/40` (which made every experimental crystal in the 14-pair benchmark — clashscore 11–56 — fall in DISCARD or SHORT_MD, contradicting the tool's own positioning) is replaced with sigmoidal `1 / (1 + (clashscore/50)^2)` and thresholds shift to `accept=0.80`, `short_md=0.50`.
- **`extract_binding_site` rewrite.** O(N) via vectorized `np.isin` instead of the original O(N·P) inner scan. Material for large structures (e.g. MmCOX2 at 33k atoms).
- **`parse_pdb_string` ENDMDL fix.** The string-API counterpart to the master commit `a65d454` fix on `parse_pdb`. **Already independently applied to master as commit `174c543`** via the PR #1 merge — this part of the technical-pass branch is redundant and will resolve cleanly during rebase.
- **LJ subscore wiring.** `n_hot_pairs_per_1000_atoms` becomes a real subscore with the same sigmoidal decay; it is folded into the composite via renormalized `CompositeConfig` weights and surfaced in both JSON and the rich-text panel. The CLI now reports `roadmap_checks` (the planned but unimplemented checks) alongside the active subscores.
- **`core/energy.py` docstring fix.** The `n_hot_pairs` / `n_clashing_pairs` key-name mismatch is corrected.
- **`tests/test_regressions.py`.** 135-line regression test suite for the calibration and wiring above. A different `tests/test_regressions.py` was independently restored on the merged feat stack at commit `e685f7d` (*"test: restore test_regressions.py (lost in causality patch base)"*); the rebase must merge the two test files.

**Natural next-PR scope:** Rebase `fixes/technical-pass-1` against the post-merge master at `e7064de`. Expected conflict points:

- `src/physics_auditor/causality/binding_site.py` — O(N) rewrite vs. the causality-layer's modifications to the same file (commit `e4fabb9`).
- `src/physics_auditor/cli.py` — composite wiring vs. CLI on master (CLI was not touched by PR #1).
- `tests/test_regressions.py` — two independent versions must be reconciled.

Open as a PR titled along the lines of *"fix: technical pass — clash subscore calibration, O(N) pocket, LJ composite wiring"*, request review, merge.

### Finding 5 — No governance documents

**Status:** Open.

Unlike Kira, Physics Auditor has no Scientific Constitution, no Claim Inflation Audit, no Disposition Plan, no `docs/` directory. The only governance content is the four-line "Rules" block in `CLAUDE.md`. The PfDHODH dossier's inline `claims` field and the `PFDHODH_SELECTIVITY_FINDINGS.md` "Non-claims" section (both now on master via PR #1) are the only per-artifact claim discipline currently practiced.

**Natural next-PR scope:** Decide between three options and implement:

1. **Adopt the Kira Constitution by reference.** Physics Auditor operates under Kira's `docs/SCIENTIFIC_CONSTITUTION.md`. A small `docs/GOVERNANCE.md` in Physics Auditor cites the Kira document as authoritative, lists which provisions apply (forbidden-term lists; evidence-tier framing), and notes any Physics-Auditor-specific exceptions.
2. **Write a Physics-Auditor-specific governance document.** Narrower scope, focused on structure-validation claims: distinguishing "the code computed X" from "X is a property of the underlying structure" from "X has biological meaning." Generalize the PfDHODH dossier's `claims` schema (`what_this_is`, `what_this_is_not`, `supported_finding`, `pending_followup`) into a schema every new result artifact must follow; add a pytest gate that rejects new `benchmark/results/**/*.json` files lacking a `claims` field with the required keys.
3. **Defer.** Document the dependency on Kira's governance in this state document (already done in §8.1 and §12) and revisit when the bridge interface (Finding 6) is implemented and demands shared discipline.

Option (2) is the most consistent with the discipline already in place. The schema is small enough to write in one sitting; the gate is small enough to implement as a single pytest.

### Finding 6 — Bridge interface to Kira absent on both sides

**Status:** Open.

The intended interface contract between Kira and Physics Auditor — `StructureAuditRequest` (Kira → Physics Auditor) and `StructureAuditResult` (Physics Auditor → Kira) — is **not implemented in either repository.** A grep for either type name returns no hits across all branches of Physics Auditor. The Kira-side document referenced in the audit brief (`docs/PHYSICS_AUDITOR_TICKET_GATE.md` in Kira) describes the intended interface; no Physics-Auditor-side counterpart exists, and no code on either side implements it.

The PfDHODH dossier JSON format (with the `claims` field) is the closest existing artifact to a `StructureAuditResult` payload, but it is not typed, not validated, and not declared as a public contract.

**Natural next-PR scope:** Design `StructureAuditRequest` and `StructureAuditResult` as frozen dataclasses in `src/physics_auditor/bridge/` (or `src/physics_auditor/contracts/`). `StructureAuditRequest` carries: target PDB path or content, optional ortholog PDB, ligand identification (HETATM resname or coords), the expected organism per structure (for the provenance gate), and the audit type (`validate`, `selectivity`, `causality_decomposition`). `StructureAuditResult` carries: the existing dossier fields, the `claims` schema from Finding 5, a `code_git_sha` field, and a `computed_at_utc` field. Implement a `handle_request(req: StructureAuditRequest) -> StructureAuditResult` entry point that dispatches to existing machinery. Add a Kira-side glue PR that issues `StructureAuditRequest`s and persists `StructureAuditResult`s as Kira's existing provenance-sidecar pattern. The two PRs should land within a release of each other.

### Finding 7 — README accuracy gaps (post-merge residual)

**Status:** Open, scope trimmed since PR #1 merged.

PR #1 brought the causality stack and the selectivity-attribution machinery onto master, which retires the README's previous overclaim about *causality and selectivity*. Three residual inaccuracies remain at the snapshot commit:

- **`physics-auditor selectivity` CLI subcommand.** The Quick Start block shows `physics-auditor selectivity --target parasite.pdb --ortholog human.pdb --ligand compound.sdf`. The subcommand does not exist; `cli.py` registers only `validate` and `info`. The selectivity machinery is exposed as a script (`benchmark/run_pfdhodh_vs_hsdhodh_selectivity.py`), not a CLI subcommand.
- **HTML reports.** The README lists "JSON (for pipelines) + HTML (for humans)" under Trust Report. `report/__init__.py` is a 1-line docstring placeholder; no HTML generator exists.
- **Per-residue heatmaps and binding-site annotations.** No plotting code exists in the repository.

**Natural next-PR scope:** Rewrite the README to match what the code does at master. Either:

1. **Demote the unimplemented features** to a "Planned" or "Roadmap" section parallel to `SETUP.md`'s "What's next" table, or
2. **Implement them as part of the README rewrite PR** — `physics-auditor selectivity` is a small wrapper around the now-on-master `compute_selectivity_map` machinery; HTML reports are a Jinja2 template against the existing JSON dossier (jinja2 is already declared in `pyproject.toml`); heatmaps need a plotting dependency to be added.

Option (1) is the doc-only, immediately-actionable form and is the natural pair with the doc-only half of Finding 3.

### Finding 8 — Wrong-protein error: pair 5 of the 14-pair clashscore benchmark used DnaK as LmPTR1

**Status:** Open. Withdrawn from benchmark in PR-A1. Restoration tracked as PR-C.

The original `BENCHMARK_PAIRS` pair 5 paired `2BPR.pdb` with `AF-Q01782` (LmPTR1 AlphaFold model) under the label "LmPTR1 — Leishmania pteridine reductase 1 — antiparasitic target". Identity audit during PR-A1's `verify_pdb_source` enforcement work surfaced that 2BPR is in fact DnaK, an NMR structure of a heat-shock chaperone from *Escherichia coli* (strain XL-1 BLUE). HEADER reads MOLECULAR CHAPERONE. COMPND MOLECULE: DNAK. SOURCE ORGANISM_SCIENTIFIC: ESCHERICHIA COLI. DBREF: UniProt P0A6Y8 (DNAK_ECOLI), residues 379–560. The committed clashscore numbers under "LmPTR1" in the previous 14-row `pocket_clashscore_comparison.json` were valid measurements of DnaK, attributed to a protein 2BPR is not. The manuscript inherited this error in Methods, Discussion (Group 2 enumeration), and Table 1.

Structurally distinct from Findings 1 and 2: pair 5 was a wrong-PDB error — the file does not contain any version of LmPTR1.

**Natural next-PR scope (PR-C):** substitute 2BPR with a real LmPTR1 crystal. The canonical PDB for *Leishmania major* PTR1 is **1E7W** (other candidates: 2BFM, 2BFA, 1E92). Commit the chosen PDB to `benchmark/structures/experimental/`. Restore the pair to `BENCHMARK_PAIRS` with verified `expected_organism_experimental = "LEISHMANIA MAJOR"`. Re-run the benchmark to produce the corrected row (after Finding 11 resolution). Update Methods, Discussion, and Table 1. Rename `benchmark/results/causality/DnaK_2BPR_causality.json` (present from PR-A1) back to `LmPTR1_causality.json` after substitution and update `CAUSALITY_FINDINGS.md` / `TOPOLOGY_FIXES.md` analogously.

### Finding 9 — Wrong-protein error: pair 3 of the 14-pair clashscore benchmark used EGFR as HsHDAC8

**Status:** Open. Withdrawn from benchmark in PR-A1. Restoration tracked as PR-D.

The original `BENCHMARK_PAIRS` pair 3 paired `4HJO.pdb` with `AF-Q9BY41` (HsHDAC8 AlphaFold model) under the label "HsHDAC8 — Kira ortholog for SmHDAC8". Identity audit during PR-A1's `verify_pdb_source` enforcement surfaced that 4HJO is in fact the EGFR tyrosine kinase domain (UniProt P00533). HEADER reads TRANSFERASE/TRANSFERASE INHIBITOR. COMPND MOLECULE: EPIDERMAL GROWTH FACTOR RECEPTOR. DBREF: UniProt P00533, residues 696–1022 (canonical EGFR kinase domain). The previous `res_seq_offset=678` was the symptom of the wrong-PDB error — 679-start is EGFR kinase-domain PDB numbering, not HDAC8. HsHDAC8 (Q9BY41) is only 377 residues long and cannot have residues 679–960. The committed clashscore numbers under "HsHDAC8" in the previous 14-row JSON were measurements of some EGFR atoms mapped via `res_seq_offset` to positions 1–282 of HDAC8's UniProt sequence — a biologically meaningless mapping. The manuscript inherited this error.

Note: the benchmark previously had TWO EGFR rows (pair 3 mislabeled as HsHDAC8, pair 12 correctly labeled as HsEGFR) and ZERO HsHDAC8 rows. After PR-A1's withdrawal, pair 12 is the only EGFR row.

**Natural next-PR scope (PR-D):** substitute 4HJO with a real HsHDAC8 crystal. Canonical PDB candidates: **1T64, 1T67, 2V5W** (all HsHDAC8 with various inhibitors). Commit the chosen PDB to `benchmark/structures/experimental/`. Restore the pair to `BENCHMARK_PAIRS` with verified `expected_organism_experimental = "HOMO SAPIENS"` and add a sanity-check that COMPND MOLECULE matches HISTONE DEACETYLASE 8 (Finding 9 is a worked example of why SOURCE-organism alone is insufficient — 4HJO passes the organism gate because it is also human). Re-run the benchmark to produce the corrected row (after Finding 11 resolution). Update Methods, Discussion (Group 2), and Table 1. Rename `benchmark/results/causality/EGFR_4HJO_causality.json` (present from PR-A1) back to `HsHDAC8_causality.json` after substitution and update `CAUSALITY_FINDINGS.md` / `TOPOLOGY_FIXES.md` analogously.

### Finding 10 — Construct-artifact annotation: pair 8 HsBRAF crystal 3OG7 is an AKAP9-BRAF fusion construct

**Status:** Annotated in place in PR-A1. No substitution planned.

Pair 8 (HsBRAF / 3OG7 / AF-P15056) passes the SOURCE-organism gate (HOMO SAPIENS on both sides), and the COMPND MOLECULE field reads "AKAP9-BRAF FUSION PROTEIN" rather than pure BRAF. The 3OG7 DBREF maps PDB residues 449–720 to UniProt **Q5IBP5** (the AKAP9-BRAF fusion accession, residues 1175–1446) rather than to the canonical BRAF UniProt P15056. The fusion's C-terminal half is the BRAF kinase domain — the residues physically present in 3OG7 are real BRAF kinase residues including V600E and the PLX4032/vemurafenib binding pocket. The clashscore measurements are scientifically meaningful BRAF kinase-pocket data; the entry was simply not declaring the construct origin.

PR-A1 annotates the `BENCHMARK_PAIRS` description for pair 8 to explicitly declare the AKAP9-BRAF fusion construct. The Table 1 footnote in the manuscript (to be added in PR-A2) will document the construct. No substitution is planned because the atomic data is correct BRAF kinase domain.

This finding is documented as a methodological flag for future provenance work: a future enhancement to `verify_pdb_source` could optionally cross-check DBREF UniProt against an expected_uniprot field per pair to surface this kind of construct-level annotation mismatch automatically. Out of scope for PR-A1.

### Finding 11 — Pocket-clashscore pipeline produces environmentally-dependent numerical results

**Status:** Open. Surfaced by PR-A1's planned rerun of `benchmark_pocket_clashscore.py`. Resolution required before PR-A2 (manuscript update + benchmark JSON regeneration).

During PR-A1's planned regeneration of `pocket_clashscore_comparison.json`, ten pairs that were not edited in this PR showed numerical drift between the committed master JSON (produced in a prior environment) and the current bio-builder environment. Within-environment reproducibility was confirmed: two consecutive runs in the current environment produce byte-identical output. The drift is exclusively across environments.

Drift magnitude (selected): HsEGFR `af_whole_clashscore` 11.39 → 3.19 (−72%), MmCOX2 `af_pocket_clashscore` 9.23 → 3.69 (−60%), CoV2Mpro `af_pocket_clashscore` 7.81 → 3.91 (−50%), HsAChE `af_whole_clashscore` 5.63 → 3.13 (−44%). Experimental-side numbers are mostly stable; predicted-side (AF model) numbers drift more, suggesting hydrogen-handling sensitivity in either `parse_pdb`, `build_bonded_mask`, or `check_clashes`.

The PfDHODH selectivity dossier (`run_pfdhodh_vs_hsdhodh_selectivity.py`) reproduces byte-identically across environments, so the LJ attribution pipeline is unaffected. The drift is specific to the clashscore pipeline.

Reference artifact: `/tmp/pocket_clashscore_current_env.json` (current-environment regeneration, not committed; preserved as evidence of the drift). The committed master `pocket_clashscore_comparison.json` is preserved as-is in PR-A1.

Hypotheses, to be investigated in a dedicated PR before PR-A2:

- **Hypothesis A:** a JAX, jaxlib, or numpy upgrade silently changed reduction semantics in `check_clashes`.
- **Hypothesis B:** hydrogen detection in `parse_pdb` is element-order or float-precision dependent.
- **Hypothesis C:** the clashscore kernel has float-precision-sensitive borderline pairs at the vdW-radius threshold; small precision differences flip which pairs count.

**Natural next-PR scope (PR-DET):** bisect the suspected dependencies (jax, jaxlib, numpy) until master JSON reproduces, identify the version where semantics changed, then choose between (a) pinning to the reproducing version with a CI determinism check, (b) hardening the kernel for cross-version determinism (float64 throughout, deterministic reductions, explicit tie-breaking at vdW threshold), or (c) accepting the new canonical and documenting the env-normalization event. Once Finding 11 is resolved, PR-A2 regenerates the JSON, updates manuscript Table 1, and ships the pair 1 new numbers.

### Status summary

| Finding | Status | Origin |
|---|---|---|
| 1. Mislabel propagation (1D3H labeled "SmDHODH" in master benchmark + manuscript) | Partially closed by PR-A1; manuscript update deferred to PR-A2 (blocked by Finding 11) | **Surfaced by this audit** |
| 2. `verify_pdb_source` built but not enforced | Closed by PR-A1 for `benchmark_pocket_clashscore.py`; three other scripts still bypass | **Surfaced by this audit** |
| 3. Eight-check suite is two checks (README contradicts SETUP.md) | Open | **Surfaced by this audit** |
| 4. `fixes/technical-pass-1` was local-only | Branch pushed; rebase + PR pending | **Surfaced by this audit; operationally mitigated by Phase 2** |
| 5. No governance documents | Open | **Surfaced by this audit** |
| 6. Bridge interface (`StructureAuditRequest` / `StructureAuditResult`) absent on both sides | Open | **Surfaced by this audit** |
| 7. README accuracy gaps (selectivity CLI, HTML, heatmaps) | Open, scope trimmed post-PR-1 | **Surfaced by this audit** |
| 8. Wrong-protein error: pair 5 used 2BPR (DnaK) as LmPTR1 | Open. Withdrawn from benchmark in PR-A1; substitution tracked as PR-C | **Surfaced by PR-A1's `verify_pdb_source` enforcement work** |
| 9. Wrong-protein error: pair 3 used 4HJO (EGFR) as HsHDAC8 | Open. Withdrawn from benchmark in PR-A1; substitution tracked as PR-D | **Surfaced by PR-A1's `verify_pdb_source` enforcement work** |
| 10. Construct-artifact annotation: pair 8 3OG7 is an AKAP9-BRAF fusion | Annotated in place in PR-A1; no substitution planned | **Surfaced by PR-A1's DBREF cross-check** |
| 11. Pocket-clashscore pipeline produces environmentally-dependent numerical results | Open. Resolution required before PR-A2; tracked as PR-DET | **Surfaced by PR-A1's planned rerun** |

---

## 10. Open questions and known unknowns

These are the things the audit producing this document surfaced that do not have an answer in committed state. They are decisions for human review, not actions.

1. **CLAUDE.md is out of date post-merge.** Line 6 still says *"45 tests, all must pass"*; the actual count is 92 + 1 skipped. The architecture section is missing the post-PR-1 additions (`utils/pdb_provenance`, `causality/{energy_decomp,divergence,selectivity_map}`). The "Active work" section documents the HDAC8 offset issue as if it were open; it has been closed. This is a small follow-up PR; it was not part of PR #1's scope.
2. **What does `test_disulfide.py` actually test?** No `checks/disulfides.py` exists. The test exercises the `S-S 2.6 Å` covalent cutoff in `topology.py::COVALENT_BOND_CUTOFFS` (i.e. it's an integration test of bond inference, not of a missing check module). Closing Finding 3's disulfide check should keep the test or expand it.
3. **CI matrix coverage.** `.github/workflows/ci.yml` runs on Python 3.11 and 3.13 only. `pyproject.toml` declares `requires-python = ">=3.10"`. Either 3.10 and 3.12 should be added to the matrix, or `requires-python` should be tightened.
4. **Boltz-2 and Chai-1 placeholder directories.** `benchmark/structures/predicted/{boltz2,chai1}/` are empty. The manuscript's "Limitations" section says extending the benchmark to these predictors is a natural next step. No code references them. Should the directories stay (as a visible roadmap signal) or be removed (until there is something to commit)?
5. **Causality decomposition coverage.** Only 6 of the 14 benchmark pairs ran to completion; 8 were skipped at the 4500-atom memory budget per `CAUSALITY_FINDINGS.md`. Is the right path forward (a) pocket-restricted decomposition (only compute LJ for `pocket × rest_of_protein` pairs), (b) chunked computation, (c) raising the budget on a larger machine? Each has different scientific implications for what "per-residue decomposition" means on a 33k-atom structure.
6. **True-ortholog pocket-divergence number.** The divergence runner produces `full_sequence_cosine = 0.989732` for true SmDHODH vs HsDHODH UniProt sequences, but the pocket leg is unmeasured because no parasite-side DHODH co-crystal is committed. Is the right path to commit `6FMD` (SmDHODH + inhibitor co-crystal, per the divergence module's own note) and rerun?
7. **Should the 14-pair benchmark re-run as part of Finding 1, or should the relabel be doc-only?** The clashscore numbers are valid for the underlying structures; only the *labels* are wrong. A purely-doc fix (relabel the JSON entries and the manuscript table) is defensible. A full re-run with the provenance gate enforced is more thorough but may produce drift on the JSON if any other unrelated changes have crept into the codebase between the original benchmark and now.
8. **Disposition of the now-stale `feat/*` branches.** With PR #1 merged, `feat/causality-energy-decomp`, `feat/selectivity-map`, `feat/divergence`, and locally-still-present `feat/pfdhodh-selectivity` are all strict prefixes of merged master. Should they be deleted in a single follow-up PR, or left as historical pointers?
9. **`reference/` data sourcing.** Closing Finding 3 requires committing Engh-Huber bond ideals (small, well-known), a Ramachandran distribution (small, multiple published sources, license-permissive options exist), and a rotamer library (Dunbrack or similar — license review needed). The disulfide-geometry data is small enough to hardcode. What is the right balance between vendoring these and depending on an external package?

---

## 11. Update protocol

This document is intended to be refreshed periodically, not continuously.

- **When to update inline (same PR):** when your PR changes the structure of what is committed or what is reproducible — adding a new active-code dependency on a data file, adding a new provenance pattern, changing the test count by more than a handful, opening or closing a numbered finding, merging or deleting a branch listed in §3.4. Notably: when a major PR merges, this document must be re-anchored to the new merge-commit SHA, the test-count dashboard must be updated, the per-artifact reproducibility section must be re-verified against the post-merge tree, and any sections that distinguished "on master" from "on a feature branch" must be collapsed.
- **Example of inline update (this document's history):** an earlier version of this document, anchored to pre-PR-1 master at `96eac03`, treated the PfDHODH selectivity work as living on the open feat branch. When PR #1 merged at `e7064de` on 2026-05-17, the document was re-anchored on the same audit branch (PR #2): snapshot commit updated, methodology map sections that distinguished "stub on master" from "implemented on feat branch" collapsed into a single on-master view, the per-artifact reproducibility check for the PfDHODH dossier was re-run on the post-merge tree (still byte-identical), and Finding 7's scope was trimmed to the residual inaccuracies that PR #1 did not address. See §13 for the post-merge update record.
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

## 13. Post-merge update record

This document was first committed against master at `96eac03` (pre-PR-1) on 2026-05-17, treating the PfDHODH selectivity work as living on the unmerged `feat/pfdhodh-selectivity` branch (PR #1, open since 2026-05-12). PR #1 merged into master later the same day at `e7064de`. This section records the re-anchoring update applied immediately after that merge.

**Changes captured by the re-anchor:**

- **Anchor commit:** `96eac03` → `e7064de`.
- **Master log:** 15 commits, all linear, zero merge commits → 28 commits, the most recent being a merge commit. §3.2 was rewritten to show the post-merge log.
- **Merged-PR count:** 0 → 1.
- **File inventory:** 67 files / ~21.0 MB → 95 files / ~22.8 MB. §4.1 and §4.2 were rewritten to reflect the new totals. The PR brought `4ORM.pdb` plus 19 other tracked files into `benchmark/`.
- **Source tree:** §4.3's tree no longer marks any module as "STUB on master." `causality/divergence.py`, `causality/energy_decomp.py`, and `causality/selectivity_map.py` are real implementations on master. A new `utils/` subpackage was added (`__init__.py` + `pdb_provenance.py`).
- **Test suite:** 45 tests → 92 + 1 skipped. §4.4 was expanded to list the six new test files.
- **Methodology map (§5):** the master-vs-feat-branch distinction was collapsed into a single on-master view. New module entries were written for `causality/energy_decomp.py`, `causality/selectivity_map.py`, `causality/divergence.py`, and `utils/pdb_provenance.py`. Each carries a note on its `NON-CLAIMS` docstring where applicable.
- **Per-artifact reproducibility (§7):** the PfDHODH dossier check was re-executed against the post-merge tree on master `e7064de`. `python benchmark/run_pfdhodh_vs_hsdhodh_selectivity.py` regenerated `benchmark/results/selectivity_maps/PfDHODH_vs_HsDHODH_selectivity.json` and `git diff --stat` returned empty — bit-identical regeneration confirmed directly on master. Sections that previously called out specific result artifacts as "branch-only" have been rewritten to drop that qualifier.
- **Governance audit (§8.2):** the README accuracy table no longer marks "Mechanistic Causality" as overclaiming — the three referenced modules are now implemented on master. The three residual inaccuracies (selectivity CLI, HTML reports, per-residue heatmaps) are tracked as Finding 7.
- **Findings (§9):**
  - Findings 1, 3, 5, 6 are unchanged in substance; their references to *"on `feat/pfdhodh-selectivity`"* have been updated to *"on master"* where applicable.
  - Finding 2's scope is unchanged but broader-reach now: the provenance gate is publicly visible on master and still unenforced at `parse_pdb()` and in most benchmark scripts.
  - Finding 4 records that `fixes/technical-pass-1` was pushed to origin during Phase 2 of the audit; the rebase target is now the post-merge master, and the `parse_pdb_string` ENDMDL fix has already been independently applied to master via commit `174c543` (so that part of the technical-pass branch is now redundant).
  - Finding 7 was originally a four-item list; the *"Mechanistic Causality"* item has been retired (PR #1 implemented it) and the finding's title was renamed to *"README accuracy gaps (post-merge residual)"*.
- **Open questions (§10):** the pre-merge questions about PR #1 size and review strategy are retired. The pre-merge question about "CLAUDE.md test-count claim post-merge" is now Open Question 1 — it is an actual stale claim, not a hypothetical. Open Question 8 (disposition of the `feat/*` stack) now references real branches that can be deleted.
- **Outstanding branches (§3.4):** the four `feat/*` branches that previously formed an unmerged stack are now reachable from master via the PR #1 merge. They are catalogued as candidates for deletion; `feat/pfdhodh-selectivity` was already removed from `origin` automatically at merge time.

**What did not change:**

- Findings 1, 3, 5, 6 in substance.
- The structure of the document (sections, ordering, dashboard format).
- The "How to use this document" and "Update protocol" sections, except for an update to the implicit-as-of commit reference and an added example of inline update history.
- The "Paired with Kira" section — the Kira snapshot date is the same day; the two snapshots remain coherent.

When the next major PR merges (likely Finding 1's cleanup PR, or `fixes/technical-pass-1` once rebased and reviewed), §13 should be replaced with a fresh post-merge update record for that event, and the snapshot anchor commit at the top of the document should be bumped accordingly.
