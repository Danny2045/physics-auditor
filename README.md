# Physics Auditor

**Physics validation and mechanistic explanation for AI-generated protein structures.**

Structure prediction tools tell you *what* a protein–ligand complex looks
like. Physics Auditor tells you *whether* the structure is physically
valid at a narrow but useful slice (steric clashes + non-bonded
Lennard-Jones energy) and — the load-bearing part of the toolkit —
*which residues* drive a binding interaction and *why* a compound binds
selectively to a parasite target over its human ortholog.

## Status

**v0.x — narrow validation, deep causality.** Two physics checks are
implemented today (steric clashes; Lennard-Jones energy). The causality
engine — per-residue energy decomposition, two convergent correspondence
methods, a selectivity map across homologous protein pairs, and the
safeguards that gate every analysis — is the differentiator and is what
the benchmark runners under `benchmark/` exercise end-to-end. Six
further physics checks (bond geometry, Ramachandran, peptide planarity,
chirality, rotamer outliers, disulfide-geometry validation) are on the
roadmap; see the "Not implemented yet — planned" section below.

## Physics validation (current)

Two checks are implemented and wired into the CLI:

- **Steric clashes** — non-bonded atom pairs closer than the sum of
  their van der Waals radii minus a tolerance. Reports a per-1000-atom
  clashscore, severe-clash count, worst overlap, and per-residue
  counts (`src/physics_auditor/checks/clashes.py:check_clashes`).
- **Lennard-Jones energy** — continuous pairwise non-bonded energy
  landscape, with total, per-atom, and per-residue rollups
  (`src/physics_auditor/core/energy.py:run_lj_analysis`).

Backbone φ/ψ/ω dihedrals are extracted as available building blocks
(`src/physics_auditor/core/geometry.py:extract_backbone_dihedrals`),
but **no check currently scores them** against Ramachandran maps,
peptide-planarity thresholds, or any reference distribution — the
dihedrals are emitted as metadata only.

S–S bonds are inferred from Cys-SG distances during topology
construction (`src/physics_auditor/core/topology.py`) so that
disulfide pairs are not miscounted as steric clashes. This is bond
inference, **not** a disulfide-geometry validation check.

### Composite score and recommendation — honest disclosure

The CLI emits a `global_score ∈ [0, 1]` and a recommendation of
`accept` / `short_md` / `discard` (thresholds in
`src/physics_auditor/config.py:CompositeConfig`).

**Today the reported score reflects steric clashes ONLY.** The LJ
pipeline runs and its output is included as a separate field in the
report, but it is **not** folded into the score. See
`src/physics_auditor/cli.py` line 97 — the score is literally
`clash_result.subscore`. The `accept` / `short_md` / `discard`
recommendation is therefore a **clash-based** recommendation, not a
multi-check composite, despite the `global_score` field name. Folding
LJ into the composite (and renaming the field) is tracked as roadmap
work.

## Causality engine

This is the differentiator. AI representation models can score ~99 %
global sequence / embedding similarity between a parasite enzyme and
its human ortholog while the two targets show large pharmacological
selectivity; the selectivity is invisible to whole-protein
representations but visible at the binding-site residue level. The
causality engine localises it.

- **Per-residue energy decomposition** — protein↔ligand LJ
  contribution for every residue on both target and ortholog sides
  (`src/physics_auditor/causality/energy_decomp.py`).
- **Binding-site extraction and comparison** — pocket residues within
  a configurable cutoff (default 5.0 Å) of a bound ligand, with paired-
  pocket divergence metrics
  (`src/physics_auditor/causality/binding_site.py`).
- **Two convergent correspondence methods:**
  - sequence — Needleman-Wunsch global alignment, including an
    identity refusal gate
    (`src/physics_auditor/causality/correspondence.py:build_pocket_alignment`);
  - structural — cofactor-anchored Kabsch superposition + nearest-CA
    pocket pairing
    (`src/physics_auditor/causality/superposition.py:superpose_and_correspond`).
- **Selectivity map** — per-residue Δ-LJ (ortholog − target) ranked
  across the pocket; a positive Δ means the residue is more favourable
  for the target than for the ortholog at the corresponding position
  (`src/physics_auditor/causality/selectivity_map.py:compute_selectivity_map`).
- **Local-vs-global divergence** — ESM-2 cosine similarity at the
  whole-protein level can coexist with substantial binding-site
  divergence; the engine measures both side-by-side
  (`src/physics_auditor/causality/divergence.py`; pulls torch +
  fair-esm, imported lazily).

## Safeguards (gates that run before analysis)

Every selectivity comparison is gated by a sequence of explicit refusals
that fail loudly rather than producing a misleading number.

- **Provenance gate** —
  `verify_pdb_source(path, expected_organism)` reads the PDB's `SOURCE`
  `ORGANISM_SCIENTIFIC` record and raises if it does not match what the
  caller asserted. Runs **before** parse so a mislabelled file never
  reaches the analysis path
  (`src/physics_auditor/utils/pdb_provenance.py`).
- **Pocket-completeness gate** — compares SEQRES (expected residues)
  against resolved ATOM records, infers the per-chain SEQRES↔ATOM
  numbering offset by name-matching, and classifies each missing
  residue as either an **internal gap** (resolved on both sequence
  sides — can corrupt a selectivity verdict if it lies in the pocket
  region) or a **terminal tail** (resolved on one side only — reported
  informationally, never flips the verdict). A reach-bound test using
  the polypeptide max-step of 3.8 Å per Cα–Cα determines whether an
  internal gap is in the pocket region. Returns `COMPLETE` or
  `INCOMPLETE`
  (`src/physics_auditor/causality/pocket_completeness.py`).
- **Sequence-correspondence refusal gate** —
  `build_pocket_alignment` refuses to pair pockets when the global
  Needleman-Wunsch identity falls below `min_global_identity` (default
  0.25). Refused alignments return an empty correspondence with
  `refused=True`; callers must inspect the flag rather than treating
  empty as success
  (`src/physics_auditor/causality/correspondence.py`).
- **NMR / multi-model safety** — the PDB parser stops at the first
  `ENDMDL`, so NMR ensembles and multi-model files contribute exactly
  one conformer's worth of atoms and the downstream JAX kernels are not
  fed a duplicate-coordinate pile
  (`src/physics_auditor/core/parser.py`).

## CLI

```bash
# Validate a single structure
physics-auditor validate structure.pdb

# Batch validation with per-file JSON reports
physics-auditor validate *.pdb --output-dir reports/

# Print basic structural information about a PDB file
physics-auditor info structure.pdb
```

The CLI prints a human-readable summary table by default, JSON to
stdout with `--json`, and a per-file JSON report into `--output-dir`
when supplied.

There is **no** `physics-auditor selectivity` subcommand at this time;
the causality engine is accessed through the Python API (next
section) and through the benchmark runners under `benchmark/`.

## Python API quickstart

```python
from physics_auditor.core.parser import parse_pdb
from physics_auditor.utils.pdb_provenance import verify_pdb_source
from physics_auditor.causality.pocket_completeness import pocket_completeness_gate
from physics_auditor.causality.selectivity_map import compute_selectivity_map

# 1. Gate at the door
verify_pdb_source("parasite.pdb", "PLASMODIUM FALCIPARUM")
verify_pdb_source("human.pdb",    "HOMO SAPIENS")

# 2. Parse
target   = parse_pdb("parasite.pdb")
ortholog = parse_pdb("human.pdb")

# 3. Pocket completeness — refuse if the target pocket has internal gaps
gate = pocket_completeness_gate(target, ligand_resname="OXM", cutoff=5.0)
assert gate.verdict == "COMPLETE", gate.summary()

# 4. Compute the per-residue selectivity map
smap = compute_selectivity_map(
    target_structure=target,    target_ligand_resname="OXM",
    ortholog_structure=ortholog, ortholog_ligand_resname="OXM",
    pocket_cutoff=5.0,
)
for r in sorted(smap.residues, key=lambda x: -x.delta_kcal):
    print(f"{r.res_name_target}{r.res_seq_target} ↔ "
          f"{r.res_name_ortholog}{r.res_seq_ortholog}  "
          f"Δ={r.delta_kcal:+.3f} kcal/mol")
```

For a realistic invocation that also runs both correspondence methods
and the comparability gate end-to-end, see the benchmark runners
below.

## Benchmarks

Worked end-to-end examples live under `benchmark/`:

- `benchmark/run_w107f_selectivity.py` — wild-type PfLDH (1LDG) vs
  HsLDH-A (1I10), the confirmatory selectivity run that exercises
  every gate, both correspondence methods, and the per-residue
  selectivity map. Result dossier at
  `benchmark/results/selectivity_maps/W107F_MECHANISTIC_RESULT.json`.
- `benchmark/run_pfdhodh_vs_hsdhodh_selectivity.py` and
  `benchmark/run_pfdhodh_realigned_comparison.py` — PfDHODH vs
  HsDHODH selectivity attribution.
- `benchmark/run_kabsch_convergent_validation.py` — convergence check
  between the sequence and structural correspondence methods.
- `benchmark/benchmark_pocket_clashscore.py` — whole-protein vs
  binding-site clashscore comparison between experimental and AI-
  predicted structures.

Curated PDB inputs live under `benchmark/structures/experimental/`.

## Installation

```bash
pip install git+https://github.com/Danny2045/physics-auditor.git
```

## Not implemented yet — planned

The items below are **not** in the engine today; they are tracked as
roadmap work. Each has a dataclass placeholder in
`src/physics_auditor/config.py`
(`BondGeometryConfig`, `RamachandranConfig`, `PeptideConfig`,
`ChiralityConfig`, `RotamerConfig`, `DisulfideConfig`) — these
**placeholders are currently unused** by any check.

- **Bond geometry** — bond-length and bond-angle deviation vs
  Engh-Huber ideals.
- **Ramachandran** — backbone φ/ψ against allowed / favoured regions.
- **Peptide planarity** — ω-angle deviation from trans/cis ideals.
- **Chirality** — L-amino-acid verification at Cα centres.
- **Rotamer outliers** — sidechain χ angles vs a rotamer library.
- **Disulfide-geometry validation** — per-disulfide scoring of S–S
  bond length, Cα-Cβ-Sγ-Sγ dihedral, and Cβ-Sγ-Sγ angle. (S–S
  *topology* is already inferred; what is missing is a per-disulfide
  validation score.)

Composite-layer roadmap item:

- **Fold the LJ subscore into the composite score** and rename the
  `global_score` field so the field name no longer implies a multi-
  check composite when only the clash subscore feeds it.

## Citation

```bibtex
@software{ngabonziza2026physics,
  author = {Ngabonziza, Daniel},
  title  = {Physics Auditor: Physics Validation and Mechanistic Explanation for AI-Generated Protein Structures},
  year   = {2026},
  url    = {https://github.com/Danny2045/physics-auditor}
}
```

## License

MIT
