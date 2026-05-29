# Archived causality dossiers

Files here are **retired historical artifacts**, kept for audit-trail
continuity but **not part of the live result set**. They are not produced by
the current benchmark code and must not be cited as current findings.

---

## `HsDHODH_causality.json` — retired 2026-05-29 (Step 3, consolidation)

### What it was
A per-residue Lennard-Jones energy-decomposition dossier produced by an early
run of `benchmark/run_causality_decomposition.py`. Its internal `pair` block:

| field | value |
|---|---|
| `experimental_pdb` | `1D3H.pdb` — **Homo sapiens** DHODH catalytic construct (UniProt Q02127, per SOURCE) |
| `predicted_pdb` | `AF-G4VFD7-F1-model_v6.pdb` — **Schistosoma mansoni** DHODH AlphaFold model |
| `description` | `"Parasite DHODH — 30.8x selectivity target"` |

### Why it is retired (mixed provenance)
The dossier pairs a **human** experimental structure with a **Schistosoma**
AlphaFold model — a cross-organism comparison — while its filename
(`HsDHODH_…`, assigned in PR-A1) and its description string disagree with each
other and with the contents:

- The PR-A1 relabel renamed the file to `HsDHODH_causality.json` because the
  *experimental* input (1D3H) is human HsDHODH.
- The `predicted_pdb` is still the Schistosoma model from the original
  (pre-PR-A1) "SmDHODH" pairing.
- The `description` still carries the original "Parasite DHODH — 30.8x
  selectivity target" label.

The interpretation this file once supported was **already invalidated** by the
PR-A1 audit: see `../CAUSALITY_FINDINGS.md` (the "Kira 30.8× selectivity
framing … is invalidated by the PR-A1 audit. This is not a parasite-vs-human
comparison" passage). The current `BENCHMARK_PAIRS["HsDHODH"]` entry pairs
1D3H with `AF-Q02127` (intra-human), so re-running the producer today would
**not** reproduce this file — it would emit a different, internally-consistent
(1D3H ↔ AF-Q02127) decomposition. Rather than regenerate (out of scope for the
consolidation), the mixed-provenance artifact was moved here intact.

### What depended on it
**Nothing in code.** No module under `src/` and no test under `tests/` reads
`HsDHODH_causality.json` or `causality_summary.csv`. The file was a leaf
artifact.

The "30.8× selectivity" figure it carried in its description string is **not**
sourced from this file. Its provenance is the Kira selectivity program (docked
compound CHEMBL155771, 23 nM, 30.8× selectivity), recorded in
`../CAUSALITY_FINDINGS.md` and attributed to Kira throughout
`STATE_OF_PHYSICS_AUDITOR.md`. Archiving this file orphans no finding.

### What was NOT done
This file was **not regenerated** and **no parasite-side co-crystal was
fetched**. Reconstructing a genuine SmDHODH-vs-HsDHODH comparison is a separate
deferred decision — see `STATE_OF_PHYSICS_AUDITOR.md` §10 item 6 (the
deferred pocket-leg divergence).
