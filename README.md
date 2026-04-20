# Physics Auditor

**Physics validation and mechanistic explanation for AI-generated protein structures.**

Structure prediction tools tell you *what* a protein-ligand complex looks like. Physics auditors tell you *whether* that prediction is physically valid. This tool does both — and then tells you *why* the interaction works the way it does.

## The Problem

AI structure generators (Boltz-2, AlphaFold 3, Chai-1, RFdiffusion3) optimize learned score functions, not physics. They produce structures that look plausible but can contain steric clashes, bond strain, forbidden backbone conformations, wrong chirality, and infinite-energy contacts.

IsoDDE (Isomorphic Labs) built internal physics violation filtering. The open ecosystem has not. Boltz-2 predictions show a 66% drop in accuracy when physics violations are filtered (Figure 5, IsoDDE technical report, Feb 2026). Academic labs and biotech startups using open models need a standardized physics referee.

But validation alone isn't enough. Drug design requires *mechanistic understanding*: which residues drive a binding interaction? Why does a compound bind selectively to a parasite target over its human ortholog? What specific binding-site differences produce 30× selectivity despite 99% global sequence similarity?

Physics Auditor answers all three questions.

## What It Does

### Physics Validation (8 checks)
- **Steric clashes** — non-bonded atoms closer than van der Waals radii
- **Bond geometry** — bond lengths and angles vs. Engh-Huber ideal values
- **Ramachandran** — backbone φ/ψ angles in allowed regions
- **Peptide planarity** — ω angle deviations from trans/cis ideals
- **Chirality** — L-amino acid verification at Cα centers
- **Rotamer outliers** — sidechain χ angles vs. rotamer library
- **Lennard-Jones energy** — continuous pairwise non-bonded energy landscape
- **Disulfide geometry** — S-S bond length, dihedral, and angle validation

### Mechanistic Causality (the differentiator)
- **Per-residue energy decomposition** — which residues contribute favorable vs. unfavorable interactions at the binding site
- **Binding site comparison** — extract, align, and compare pockets between homologous proteins (e.g., parasite target vs. human ortholog)
- **Selectivity attribution** — ranked list of residues that drive selective binding, with per-residue energy contributions

### Trust Report
- Composite physics score (0–1)
- Actionable recommendation: **accept** / **run short MD** / **discard**
- Per-residue heatmaps and binding-site annotations
- JSON (for pipelines) + HTML (for humans)

## Quick Start

```bash
pip install git+https://github.com/Danny2045/physics-auditor.git

# Validate a single structure
physics-auditor validate structure.pdb

# Batch validation
physics-auditor validate *.pdb --output-dir reports/

# Selectivity analysis (two homologous structures + ligand)
physics-auditor selectivity --target parasite.pdb --ortholog human.pdb --ligand compound.sdf
```

## Why Not Molprobity?

1. **Calibrated for AI structures.** Molprobity's thresholds are tuned for X-ray/cryo-EM artifacts, not generative model failure modes.
2. **Speed.** JIT-compiled JAX pipeline processes thousands of structures in batch on a single machine.
3. **Actionable output.** A single trust score + decision (accept/relax/discard) for automated pipelines.
4. **Mechanistic causality.** No existing tool provides per-residue selectivity attribution across homologous protein pairs.

## Citation

If you use Physics Auditor in your research, please cite:

```bibtex
@software{ngabonziza2026physics,
  author = {Ngabonziza, Daniel},
  title = {Physics Auditor: Physics Validation and Mechanistic Explanation for AI-Generated Protein Structures},
  year = {2026},
  url = {https://github.com/Danny2045/physics-auditor}
}
```

## License

MIT
