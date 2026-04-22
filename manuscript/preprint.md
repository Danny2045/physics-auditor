# Binding-Site Physics Quality in AI-Predicted Protein Structures Is Target-Dependent: A 14-Protein Benchmark

**Danny Ngabonziza**

Independent researcher, Franklin, TN, USA

Correspondence: ngabondanny@gmail.com

---

## Abstract

AlphaFold predictions consistently outscore experimental crystal structures on whole-protein physics metrics such as clashscore, leading to the assumption that predicted structures are uniformly superior for downstream applications including drug design. We tested this assumption by comparing binding-site clashscores between 14 matched pairs of experimental crystal structures and AlphaFold v6 predictions spanning kinases, metalloenzymes, proteases, oxidoreductases, and other major drug target families. Whole-protein clashscores confirmed AlphaFold's advantage (1.2--11.4x lower, median 4.7x). However, at the binding site, this advantage was target-dependent: it shrank by 13--58% at six targets with open or substrate-channel active sites and grew by 13--490% at seven targets with buried or cofactor-dependent active sites. The most extreme divergence occurred at human aromatase (CYP19A1), a heme-containing metalloenzyme where AlphaFold's binding-site advantage was 490% larger than its whole-protein advantage. These results indicate that whole-protein clashscore conceals functionally relevant variation at binding sites and cannot serve as a reliable quality filter for structure-based drug design. Physics validation for drug discovery applications must be performed at the binding-site level, with target-class-specific interpretation.

**Keywords:** AlphaFold, protein structure prediction, clashscore, binding site, drug design, physics validation

---

## Introduction

The release of AlphaFold 2 transformed structural biology by providing predicted structures for nearly all known protein sequences at a quality rivaling experimental determination for many targets (Jumper et al., 2021; Varadi et al., 2022). These predictions routinely achieve sub-angstrom backbone RMSD to experimental structures for well-folded domains, and subsequent methods including AlphaFold 3 (Abramson et al., 2024), RoseTTAFold (Baek et al., 2021), and ESMFold (Lin et al., 2023) have further expanded the scope and accuracy of structure prediction.

A natural consequence has been the adoption of predicted structures in drug discovery pipelines. Virtual screening, molecular docking, and binding-site analysis increasingly rely on AlphaFold models, particularly for targets lacking experimental structures (Buel & Bhatt, 2024). This raises a critical quality question: are AI-predicted structures reliable specifically at the binding site, where geometric accuracy directly determines the utility of structure-based drug design?

Standard physics-based validation tools such as MolProbity (Chen et al., 2010) evaluate structures at the whole-protein level, reporting metrics including clashscore (steric clashes per 1000 atoms), Ramachandran outliers, and rotamer quality. By these measures, AlphaFold predictions typically outperform experimental crystal structures, which carry artifacts from crystal packing, radiation damage, and refinement limitations (Read & Adams, 2023). This has led to a widespread but largely untested assumption: if the whole-protein physics are better, the binding-site physics must also be better.

We tested this assumption directly. By decomposing clashscore into whole-protein and binding-site components across 14 matched experimental--predicted structure pairs, we found that the relationship between whole-protein and binding-site physics quality is target-dependent. For some targets AlphaFold's advantage shrinks substantially at the binding site; for others it grows dramatically. This variation is invisible to whole-protein metrics and has direct implications for how predicted structures should be evaluated before use in drug design.

---

## Methods

### Structure selection

We assembled 14 matched pairs of experimental crystal structures from the Protein Data Bank (Berman et al., 2000) and AlphaFold Database v6 predictions (Varadi et al., 2022). Targets were selected to span major drug target families: oxidoreductases (HsDHODH, SmDHODH, LmPTR1), metalloenzymes (HsHDAC8, HsAromatase), kinases (HsABL1, HsBRAF, HsEGFR), proteases (CoV2Mpro), cholinesterases (HsAChE), cyclooxygenases (MmCOX2), reductases (HsHMGCR), peptidases (HsDPP4), and a ligand-free control (ubiquitin). Table 1 provides PDB codes, UniProt accessions, and binding-site characteristics for all pairs.

Experimental structures were selected based on ligand occupancy (bound to a drug or drug-like inhibitor) and resolution (all below 3.0 A). AlphaFold structures are single-chain, ligand-free (apo) predictions. Only the first model was used for NMR-deposited structures.

### Binding-site definition

For each experimental structure, the binding site was defined as all protein residues with any non-hydrogen atom within 8.0 A of any ligand heavy atom. Ligands were identified as non-water HETATM records. For ubiquitin, which lacks a bound ligand, the protein centroid was used as a fallback reference point; this target serves as a ligand-free control and is excluded from binding-site-specific claims.

To apply the same pocket definition to the corresponding AlphaFold prediction (which contains no ligand), we transferred the binding-site residue identities by sequence number matching. Where the experimental and predicted structures use different residue numbering schemes (e.g., PDB numbering vs. UniProt numbering), we applied manually determined offsets. Six of 14 pairs required offset correction; offsets ranged from -31 to +678 residues.

### Clashscore computation

Clashscore was computed using Physics Auditor, an open-source JAX-based structural validation tool. A steric clash is defined as a non-bonded atom pair whose interatomic distance is less than the sum of their van der Waals radii minus a 0.4 A tolerance, following the MolProbity convention (Word et al., 1999). Bonded pairs (1-2 and 1-3 neighbors) are excluded via topology inference from residue connectivity. Clashscore is reported as clashes per 1000 atoms.

For whole-protein clashscore, all protein atoms are considered. For binding-site clashscore, a clash is counted if at least one of the two atoms in the clashing pair belongs to the binding pocket. This captures both intra-pocket clashes and pocket-to-rest-of-protein clashes, both of which affect drug design utility.

Van der Waals radii are assigned at the element level (C, N, O, S, H) using standard values. All computation uses JAX arrays for GPU-accelerable distance matrix and overlap calculations, with NumPy at I/O boundaries.

### Metrics

For each matched pair, we computed four values: experimental whole-protein clashscore, experimental binding-site clashscore, AlphaFold whole-protein clashscore, and AlphaFold binding-site clashscore. We then derived two ratios:

- **Whole-protein ratio** = experimental whole clashscore / AlphaFold whole clashscore
- **Binding-site ratio** = experimental pocket clashscore / AlphaFold pocket clashscore

A ratio greater than 1 indicates AlphaFold has fewer clashes. The percentage change from whole-protein to binding-site ratio quantifies whether AlphaFold's advantage grows or shrinks at the functional site:

- **Delta** = (binding-site ratio / whole-protein ratio - 1) x 100%

Negative delta indicates the advantage shrinks at the binding site; positive delta indicates it grows.

---

## Results

### Whole-protein clashscore confirms AlphaFold advantage

Across all 14 pairs, AlphaFold predictions had lower whole-protein clashscores than their experimental counterparts (Table 1). The whole-protein ratio ranged from 1.2x (HsEGFR) to 11.4x (HsDHODH), with a median of 4.7x and mean of 5.8x. Twelve of 14 targets showed ratios of 3.7x or higher; the two exceptions were HsEGFR (1.2x), where the experimental structure (1M17, 2.6 A) was already of relatively high quality, and HsHMGCR (2.8x).

This result is consistent with prior reports that AlphaFold predictions carry fewer steric clashes than experimental crystal structures, reflecting the absence of crystal packing forces, radiation damage, and refinement artifacts in predicted models.

### Binding-site advantage is target-dependent

At the binding site, AlphaFold's advantage did not uniformly mirror its whole-protein performance (Table 1). Instead, the binding-site ratio diverged from the whole-protein ratio in a target-dependent manner, falling into two broad groups.

**Group 1: Advantage shrinks (6 targets).** At HsDHODH, SmDHODH, HsABL1, CoV2Mpro, HsHMGCR, and MmCOX2, the binding-site ratio was lower than the whole-protein ratio, meaning AlphaFold's advantage was reduced at the functional site. The reduction ranged from 13% (MmCOX2) to 58% (CoV2Mpro). Excluding the borderline case of MmCOX2, the five clear cases showed 29--58% shrinkage.

**Group 2: Advantage grows (8 targets).** At HsHDAC8, LmPTR1, HsAromatase, HsBRAF, HsAChE, HsEGFR, HsDPP4, and ubiquitin, the binding-site ratio exceeded the whole-protein ratio. Excluding ubiquitin (no true binding site) and the borderline HsAChE (+13%), the growth ranged from 19% (HsHDAC8) to 490% (HsAromatase).

**Table 1.** Whole-protein and binding-site clashscores for 14 matched experimental--AlphaFold structure pairs.

| Target | PDB | UniProt | Site description | Pocket residues | Exp whole CS | Exp pocket CS | AF whole CS | AF pocket CS | Whole ratio | Pocket ratio | Delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HsDHODH | 1MVS | Q02127 | FMN substrate tunnel | 68 | 45.40 | 35.27 | 3.97 | 5.73 | 11.4x | 6.2x | -46% |
| SmDHODH | 1D3H | G4VFD7 | FMN substrate tunnel | 129 | 32.74 | 23.26 | 5.16 | 5.19 | 6.4x | 4.5x | -29% |
| HsHDAC8 | 4HJO | Q9BY41 | Zn-dependent active site | 46 | 19.43 | 24.93 | 5.11 | 5.49 | 3.8x | 4.5x | +19% |
| Ubiquitin | 1UBQ | P0CG48 | No ligand (centroid) | 24 | 39.39 | 10.26 | 3.51 | 0.00 | 11.2x | >1000x | * |
| LmPTR1 | 2BPR | Q01782 | NADPH-dependent site | 24 | 18.87 | 32.97 | 5.14 | 6.13 | 3.7x | 5.4x | +46% |
| HsABL1 | 2HYY | P00519 | ATP site, DFG-out | 239 | 29.30 | 32.88 | 3.93 | 9.92 | 7.5x | 3.3x | -56% |
| HsAromatase | 3EQM | P11511 | Heme-containing CYP | 109 | 22.03 | 24.92 | 5.91 | 1.13 | 3.7x | 22.0x | +490% |
| HsBRAF | 3OG7 | P15056 | ATP site, DFG-in | 52 | 19.27 | 21.63 | 3.54 | 2.40 | 5.5x | 9.0x | +65% |
| CoV2Mpro | 5R82 | P0DTD1 | Cys protease active site | 99 | 56.17 | 35.67 | 5.21 | 7.81 | 10.8x | 4.6x | -58% |
| HsAChE | 4EY7 | P22303 | Deep aromatic gorge | 250 | 24.99 | 36.54 | 5.63 | 7.29 | 4.4x | 5.0x | +13% |
| MmCOX2 | 3LN1 | Q05769 | Buried hydrophobic channel | 684 | 33.54 | 38.36 | 6.99 | 9.23 | 4.8x | 4.2x | -13% |
| HsEGFR | 1M17 | P00533 | ATP site, active kinase | 51 | 14.14 | 17.37 | 11.39 | 4.96 | 1.2x | 3.5x | +182% |
| HsHMGCR | 1HWL | P04035 | Large open substrate site | 296 | 11.10 | 13.84 | 3.96 | 8.28 | 2.8x | 1.7x | -40% |
| HsDPP4 | 1X70 | P27487 | Beta-propeller cavity | 343 | 41.43 | 49.74 | 8.98 | 6.65 | 4.6x | 7.5x | +62% |

CS = clashscore (clashes per 1000 atoms). Whole ratio and pocket ratio = experimental CS / AlphaFold CS. Delta = percentage change from whole-protein to binding-site ratio; negative values indicate AlphaFold's advantage shrinks at the binding site, positive values indicate it grows. *Ubiquitin excluded from binding-site claims (no ligand; centroid-based pocket).

### The aromatase extreme case

Human aromatase (CYP19A1) displayed the most dramatic divergence. Its whole-protein ratio was an unremarkable 3.7x, yet its binding-site ratio was 22.0x -- a 490% amplification of AlphaFold's advantage. The experimental structure (3EQM) contains a heme prosthetic group coordinating the iron center and the steroidal substrate androstenedione; the crystal structure's binding-site clashscore of 24.92 likely reflects the geometric strain imposed by the metal-ligand coordination geometry and substrate contacts. AlphaFold's apo prediction, unconstrained by cofactor or substrate, achieves a pocket clashscore of just 1.13 -- effectively relaxing the active site into a low-energy state that bears no geometric imprint of the catalytic machinery.

This illustrates a fundamental asymmetry: AlphaFold predicts the apo protein, while crystal structures capture the holo state with its functionally essential but physically strained cofactor interactions. At metalloenzyme active sites, this asymmetry produces artificially favorable clashscores for the predicted structure.

### Variation across kinases

Kinases provided an instructive within-family comparison. Three kinases were included (HsABL1, HsBRAF, HsEGFR), all with inhibitor-bound experimental structures targeting the ATP-binding site. Despite targeting the same fold and binding region, their deltas spanned -56% to +182%:

- **HsABL1** (DFG-out, imatinib-bound): the advantage shrank 56%, consistent with imatinib inducing a non-standard DFG-out conformation that AlphaFold's apo prediction may not capture, resulting in a geometrically divergent pocket.
- **HsBRAF** (DFG-in, inhibitor-bound): the advantage grew 65%, suggesting the DFG-in conformation is well-predicted and the experimental pocket carries more packing-related strain.
- **HsEGFR** (active conformation, erlotinib-bound): the advantage grew 182%, though this target had the smallest whole-protein ratio (1.2x), so the absolute pocket clashscores were modest.

This within-family variation underscores that binding-site physics quality depends on the specific conformational state captured by the experimental structure, not merely the target family.

---

## Discussion

### Mechanistic interpretation

The two-group pattern admits a plausible mechanistic interpretation. AlphaFold predicts single-chain apo structures optimized for internal consistency (low self-clashes) but without the geometric constraints imposed by bound ligands, cofactors, metal ions, or crystal packing forces.

At **open, solvent-exposed active sites** (DHODH substrate tunnels, HsABL1 in the DFG-out conformation, CoV2Mpro catalytic cleft), the experimental structure's clashscore reflects both crystal artifacts and the conformational strain of accommodating a bound ligand. AlphaFold's prediction, while clash-free, may adopt a conformation that differs from the drug-relevant holo state. The reduced binding-site ratio in these cases does not necessarily mean AlphaFold is worse -- it means the binding site is a region where the experimental structure's crystal artifacts are concentrated, and AlphaFold's advantage is correspondingly less dramatic.

At **buried, cofactor-dependent active sites** (aromatase heme pocket, HDAC8 zinc site, LmPTR1 NADPH site), the experimental structure carries geometric strain from metal coordination and cofactor contacts that inflate its pocket clashscore. AlphaFold, predicting without these cofactors, produces a relaxed pocket with minimal clashes. The amplified binding-site ratio in these cases reflects AlphaFold's apo relaxation rather than superior prediction of the functional geometry.

### Implications for drug design workflows

These findings have three practical implications:

**First, whole-protein clashscore is insufficient for drug design quality assessment.** A predicted structure with excellent whole-protein physics may have a binding-site geometry that is either appropriately relaxed (for cofactor-dependent targets, where the apo state differs fundamentally from the drug-relevant holo state) or inappropriately divergent (for targets requiring specific induced-fit conformations). Whole-protein metrics cannot distinguish these cases.

**Second, binding-site physics must be evaluated in the context of target class.** For metalloenzymes and cofactor-dependent enzymes, a low AlphaFold binding-site clashscore may indicate the absence of functionally essential geometric constraints rather than high-quality prediction. For open active sites, a relatively elevated binding-site clashscore may still represent a drug-relevant geometry.

**Third, the choice between experimental and predicted structures for docking should be target-informed, not based on global quality metrics alone.** In six of our 14 cases, AlphaFold's binding-site advantage was substantially reduced compared to its whole-protein advantage, suggesting that the experimental structure -- despite its globally worse physics -- may better represent the drug-relevant binding-site geometry.

### Limitations

This study has several important limitations that constrain the scope of its conclusions.

**Element-level van der Waals radii.** Our clash detection uses element-level (C, N, O, S) rather than atom-type-level radii. This is less granular than MolProbity's all-atom contact analysis, which uses probe-based dot surfaces and atom-type-specific radii. Our absolute clashscore values are therefore not directly comparable to MolProbity values, though the relative comparisons (experimental vs. predicted, whole vs. pocket) remain valid because the same methodology is applied consistently.

**Sample size.** Fourteen target pairs is sufficient to demonstrate that binding-site variation exists and is substantial, but insufficient to establish robust quantitative relationships between target properties (solvent accessibility, cofactor dependence, pocket depth) and the direction or magnitude of the binding-site effect. Larger studies across hundreds of targets are needed to build predictive models of when AlphaFold binding-site physics can be trusted.

**AlphaFold only.** We benchmarked AlphaFold Database v6 predictions exclusively. Other structure prediction methods -- including Boltz-2 (Wohlwend et al., 2025), Chai-1 (Chai Discovery, 2024), and AlphaFold 3 (Abramson et al., 2024) -- may show different binding-site physics patterns, particularly those that predict protein--ligand complexes directly. Extending this benchmark to complex-prediction methods is a natural next step.

**Apo predictions only.** All AlphaFold structures are single-chain apo predictions. Methods that predict bound complexes (AlphaFold 3, Boltz-2) would produce holo-like geometries that could change the binding-site clashscore relationship substantially. Our results specifically characterize the apo prediction regime that dominates current AlphaFold Database usage.

**Clashscore only.** We assessed a single physics metric. Rotamer quality, backbone geometry, local strain energy, and electrostatic complementarity may show different or complementary patterns at binding sites. A comprehensive binding-site physics assessment should integrate multiple validation checks.

**Pocket definition sensitivity.** Our 8.0 A cutoff from ligand atoms is standard but arbitrary. Tighter cutoffs would focus on the most proximal residues (where cofactor effects are strongest), while broader cutoffs would dilute binding-site signal with bulk protein. We did not systematically vary this parameter.

---

## Data and Code Availability

All code, structures, and benchmark results are publicly available at https://github.com/Danny2045/physics-auditor. The Physics Auditor tool, benchmark scripts, experimental PDB files, and AlphaFold predictions used in this study are included in the repository under `benchmark/`. The full results in machine-readable format are at `benchmark/results/pocket_clashscore_comparison.json`.

---

## References

Abramson, J., Adler, J., Dunger, J., et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature*, 630, 493--500.

Baek, M., DiMaio, F., Anishchenko, I., et al. (2021). Accurate prediction of protein structures and interactions using a three-track neural network. *Science*, 373(6557), 871--876.

Berman, H. M., Westbrook, J., Feng, Z., et al. (2000). The Protein Data Bank. *Nucleic Acids Research*, 28(1), 235--242.

Buel, G. R. & Bhatt, D. K. (2024). Can AlphaFold2 predict the impact of missense mutations on structure? *Nature Structural & Molecular Biology*, 31, 1--2.

Chai Discovery. (2024). Chai-1: Decoding the molecular interactions of life. *bioRxiv*, 2024.10.10.615955.

Chen, V. B., Arendall, W. B., Headd, J. J., et al. (2010). MolProbity: all-atom structure validation for macromolecular crystallography. *Acta Crystallographica Section D*, 66(1), 12--21.

Jumper, J., Evans, R., Pritzel, A., et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596, 583--589.

Lin, Z., Akin, H., Rao, R., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123--1130.

Read, R. J. & Adams, P. D. (2023). Evaluation of predicted protein structures: A cautionary tale. *Structure*, 31(7), 735--738.

Varadi, M., Anyango, S., Deshpande, M., et al. (2022). AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space with high-accuracy models. *Nucleic Acids Research*, 50(D1), D431--D437.

Wohlwend, J., Corso, G., Passaro, S., et al. (2025). Boltz-2: Exploring the frontiers of biomolecular prediction. *bioRxiv*, 2025.03.07.642143.

Word, J. M., Lovell, S. C., Richardson, J. S., & Richardson, D. C. (1999). Asparagin and glutamine: using hydrogen atom contacts in the choice of side-chain amide orientation. *Journal of Molecular Biology*, 285(4), 1735--1747.
