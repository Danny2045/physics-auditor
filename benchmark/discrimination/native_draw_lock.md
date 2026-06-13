# Native draw lock — 3DRobot K=30 (charter v0.1 §4)

**Status:** PRE-SCORING LOCK. The 30 native target IDs in §4 were drawn and
committed **before any clashscore, AUROC, perturbation, or scoring code ran**,
per `CHARTER_clash_discrimination_v0.1.md` (committed `6dad2dc`) §4 and §5. No
discrimination result exists yet.

**Date:** 2026-06-13.

## 1. Decoy set: source + identity verification

- **Set:** 3DRobot decoy set (`3DRobot_set`), Yang Zhang lab.
- **Citation:** Deng H, Jia Y, Zhang Y. *3DRobot: automated generation of
  diverse and well-packed protein structure decoys.* Bioinformatics 2016,
  32(3):378–387. PMID 26471454; PMCID PMC5006309. (Matches charter §3 Tier 2.)
- **Source page:** https://zhanggroup.org/3DRobot/decoys/
- **Archive URL:** https://zhanggroup.org/3DRobot/decoys/3DRobot_set.tar.bz2
  - HTTP 200, `content-type: application/x-bzip2`, `content-length: 655143405`
    (≈625 MB), `last-modified: Fri, 16 Jan 2026 05:45:43 GMT`.
  - Downloaded size verified == **655143405** bytes.
  - Archive sha256: `cd2b6be25ea8fa20af080a54d505e72804a21500adbc7be5bc813117ed045192`

**Official set description — quoted verbatim from the Zhang-lab decoys page:**

> The '3DRobot_set' includes structural decoys for 200 non-homologous proteins
> that are randomly selected from the PDB library. This protein set contains
> 48 α-, 40 β-, and 112 α/β-single-domain proteins with length ranging from 80
> to 250 residues. Each protein contains 300 structural decoys with RMSD
> ranging from 0 to 12 Å.

**Independent verification against the downloaded distribution itself (not the
web page):**

- The outer archive contains 1 directory plus **200** nested per-target
  archives `3DRobot_set/<ID>.tar.bz2` (201 entries total).
- Each per-target archive expands to `<ID>/` containing **exactly 300**
  `decoy*.pdb` files **+ 1** `native.pdb` (plus `rst.dat`, `list.txt`
  metadata). Swept across **all 200** targets: decoy-count histogram
  `{300: 200}`, native-count histogram `{1: 200}`, **0 anomalies**.
- **PDB format** confirmed on real extracted files — fixed-width `ATOM`
  records with parseable coordinate columns. E.g. `4BK7A/native.pdb` first
  record `ATOM      1  N   GLY A   1      -7.018  -9.419  20.295` (1403 atoms);
  `4BK7A/decoy1_11.pdb` (1382 atoms). The auditor's parser consumes PDB; this
  satisfies charter §4 ("3DRobot decoys are PDB-format").

→ **200 targets / 300 decoys each / PDB format — confirmed from the
distribution, not just the page.**

## 2. Draw basis (the sorted 200)

- **Target-ID form:** `<PDBID><chain>` (5 characters), taken from the
  distribution's own per-target archive basenames `3DRobot_set/<ID>.tar.bz2` —
  the authoritative naming. All 200 IDs are unique and 5-char alphanumeric
  (chains include A, and a few B/M — e.g. `4EETB`, `2FHZB`, `2BMOB`, `3HNYM`).
- **Basis file:** `sorted_200_targets.txt` — those 200 IDs sorted
  lexicographically with Python `sorted()`, one per line.
  - sha256: `fcca0dc2f4391c40338fbeb880ce7a1e34413fe94f9b5b36573f7f7f16f45e85`
- Sorting makes the draw independent of download / filesystem order, as charter
  §4 intends ("Sort them lexicographically … deterministic regardless of order").

## 3. The draw

- **Method:** `numpy.random.default_rng(20260613).choice(sorted_200_ids,
  size=30, replace=False)`.
- **Seed:** `20260613` (charter §4, fixed before any draw).
- **K:** `30` (charter §4).
- **numpy version:** `2.4.4` (default PCG64 BitGenerator). `choice` selects
  indices from a stream determined only by the seed and population size (200),
  then maps them onto the sorted basis — so the draw is stable across numpy
  versions; the version is recorded here per charter.
- **One-command reproducer (no scoring):**
  `python benchmark/discrimination/draw_native_set.py` re-derives the 30 IDs
  and rewrites `native_set_30.txt` byte-for-byte.

## 4. The 30 locked native target IDs

Draw order (as returned by `rng.choice`):

```
4FTFA 4B97A 2C4JA 2OVJA 3M9QA 1W0NA 4I2UA 4IZXA 3SY1A 4EETB
4GAIA 2FHZB 3HNYM 4AQOA 2BT9A 1TJXA 256BA 1SAUA 3NE0A 3QZXA
2NSZA 4JFIA 2BMOB 1WPUA 2C2UA 4BK7A 4J1PA 4KZVA 4B62A 3RHBA
```

Sorted (for reference):

```
1SAUA 1TJXA 1W0NA 1WPUA 256BA 2BMOB 2BT9A 2C2UA 2C4JA 2FHZB
2NSZA 2OVJA 3HNYM 3M9QA 3NE0A 3QZXA 3RHBA 3SY1A 4AQOA 4B62A
4B97A 4BK7A 4EETB 4FTFA 4GAIA 4I2UA 4IZXA 4J1PA 4JFIA 4KZVA
```

- Machine-readable: `native_set_30.txt` (draw order, one ID per line).
  - sha256: `e709fded3f057dc3fd4dd0ddb85243f5c6a690aa408efdb53b443edb9c09c10c`
- All 30 ∈ the sorted-200 basis (verified). For each, the native + its 300
  decoys live in `3DRobot_set/<ID>.tar.bz2` of the distribution above.

## 5. Local data (not committed)

The 625 MB archive and its extracted decoys are **not** committed — only this
lock, the two ID manifests (`sorted_200_targets.txt`, `native_set_30.txt`), and
the draw script (`draw_native_set.py`) are. For PART B the data is reproducible
from the §1 archive URL. The local extraction used for verification here was
`~/3DRobot_data/3DRobot_set/` (machine-specific; the §1 URL is the canonical,
reproducible path).

## 6. Pre-registration wall — attestation

No clashscore, AUROC, perturbation, or scoring code ran in producing this lock.
The only computation was: archive download + integrity check, archive
enumeration, PDB-format / structure verification, and the numpy draw over
target-ID strings. The committed clash check is untouched. Per charter §5, the
30 native IDs are now fixed **before** any clashscore is computed.
