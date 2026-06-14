#!/usr/bin/env python3
"""Clash-check native-vs-decoy discrimination benchmark (charter v0.1).

Governing doc: ``CHARTER_clash_discrimination_v0.1.md`` (committed ``6dad2dc``).
Native draw locked at ``575a4af`` (``native_set_30.txt``). The 30 target IDs
are frozen — this runner never redraws or alters the set.

What this does (and only this):

- **Tier 1 (positive control):** for each native, generate perturbation decoys
  by Cartesian Gaussian displacement of all (heavy) atoms at σ ∈ {0.5, 1.0,
  1.5, 2.0} Å, K_p=50 per native per σ; score native + decoys; report AUROC
  pooled across targets per σ and the per-σ mean clashscore (dose-response).
  PASS bar: AUROC ≥ 0.95 at σ=2.0. If it fails, the run STOPS before Tier 2
  (the clash check would be defective on gross corruption — charter §3/§6).
- **Tier 2 (the real test):** for each target, score native + its 300 3DRobot
  decoys; report pooled AUROC (bar 0.65), native-recognition rate, and the
  per-target native-vs-decoy Z-score distribution (charter §2).

INTEGRITY INVARIANTS (charter §5, and the task brief):

1. IDENTICAL PREPROCESSING. Every structure — native, perturbation decoy, and
   3DRobot decoy — is reduced to (atoms, coords) by the committed
   ``parse_pdb`` (which strips hydrogens by default and keeps altloc "A"), and
   then scored by exactly one function, :func:`clashscore_of_structure`, which
   runs the committed ``infer_bonds_from_topology`` → ``build_bonded_mask`` →
   ``compute_distance_matrix`` → ``check_clashes`` with the default frozen
   ``ClashConfig``. Perturbation decoys reuse the native's parsed heavy-atom
   set with displaced coordinates (cast to float32, matching the file-parse
   path), so their preprocessing is identical to the native's by construction.
   No class of structure is treated differently at any step.

2. COMMITTED CLASH CHECK, UNMODIFIED. This runner imports and calls the
   existing ``check_clashes`` and its preprocessing from ``src/``; it does not
   reimplement, retune, or touch the frozen sigmoid/thresholds. Fidelity to the
   library ``audit()`` clashscore is asserted in the regression test.

3. PARSE FAILURES ARE RECORDED AS EXCLUSIONS, NEVER SILENTLY DROPPED. A native
   that fails to parse excludes its whole target (recorded). A decoy that fails
   to parse is recorded (id, which structure, error) and skipped; scored-vs-
   excluded counts are reported per target and in totals.

AUROC is implemented directly (Mann-Whitney / average-rank form, ties = 0.5) so
no new dependency is added.

Run (one command)::

    python benchmark/discrimination/run_discrimination.py

Writes ``discrimination_results.json`` and ``discrimination_results.md`` next to
itself. Deterministic: same seed, same committed clashscore → same AUROCs.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import tarfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from physics_auditor.checks.clashes import check_clashes
from physics_auditor.config import ClashConfig
from physics_auditor.core.geometry import compute_distance_matrix
from physics_auditor.core.parser import Structure, parse_pdb
from physics_auditor.core.topology import build_bonded_mask, infer_bonds_from_topology

# --- Charter-fixed constants (do not change without a dated charter amendment) ---
SEED = 20260613  # charter §3/§4
SIGMAS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)  # charter §3 Tier 1
K_P = 50  # perturbation decoys per native per sigma, charter §3
TIER1_BAR = 0.95  # AUROC at sigma=2.0, charter §3/§6
TIER2_BAR = 0.65  # pooled AUROC, charter §3/§6

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
NATIVE_SET_FILE = HERE / "native_set_30.txt"
RESULTS_JSON = HERE / "discrimination_results.json"
RESULTS_MD = HERE / "discrimination_results.md"

# 3DRobot data dir (machine-local; NOT committed — reproducible from the
# charter §1 archive URL). Override with PHYSICS_AUDITOR_3DROBOT_DIR.
DEFAULT_DATA_DIR = Path(
    os.environ.get("PHYSICS_AUDITOR_3DROBOT_DIR", "~/3DRobot_data/3DRobot_set")
).expanduser()

CLASH_CONFIG = ClashConfig()  # frozen defaults; never tuned (charter §5)


@dataclasses.dataclass
class Score:
    """Clashscore of one structure under the committed pipeline."""

    clashscore: float
    n_atoms: int
    n_clashes: int


def clashscore_of_structure(structure: Structure) -> Score:
    """Score one structure with the committed clash pipeline, unmodified.

    Replicates exactly the clash leg of ``audit()`` (parse → bonds → mask →
    distance → ``check_clashes``) using the committed functions and the frozen
    default ``ClashConfig``. This is the single scoring path used for natives
    and all decoys.
    """
    bonds = infer_bonds_from_topology(structure)
    mask = build_bonded_mask(structure.n_atoms, bonds)
    dist = compute_distance_matrix(jnp.array(structure.coords))
    result = check_clashes(
        dist,
        structure.elements,
        jnp.array(mask),
        structure.res_indices,
        structure.n_residues,
        CLASH_CONFIG,
    )
    return Score(
        clashscore=float(result.clashscore),
        n_atoms=int(structure.n_atoms),
        n_clashes=int(result.n_clashes),
    )


def perturbed_structure(native: Structure, sigma: float, rng: np.random.Generator) -> Structure:
    """Return a copy of ``native`` with all atom coords Gaussian-displaced.

    Cartesian displacement: each coordinate gets independent N(0, sigma) noise.
    Coords are cast to float32 to match the file-parse path exactly. Only the
    coordinate array changes; the atom set / elements / residue indices are the
    native's, so downstream preprocessing is identical to the native's.
    """
    noise = rng.normal(loc=0.0, scale=sigma, size=native.coords.shape)
    new_coords = (native.coords + noise).astype(np.float32)
    return dataclasses.replace(native, coords=new_coords)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC with clashscore as the decoy-score (positive label = 1 = decoy).

    Mann-Whitney / average-rank formulation (ties contribute 0.5), identical to
    ``sklearn.metrics.roc_auc_score`` but dependency-free. Returns NaN if either
    class is empty.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0  # average of 1-based ranks i+1..j
        ranks[order[i:j]] = avg_rank
        i = j
    sum_ranks_pos = float(ranks[labels == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_native_ids() -> list[str]:
    """Load the 30 locked native target IDs, sorted (order-independent)."""
    ids = [ln.strip() for ln in NATIVE_SET_FILE.read_text().splitlines() if ln.strip()]
    if len(ids) != 30:
        raise ValueError(f"expected 30 locked IDs, found {len(ids)}")
    return sorted(ids)


def ensure_extracted(data_dir: Path, target_id: str) -> Path:
    """Ensure ``<data_dir>/<target_id>/`` exists, extracting its tarball if not."""
    target_dir = data_dir / target_id
    if target_dir.is_dir():
        return target_dir
    tarball = data_dir / f"{target_id}.tar.bz2"
    if not tarball.exists():
        raise FileNotFoundError(f"missing 3DRobot data for {target_id}: {tarball}")
    with tarfile.open(tarball, "r:bz2") as tf:
        tf.extractall(data_dir)
    return target_dir


def read_rmsd_map(list_path: Path) -> dict[str, float]:
    """Parse a 3DRobot ``list.txt`` into ``{decoy_filename: rmsd_to_native}``.

    Used only for the post-hoc near-native control (NOT a pre-registered metric).
    The header line ("NAME RMSD") and any non-numeric row are skipped.
    """
    out: dict[str, float] = {}
    if not list_path.exists():
        return out
    for line in list_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = float(parts[-1])
            except ValueError:
                continue
    return out


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def run_benchmark(data_dir: Path, native_ids: list[str]) -> dict:
    """Run Tier 1 then (if it passes) Tier 2; return the full results dict."""
    excluded_targets: list[dict] = []

    # --- Parse + score every native once (reused by both tiers) ---
    native_struct: dict[str, Structure] = {}
    native_score: dict[str, Score] = {}
    for tid in native_ids:
        target_dir = ensure_extracted(data_dir, tid)
        npath = target_dir / "native.pdb"
        try:
            s = parse_pdb(npath)
            native_struct[tid] = s
            native_score[tid] = clashscore_of_structure(s)
        except Exception as exc:  # parse/score failure → exclude whole target
            excluded_targets.append(
                {"id": tid, "structure": "native", "error": f"{type(exc).__name__}: {exc}"}
            )

    scored_ids = [tid for tid in native_ids if tid in native_score]

    # ===================== TIER 1: perturbation positive control =====================
    sigma_decoy_cs: dict[float, list[float]] = {sig: [] for sig in SIGMAS}
    n_tier1_decoys = 0
    for ti, tid in enumerate(scored_ids):
        native = native_struct[tid]
        for si, sigma in enumerate(SIGMAS):
            # Per-(target, sigma) independent RNG rooted at the charter seed, so
            # the draw is reproducible and immune to target ordering / exclusions.
            rng = np.random.default_rng([SEED, ti, si])
            for _ in range(K_P):
                pert = perturbed_structure(native, sigma, rng)
                sigma_decoy_cs[sigma].append(clashscore_of_structure(pert).clashscore)
                n_tier1_decoys += 1

    native_cs_list = [native_score[tid].clashscore for tid in scored_ids]
    tier1_per_sigma: dict[str, dict] = {}
    for sigma in SIGMAS:
        decoys = sigma_decoy_cs[sigma]
        scores = np.array(native_cs_list + decoys, dtype=np.float64)
        labels = np.array([0] * len(native_cs_list) + [1] * len(decoys), dtype=np.int64)
        tier1_per_sigma[f"{sigma}"] = {
            "auroc": round(auroc(scores, labels), 6),
            "mean_decoy_clashscore": round(float(np.mean(decoys)), 4),
            "n_decoys": len(decoys),
        }
    tier1_auroc_sigma2 = tier1_per_sigma[f"{SIGMAS[-1]}"]["auroc"]
    tier1_passed = tier1_auroc_sigma2 >= TIER1_BAR

    results: dict = {
        "charter": "CHARTER_clash_discrimination_v0.1.md (6dad2dc)",
        "native_draw_lock_commit": "575a4af",
        "seed": SEED,
        "numpy_version": np.__version__,
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "repo_commit": _git_head(),
        "clashscore_source": "physics_auditor.checks.clashes.check_clashes (committed, unmodified)",
        "clash_config": {
            "vdw_tolerance": CLASH_CONFIG.vdw_tolerance,
            "ignore_bonded_neighbors": CLASH_CONFIG.ignore_bonded_neighbors,
            "severe_clash_threshold": CLASH_CONFIG.severe_clash_threshold,
        },
        "preprocessing": (
            "parse_pdb defaults (keep_hydrogens=False → H stripped; keep_altloc='A'); "
            "then infer_bonds_from_topology → build_bonded_mask → compute_distance_matrix "
            "→ check_clashes; identical for natives and all decoys."
        ),
        "n_targets_requested": len(native_ids),
        "n_targets_scored": len(scored_ids),
        "excluded_targets": excluded_targets,
        "tier1": {
            "design": {
                "sigmas": list(SIGMAS),
                "k_p_per_native_per_sigma": K_P,
                "rng_method": (
                    "numpy.random.default_rng([20260613, target_index, sigma_index]); "
                    "target_index over sorted(native_ids), sigma_index over "
                    "[0.5,1.0,1.5,2.0]; decoy = native_coords + N(0,sigma), float32"
                ),
            },
            "native_mean_clashscore": round(float(np.mean(native_cs_list)), 4),
            "per_sigma": tier1_per_sigma,
            "auroc_sigma_2.0": tier1_auroc_sigma2,
            "bar": TIER1_BAR,
            "passed": tier1_passed,
            "n_decoys_total": n_tier1_decoys,
            "n_decoys_excluded": 0,
        },
    }

    if not tier1_passed:
        results["stopped_after_tier1"] = True
        results["outcome_table_mapping"] = {
            "tier1_row": (
                f"AUROC {tier1_auroc_sigma2:.4f} < {TIER1_BAR} @ σ=2.0 — the clash "
                "check is DEFECTIVE on gross corruption. STOP, investigate (a finding "
                "about the check itself, not a tuning opportunity). [charter §6 row 2]"
            ),
            "tier2_row": "NOT RUN — Tier 1 positive control failed.",
        }
        return results

    # ===================== TIER 2: 3DRobot decoys (the real test) =====================
    excluded_decoys: list[dict] = []
    per_target: list[dict] = []
    pooled_scores: list[float] = []
    pooled_labels: list[int] = []
    n_decoys_scored = 0
    n_decoys_excluded = 0
    n_recognized = 0
    zscores: list[float] = []
    nearnative: list[dict] = []  # post-hoc near-native control (not pre-registered)

    for tid in scored_ids:
        target_dir = ensure_extracted(data_dir, tid)
        native_cs = native_score[tid].clashscore
        rmsd_map = read_rmsd_map(target_dir / "list.txt")
        decoy_cs: list[float] = []
        decoy_rmsd_cs: list[tuple[float, float]] = []  # (rmsd, clashscore) when rmsd known
        n_excl_target = 0
        for dpath in sorted(target_dir.glob("decoy*.pdb")):
            try:
                ds = parse_pdb(dpath)
                cs = clashscore_of_structure(ds).clashscore
            except Exception as exc:
                n_excl_target += 1
                if len(excluded_decoys) < 200:  # cap the recorded list
                    excluded_decoys.append(
                        {
                            "target": tid,
                            "decoy": dpath.name,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                continue
            decoy_cs.append(cs)
            rmsd = rmsd_map.get(dpath.name)
            if rmsd is not None:
                decoy_rmsd_cs.append((rmsd, cs))
        n_decoys_scored += len(decoy_cs)
        n_decoys_excluded += n_excl_target

        # pool for AUROC
        pooled_scores.append(native_cs)
        pooled_labels.append(0)
        pooled_scores.extend(decoy_cs)
        pooled_labels.extend([1] * len(decoy_cs))

        # Post-hoc near-native control: the single lowest-RMSD decoy vs native.
        if decoy_rmsd_cs:
            min_rmsd, nn_cs = min(decoy_rmsd_cs, key=lambda t: t[0])
            nearnative.append(
                {
                    "id": tid,
                    "min_rmsd_angstrom": round(min_rmsd, 3),
                    "nearnative_decoy_clashscore": round(nn_cs, 4),
                    "native_clashscore": round(native_cs, 4),
                    "nearnative_exceeds_native": bool(nn_cs > native_cs),
                }
            )

        decoy_arr = np.array(decoy_cs, dtype=np.float64)
        decoy_mean = float(decoy_arr.mean()) if len(decoy_arr) else float("nan")
        decoy_std = float(decoy_arr.std(ddof=0)) if len(decoy_arr) else float("nan")
        native_is_min = bool(len(decoy_arr) and native_cs < decoy_arr.min())
        if native_is_min:
            n_recognized += 1
        # native's z-score relative to the decoy distribution (negative = native
        # has lower clashscore than the decoy mean = good discrimination)
        zscore = (
            (native_cs - decoy_mean) / decoy_std
            if (len(decoy_arr) and decoy_std > 0)
            else float("nan")
        )
        if not np.isnan(zscore):
            zscores.append(zscore)
        per_target.append(
            {
                "id": tid,
                "native_clashscore": round(native_cs, 4),
                "n_decoys_scored": len(decoy_cs),
                "n_decoys_excluded": n_excl_target,
                "decoy_mean_clashscore": round(decoy_mean, 4),
                "decoy_std_clashscore": round(decoy_std, 4),
                "native_zscore": round(zscore, 4) if not np.isnan(zscore) else None,
                "native_is_lowest": native_is_min,
            }
        )

    pooled_auroc = round(
        auroc(np.array(pooled_scores, dtype=np.float64), np.array(pooled_labels, dtype=np.int64)),
        6,
    )
    n_recog_targets = len(per_target)
    native_recognition_rate = round(n_recognized / n_recog_targets, 6) if n_recog_targets else float("nan")
    z = np.array(zscores, dtype=np.float64)
    z_summary = {
        "mean": round(float(z.mean()), 4) if len(z) else None,
        "median": round(float(np.median(z)), 4) if len(z) else None,
        "min": round(float(z.min()), 4) if len(z) else None,
        "max": round(float(z.max()), 4) if len(z) else None,
        "std": round(float(z.std(ddof=0)), 4) if len(z) else None,
        "n_targets": len(z),
    }
    meets_bar = pooled_auroc >= TIER2_BAR

    # Post-hoc near-native control summary (only meaningful when AUROC ≥ bar).
    nn_native = np.array([d["native_clashscore"] for d in nearnative], dtype=np.float64)
    nn_decoy = np.array([d["nearnative_decoy_clashscore"] for d in nearnative], dtype=np.float64)
    nn_rmsd = np.array([d["min_rmsd_angstrom"] for d in nearnative], dtype=np.float64)
    n_nn_exceeds = sum(1 for d in nearnative if d["nearnative_exceeds_native"])
    nearnative_control = {
        "description": (
            "POST-HOC (not pre-registered): for each target, the single lowest-RMSD "
            "3DRobot decoy (closest to the native fold) vs the native clashscore. If "
            "near-native-fold decoys still clash far more than the native, the Tier-2 "
            "separation is a preparation/provenance signal (experimentally-refined "
            "natives vs computationally-generated decoys), not fold-quality discrimination."
        ),
        "n_targets": len(nearnative),
        "mean_min_rmsd_angstrom": round(float(nn_rmsd.mean()), 3) if len(nn_rmsd) else None,
        "max_min_rmsd_angstrom": round(float(nn_rmsd.max()), 3) if len(nn_rmsd) else None,
        "mean_native_clashscore": round(float(nn_native.mean()), 4) if len(nn_native) else None,
        "mean_nearnative_decoy_clashscore": (
            round(float(nn_decoy.mean()), 4) if len(nn_decoy) else None
        ),
        "n_targets_nearnative_exceeds_native": n_nn_exceeds,
        "per_target": nearnative,
    }

    results["tier2"] = {
        "auroc_pooled": pooled_auroc,
        "bar": TIER2_BAR,
        "meets_bar": meets_bar,
        "native_recognition_rate": native_recognition_rate,
        "n_targets_recognized": n_recognized,
        "zscore_summary": z_summary,
        "zscore_definition": "(native_clashscore - decoy_mean) / decoy_std; negative = native below decoy mean",
        "n_native_scored": len(scored_ids),
        "n_decoys_scored": n_decoys_scored,
        "n_decoys_excluded": n_decoys_excluded,
        "nearnative_control": nearnative_control,
        "per_target": per_target,
        "excluded_decoys": excluded_decoys,
    }
    results["stopped_after_tier1"] = False
    results["outcome_table_mapping"] = {
        "tier1_row": (
            f"AUROC {tier1_auroc_sigma2:.4f} ≥ {TIER1_BAR} @ σ=2.0 — positive control "
            "holds: the scored check fires on real packing defects. [charter §6 row 1]"
        ),
        "tier2_row": (
            (
                f"AUROC {pooled_auroc:.4f} ≥ {TIER2_BAR} — clash UNEXPECTEDLY discriminates "
                "3DRobot decoys: surprising positive, investigate (e.g. a clash-rich decoy "
                "subpopulation). [charter §6 row 4]"
            )
            if meets_bar
            else (
                f"AUROC {pooled_auroc:.4f} < {TIER2_BAR} (the pre-registered expectation) — "
                "clash alone does NOT discriminate well-packed realistic decoys; the "
                "instrument's abstention on realistic structures is justified, consistent "
                "with 3DRobot's design against trivial potentials. Expected honest envelope "
                "limit, NOT a failure. [charter §6 row 3]"
            )
        ),
    }
    return results


def _render_md(r: dict) -> str:
    """Render a human-readable summary mirroring the JSON."""
    lines: list[str] = []
    A = lines.append
    A("# Clash-check native-vs-decoy discrimination — results (charter v0.1)")
    A("")
    A(f"- Charter: `{r['charter']}`; native draw lock: `{r['native_draw_lock_commit']}`")
    A(f"- Clashscore source: `{r['clashscore_source']}`")
    A(f"- ClashConfig (frozen, untouched): `{r['clash_config']}`")
    A(f"- Preprocessing (identical natives/decoys): {r['preprocessing']}")
    A(f"- Seed: `{r['seed']}` | numpy `{r['numpy_version']}` | jax `{r['jax_version']}` | repo `{r['repo_commit'][:7]}`")
    A(f"- Targets: {r['n_targets_scored']} scored of {r['n_targets_requested']} requested")
    if r["excluded_targets"]:
        A(f"- EXCLUDED targets ({len(r['excluded_targets'])}): {r['excluded_targets']}")
    else:
        A("- Excluded targets: 0")
    A("")
    A("## Tier 1 — perturbation positive control (clash MUST fire)")
    A("")
    A(f"RNG: {r['tier1']['design']['rng_method']}")
    A(f"K_p={r['tier1']['design']['k_p_per_native_per_sigma']} per native per σ; "
      f"native mean clashscore = {r['tier1']['native_mean_clashscore']}")
    A("")
    A("| σ (Å) | AUROC | mean decoy clashscore | n decoys |")
    A("|------:|------:|----------------------:|---------:|")
    for sig in (str(s) for s in r["tier1"]["design"]["sigmas"]):
        ps = r["tier1"]["per_sigma"][sig]
        A(f"| {sig} | {ps['auroc']:.6f} | {ps['mean_decoy_clashscore']:.4f} | {ps['n_decoys']} |")
    A("")
    A(f"**Tier 1 bar: AUROC ≥ {r['tier1']['bar']} at σ=2.0 — "
      f"observed {r['tier1']['auroc_sigma_2.0']:.6f} → "
      f"{'PASS' if r['tier1']['passed'] else 'FAIL'}.**")
    A(f"Total Tier 1 decoys scored: {r['tier1']['n_decoys_total']} "
      f"(excluded: {r['tier1']['n_decoys_excluded']}).")
    A("")
    if r.get("stopped_after_tier1"):
        A("## Tier 2 — NOT RUN")
        A("")
        A(f"> {r['outcome_table_mapping']['tier1_row']}")
        A("")
        return "\n".join(lines) + "\n"
    t2 = r["tier2"]
    A("## Tier 2 — 3DRobot decoys (the real test; expected AUROC ~0.5)")
    A("")
    A(f"- **Pooled AUROC = {t2['auroc_pooled']:.6f}** (bar ≥ {t2['bar']} → "
      f"{'MEETS' if t2['meets_bar'] else 'below'} bar)")
    A(f"- Native-recognition rate = {t2['native_recognition_rate']:.6f} "
      f"({t2['n_targets_recognized']}/{r['n_targets_scored']} targets where native has the lowest clashscore)")
    A(f"- Native z-score vs decoy distribution ({t2['zscore_definition']}):")
    A(f"  mean={t2['zscore_summary']['mean']}, median={t2['zscore_summary']['median']}, "
      f"min={t2['zscore_summary']['min']}, max={t2['zscore_summary']['max']}, "
      f"std={t2['zscore_summary']['std']} (n={t2['zscore_summary']['n_targets']})")
    A(f"- Scored: {t2['n_native_scored']} natives + {t2['n_decoys_scored']} decoys; "
      f"decoys excluded (parse failures): {t2['n_decoys_excluded']}")
    A("")
    nn = t2["nearnative_control"]
    A("## Post-hoc near-native control (NOT pre-registered)")
    A("")
    A(nn["description"])
    A("")
    A(f"- Lowest-RMSD decoy per target: mean min-RMSD = {nn['mean_min_rmsd_angstrom']} Å "
      f"(max {nn['max_min_rmsd_angstrom']} Å) across {nn['n_targets']} targets")
    A(f"- Mean native clashscore = {nn['mean_native_clashscore']} vs mean "
      f"lowest-RMSD-decoy clashscore = {nn['mean_nearnative_decoy_clashscore']}")
    A(f"- Lowest-RMSD decoy clashscore EXCEEDS its native in "
      f"{nn['n_targets_nearnative_exceeds_native']}/{nn['n_targets']} targets")
    A("")
    A("Reading: if even the closest-to-native decoys clash far more than the native, the "
      "Tier-2 separation reflects experimental refinement vs computational generation (a "
      "preparation/provenance confound), not native-fold recognition. See "
      "DISCRIMINATION_FINDINGS.md.")
    A("")
    A("## Charter §6 pre-committed outcome mapping")
    A("")
    A(f"- **Tier 1:** {r['outcome_table_mapping']['tier1_row']}")
    A(f"- **Tier 2:** {r['outcome_table_mapping']['tier2_row']}")
    A("")
    return "\n".join(lines) + "\n"


def main() -> None:
    t0 = time.time()
    native_ids = load_native_ids()
    print(f"Loaded {len(native_ids)} locked native IDs. Data dir: {DEFAULT_DATA_DIR}")
    results = run_benchmark(DEFAULT_DATA_DIR, native_ids)
    results["runtime_seconds"] = round(time.time() - t0, 1)

    RESULTS_JSON.write_text(json.dumps(results, indent=2) + "\n")
    RESULTS_MD.write_text(_render_md(results))

    print(f"\nTier 1 AUROC @ σ=2.0 = {results['tier1']['auroc_sigma_2.0']:.6f} "
          f"(bar {TIER1_BAR}) → {'PASS' if results['tier1']['passed'] else 'FAIL'}")
    if results.get("stopped_after_tier1"):
        print("STOPPED after Tier 1 — clash check defective on gross corruption (charter §6).")
    else:
        t2 = results["tier2"]
        print(f"Tier 2 pooled AUROC = {t2['auroc_pooled']:.6f} (bar {TIER2_BAR}) → "
              f"{'MEETS' if t2['meets_bar'] else 'below'} bar")
        print(f"Tier 2 native-recognition rate = {t2['native_recognition_rate']:.6f}")
    print(f"Wrote {RESULTS_JSON.name} and {RESULTS_MD.name} in {results['runtime_seconds']}s")


if __name__ == "__main__":
    main()
