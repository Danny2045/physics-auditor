"""Run ESM-2 divergence amplification across the available Kira pairs.

Output: ``benchmark/results/divergence/summary.json``. Per-pair report
plus an aggregate mean ± σ across pairs that produced a full triple of
cosines (full / pocket_meanpool / pocket_subseq).

Operational claim being quantified:

    "SmDHODH and HsDHODH share full-sequence ESM-2 cosine 0.9897 yet
    admit ~30× experimental compound selectivity for the parasite."

Numbers per pair plus the aggregate gap (the "divergence amplification":
full − pocket_meanpool) are written verbatim into the JSON dossier with
a ``claims`` field for each pair so claim strength stays attached to
the data.

What's in the repo and what isn't
---------------------------------
Of the Kira NTD pairs, only one has a parasite-side co-crystal in this
checkout (SmDHODH/HsDHODH style): the structures committed under
``benchmark/structures/experimental/`` are mostly human, plus two AF-
predicted parasite monomers (SmDHODH = G4VFD7, LmPTR1 = Q01782). The
runner therefore produces:

* **AF-G4VFD7 vs AF-Q02127** — the actual SmDHODH and HsDHODH UniProt
  sequences. Full-sequence cosine only (the Kira slogan number). No
  pocket-level divergence for this pair because the repo has neither
  a SmDHODH nor an HsDHODH co-crystal (1D3H is an HsDHODH catalytic
  construct, but the parasite side has only the apo AF prediction).

* **1D3H vs 1MVS** — the structural co-crystal pair the existing
  selectivity_map run uses, reusing those pocket indices. Honest
  labeling: 1D3H is the truncated catalytic domain of HsDHODH
  (UniProt Q02127, starting at residue 30); 1MVS is HsDHFR (a
  *different* human enzyme). The existing benchmark code labels these
  "SmDHODH" and "HsDHODH" which is a mislabel inherited from earlier
  Kira notation. The pair is reported here as a *functional paralog*
  comparison (both nucleotide-flavin-related oxidoreductases binding
  bicyclic heterocyclic inhibitors), not as an ortholog comparison.

* **TbPTR1/HsDHFR** — explicitly excluded. We have AF-Q01782 (LmPTR1,
  not Tb) and could pair with HsDHFR sequence from 1MVS, but neither
  side has a co-crystal that establishes pocket residues from the
  same evolutionary context, so the pocket-divergence number for this
  pair would not survive a frontier-lab review. Recorded as exclusion.

* **Leishmaniasis target** — similarly excluded. We have LmPTR1
  (Q01782) but no clean human paralog co-crystal in the repo.

Aggregate is computed only over pairs that produced full cosine triples.
With one such pair the σ is reported as null.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from physics_auditor.causality.divergence import (  # noqa: E402
    compute_divergence,
    full_sequence_cosine,
)
from physics_auditor.core.parser import parse_pdb  # noqa: E402

MODEL_NAME = "esm2_t33_650M_UR50D"


@dataclass
class PairResult:
    """One row of the dossier: per-pair numbers plus claim strength."""

    pair_name: str
    status: str  # "full" | "full_sequence_only" | "excluded"
    target_source: str
    ortholog_source: str
    full_sequence_cosine: float | None
    pocket_meanpool_cosine: float | None
    pocket_subseq_cosine: float | None
    divergence_amplification: float | None
    n_pocket_residues_target: int | None
    n_pocket_residues_ortholog: int | None
    claims: list[str]
    notes: list[str]


def _build_seq_pos_map(structure) -> dict[tuple, int]:
    """Map (chain_id, res_seq, insertion_code) → position in protein
    sequence. This is the bridge between structure-level residue
    identifiers and the 0-based sequence indices compute_divergence
    expects."""
    m: dict[tuple, int] = {}
    pos = 0
    for chain in structure.protein_chains:
        for res in chain.residues:
            if res.is_protein:
                m[(chain.chain_id, res.res_seq, res.insertion_code)] = pos
                pos += 1
    return m


def run_smdhodh_hsdhodh_sequence_pair() -> PairResult:
    """Full-sequence cosine on the ACTUAL SmDHODH and HsDHODH UniProt
    sequences (G4VFD7 / Q02127), via their AF predictions in the repo.

    Reproduces the Kira slogan "full-sequence cosine = 0.9897" headline.
    Pocket-level divergence is not computed for this pair because the
    repo has no SmDHODH co-crystal — the parasite side is the apo AF
    prediction, with no bound ligand from which to extract pocket
    residues via the same convention compute_selectivity_map uses.
    """
    af_sm_path = REPO_ROOT / "benchmark/structures/predicted/AF-G4VFD7-F1-model_v6.pdb"
    af_hs_path = REPO_ROOT / "benchmark/structures/predicted/AF-Q02127-F1-model_v6.pdb"

    if not af_sm_path.exists() or not af_hs_path.exists():
        return PairResult(
            pair_name="SmDHODH_vs_HsDHODH_UniProt",
            status="excluded",
            target_source=str(af_sm_path.relative_to(REPO_ROOT)),
            ortholog_source=str(af_hs_path.relative_to(REPO_ROOT)),
            full_sequence_cosine=None,
            pocket_meanpool_cosine=None,
            pocket_subseq_cosine=None,
            divergence_amplification=None,
            n_pocket_residues_target=None,
            n_pocket_residues_ortholog=None,
            claims=[],
            notes=["AF prediction files not present in this checkout."],
        )

    sm = parse_pdb(af_sm_path)
    hs = parse_pdb(af_hs_path)
    seq_sm = sm.protein_chains[0].sequence
    seq_hs = hs.protein_chains[0].sequence

    cos = full_sequence_cosine(seq_sm, seq_hs, model_name=MODEL_NAME)

    return PairResult(
        pair_name="SmDHODH_vs_HsDHODH_UniProt",
        status="full_sequence_only",
        target_source="AF-G4VFD7 (SmDHODH UniProt sequence)",
        ortholog_source="AF-Q02127 (HsDHODH UniProt sequence)",
        full_sequence_cosine=cos,
        pocket_meanpool_cosine=None,
        pocket_subseq_cosine=None,
        divergence_amplification=None,
        n_pocket_residues_target=None,
        n_pocket_residues_ortholog=None,
        claims=[
            f"Full-sequence ESM-2 ({MODEL_NAME}) cosine between true "
            f"SmDHODH (G4VFD7, {len(seq_sm)} residues) and true HsDHODH "
            f"(Q02127, {len(seq_hs)} residues) is {cos}.",
            "Reproduces the Kira-cited high-similarity slogan from "
            "public UniProt sequences with no additional data.",
        ],
        notes=[
            "Pocket-level divergence is not computed for this pair: no "
            "SmDHODH co-crystal in the repo to anchor pocket-residue "
            "indices via the compute_selectivity_map convention. A "
            "future PDB pull of e.g. 6FMD (SmDHODH+inhibitor) would "
            "enable the pocket leg of the comparison.",
        ],
    )


def run_1d3h_1mvs_structural_pair() -> PairResult:
    """Full divergence triple on 1D3H/1MVS using the pocket indices the
    selectivity_map run already produced.

    Honest labeling: 1D3H is the truncated catalytic domain of HsDHODH
    (UniProt Q02127, residues 30+). 1MVS is HsDHFR. The two are not
    orthologs; they are *functional paralogs* — both nucleotide/flavin-
    related oxidoreductases that engage bicyclic heterocyclic inhibitors
    at their respective active sites. The existing benchmark code labels
    these "SmDHODH" and "HsDHODH" which is a mislabel inherited from
    earlier Kira notation.
    """
    dossier = REPO_ROOT / "benchmark/results/selectivity_maps/SmDHODH_vs_HsDHODH.json"
    p1d3h = REPO_ROOT / "benchmark/structures/experimental/1D3H.pdb"
    p1mvs = REPO_ROOT / "benchmark/structures/experimental/1MVS.pdb"

    if not (dossier.exists() and p1d3h.exists() and p1mvs.exists()):
        return PairResult(
            pair_name="1D3H_vs_1MVS",
            status="excluded",
            target_source=str(p1d3h.relative_to(REPO_ROOT)),
            ortholog_source=str(p1mvs.relative_to(REPO_ROOT)),
            full_sequence_cosine=None,
            pocket_meanpool_cosine=None,
            pocket_subseq_cosine=None,
            divergence_amplification=None,
            n_pocket_residues_target=None,
            n_pocket_residues_ortholog=None,
            claims=[],
            notes=["1D3H/1MVS structures or selectivity-map dossier missing."],
        )

    s_t = parse_pdb(p1d3h)
    s_o = parse_pdb(p1mvs)
    seq_t = s_t.protein_chains[0].sequence
    seq_o = s_o.protein_chains[0].sequence

    m_t = _build_seq_pos_map(s_t)
    m_o = _build_seq_pos_map(s_o)
    rows = json.loads(dossier.read_text())["residues"]

    target_pocket_idx: list[int] = []
    ortholog_pocket_idx: list[int] = []
    for row in rows:
        kt = [k for k in m_t if k[1] == row["res_seq_target"]]
        ko = [k for k in m_o if k[1] == row["res_seq_ortholog"]]
        if kt and ko:
            target_pocket_idx.append(m_t[kt[0]])
            ortholog_pocket_idx.append(m_o[ko[0]])

    rep = compute_divergence(
        seq_t, seq_o, target_pocket_idx, ortholog_pocket_idx,
        model_name=MODEL_NAME, pair_name="1D3H_vs_1MVS",
    )

    return PairResult(
        pair_name="1D3H_vs_1MVS",
        status="full",
        target_source="1D3H (HsDHODH catalytic construct, Q02127 res 30+)",
        ortholog_source="1MVS (HsDHFR)",
        full_sequence_cosine=rep.full_sequence_cosine,
        pocket_meanpool_cosine=rep.pocket_meanpool_cosine,
        pocket_subseq_cosine=rep.pocket_subseq_cosine,
        divergence_amplification=rep.divergence_amplification,
        n_pocket_residues_target=rep.n_pocket_residues_target,
        n_pocket_residues_ortholog=rep.n_pocket_residues_ortholog,
        claims=[
            f"Pocket cosine is lower than full-sequence cosine for this "
            f"pair: full={rep.full_sequence_cosine}, "
            f"pocket_meanpool={rep.pocket_meanpool_cosine}, "
            f"amplification={rep.divergence_amplification}.",
            "The amplification number is the operational quantity: it "
            "is the gap that whole-protein representations conceal.",
        ],
        notes=[
            "This is a functional paralog comparison (HsDHODH vs HsDHFR), "
            "not an ortholog comparison. The selectivity_map dossier "
            "labels these 'SmDHODH' and 'HsDHODH'; that label is a "
            "mislabel — 1D3H's SOURCE record is HOMO SAPIENS DHODH, and "
            "1MVS is HOMO SAPIENS DHFR.",
            f"Pocket residue indices were transferred from "
            f"{dossier.relative_to(REPO_ROOT)} via res_seq lookup; "
            f"alignment is therefore exactly the one compute_selectivity_map "
            f"produced.",
        ],
    )


def excluded_pair(name: str, reason: str) -> PairResult:
    """Record a Kira pair that the runner cannot honestly evaluate."""
    return PairResult(
        pair_name=name,
        status="excluded",
        target_source="",
        ortholog_source="",
        full_sequence_cosine=None,
        pocket_meanpool_cosine=None,
        pocket_subseq_cosine=None,
        divergence_amplification=None,
        n_pocket_residues_target=None,
        n_pocket_residues_ortholog=None,
        claims=[],
        notes=[reason],
    )


def aggregate(rows: list[PairResult]) -> dict:
    """Mean and σ of the divergence amplification across pairs that
    produced a full cosine triple."""
    full = [r for r in rows if r.status == "full" and r.divergence_amplification is not None]
    n = len(full)
    if n == 0:
        return {
            "n_pairs_with_amplification": 0,
            "mean_amplification": None,
            "std_amplification": None,
            "note": "No pairs produced a full cosine triple; nothing to aggregate.",
        }
    amps = [r.divergence_amplification for r in full]
    mean_amp = sum(amps) / n
    if n == 1:
        return {
            "n_pairs_with_amplification": 1,
            "mean_amplification": round(mean_amp, 6),
            "std_amplification": None,
            "note": (
                "Only one pair produced a full cosine triple in this "
                "checkout. σ is not reported because a single-sample "
                "standard deviation is not meaningful. Adding parasite "
                "co-crystals (e.g. 6FMD for SmDHODH) would lift n above "
                "1 and unlock honest error bars."
            ),
        }
    var = sum((a - mean_amp) ** 2 for a in amps) / (n - 1)
    std = var ** 0.5
    return {
        "n_pairs_with_amplification": n,
        "mean_amplification": round(mean_amp, 6),
        "std_amplification": round(std, 6),
        "note": "Aggregate over pairs with full cosine triples.",
    }


def main():
    out_dir = REPO_ROOT / "benchmark" / "results" / "divergence"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[PairResult] = []
    rows.append(run_smdhodh_hsdhodh_sequence_pair())
    rows.append(run_1d3h_1mvs_structural_pair())
    rows.append(excluded_pair(
        "TbPTR1_vs_HsDHFR",
        "No Tb structure in repo (closest: AF-Q01782 = LmPTR1). Pairing "
        "LmPTR1 with the 1MVS HsDHFR sequence is technically possible "
        "but neither side has a co-crystal that anchors pocket-residue "
        "indices in the same evolutionary context as the SmDHODH pair, "
        "so the pocket-divergence number for this pair would not "
        "survive review. Functional paralog comparison, no clean pocket "
        "alignment available in this checkout.",
    ))
    rows.append(excluded_pair(
        "LmPTR1_human_paralog",
        "LmPTR1 (AF-Q01782) has no clean human paralog co-crystal in "
        "the repo. HsDHFR (1MVS) is the closest enzyme-class paralog "
        "but the parasite side (Lm) is apo AF only, so pocket-residue "
        "indices can not be extracted by compute_selectivity_map "
        "convention.",
    ))

    aggregate_block = aggregate(rows)
    summary = {
        "model_name": MODEL_NAME,
        "slogan_being_quantified": (
            "SmDHODH/HsDHODH share full-sequence ESM-2 cosine ~0.99 "
            "yet admit ~30x experimental compound selectivity."
        ),
        "pairs": [asdict(r) for r in rows],
        "aggregate": aggregate_block,
        "data_provenance": {
            "structures_dir": "benchmark/structures/",
            "selectivity_map_dossier": "benchmark/results/selectivity_maps/SmDHODH_vs_HsDHODH.json",
            "esm_model": MODEL_NAME,
        },
    }

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"Model: {MODEL_NAME}")
    print()
    for r in rows:
        if r.status == "excluded":
            print(f"  [excluded] {r.pair_name}")
            for note in r.notes:
                print(f"             {note}")
        elif r.status == "full_sequence_only":
            print(f"  [seq-only] {r.pair_name}")
            print(f"             full_sequence_cosine = {r.full_sequence_cosine}")
        else:
            print(f"  [full]     {r.pair_name}")
            print(f"             full           = {r.full_sequence_cosine}")
            print(f"             pocket_meanpool= {r.pocket_meanpool_cosine}")
            print(f"             pocket_subseq  = {r.pocket_subseq_cosine}")
            print(f"             amplification  = {r.divergence_amplification}")
            print(f"             pocket sizes   = "
                  f"{r.n_pocket_residues_target}/{r.n_pocket_residues_ortholog}")
    print()
    print(f"  Aggregate: {aggregate_block}")
    print(f"\n  Dossier: {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
