"""Physics Auditor CLI.

Entry point for structure validation and mechanistic analysis.

Usage:
    physics-auditor validate structure.pdb
    physics-auditor validate *.pdb --output-dir reports/
    physics-auditor info structure.pdb
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="physics-auditor",
    help="Physics validation and mechanistic explanation for AI-generated protein structures.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def validate(
    paths: list[Path] = typer.Argument(..., help="PDB file(s) to validate"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o", help="Directory for JSON reports"),
    config: Path | None = typer.Option(None, "--config", "-c", help="YAML config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-residue details"),
    json_output: bool = typer.Option(False, "--json", help="Print JSON to stdout"),
):
    """Validate one or more protein structures."""
    import jax.numpy as jnp

    from physics_auditor.checks.clashes import check_clashes
    from physics_auditor.config import load_config
    from physics_auditor.core.energy import run_lj_analysis
    from physics_auditor.core.geometry import compute_distance_matrix, extract_backbone_dihedrals
    from physics_auditor.core.parser import parse_pdb
    from physics_auditor.core.topology import build_bonded_mask, infer_bonds_from_topology

    cfg = load_config(str(config) if config else None)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        if not path.exists():
            console.print(f"[red]File not found: {path}[/red]")
            continue

        t0 = time.time()

        try:
            struct = parse_pdb(path)
        except (ValueError, Exception) as e:
            console.print(f"[red]Failed to parse {path}: {e}[/red]")
            continue

        # Build topology
        bonds = infer_bonds_from_topology(struct)
        mask = build_bonded_mask(struct.n_atoms, bonds)

        # Compute distance matrix (the foundation)
        coords = jnp.array(struct.coords)
        dist_matrix = compute_distance_matrix(coords)
        mask_jnp = jnp.array(mask)

        # Run checks
        clash_result = check_clashes(
            dist_matrix, struct.elements, mask_jnp,
            struct.res_indices, struct.n_residues, cfg.clashes,
        )

        # Run LJ analysis
        lj_result = run_lj_analysis(
            dist_matrix, struct.elements, mask_jnp,
            struct.res_indices, struct.n_residues, cfg.lennard_jones.energy_cap,
        )

        # Extract backbone dihedrals
        dihedrals = extract_backbone_dihedrals(
            struct.coords, struct.atom_names, struct.res_indices,
            struct.is_protein_mask, struct.chain_ids_array,
        )

        runtime = time.time() - t0

        # Compute preliminary composite (using available checks)
        # Full composite will include all 8 checks — for now use clashes + LJ
        composite = clash_result.subscore  # Simplified for v0.1

        # Recommendation
        if composite >= cfg.composite.accept_threshold:
            recommendation = "accept"
            rec_color = "green"
        elif composite >= cfg.composite.short_md_threshold:
            recommendation = "short_md"
            rec_color = "yellow"
        else:
            recommendation = "discard"
            rec_color = "red"

        # Build report dict
        report = {
            "file": str(path),
            "global_score": round(composite, 3),
            "recommendation": recommendation,
            "checks": {
                "steric_clashes": {
                    "n_clashes": clash_result.n_clashes,
                    "n_severe": clash_result.n_severe_clashes,
                    "clashscore": round(clash_result.clashscore, 2),
                    "worst_overlap_angstrom": round(clash_result.worst_overlap, 3),
                    "subscore": round(clash_result.subscore, 3),
                },
                "lennard_jones": {
                    "total_energy_kcal": round(lj_result["total_energy"], 2),
                    "n_hot_pairs": lj_result["n_hot_pairs"],
                },
                "backbone_dihedrals": {
                    "n_phi": len(dihedrals["phi"]["res_indices"]),
                    "n_psi": len(dihedrals["psi"]["res_indices"]),
                    "n_omega": len(dihedrals["omega"]["res_indices"]),
                },
            },
            "metadata": {
                "n_atoms": struct.n_atoms,
                "n_residues": struct.n_residues,
                "n_chains": struct.n_chains,
                "n_bonds_inferred": len(bonds),
                "protein_chains": len(struct.protein_chains),
                "runtime_seconds": round(runtime, 3),
            },
        }

        if json_output:
            print(json.dumps(report, indent=2))
        else:
            _print_rich_report(report, verbose, rec_color)

        if output_dir:
            out_path = output_dir / f"{path.stem}_report.json"
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)
            console.print(f"  Report saved: {out_path}")


@app.command()
def info(
    path: Path = typer.Argument(..., help="PDB file to inspect"),
):
    """Print basic structural information about a PDB file."""
    from physics_auditor.core.parser import parse_pdb

    struct = parse_pdb(path)

    table = Table(title=f"Structure: {struct.name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Atoms", str(struct.n_atoms))
    table.add_row("Residues", str(struct.n_residues))
    table.add_row("Chains", str(struct.n_chains))
    table.add_row("Protein chains", str(len(struct.protein_chains)))

    for chain in struct.chains.values():
        label = f"Chain {chain.chain_id}"
        seq = chain.sequence
        if len(seq) > 40:
            seq = seq[:37] + "..."
        table.add_row(label, f"{len(chain.residues)} res | {seq}")

    console.print(table)


def _print_rich_report(report: dict, verbose: bool, rec_color: str) -> None:
    """Print a formatted report to the terminal."""
    name = Path(report["file"]).stem
    meta = report["metadata"]
    checks = report["checks"]

    # Header
    score = report["global_score"]
    rec = report["recommendation"]
    console.print()
    console.print(Panel(
        f"[bold]{name}[/bold]  |  "
        f"Score: [bold]{score:.3f}[/bold]  |  "
        f"Recommendation: [{rec_color}][bold]{rec.upper()}[/bold][/{rec_color}]",
        title="Physics Auditor",
        border_style="blue",
    ))

    # Summary table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", style="cyan")
    table.add_column("Result", style="white")
    table.add_column("Score", style="white")

    # Clashes
    cl = checks["steric_clashes"]
    clash_str = f"{cl['n_clashes']} clashes ({cl['n_severe']} severe), clashscore={cl['clashscore']}"
    table.add_row("Steric Clashes", clash_str, f"{cl['subscore']:.3f}")

    # LJ
    lj = checks["lennard_jones"]
    lj_str = f"E={lj['total_energy_kcal']:.1f} kcal/mol, {lj['n_hot_pairs']} hot pairs"
    table.add_row("Lennard-Jones", lj_str, "—")

    # Backbone
    bb = checks["backbone_dihedrals"]
    bb_str = f"φ={bb['n_phi']}, ψ={bb['n_psi']}, ω={bb['n_omega']}"
    table.add_row("Backbone Dihedrals", bb_str, "—")

    console.print(table)

    # Metadata
    console.print(
        f"  [dim]{meta['n_atoms']} atoms | {meta['n_residues']} residues | "
        f"{meta['n_chains']} chains | {meta['n_bonds_inferred']} bonds | "
        f"{meta['runtime_seconds']:.3f}s[/dim]"
    )
    console.print()


if __name__ == "__main__":
    app()
