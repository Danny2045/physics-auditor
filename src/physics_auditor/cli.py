"""Physics Auditor CLI.

Entry point for structure validation and mechanistic analysis. The CLI is a
thin renderer over the library-level ``audit()`` function: it builds a
``StructureAuditRequest``, invokes ``audit()``, and prints the typed
``StructureAuditResult`` as either JSON or a Rich-formatted table.

Usage:
    physics-auditor validate structure.pdb
    physics-auditor validate *.pdb --output-dir reports/
    physics-auditor info structure.pdb
"""

from __future__ import annotations

import json
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
    ligand_resname: str | None = typer.Option(
        None,
        "--ligand-resname",
        help=(
            "3-letter HETATM resname identifying the bound ligand. When set, "
            "the pocket-completeness check runs in addition to the always-on "
            "structure-level gap check."
        ),
    ),
    pocket_cutoff: float = typer.Option(
        5.0,
        "--pocket-cutoff",
        help="Pocket-defining distance in Å (used when --ligand-resname is set).",
    ),
):
    """Validate one or more protein structures."""
    from physics_auditor.audit import StructureAuditRequest, audit
    from physics_auditor.config import load_config

    cfg = load_config(str(config) if config else None)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        if not path.exists():
            console.print(f"[red]File not found: {path}[/red]")
            continue

        # Only catch expected user/data errors here. Anything else is a real
        # bug and must propagate — silently masking an unexpected exception as
        # "Failed to audit ..." and continuing risks a green exit on a batch
        # run where every file actually failed.
        try:
            request = StructureAuditRequest(
                pdb_path=path,
                config=cfg,
                ligand_resname=ligand_resname,
                pocket_cutoff=pocket_cutoff,
            )
            result = audit(request)
        except (FileNotFoundError, ValueError) as e:
            console.print(f"[red]Failed to audit {path}: {e}[/red]")
            continue

        report = result.to_dict()

        if json_output:
            print(json.dumps(report, indent=2))
        else:
            _print_rich_report(report, verbose)

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


def _print_rich_report(report: dict, verbose: bool) -> None:
    """Print a formatted audit report to the terminal."""
    name = Path(report["file"]).stem
    meta = report["metadata"]
    checks = report["checks"]
    comp = report.get("composite") or {}
    readiness = report.get("readiness") or {}

    score = report.get("global_score")
    rec = report.get("recommendation")
    ready = readiness.get("ready", True)

    if ready and score is not None and rec is not None:
        rec_color = (
            "green" if rec == "accept"
            else "yellow" if rec == "short_md"
            else "red"
        )
        header = (
            f"[bold]{name}[/bold]  |  "
            f"Score: [bold]{score:.3f}[/bold]  |  "
            f"Recommendation: [{rec_color}][bold]{rec.upper()}[/bold][/{rec_color}]"
        )
    else:
        header = (
            f"[bold]{name}[/bold]  |  "
            f"Score: [bold]n/a (unready)[/bold]  |  "
            f"Recommendation: [red][bold]WITHHELD[/bold][/red]"
        )

    console.print()
    console.print(Panel(header, title="Physics Auditor", border_style="blue"))

    # Readiness banner — always show; UNREADY is loud, READY is one dim line.
    if not ready:
        console.print(
            f"  [red]✗ UNREADY[/red] [dim]({readiness.get('reason', '')})[/dim]"
        )
        n_gaps = len(readiness.get("structure_internal_gaps", []))
        n_term = len(readiness.get("structure_unresolved_termini", []))
        if n_gaps or n_term:
            console.print(
                f"  [dim]structure: {n_gaps} internal gap(s), "
                f"{n_term} terminal residue(s)[/dim]"
            )
        n_pocket = len(readiness.get("pocket_missing", []))
        if n_pocket:
            console.print(
                f"  [dim]pocket: {n_pocket} pocket-region residue(s) missing[/dim]"
            )
    else:
        struct_lvl = readiness.get("structure_level", "?")
        pocket_lvl = readiness.get("pocket_level", "?")
        console.print(
            f"  [dim]✓ ready (structure: {struct_lvl}, "
            f"pocket: {pocket_lvl})[/dim]"
        )

    # Partial-composite honesty: never let the score read as a full audit.
    if ready and comp.get("is_partial", False):
        scored = ", ".join(c["check"] for c in comp.get("contributing_checks", []))
        console.print(
            f"  [yellow]⚠ PARTIAL composite[/yellow] [dim]over "
            f"{comp.get('n_scored_checks')} of {comp.get('n_total_checks')} "
            f"checks ({scored}); "
            f"{len(comp.get('computed_but_unscored', []))} computed-but-unscored, "
            f"{len(comp.get('not_implemented', []))} not-implemented[/dim]"
        )

    # Summary table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", style="cyan")
    table.add_column("Result", style="white")
    table.add_column("Score", style="white")

    cl = checks["steric_clashes"]
    clash_str = (
        f"{cl['n_clashes']} clashes ({cl['n_severe']} severe), "
        f"clashscore={cl['clashscore']}"
    )
    table.add_row("Steric Clashes", clash_str, f"{cl['subscore']:.3f}")

    lj = checks["lennard_jones"]
    lj_str = f"E={lj['total_energy_kcal']:.1f} kcal/mol, {lj['n_hot_pairs']} hot pairs"
    table.add_row("Lennard-Jones", lj_str, "—")

    bb = checks["backbone_dihedrals"]
    bb_str = f"φ={bb['n_phi']}, ψ={bb['n_psi']}, ω={bb['n_omega']}"
    table.add_row("Backbone Dihedrals", bb_str, "—")

    console.print(table)

    console.print(
        f"  [dim]{meta['n_atoms']} atoms | {meta['n_residues']} residues | "
        f"{meta['n_chains']} chains | {meta['n_bonds_inferred']} bonds | "
        f"{meta['runtime_seconds']:.3f}s[/dim]"
    )
    console.print()


if __name__ == "__main__":
    app()
