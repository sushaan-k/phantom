"""Command-line interface for Phantom."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from phantom.logging import configure_logging

if TYPE_CHECKING:
    from phantom.redteam import RedTeamResults

app = typer.Typer(
    name="phantom",
    help="RL-based adversarial red-team agent for LLM systems.",
    no_args_is_help=True,
)
console = Console(stderr=True)
output_console = Console()


@app.command()
def scan(
    target: str = typer.Option(
        ...,
        "--target",
        "-t",
        help="Target LLM endpoint URL.",
    ),
    output: str = typer.Option(
        "json",
        "--output",
        "-o",
        help="Output format: json, html, sarif, or all.",
    ),
    output_path: str = typer.Option(
        "phantom-results",
        "--output-path",
        "-p",
        help="Output file path (without extension).",
    ),
    categories: str | None = typer.Option(
        None,
        "--categories",
        "-c",
        help="Comma-separated attack categories.",
    ),
    max_interactions: int = typer.Option(
        500,
        "--max-interactions",
        "-n",
        help="Maximum number of probes to send.",
    ),
    attack_model: str = typer.Option(
        "gpt-4",
        "--attack-model",
        "-m",
        help="Model to use for attack generation.",
    ),
    multi_turn: bool = typer.Option(
        True,
        "--multi-turn/--no-multi-turn",
        help="Enable or disable multi-turn attacks.",
    ),
    auth_header: str | None = typer.Option(
        None,
        "--auth",
        "-a",
        help='Auth header value (e.g., "Bearer sk-...").',
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging.",
    ),
    json_logs: bool = typer.Option(
        False,
        "--json-logs",
        help="Output logs in JSON format.",
    ),
) -> None:
    """Run a red-team security scan against a target LLM endpoint."""
    log_level = "DEBUG" if verbose else "INFO"
    configure_logging(level=log_level, json_output=json_logs)

    console.print(
        Panel.fit(
            "[bold blue]Phantom[/bold blue] "
            "[dim]RL-based adversarial red-team agent[/dim]",
            border_style="blue",
        )
    )

    parsed_categories = None
    if categories:
        parsed_categories = [c.strip() for c in categories.split(",")]

    auth: dict[str, str] = {}
    if auth_header:
        auth["Authorization"] = auth_header

    asyncio.run(
        _run_scan(
            target_url=target,
            output_format=output,
            output_path=output_path,
            categories=parsed_categories,
            max_interactions=max_interactions,
            attack_model=attack_model,
            multi_turn=multi_turn,
            auth=auth,
        )
    )


async def _run_scan(
    target_url: str,
    output_format: str,
    output_path: str,
    categories: list[str] | None,
    max_interactions: int,
    attack_model: str,
    multi_turn: bool,
    auth: dict[str, str],
) -> None:
    """Execute the scan asynchronously.

    Args:
        target_url: The target endpoint URL.
        output_format: Output format string.
        output_path: Base path for output files.
        categories: Attack categories to test.
        max_interactions: Maximum probe count.
        attack_model: Model for attack generation.
        multi_turn: Whether to use multi-turn strategies.
        auth: Authentication headers.
    """
    from phantom.atlas.report import ATLASReport
    from phantom.redteam import RedTeam
    from phantom.target import Target

    target = Target(endpoint=target_url, auth=auth)
    red_team = RedTeam(
        target=target,
        attack_model=attack_model,
        categories=categories,
        max_interactions=max_interactions,
        multi_turn=multi_turn,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running security assessment...", total=None)
        results = await red_team.run()
        progress.update(task, completed=True)

    _print_summary(results)

    report = ATLASReport(results)

    formats = ["json", "html", "sarif"] if output_format == "all" else [output_format]

    for fmt in formats:
        ext = fmt if fmt != "json" else "json"
        path = f"{output_path}.{ext}"
        if fmt == "json":
            report.to_json(path)
        elif fmt == "html":
            report.to_html(path)
        elif fmt == "sarif":
            report.to_sarif(path)
        console.print(f"  [green]Wrote {fmt.upper()} report:[/green] {path}")


def _print_summary(results: RedTeamResults) -> None:
    """Print a summary table of results to the console.

    Args:
        results: The RedTeamResults to summarize.
    """

    table = Table(title="Assessment Summary", border_style="blue")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total Probes", str(results.total_probes))
    table.add_row("Total Bypasses", str(results.total_bypasses))
    table.add_row("Bypass Rate", f"{results.bypass_rate:.1%}")
    table.add_row("Findings", str(len(results.findings)))
    table.add_row(
        "Critical",
        f"[red]{results.count_by_severity('CRITICAL')}[/red]",
    )
    table.add_row(
        "High",
        f"[yellow]{results.count_by_severity('HIGH')}[/yellow]",
    )
    table.add_row(
        "Medium",
        f"[bright_yellow]{results.count_by_severity('MEDIUM')}[/bright_yellow]",
    )
    table.add_row(
        "Low",
        f"[green]{results.count_by_severity('LOW')}[/green]",
    )
    table.add_row("Novel Attacks", str(results.novel_attack_count))

    console.print()
    console.print(table)
    console.print()


@app.command()
def report(
    input_path: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to a phantom results JSON file.",
    ),
    output_format: str = typer.Option(
        "html",
        "--output",
        "-o",
        help="Output format: html, sarif, or all.",
    ),
    output_path: str = typer.Option(
        "phantom-report",
        "--output-path",
        "-p",
        help="Output file path (without extension).",
    ),
) -> None:
    """Generate reports from a previous scan's JSON results."""
    from phantom.atlas.report import ATLASReport

    configure_logging()

    path = Path(input_path)
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {input_path}")
        raise typer.Exit(code=1)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError, Exception) as exc:
        console.print(f"[red]Error:[/red] Failed to parse input: {exc}")
        raise typer.Exit(code=1) from exc

    atlas_report = ATLASReport(data)

    formats = ["html", "sarif"] if output_format == "all" else [output_format]

    for fmt in formats:
        out = f"{output_path}.{fmt}"
        if fmt == "html":
            atlas_report.to_html(out)
        elif fmt == "sarif":
            atlas_report.to_sarif(out)
        elif fmt == "json":
            atlas_report.to_json(out)
        console.print(f"  [green]Wrote {fmt.upper()} report:[/green] {out}")


@app.command()
def version() -> None:
    """Print the Phantom version."""
    from phantom import __version__

    output_console.print(f"phantom {__version__}")


if __name__ == "__main__":
    app()
