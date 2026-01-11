#!/usr/bin/env python3
"""Run tinker-agent CLI for each row in tinkerbench.jsonl with timeout."""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import chz
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@chz.chz
class BenchmarkConfig:
    """Benchmark runner configuration."""

    timeout: int = 30  # Timeout in minutes for each benchmark
    start_from: int = 0  # Start from benchmark index (0-based)
    limit: int | None = None  # Limit number of benchmarks to run
    benchmark_file: str = str(
        Path(__file__).parent / "tinkerbench.jsonl"
    )  # Path to benchmark file


def run_single_benchmark(
    dataset: str,
    task: str,
    model: str,
    timeout_minutes: int,
    benchmark_idx: int,
    total_benchmarks: int,
) -> dict:
    """
    Run a single benchmark with timeout.

    Returns:
        dict with keys: success (bool), duration (float), error (str | None)
    """
    console.print()
    console.print(
        Panel(
            f"[bold]Dataset:[/bold] {dataset}\n"
            f"[bold]Task:[/bold] {task}\n"
            f"[bold]Model:[/bold] {model}\n"
            f"[bold]Timeout:[/bold] {timeout_minutes} minutes",
            title=f"[bold cyan]Benchmark {benchmark_idx + 1}/{total_benchmarks}[/bold cyan]",
            border_style="cyan",
        )
    )

    # Build command
    cmd = [
        "tinker-agent",
        f"dataset={dataset}",
        f"task_type={task.lower()}",
        f"model={model}",
    ]

    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    try:
        # Run with timeout
        result = subprocess.run(
            cmd,
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            console.print(f"[green]✓ Completed in {duration / 60:.1f} minutes[/green]")
            return {"success": True, "duration": duration, "error": None}
        else:
            error_msg = result.stderr or result.stdout or "Unknown error"
            console.print(f"[red]✗ Failed: {error_msg[:200]}[/red]")
            return {"success": False, "duration": duration, "error": error_msg}

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        console.print(f"[yellow]⏱ Timeout after {timeout_minutes} minutes[/yellow]")
        return {
            "success": False,
            "duration": duration,
            "error": f"Timeout after {timeout_minutes} minutes",
        }
    except Exception as e:
        duration = time.time() - start_time
        console.print(f"[red]✗ Error: {e}[/red]")
        return {"success": False, "duration": duration, "error": str(e)}


def run_benchmarks(config: BenchmarkConfig):
    """Run all benchmarks with the given configuration."""
    # Read benchmarks
    benchmark_path = Path(config.benchmark_file)
    if not benchmark_path.exists():
        console.print(f"[red]Error: Benchmark file not found: {benchmark_path}[/red]")
        sys.exit(1)

    benchmarks = []
    with open(benchmark_path) as f:
        for line in f:
            line = line.strip()
            if line:
                benchmarks.append(json.loads(line))

    # Apply filters
    total_benchmarks = len(benchmarks)
    benchmarks = benchmarks[config.start_from :]
    if config.limit:
        benchmarks = benchmarks[: config.limit]

    if not benchmarks:
        console.print("[yellow]No benchmarks to run[/yellow]")
        sys.exit(0)

    # Header
    console.print()
    console.print(
        Panel(
            f"[bold]Total benchmarks:[/bold] {len(benchmarks)}\n"
            f"[bold]Timeout per benchmark:[/bold] {config.timeout} minutes\n"
            f"[bold]Estimated total time:[/bold] {len(benchmarks) * config.timeout} minutes",
            title="[bold cyan]Tinker-Agent Benchmark Runner[/bold cyan]",
            border_style="cyan",
        )
    )

    # Run benchmarks
    results = []
    start_time = datetime.now()

    for idx, benchmark in enumerate(benchmarks):
        actual_idx = config.start_from + idx
        result = run_single_benchmark(
            dataset=benchmark["dataset"],
            task=benchmark["task"],
            model=benchmark["model"],
            timeout_minutes=config.timeout,
            benchmark_idx=actual_idx,
            total_benchmarks=total_benchmarks,
        )

        results.append(
            {
                "benchmark": benchmark,
                "index": actual_idx,
                **result,
            }
        )

    # Summary
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    console.print()
    console.print("=" * 80)
    console.print()

    # Results table
    table = Table(title="Benchmark Results", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Dataset", style="cyan")
    table.add_column("Task", style="magenta")
    table.add_column("Status", justify="center")
    table.add_column("Duration", justify="right")

    for result in results:
        idx = result["index"]
        benchmark = result["benchmark"]
        status = "[green]✓[/green]" if result["success"] else "[red]✗[/red]"
        duration = f"{result['duration'] / 60:.1f}m"

        table.add_row(
            str(idx + 1),
            benchmark["dataset"],
            benchmark["task"],
            status,
            duration,
        )

    console.print(table)

    # Statistics
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count

    console.print()
    console.print(
        Panel(
            f"[bold]Total:[/bold] {len(results)}\n"
            f"[bold green]Successful:[/bold green] {success_count}\n"
            f"[bold red]Failed:[/bold red] {fail_count}\n"
            f"[bold]Total time:[/bold] {total_duration / 60:.1f} minutes",
            title="[bold cyan]Summary[/bold cyan]",
            border_style="cyan",
        )
    )

    # Save results
    results_file = Path("benchmark_results.jsonl")
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    console.print(f"\n[dim]Results saved to: {results_file}[/dim]")

    # Exit with error if any failed
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    chz.entrypoint(run_benchmarks)
