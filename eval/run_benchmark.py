#!/usr/bin/env python3
"""Run tinker-agent CLI for each row in tinkerbench.jsonl with timeout."""

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import chz
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tinker_agent.eval import validate_results

console = Console()

# Stagger delay between parallel benchmark starts (seconds)
PARALLEL_STAGGER_DELAY = 5


@chz.chz
class BenchmarkConfig:
    """Benchmark runner configuration."""

    timeout: int = 30  # Timeout in minutes for each benchmark
    start_from: int = 0  # Start from benchmark index (0-based)
    limit: int | None = None  # Limit number of benchmarks to run
    parallel: bool = False  # Run benchmarks in parallel (with 5s stagger)
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
    runs_dir: Path,
) -> dict:
    """
    Run a single benchmark with timeout.

    Returns:
        dict with keys: success (bool), duration (float), error (str | None),
                       validation (ValidationResult | None), run_dir (str | None)
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

    # Build command (chz requires config. prefix for nested config)
    cmd = [
        "tinker-agent",
        f"config.dataset={dataset}",
        f"config.task_type={task.lower()}",
        f"config.model={model}",
    ]

    # Record existing run dirs before starting
    existing_runs = (
        set(d.name for d in runs_dir.iterdir() if d.is_dir())
        if runs_dir.exists()
        else set()
    )

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

        # Find the new run directory
        run_dir = None
        if runs_dir.exists():
            new_runs = [
                d
                for d in runs_dir.iterdir()
                if d.is_dir() and d.name not in existing_runs
            ]
            if new_runs:
                run_dir = max(new_runs, key=lambda d: d.name)

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            console.print(f"[red]✗ CLI failed: {error_msg[:200]}[/red]")
            return {
                "success": False,
                "duration": duration,
                "error": error_msg,
                "validation": None,
                "run_dir": str(run_dir) if run_dir else None,
            }

        # CLI succeeded - now validate results
        if run_dir is None:
            console.print("[red]✗ No run directory created[/red]")
            return {
                "success": False,
                "duration": duration,
                "error": "No run directory created",
                "validation": None,
                "run_dir": None,
            }

        results_dir = run_dir / "results"
        validation = validate_results(results_dir)

        if validation.valid:
            # Show improvement metrics
            if validation.task_type == "rl":
                improvement = (
                    (validation.trained_score - validation.base_score)
                    / validation.base_score
                    * 100
                )
                console.print(
                    f"[green]✓ Completed in {duration / 60:.1f}m - "
                    f"Accuracy: {validation.base_score:.1%} → {validation.trained_score:.1%} (+{improvement:.1f}%)[/green]"
                )
            elif validation.task_type == "sft":
                reduction = (
                    (validation.base_nll - validation.trained_nll)
                    / validation.base_nll
                    * 100
                )
                console.print(
                    f"[green]✓ Completed in {duration / 60:.1f}m - "
                    f"NLL: {validation.base_nll:.4f} → {validation.trained_nll:.4f} (-{reduction:.1f}%)[/green]"
                )
            else:
                console.print(
                    f"[green]✓ Completed in {duration / 60:.1f}m - Validation passed[/green]"
                )

            return {
                "success": True,
                "duration": duration,
                "error": None,
                "validation": validation,
                "run_dir": str(run_dir),
            }
        else:
            error_summary = "; ".join(validation.errors[:3])
            if len(validation.errors) > 3:
                error_summary += f" (+{len(validation.errors) - 3} more)"
            console.print(f"[red]✗ Validation failed: {error_summary}[/red]")
            return {
                "success": False,
                "duration": duration,
                "error": f"Validation failed: {error_summary}",
                "validation": validation,
                "run_dir": str(run_dir),
            }

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        console.print(f"[yellow]⏱ Timeout after {timeout_minutes} minutes[/yellow]")
        return {
            "success": False,
            "duration": duration,
            "error": f"Timeout after {timeout_minutes} minutes",
            "validation": None,
            "run_dir": None,
        }
    except Exception as e:
        duration = time.time() - start_time
        console.print(f"[red]✗ Error: {e}[/red]")
        return {
            "success": False,
            "duration": duration,
            "error": str(e),
            "validation": None,
            "run_dir": None,
        }


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

    # Runs directory (where tinker-agent creates run directories)
    runs_dir = Path.cwd() / "runs"

    # Header
    mode_str = "parallel" if config.parallel else "sequential"
    console.print()
    console.print(
        Panel(
            f"[bold]Total benchmarks:[/bold] {len(benchmarks)}\n"
            f"[bold]Mode:[/bold] {mode_str}\n"
            f"[bold]Timeout per benchmark:[/bold] {config.timeout} minutes",
            title="[bold cyan]Tinker-Agent Benchmark Runner[/bold cyan]",
            border_style="cyan",
        )
    )

    # Run benchmarks
    results = []
    start_time = datetime.now()

    def process_result(result: dict, benchmark: dict, actual_idx: int) -> dict:
        """Convert ValidationResult to dict and add benchmark info."""
        if result.get("validation"):
            v = result["validation"]
            result["validation"] = {
                "valid": v.valid,
                "errors": v.errors,
                "task_type": v.task_type,
                "base_score": v.base_score,
                "trained_score": v.trained_score,
                "base_nll": v.base_nll,
                "trained_nll": v.trained_nll,
            }
        return {
            "benchmark": benchmark,
            "index": actual_idx,
            **result,
        }

    if config.parallel:
        # Parallel execution with staggered starts
        console.print(
            f"[dim]Starting benchmarks with {PARALLEL_STAGGER_DELAY}s stagger...[/dim]"
        )
        futures = {}

        with ThreadPoolExecutor(max_workers=len(benchmarks)) as executor:
            for idx, benchmark in enumerate(benchmarks):
                actual_idx = config.start_from + idx

                # Stagger start times
                if idx > 0:
                    time.sleep(PARALLEL_STAGGER_DELAY)

                future = executor.submit(
                    run_single_benchmark,
                    dataset=benchmark["dataset"],
                    task=benchmark["task"],
                    model=benchmark["model"],
                    timeout_minutes=config.timeout,
                    benchmark_idx=actual_idx,
                    total_benchmarks=total_benchmarks,
                    runs_dir=runs_dir,
                )
                futures[future] = (benchmark, actual_idx)

            # Collect results as they complete
            for future in as_completed(futures):
                benchmark, actual_idx = futures[future]
                result = future.result()
                results.append(process_result(result, benchmark, actual_idx))

        # Sort by index for consistent display
        results.sort(key=lambda r: r["index"])
    else:
        # Sequential execution
        for idx, benchmark in enumerate(benchmarks):
            actual_idx = config.start_from + idx
            result = run_single_benchmark(
                dataset=benchmark["dataset"],
                task=benchmark["task"],
                model=benchmark["model"],
                timeout_minutes=config.timeout,
                benchmark_idx=actual_idx,
                total_benchmarks=total_benchmarks,
                runs_dir=runs_dir,
            )
            results.append(process_result(result, benchmark, actual_idx))

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
    table.add_column("Improvement", justify="right")
    table.add_column("Duration", justify="right")

    for result in results:
        idx = result["index"]
        benchmark = result["benchmark"]
        status = "[green]✓[/green]" if result["success"] else "[red]✗[/red]"
        duration = f"{result['duration'] / 60:.1f}m"

        # Calculate improvement from validation results
        improvement = "-"
        v = result.get("validation")
        if v:
            if (
                v.get("task_type") == "rl"
                and v.get("base_score")
                and v.get("trained_score")
            ):
                pct = (v["trained_score"] - v["base_score"]) / v["base_score"] * 100
                improvement = f"+{pct:.1f}%" if pct > 0 else f"{pct:.1f}%"
            elif (
                v.get("task_type") == "sft"
                and v.get("base_nll")
                and v.get("trained_nll")
            ):
                pct = (v["base_nll"] - v["trained_nll"]) / v["base_nll"] * 100
                improvement = f"-{pct:.1f}% NLL" if pct > 0 else f"+{abs(pct):.1f}% NLL"

        table.add_row(
            str(idx + 1),
            benchmark["dataset"],
            benchmark["task"],
            status,
            improvement,
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

    # Save results to eval/results/{timestamp}.jsonl
    eval_results_dir = Path(__file__).parent / "results"
    eval_results_dir.mkdir(exist_ok=True)

    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    results_file = eval_results_dir / f"{timestamp}.jsonl"

    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    console.print(f"\n[dim]Results saved to: {results_file}[/dim]")

    # Exit with error if any failed
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    chz.entrypoint(run_benchmarks)
