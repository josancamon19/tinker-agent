import asyncio
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from functools import partial
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

sys.path.insert(0, str(Path(__file__).parent.parent))

from tinker_agent.agent import run_agent, Config

console = Console()

# Trace file location
TRACES_FILE = "traces.json"


def update_runs_index(project_root: Path) -> None:
    """Update runs/index.json with all runs that have traces."""
    runs_dir = project_root / "runs"
    index_path = runs_dir / "index.json"

    if not runs_dir.exists():
        return

    index = {}
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir() and (run_dir / "traces.json").exists():
            # Count traces from JSON array
            trace_count = 0
            try:
                with open(run_dir / "traces.json") as f:
                    traces = json.load(f)
                    if isinstance(traces, list):
                        trace_count = len(traces)
            except Exception:
                pass

            index[run_dir.name] = {
                "path": f"runs/{run_dir.name}",
                "traceCount": trace_count,
            }

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)


def start_trace_server(project_root: Path, port: int = 8765) -> int:
    """Start a simple HTTP server at project root to serve the viewer."""
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(project_root))

    def serve():
        with http.server.HTTPServer(("", port), handler) as httpd:
            httpd.serve_forever()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return port


PYPROJECT_DEPENDENCIES = """dependencies = [
    "tinker",
    "tinker-cookbook",
    "wandb",
    "python-dotenv",
    "datasets",
    "numpy",
]
"""

PYPROJECT_UV_SOURCES = """
[tool.uv.sources]
tinker-cookbook = { git = "https://github.com/thinking-machines-lab/tinker-cookbook.git" }
"""


def setup_project(runs_dir: Path) -> None:
    """Initialize uv project and install dependencies."""
    console.print("[dim]Setting up project...[/dim]")

    # Force uv to use a local .venv and ignore parent workspace
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(runs_dir / ".venv")
    env["UV_NO_WORKSPACE"] = "1"  # Prevent discovering parent workspace

    # Run uv init (--no-workspace prevents modifying parent pyproject.toml)
    subprocess.run(
        ["uv", "init", "--no-workspace"],
        cwd=runs_dir,
        env=env,
        check=True,
        capture_output=True,
    )

    # Update pyproject.toml with dependencies
    pyproject_path = runs_dir / "pyproject.toml"
    content = pyproject_path.read_text()

    # Replace the default dependencies line
    content = content.replace("dependencies = []", PYPROJECT_DEPENDENCIES.strip())

    # Add uv sources section at the end
    content = content.rstrip() + "\n" + PYPROJECT_UV_SOURCES

    pyproject_path.write_text(content)

    # Run uv sync with isolated environment
    subprocess.run(
        ["uv", "sync"], cwd=runs_dir, env=env, check=True, capture_output=True
    )

    console.print("[green]✓[/green] Project setup complete")


def setup_run_directory(project_root: Path) -> Path:
    """Create a new run directory with uv project setup."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = project_root / "runs" / timestamp
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Copy .env
    env_file = project_root / ".env"
    if env_file.exists():
        shutil.copy(env_file, runs_dir / ".env")

    # Setup project
    setup_project(runs_dir)

    return runs_dir


def run_training_agent(
    dataset: str,
    task_type: str,
    model: str,
    project_root: Path | None = None,
    data_dir: str | None = None,
) -> None:
    """
    Run the agent with the given configuration.

    Args:
        dataset: HuggingFace dataset name (e.g., "HuggingFaceFW/fineweb") or local path
        task_type: Task type ("sft", "rl", or "cpt")
        model: Model name
        project_root: Project root directory (defaults to cwd)
        data_dir: Read-only data directory for local datasets (if dataset is a local path)
    """
    if project_root is None:
        project_root = Path.cwd()

    # Setup run directory
    console.print("[cyan]Setting up environment...[/cyan]")
    runs_dir = setup_run_directory(project_root)

    # Construct prompt
    if data_dir:
        # For local directories, tell the agent where to find the data
        prompt = f"Do {task_type.upper()} training on {model} using the local dataset at {data_dir}"
    else:
        prompt = f"Do {task_type.upper()} training on {model} using dataset {dataset}"

    # Trace file path
    trace_path = runs_dir / TRACES_FILE

    # Build dataset info for display
    dataset_display = dataset
    if data_dir:
        dataset_display += " [dim](local, read-only)[/dim]"

    console.print()
    console.print(
        Panel(
            f"[bold]Dataset:[/bold] {dataset_display}\n"
            f"[bold]Task Type:[/bold] {task_type}\n"
            f"[bold]Model:[/bold] {model}\n"
            f"[bold]Run Directory:[/bold] {runs_dir}",
            title="[bold cyan]Starting Agent[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()

    # Run agent
    config = Config(
        prompt=prompt,
        cwd=str(runs_dir),
        trace_path=str(trace_path),
        data_dir=data_dir,
    )

    try:
        asyncio.run(run_agent(config))
        console.print("\n[green]✓ Agent completed successfully[/green]")
    except Exception as e:
        console.print(f"\n[red]✗ Agent error: {e}[/red]")
        raise

    # Update runs index
    update_runs_index(project_root)
