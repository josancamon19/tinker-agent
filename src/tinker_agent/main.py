import asyncio
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

sys.path.insert(0, str(Path(__file__).parent.parent))

from tinker_agent.agent import run_agent, Config

console = Console()

PYPROJECT_DEPENDENCIES = """dependencies = [
    "tinker",
    "tinker-cookbook",
    "wandb",
    "python-dotenv",
]
"""

PYPROJECT_UV_SOURCES = """
[tool.uv.sources]
tinker-cookbook = { git = "https://github.com/thinking-machines-lab/tinker-cookbook.git" }
"""


def setup_project(runs_dir: Path) -> None:
    """Initialize uv project and install dependencies."""
    console.print("[dim]Setting up project...[/dim]")

    # Force uv to use a local .venv in the runs directory
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(runs_dir / ".venv")

    # Run uv init
    subprocess.run(
        ["uv", "init"], cwd=runs_dir, env=env, check=True, capture_output=True
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


def main() -> None:
    """CLI entrypoint with interactive prompt."""
    # Create timestamped runs directory (resolve before chdir)
    project_root = Path.cwd().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = (project_root / "runs" / timestamp).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Copy .env to runs directory if it exists
    env_file = project_root / ".env"
    if env_file.exists():
        shutil.copy(env_file, runs_dir / ".env")

    # Setup uv project with dependencies
    setup_project(runs_dir)

    # Change to the runs directory
    os.chdir(runs_dir)

    # Header
    console.print()
    console.print(
        Panel(
            Text("tinker-agent", style="bold cyan", justify="center"),
            subtitle=f"[dim]{runs_dir}[/dim]",
            border_style="cyan",
        )
    )

    # Interactive CLI loop
    while True:
        try:
            console.print()
            prompt = Prompt.ask("[bold cyan]>[/bold cyan]")
            if not prompt.strip():
                continue
            if prompt.strip().lower() in ("exit", "quit", "q"):
                break

            config = Config(prompt=prompt, cwd=str(runs_dir))
            asyncio.run(run_agent(config))

        except KeyboardInterrupt:
            console.print("\n[dim]Exiting...[/dim]")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()
