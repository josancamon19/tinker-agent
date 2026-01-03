import asyncio
import os
import shutil
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
