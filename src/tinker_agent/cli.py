"""Interactive and non-interactive CLI for tinker-agent."""

import os
import subprocess
import time
from enum import Enum
from pathlib import Path

import chz
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text
from rich.align import Align

console = Console()


def show_welcome_animation():
    """Display welcome screen."""
    # Welcome message
    message = Text()
    message.append("Welcome to ", style="bold cyan")
    message.append("tinker-agent", style="bold yellow")
    message.append("\nPost training agent for @thinkymachines Tinker platform", style="dim")

    # Warning message
    warning = Text()
    warning.append("\n\n⚠️  Preview Release\n", style="bold yellow")
    warning.append("• Claude is sandboxed to prevent access outside its directory\n", style="dim")
    warning.append("• Some bypass permissions exist for system operations\n", style="dim")
    warning.append("• Use with caution in production environments", style="dim")

    # Combine everything
    content = Text()
    content.append(message)
    content.append(warning)

    panel = Panel(
        Align.center(content),
        border_style="cyan",
        padding=(1, 2),
    )

    console.print()
    console.print(panel)
    console.print()


def is_viewer_running() -> bool:
    """Check if streamlit viewer is running on default port."""
    try:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("127.0.0.1", 8501))
        sock.close()
        return result == 0
    except Exception:
        return False


def launch_viewer() -> str:
    """Launch streamlit viewer if not already running. Returns URL."""
    viewer_url = "http://localhost:8501"

    if is_viewer_running():
        console.print(f"[dim]Viewer already running at {viewer_url}[/dim]")
        return viewer_url

    console.print("[cyan]Launching trace viewer...[/cyan]")

    # Launch streamlit in background
    try:
        subprocess.Popen(
            [
                "streamlit",
                "run",
                str(Path(__file__).parent / "viewer.py"),
                "--server.port=8501",
                "--server.headless=true",
                "--browser.gatherUsageStats=false",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for viewer to start
        for _ in range(10):
            time.sleep(0.5)
            if is_viewer_running():
                console.print(f"[green]✓ Viewer launched at {viewer_url}[/green]")
                return viewer_url

        console.print(f"[yellow]Viewer may be starting... Check {viewer_url}[/yellow]")
    except FileNotFoundError:
        console.print(
            "[yellow]streamlit not found. Install with: pip install streamlit[/yellow]"
        )
    except Exception as e:
        console.print(f"[yellow]Could not launch viewer: {e}[/yellow]")

    return viewer_url


# Environment variables configuration
ENV_VARS = {
    "TINKER_API_KEY": {
        "description": "Your Tinker API key for authentication with the fine-tuning service",
        "help": "Get your API key from https://tinker.anthropic.com/settings/api-keys",
        "secret": True,
    },
    "WANDB_API_KEY": {
        "description": "Weights & Biases API key for experiment tracking and logging",
        "help": "Get your API key from https://wandb.ai/authorize",
        "secret": True,
    },
    "WANDB_PROJECT": {
        "description": "W&B project name to organize your fine-tuning experiments",
        "help": "Example: 'my-finetuning-experiments' or 'company/project-name'",
        "secret": False,
        "default": "tinker-finetuning",
    },
}


class TaskType(str, Enum):
    """Post-training task types."""

    SFT = "sft"
    RL = "rl"
    CPT = "cpt"

    def __str__(self) -> str:
        return self.value


# Available models for fine-tuning with metadata
AVAILABLE_MODELS = [
    {"name": "Qwen/Qwen3-VL-235B-A22B-Instruct", "type": "Vision", "size": "Large"},
    {"name": "Qwen/Qwen3-VL-30B-A3B-Instruct", "type": "Vision", "size": "Medium"},
    {
        "name": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "type": "Instruction",
        "size": "Large",
    },
    {
        "name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "type": "Instruction",
        "size": "Medium",
    },
    {"name": "Qwen/Qwen3-30B-A3B", "type": "Hybrid", "size": "Medium"},
    {"name": "Qwen/Qwen3-30B-A3B-Base", "type": "Base", "size": "Medium"},
    {"name": "Qwen/Qwen3-32B", "type": "Hybrid", "size": "Medium"},
    {"name": "Qwen/Qwen3-8B", "type": "Hybrid", "size": "Small"},
    {"name": "Qwen/Qwen3-8B-Base", "type": "Base", "size": "Small"},
    {"name": "Qwen/Qwen3-4B-Instruct-2507", "type": "Instruction", "size": "Compact"},
    {"name": "openai/gpt-oss-120b", "type": "Reasoning", "size": "Medium"},
    {"name": "openai/gpt-oss-20b", "type": "Reasoning", "size": "Small"},
    {"name": "deepseek-ai/DeepSeek-V3.1", "type": "Hybrid", "size": "Large"},
    {"name": "deepseek-ai/DeepSeek-V3.1-Base", "type": "Base", "size": "Large"},
    {"name": "meta-llama/Llama-3.1-70B", "type": "Base", "size": "Large"},
    {
        "name": "meta-llama/Llama-3.3-70B-Instruct",
        "type": "Instruction",
        "size": "Large",
    },
    {"name": "meta-llama/Llama-3.1-8B", "type": "Base", "size": "Small"},
    {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "type": "Instruction",
        "size": "Small",
    },
    {"name": "meta-llama/Llama-3.2-3B", "type": "Base", "size": "Compact"},
    {"name": "meta-llama/Llama-3.2-1B", "type": "Base", "size": "Compact"},
    {"name": "moonshotai/Kimi-K2-Thinking", "type": "Reasoning", "size": "Large"},
]

# Helper to get model names
MODEL_NAMES = [m["name"] for m in AVAILABLE_MODELS]


def validate_dataset(dataset: str) -> tuple[bool, str, bool]:
    """
    Validate that a dataset exists (either HuggingFace or local directory).

    Returns (is_valid, message, is_local_directory).
    """
    # Check if it's a local directory path
    dataset_path = Path(dataset).expanduser()
    if dataset_path.exists() and dataset_path.is_dir():
        # Verify it has some files (not empty)
        files = list(dataset_path.iterdir())
        if not files:
            return False, f"Local directory '{dataset}' is empty", True
        
        # Check for common data file formats
        data_files = [f for f in files if f.suffix in {'.json', '.jsonl', '.parquet', '.csv', '.txt', '.arrow'}]
        if data_files:
            return True, f"Local directory '{dataset}' found with {len(data_files)} data file(s)", True
        
        # Check subdirectories for data files
        all_files = list(dataset_path.rglob('*'))
        data_files = [f for f in all_files if f.is_file() and f.suffix in {'.json', '.jsonl', '.parquet', '.csv', '.txt', '.arrow'}]
        if data_files:
            return True, f"Local directory '{dataset}' found with {len(data_files)} data file(s)", True
        
        return True, f"Local directory '{dataset}' found (no standard data files detected, but accessible)", True
    
    # Check format for HuggingFace: must be "organization/dataset-name" or "organization/dataset-name:config"
    if "/" not in dataset:
        return False, (
            f"Invalid dataset: '{dataset}'\n"
            "Dataset must be either:\n"
            "  - A HuggingFace dataset: 'organization/dataset-name'\n"
            "  - A local directory path: '/path/to/data' or './data'\n"
            "Examples:\n"
            "  - 'ServiceNow-AI/R1-Distill-SFT' (HuggingFace)\n"
            "  - '/home/user/my-sft-data' (local directory)"
        ), False

    try:
        from huggingface_hub import HfApi

        api = HfApi()

        # Handle dataset with config (e.g., "dataset/name:config")
        if ":" in dataset:
            dataset_name, _ = dataset.split(":", 1)
        else:
            dataset_name = dataset

        # Check if dataset exists
        try:
            api.dataset_info(dataset_name)
            return True, f"Dataset '{dataset}' found on HuggingFace", False
        except Exception:
            return False, f"Dataset '{dataset}' not found on HuggingFace", False

    except ImportError:
        # If huggingface_hub not installed, skip validation
        return True, "HuggingFace validation skipped (huggingface_hub not installed)", False


def validate_hf_dataset(dataset: str) -> tuple[bool, str]:
    """
    Validate that a HuggingFace dataset exists.
    
    Backwards-compatible wrapper around validate_dataset.
    Returns (is_valid, message).
    """
    is_valid, message, _ = validate_dataset(dataset)
    return is_valid, message


def interactive_select_task_type() -> TaskType:
    """Interactively select task type."""
    console.print()
    console.print("[bold]Select task type:[/bold]")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="cyan bold")
    table.add_column("Type", style="white")
    table.add_column("Description", style="dim")

    table.add_row("1", "sft", "Supervised Fine-Tuning")
    table.add_row("2", "rl", "Reinforcement Learning")
    table.add_row("3", "cpt", "Continued Pre-Training")

    console.print(table)

    while True:
        choice = Prompt.ask(
            "[cyan]Choice[/cyan]", choices=["1", "2", "3", "sft", "rl", "cpt"]
        )
        if choice in ("1", "sft"):
            return TaskType.SFT
        elif choice in ("2", "rl"):
            return TaskType.RL
        elif choice in ("3", "cpt"):
            return TaskType.CPT


def interactive_select_model() -> str:
    """Interactively select model."""
    console.print()
    console.print("[bold]Select model:[/bold]")

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Key", style="cyan bold", width=4)
    table.add_column("Model", style="white")
    table.add_column("Type", style="yellow", width=12)
    table.add_column("Size", style="green", width=8)

    for i, model in enumerate(AVAILABLE_MODELS, 1):
        table.add_row(str(i), model["name"], model["type"], model["size"])

    console.print(table)

    while True:
        choice = Prompt.ask(
            "[cyan]Choice (enter number 1-21)[/cyan]",
            default="1",
        )

        # Try to parse as number
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(AVAILABLE_MODELS):
                return AVAILABLE_MODELS[idx - 1]["name"]
            console.print(
                f"[red]Invalid choice. Enter a number between 1 and {len(AVAILABLE_MODELS)}[/red]"
            )
        # Check if it's a valid model name
        elif choice in MODEL_NAMES:
            return choice
        else:
            console.print(
                f"[red]Invalid choice. Enter a number between 1 and {len(AVAILABLE_MODELS)} or a valid model name[/red]"
            )


def interactive_get_dataset() -> tuple[str, bool]:
    """Interactively get and validate dataset.
    
    Returns (dataset, is_local_directory).
    """
    console.print()
    console.print("[bold]Dataset:[/bold]")
    console.print("[dim]  • HuggingFace: org/dataset-name (e.g. ServiceNow-AI/R1-Distill-SFT)[/dim]")
    console.print("[dim]  • Local path:  ~/my-obsidian-vault or /path/to/data[/dim]")
    while True:
        dataset = Prompt.ask("[cyan]Enter dataset[/cyan]")
        if not dataset.strip():
            console.print("[red]Dataset cannot be empty[/red]")
            continue

        with console.status("[dim]Validating dataset...[/dim]"):
            valid, message, is_local = validate_dataset(dataset)

        if valid:
            console.print(f"[green]{message}[/green]")
            if is_local:
                console.print("[dim]Note: Local directory will be mounted as read-only[/dim]")
            return dataset, is_local
        else:
            console.print(f"[red]{message}[/red]")
            if not Confirm.ask("Try another dataset?", default=True):
                raise ValueError(f"Invalid dataset: {dataset}")


def load_env_file(path: Path) -> dict[str, str]:
    """Load environment variables from a .env file."""
    env = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def save_env_file(path: Path, env: dict[str, str]) -> None:
    """Save environment variables to a .env file."""
    with open(path, "w") as f:
        for key, value in env.items():
            f.write(f"{key}={value}\n")


def check_env_configured() -> tuple[bool, list[str]]:
    """Check if required environment variables are configured."""
    # First, try loading from .env file if it exists
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        env_vars = load_env_file(env_path)
        # Load into os.environ if not already set
        for key, value in env_vars.items():
            if key not in os.environ:
                os.environ[key] = value

    missing = []
    for var in ENV_VARS:
        if not os.environ.get(var):
            missing.append(var)
    return len(missing) == 0, missing


def run_setup(env_path: Path | None = None) -> dict[str, str]:
    """Run interactive setup to configure environment variables."""
    console.print()
    console.print(
        Panel(
            Text("tinker-agent setup", style="bold cyan", justify="center"),
            subtitle="[dim]Configure your environment[/dim]",
            border_style="cyan",
        )
    )

    # Determine .env path
    if env_path is None:
        env_path = Path.cwd() / ".env"

    # Load existing values
    existing = load_env_file(env_path)

    console.print()
    console.print(f"[dim]Configuration will be saved to: {env_path}[/dim]")
    console.print()

    env = {}
    for var, config in ENV_VARS.items():
        console.print(f"[bold]{var}[/bold]")
        console.print(f"[dim]{config['description']}[/dim]")
        console.print(f"[dim italic]{config['help']}[/dim italic]")

        # Show current value if exists
        current = existing.get(var) or os.environ.get(var)
        if current:
            if config.get("secret"):
                display = (
                    current[:4] + "..." + current[-4:] if len(current) > 8 else "****"
                )
            else:
                display = current
            console.print(f"[green]Current: {display}[/green]")

        # Get value
        default = config.get("default", "")
        if current:
            default = current

        if config.get("secret"):
            value = Prompt.ask(
                f"[cyan]{var}[/cyan]",
                default=default if default else None,
                password=True,
            )
        else:
            value = Prompt.ask(
                f"[cyan]{var}[/cyan]",
                default=default if default else None,
            )

        if value:
            env[var] = value
        console.print()

    # Save to .env file
    save_env_file(env_path, env)
    console.print(f"[green]Configuration saved to {env_path}[/green]")

    return env


def ensure_env_configured() -> bool:
    """Ensure environment is configured, running setup if needed."""
    configured, missing = check_env_configured()
    if not configured:
        console.print()
        console.print("[yellow]Missing required environment variables:[/yellow]")
        for var in missing:
            console.print(f"  [dim]- {var}[/dim]")
        console.print()

        if Confirm.ask("Run setup now?", default=True):
            run_setup()
            # Re-check after setup (check_env_configured will reload .env)
            configured, missing = check_env_configured()
            if not configured:
                console.print("[red]Setup incomplete. Missing:[/red]")
                for var in missing:
                    console.print(f"  [dim]- {var}[/dim]")
                return False
            return True
        else:
            console.print("[red]Cannot continue without configuration[/red]")
            return False
    return True


@chz.chz
class TinkerConfig:
    """Configuration for tinker-agent fine-tuning."""

    dataset: str
    task_type: str  # Will be validated and converted to TaskType
    model: str = MODEL_NAMES[0]  # Default to first model
    interactive: bool = True
    data_dir: str | None = None  # Local directory path (read-only, for local dataset)


def run_interactive() -> TinkerConfig:
    """Run interactive mode to gather configuration."""
    console.print()
    console.print(
        Panel(
            Text("tinker-agent", style="bold cyan", justify="center"),
            subtitle="[dim]Fine-tuning configuration[/dim]",
            border_style="cyan",
        )
    )

    # Gather config interactively
    task_type = interactive_select_task_type()
    model = interactive_select_model()
    dataset, is_local = interactive_get_dataset()

    # If local directory, resolve to absolute path for data_dir
    data_dir = None
    if is_local:
        data_dir = str(Path(dataset).expanduser().resolve())

    config = TinkerConfig(
        dataset=dataset,
        task_type=task_type.value,  # Convert enum to string
        model=model,
        interactive=True,
        data_dir=data_dir,
    )

    # Show summary
    dataset_info = f"[bold]Dataset:[/bold] {dataset}"
    if is_local:
        dataset_info += " [dim](local, read-only)[/dim]"
    
    console.print()
    console.print(
        Panel(
            f"[bold]Task Type:[/bold] {task_type}\n"
            f"[bold]Model:[/bold] {model}\n"
            f"{dataset_info}",
            title="[bold green]Configuration Summary[/bold green]",
            border_style="green",
        )
    )

    if not Confirm.ask("\nProceed with this configuration?", default=True):
        raise KeyboardInterrupt("User cancelled")

    return config


def run_non_interactive(config: TinkerConfig) -> TinkerConfig:
    """Validate non-interactive configuration."""
    # Validate task_type string
    try:
        TaskType(config.task_type)  # Validate it's a valid enum value
    except ValueError:
        raise ValueError(
            f"Invalid task_type '{config.task_type}'. "
            f"Must be one of: {', '.join(t.value for t in TaskType)}"
        )

    # Validate model
    if config.model not in MODEL_NAMES:
        raise ValueError(
            f"Invalid model '{config.model}'. Must be one of: {', '.join(MODEL_NAMES)}"
        )

    # Validate dataset (supports both HuggingFace and local directories)
    valid, message, is_local = validate_dataset(config.dataset)
    if not valid:
        raise ValueError(message)

    console.print(f"[green]{message}[/green]")
    
    # If local directory and data_dir not already set, set it
    if is_local and not config.data_dir:
        # Create new config with data_dir set
        config = TinkerConfig(
            dataset=config.dataset,
            task_type=config.task_type,
            model=config.model,
            interactive=config.interactive,
            data_dir=str(Path(config.dataset).expanduser().resolve()),
        )
        console.print("[dim]Note: Local directory will be mounted as read-only[/dim]")
    
    return config


def setup() -> None:
    """Setup command entrypoint."""
    run_setup()


def viewer() -> None:
    """Launch the trace viewer."""
    viewer_url = launch_viewer()
    console.print(f"\n[bold]Open viewer at:[/bold] {viewer_url}")
    console.print("[dim]Press Ctrl+C to exit[/dim]\n")

    # Keep process alive
    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Exiting...[/dim]")


def main() -> None:
    """CLI entrypoint supporting both interactive and non-interactive modes."""
    import sys

    # Handle 'setup' subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        run_setup()
        return

    # Handle 'viewer' subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "viewer":
        viewer()
        return

    # Show welcome animation
    show_welcome_animation()

    # Check if running with chz arguments (non-interactive)
    if len(sys.argv) > 1 and "=" in sys.argv[1]:
        # Non-interactive mode via chz
        def run_with_config(config: TinkerConfig) -> None:
            # Check env is configured
            if not ensure_env_configured():
                sys.exit(1)

            config = run_non_interactive(config)

            # Launch viewer
            viewer_url = launch_viewer()

            # Import and run agent
            from tinker_agent.main import run_training_agent

            try:
                run_training_agent(
                    dataset=config.dataset,
                    task_type=config.task_type,
                    model=config.model,
                    data_dir=config.data_dir,
                )
                console.print(f"\n[bold]View results at:[/bold] {viewer_url}\n")
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                sys.exit(1)

        chz.entrypoint(run_with_config)
    else:
        # Interactive mode
        try:
            # Check env is configured first
            if not ensure_env_configured():
                sys.exit(1)

            config = run_interactive()

            # Launch viewer
            viewer_url = launch_viewer()

            # Import and run agent
            from tinker_agent.main import run_training_agent

            try:
                run_training_agent(
                    dataset=config.dataset,
                    task_type=config.task_type,
                    model=config.model,
                    data_dir=config.data_dir,
                )
                console.print(f"\n[bold]View results at:[/bold] {viewer_url}\n")
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                sys.exit(1)
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled[/dim]")
        except ValueError as e:
            console.print(f"\n[red]Error: {e}[/red]")
            sys.exit(1)


if __name__ == "__main__":
    main()
