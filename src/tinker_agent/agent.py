import asyncio
import json
import uuid
from pathlib import Path

import chz
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, Message, HookMatcher
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from tinker_agent.eval import validate_results
from tinker_agent.tracer import Tracer

console = Console()

# Global tracer instance (set per run)
_tracer: Tracer | None = None

# Per-session state tracking (supports concurrent sessions)
# Key: session_id, Value: {"tool_output_chars": int}
_session_state: dict[str, dict] = {}


def _find_latest_results_dir(cwd: Path | None = None) -> Path | None:
    """
    Find the results directory.

    Checks in order:
    1. cwd/results/ (if cwd is a run directory)
    2. cwd/runs/*/results/ (if cwd is project root)
    3. ./results/ or ./runs/*/results/ (if cwd is None)

    Returns the path to the results/ directory, or None if not found.
    """
    base = cwd or Path.cwd()

    # First check: results/ directly in cwd (when cwd is a run directory)
    direct_results = base / "results"
    if direct_results.exists() and direct_results.is_dir():
        return direct_results

    # Second check: runs/*/results/ pattern (when cwd is project root)
    runs_dir = base / "runs"
    if runs_dir.exists():
        run_dirs_with_results = []
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir() and (run_dir / "results").exists():
                run_dirs_with_results.append(run_dir)

        if run_dirs_with_results:
            # Sort by name (YYYYMMDD_HHMMSS format sorts chronologically)
            latest = sorted(run_dirs_with_results, key=lambda d: d.name, reverse=True)[
                0
            ]
            return latest / "results"

    return None


@chz.chz
class Config:
    """Agent configuration."""

    prompt: str
    use_custom_prompt: bool = True
    cwd: str | None = None
    verbose: bool = False
    trace_path: str | None = None  # Path to JSONL trace file (None = no tracing)


def _create_log_tool_output_hook(session_id: str, tracer: Tracer | None = None):
    """Factory: creates a PostToolUse hook bound to a specific session."""

    # Tools that execute commands and produce stdout/stderr
    COMMAND_TOOLS = {
        "Bash",
        "bash",
        "Shell",
        "shell",
        "Execute",
        "execute",
        "Run",
        "run",
    }

    async def log_tool_output(input_data, tool_use_id, context):
        """PostToolUse hook: log tool output size to debug context growth."""
        state = _session_state.get(session_id, {})

        tool_name = input_data.get("tool_name", "unknown")
        # SDK uses "tool_response" not "tool_result" for PostToolUse hooks
        tool_response = input_data.get("tool_response", "")
        is_error = input_data.get("is_error", False)

        # Log to tracer (captures the actual tool output!)
        if tracer:
            tracer.log_tool_result(
                tool_id=tool_use_id,
                result=tool_response,
                is_error=is_error,
                tool_name=tool_name,
            )

        # Calculate size of this tool's output
        result_str = str(tool_response)
        result_chars = len(result_str)
        state["tool_output_chars"] = state.get("tool_output_chars", 0) + result_chars
        _session_state[session_id] = state

        # Estimate tokens (~4 chars per token)
        result_tokens = result_chars // 4
        total_tokens = state["tool_output_chars"] // 4

        # Log with color coding based on size
        if result_chars > 50000:
            style = "bold red"
            warning = " ⚠️ VERY LARGE"
        elif result_chars > 10000:
            style = "yellow"
            warning = " ⚠️ large"
        else:
            style = "dim"
            warning = ""

        console.print(
            f"[{style}]📊 PostToolUse: {tool_name} | "
            f"output: {result_chars:,} chars (~{result_tokens:,} tokens){warning} | "
            f"cumulative: {state['tool_output_chars']:,} chars (~{total_tokens:,} tokens)[/{style}]"
        )

        # For command tools (Bash, etc.), show the actual stdout/stderr output
        if tool_name in COMMAND_TOOLS and result_str:
            # Show up to 2000 chars of command output
            output_preview = result_str[:2000]
            if len(result_str) > 2000:
                output_preview += f"\n... [{len(result_str) - 2000:,} more chars]"

            border_style = "red" if is_error else "blue"
            title = (
                "[bold red]stderr/error[/bold red]"
                if is_error
                else "[bold blue]stdout[/bold blue]"
            )
            console.print(
                Panel(
                    output_preview,
                    title=title,
                    border_style=border_style,
                )
            )
        # Preview first 200 chars if large (for non-command tools)
        elif result_chars > 10000:
            preview = result_str[:200].replace("\n", " ")
            console.print(f"[dim]   Preview: {preview}...[/dim]")

        return {}

    return log_tool_output


def _create_stop_hook(session_id: str, cwd: Path | None = None):
    """Factory: creates a Stop hook for logging (validation is handled in main loop)."""

    async def on_stop(input_data, tool_use_id, context):
        """Stop hook: log agent stop (validation retry is handled externally)."""
        # Note: The Stop hook cannot prevent stopping - it's for notification only.
        # Validation-based retries are handled in run_agent() after receive_response() completes.
        return {}

    return on_stop


# TODO: reuse tinker cookbook agents.md
async def run_agent(config: Config) -> None:
    """Run the post-training agent."""
    global _tracer

    # Create unique session ID for this run (supports concurrent sessions)
    session_id = str(uuid.uuid4())
    _session_state[session_id] = {"tool_output_chars": 0}

    # Initialize tracer if path provided
    tracer = None
    if config.trace_path:
        tracer = Tracer(config.trace_path)
        tracer.start_trace(
            prompt=config.prompt,
            model="claude-opus-4-5-20251101",
            metadata={"cwd": config.cwd or str(Path.cwd())},
        )
        _tracer = tracer

    system_prompt_config = None
    if config.use_custom_prompt:
        prompt_path = Path(__file__).parent.parent.parent / "prompt.md"
        if prompt_path.exists():
            custom_prompt = prompt_path.read_text()
            system_prompt_config = {
                "type": "preset",
                "preset": "claude_code",
                "append": custom_prompt,
            }
        else:
            console.print(
                f"[yellow]Warning: prompt.md not found at {prompt_path}, using default system prompt[/yellow]"
            )
            system_prompt_config = {"type": "preset", "preset": "claude_code"}
    else:
        system_prompt_config = {"type": "preset", "preset": "claude_code"}

    # TODO: check planning mode
    # TODO: when in a sandbox, make sure it can't go outside it's directory
    options = ClaudeAgentOptions(
        tools={"type": "preset", "preset": "claude_code"},
        system_prompt=system_prompt_config,
        model="claude-opus-4-5-20251101",
        permission_mode="bypassPermissions",
        mcp_servers={},
        continue_conversation=False,
        cwd=config.cwd or str(Path.cwd()),
        hooks={
            "PostToolUse": [
                HookMatcher(
                    hooks=[_create_log_tool_output_hook(session_id, tracer=tracer)]
                )
            ],
            "Stop": [
                HookMatcher(
                    hooks=[
                        _create_stop_hook(
                            session_id, cwd=Path(config.cwd) if config.cwd else None
                        )
                    ]
                )
            ],
        },
    )

    result_text = None
    error_text = None
    cwd_path = Path(config.cwd) if config.cwd else None
    MAX_VALIDATION_RETRIES = 2
    retry_count = 0

    try:
        # Use ClaudeSDKClient instead of query() to support hooks
        async with ClaudeSDKClient(options=options) as client:
            # Send initial prompt
            await client.query(config.prompt)

            while retry_count <= MAX_VALIDATION_RETRIES:
                # Receive agent response
                async for message in client.receive_response():
                    _print_message(message, verbose=config.verbose)
                    if tracer:
                        tracer.record_message(message)
                        if hasattr(message, "result") and message.result:
                            result_text = message.result

                # After agent completes, validate results
                results_dir = _find_latest_results_dir(cwd_path)
                if results_dir:
                    console.print(f"[dim]🔍 Validating results at: {results_dir}[/dim]")
                    validation = validate_results(results_dir)

                    if validation.valid:
                        console.print("[green]✅ Validation passed![/green]")
                        break  # Success, exit retry loop
                    else:
                        retry_count += 1
                        if retry_count > MAX_VALIDATION_RETRIES:
                            console.print(
                                Panel(
                                    f"[bold red]Validation failed after {MAX_VALIDATION_RETRIES} retries.[/bold red]\n\n"
                                    "Errors:\n"
                                    + "\n".join(
                                        f"  • {e}" for e in validation.errors[:5]
                                    ),
                                    title="[bold red]❌ Max Retries Reached[/bold red]",
                                    border_style="red",
                                )
                            )
                            break

                        # Continue conversation with validation errors
                        error_summary = "; ".join(validation.errors[:3])
                        console.print(
                            f"[yellow]⚠️ Validation failed (attempt {retry_count}/{MAX_VALIDATION_RETRIES}), continuing...[/yellow]"
                        )
                        # Send follow-up in same conversation (agent remembers context)
                        await client.query(
                            f"Validation failed: {error_summary}. Please fix these issues."
                        )
                else:
                    # No results dir found, nothing to validate
                    break

    except Exception as e:
        error_text = str(e)
        console.print(f"[red]Agent error: {error_text}[/red]")
        raise
    finally:
        # End trace
        if tracer:
            tracer.end_trace(result=result_text, error=error_text)
            _tracer = None
        # Clean up session state
        _session_state.pop(session_id, None)


def _print_message(message: Message, verbose: bool = False) -> None:
    """Print a message from the agent with rich formatting."""
    # Handle final result
    if hasattr(message, "result") and message.result:
        console.print(
            Panel(
                message.result,
                title="[bold green]Result[/bold green]",
                border_style="green",
            )
        )
        return

    # Handle content blocks
    if hasattr(message, "content"):
        for block in message.content:
            block_type = getattr(block, "type", None)

            # Text content
            if hasattr(block, "text"):
                console.print(block.text)

            # Thinking block
            elif block_type == "thinking" or hasattr(block, "thinking"):
                thinking = getattr(block, "thinking", None) or getattr(
                    block, "text", ""
                )
                if thinking and verbose:
                    console.print(
                        Panel(
                            Text(thinking, style="dim italic"),
                            title="[bold magenta]Thinking[/bold magenta]",
                            border_style="magenta",
                        )
                    )

            # Tool use block
            elif block_type == "tool_use" or hasattr(block, "name"):
                tool_name = getattr(block, "name", "unknown")
                tool_input = getattr(block, "input", {})
                tool_id = getattr(block, "id", "")

                # Format tool input
                if isinstance(tool_input, dict):
                    input_str = json.dumps(tool_input, indent=2, default=str)
                    if len(input_str) > 500:
                        input_str = input_str[:500] + "\n..."
                else:
                    input_str = str(tool_input)[:500]

                console.print(
                    Panel(
                        Syntax(input_str, "json", theme="monokai", word_wrap=True),
                        title=f"[bold cyan]Tool: {tool_name}[/bold cyan]",
                        subtitle=f"[dim]{tool_id}[/dim]" if tool_id else None,
                        border_style="cyan",
                    )
                )

            # Tool result block
            elif block_type == "tool_result":
                tool_use_id = getattr(block, "tool_use_id", "")
                content = getattr(block, "content", "")
                is_error = getattr(block, "is_error", False)

                if isinstance(content, list):
                    content = "\n".join(getattr(c, "text", str(c)) for c in content)

                # Truncate long outputs
                if len(str(content)) > 1000:
                    content = str(content)[:1000] + "\n... [truncated]"

                style = "red" if is_error else "green"
                title = (
                    "[bold red]Tool Error[/bold red]"
                    if is_error
                    else "[bold green]Tool Result[/bold green]"
                )

                if verbose or is_error:
                    console.print(
                        Panel(
                            str(content),
                            title=title,
                            subtitle=f"[dim]{tool_use_id}[/dim]"
                            if tool_use_id
                            else None,
                            border_style=style,
                        )
                    )

    # Handle error messages
    if hasattr(message, "error") and message.error:
        console.print(
            Panel(
                str(message.error),
                title="[bold red]Error[/bold red]",
                border_style="red",
            )
        )

    # Handle stop reason
    if hasattr(message, "stop_reason") and message.stop_reason:
        if verbose:
            console.print(f"[dim]Stop reason: {message.stop_reason}[/dim]")


def main() -> None:
    """CLI entrypoint."""
    chz.entrypoint(lambda cfg: asyncio.run(run_agent(cfg)))


if __name__ == "__main__":
    main()

# streamlit run src/tinker_agent/viewer.py --server.headless=true --browser.gatherUsageStats=false
