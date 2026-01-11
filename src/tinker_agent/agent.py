import asyncio
import json
import re
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
    data_dir: str | None = None  # Read-only data directory (user's SFT data)


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


# Minimum timeout (ms) to enable streaming - 30 seconds
MIN_TIMEOUT_FOR_STREAMING = 30000


def _is_path_within_root(path_str: str, root_dir: Path) -> bool:
    """Check if a path is within the allowed root directory."""
    try:
        # Resolve to absolute path (handles .., symlinks, etc.)
        path = Path(path_str).resolve()
        root = root_dir.resolve()
        # Check if path is within root
        return path == root or root in path.parents
    except (OSError, ValueError):
        # Invalid path
        return False


def _extract_paths_from_input(
    tool_name: str, input_data: dict, root_dir: Path | None = None
) -> list[str]:
    """Extract file/directory paths from tool input based on tool type."""
    paths = []
    tool_lower = tool_name.lower()

    # File operation tools with explicit path parameters
    path_params = ["file_path", "path", "notebook_path", "directory"]
    for param in path_params:
        if param in input_data and input_data[param]:
            paths.append(input_data[param])

    # Bash/shell commands need path extraction from command string
    if tool_lower in {"bash", "shell", "execute", "run"}:
        command = input_data.get("command", "")

        # Detect ~ or $HOME (home directory) - always outside sandbox
        if (
            re.search(r'(?:^|[\s"\'=])~(?:/|[\s"\';&|><]|$)', command)
            or "$HOME" in command
        ):
            # Return a path guaranteed to be outside sandbox
            paths.append("/home/user")

        # Match absolute paths (Unix-style)
        abs_paths = re.findall(r'(?:^|[\s"\'=])(/[^\s"\';&|><]+)', command)
        paths.extend(abs_paths)

        # Detect relative paths with .. that could escape sandbox
        if root_dir:
            # Find relative paths in the command
            rel_paths = re.findall(r'(?:^|[\s"\'=])(\.\.[^\s"\';&|><]*)', command)
            for rel_path in rel_paths:
                # Resolve relative to root_dir to check if it escapes
                try:
                    resolved = (root_dir / rel_path).resolve()
                    paths.append(str(resolved))
                except (OSError, ValueError):
                    pass

    return paths


def _is_write_tool(tool_name: str) -> bool:
    """Check if a tool performs write operations."""
    write_tools = {
        # File modification tools
        "write",
        "edit",
        "delete",
        "strreplace",
        "str_replace",
        "create",
        "remove",
        "mkdir",
        "rmdir",
        "move",
        "copy",
        "editnotebook",
        "notebook_edit",
    }
    return tool_name.lower() in write_tools


def _is_write_command(command: str) -> bool:
    """Check if a bash command performs write operations."""
    # Commands that modify the filesystem
    write_commands = [
        # File/directory operations
        r"\brm\b",
        r"\brmdir\b",
        r"\bmkdir\b",
        r"\bmv\b",
        r"\btouch\b",
        r"\bcp\b",  # cp can write to destination
        # File content modification
        r"\becho\b.*>",
        r"\bcat\b.*>",
        r"\bprintf\b.*>",
        r"\btee\b",
        r"\btruncate\b",
        # In-place editing
        r"\bsed\b.*-i",
        r"\bawk\b.*-i",
        # Git operations that modify
        r"\bgit\b.*\b(add|commit|push|reset|checkout|merge|rebase|stash|rm|mv)\b",
        # Permission changes
        r"\bchmod\b",
        r"\bchown\b",
        r"\bchgrp\b",
        # Archive extraction (writes files)
        r"\btar\b.*-?x",
        r"\bunzip\b",
        r"\bgunzip\b",
        # Python/pip that might write
        r"\bpip\b.*install",
        r"\buv\b.*(add|remove|sync)",
    ]

    for pattern in write_commands:
        if re.search(pattern, command):
            return True
    return False


def _create_can_use_tool_handler(
    root_dir: Path | None = None,
    data_dir: Path | None = None,
    tracer: Tracer | None = None,
):
    """Factory: creates a can_use_tool handler that validates paths and modifies Bash commands.

    Args:
        root_dir: The sandbox root directory (agent can read/write here)
        data_dir: Read-only data directory (agent can only read from here)
        tracer: Optional tracer for logging
    """
    from claude_agent_sdk.types import (
        PermissionResultAllow,
        PermissionResultDeny,
        ToolPermissionContext,
    )

    async def can_use_tool(
        tool_name: str, input_data: dict, context: ToolPermissionContext
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Permission handler that validates paths and wraps long-running Bash commands."""

        # Check if tool/command is a write operation
        is_write = _is_write_tool(tool_name)
        tool_lower = tool_name.lower()
        if tool_lower in {"bash", "shell", "execute", "run"}:
            command = input_data.get("command", "")
            is_write = is_write or _is_write_command(command)

        # Path sandboxing logic
        if root_dir is not None or data_dir is not None:
            paths = _extract_paths_from_input(tool_name, input_data, root_dir=root_dir)

            for path_str in paths:
                in_root = root_dir and _is_path_within_root(path_str, root_dir)
                in_data = data_dir and _is_path_within_root(path_str, data_dir)

                # Case 1: Path is in data_dir (read-only)
                if in_data:
                    if is_write:
                        console.print(
                            f"[bold red]🚫 BLOCKED: {tool_name} tried to write to '{path_str}' "
                            f"in read-only data directory '{data_dir}'[/bold red]"
                        )
                        return PermissionResultDeny(
                            message=f"Access denied: path '{path_str}' is in the read-only data directory '{data_dir}'. "
                            f"You can read from this directory but cannot modify it."
                        )
                    # Read access to data_dir is allowed
                    continue

                # Case 2: Path is in root_dir (read/write allowed)
                if in_root:
                    continue

                # Case 3: Path is outside both directories - deny
                console.print(
                    f"[bold red]🚫 BLOCKED: {tool_name} tried to access '{path_str}' "
                    f"outside allowed directories[/bold red]"
                )
                allowed_dirs = []
                if root_dir:
                    allowed_dirs.append(f"sandbox '{root_dir}'")
                if data_dir:
                    allowed_dirs.append(f"data directory '{data_dir}' (read-only)")
                return PermissionResultDeny(
                    message=f"Access denied: path '{path_str}' is outside allowed directories. "
                    f"You can access: {', '.join(allowed_dirs)}."
                )

        # Only intercept Bash commands with significant timeout
        if tool_name.lower() not in {"bash", "shell", "execute", "run"}:
            return PermissionResultAllow(updated_input=input_data)

        # Activate virtual environment for all bash commands
        command = input_data.get("command", "")
        if command and root_dir:
            venv_path = root_dir / ".venv"
            activate_script = venv_path / "bin" / "activate"
            if activate_script.exists():
                # Prepend venv activation to command
                command = f"source {activate_script} && {command}"
                input_data = {**input_data, "command": command}

        timeout = input_data.get("timeout", 0)
        if timeout < MIN_TIMEOUT_FOR_STREAMING:
            return PermissionResultAllow(updated_input=input_data)

        command = input_data.get("command", "")
        if not command:
            return PermissionResultAllow(updated_input=input_data)

        # Create log directory in the run directory
        bash_logs_dir = (
            root_dir / "bash_logs" if root_dir else Path("/tmp/tinker_bash_logs")
        )
        bash_logs_dir.mkdir(parents=True, exist_ok=True)

        # Create unique log file for this tool call (use timestamp + hash since we don't have tool_use_id here)
        import hashlib
        import time

        tool_hash = hashlib.md5(f"{command}{time.time()}".encode()).hexdigest()[:12]
        log_file = bash_logs_dir / f"{tool_hash}.log"

        # Clear any existing log file
        log_file.write_text("")

        # Wrap command to tee output to log file
        # Use stdbuf to disable buffering so output appears immediately
        modified_command = f"({command}) 2>&1 | stdbuf -oL tee {log_file}"

        console.print(f"[dim]📡 Streaming output to: {log_file}[/dim]")

        # Log to tracer so viewer knows where to find the log
        if tracer:
            tracer.add_event(
                "bash_stream_start",
                {
                    "tool_id": tool_hash,
                    "log_file": str(log_file),
                    "original_command": command,
                },
            )

        # Return with modified input
        return PermissionResultAllow(
            updated_input={**input_data, "command": modified_command}
        )

    return can_use_tool


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
    # Sandbox: restrict agent to cwd directory
    root_dir = Path(config.cwd).resolve() if config.cwd else Path.cwd().resolve()

    # Resolve data_dir if provided (read-only access for user's SFT data)
    data_dir = Path(config.data_dir).resolve() if config.data_dir else None
    if data_dir and not data_dir.exists():
        console.print(f"[yellow]Warning: data_dir '{data_dir}' does not exist[/yellow]")
        data_dir = None

    options = ClaudeAgentOptions(
        tools={"type": "preset", "preset": "claude_code"},
        system_prompt=system_prompt_config,
        model="claude-opus-4-5-20251101",
        permission_mode="acceptEdits",  # Use acceptEdits so can_use_tool still runs
        mcp_servers={},
        continue_conversation=False,
        cwd=str(root_dir),
        can_use_tool=_create_can_use_tool_handler(
            root_dir=root_dir, data_dir=data_dir, tracer=tracer
        ),
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
