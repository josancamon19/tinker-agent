import asyncio
import json
import uuid
from pathlib import Path

import chz
from claude_agent_sdk import query, ClaudeAgentOptions, Message, HookMatcher
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from tinker_agent.eval import validate_results

console = Console()

# Per-session state tracking (supports concurrent sessions)
# Key: session_id, Value: {"tool_output_chars": int, "validation_retries": int}
_session_state: dict[str, dict] = {}
MAX_VALIDATION_RETRIES = 2


@chz.chz
class Config:
    """Agent configuration."""

    prompt: str
    use_custom_prompt: bool = True
    cwd: str | None = None
    verbose: bool = False


def _create_log_tool_output_hook(session_id: str):
    """Factory: creates a PostToolUse hook bound to a specific session."""

    async def log_tool_output(input_data, tool_use_id, context):
        """PostToolUse hook: log tool output size to debug context growth."""
        state = _session_state.get(session_id, {})

        tool_name = input_data.get("tool_name", "unknown")
        tool_result = input_data.get("tool_result", "")

        # Calculate size of this tool's output
        result_str = str(tool_result)
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

        # Preview first 200 chars if large
        if result_chars > 10000:
            preview = result_str[:200].replace("\n", " ")
            console.print(f"[dim]   Preview: {preview}...[/dim]")

        return {}

    return log_tool_output


def _create_validate_on_stop_hook(session_id: str):
    """Factory: creates a Stop hook bound to a specific session."""

    async def validate_on_stop(input_data, tool_use_id, context):
        """Stop hook: validate results and continue if validation fails (max 2 retries)."""
        state = _session_state.get(session_id, {})

        result = validate_results("results")

        if not result.valid:
            state["validation_retries"] = state.get("validation_retries", 0) + 1
            _session_state[session_id] = state
            retry_count = state["validation_retries"]
            error_summary = "; ".join(result.errors[:3])  # First 3 errors

            if retry_count >= MAX_VALIDATION_RETRIES:
                # Max retries reached, inform user and stop
                console.print(
                    Panel(
                        f"[bold red]Validation failed after {MAX_VALIDATION_RETRIES} attempts.[/bold red]\n\n"
                        f"Errors:\n" + "\n".join(f"  • {e}" for e in result.errors),
                        title="[bold red]❌ Max Retries Reached[/bold red]",
                        border_style="red",
                    )
                )
                return {}  # Allow stop

            # Retry - continue the agent
            console.print(
                f"[yellow]⚠️ Validation failed (attempt {retry_count}/{MAX_VALIDATION_RETRIES}), retrying...[/yellow]"
            )
            return {
                "hookSpecificOutput": {
                    "hookEventName": input_data["hook_event_name"],
                    "decision": "continue",
                    "updatedStopReason": f"Validation failed: {error_summary}. Please fix these issues.",
                }
            }

        # Validation passed, allow stop
        return {}

    return validate_on_stop


# TODO: reuse tinker cookbook agents.md
async def run_agent(config: Config) -> None:
    """Run the post-training agent."""
    # Create unique session ID for this run (supports concurrent sessions)
    session_id = str(uuid.uuid4())
    _session_state[session_id] = {"tool_output_chars": 0, "validation_retries": 0}

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
                HookMatcher(hooks=[_create_log_tool_output_hook(session_id)])
            ],
            "Stop": [HookMatcher(hooks=[_create_validate_on_stop_hook(session_id)])],
        },
    )

    try:
        async for message in query(prompt=config.prompt, options=options):
            _print_message(message, verbose=config.verbose)
    finally:
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
