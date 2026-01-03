import asyncio
from pathlib import Path

import chz
from claude_agent_sdk import query, ClaudeAgentOptions, Message


@chz.chz
class Config:
    """Agent configuration."""

    prompt: str
    use_custom_prompt: bool = True
    cwd: str | None = None


async def run_agent(config: Config) -> None:
    """Run the post-training agent."""
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
            print(
                f"Warning: prompt.md not found at {prompt_path}, using default system prompt"
            )
            system_prompt_config = {"type": "preset", "preset": "claude_code"}
    else:
        system_prompt_config = {"type": "preset", "preset": "claude_code"}

    options = ClaudeAgentOptions(
        tools={"type": "preset", "preset": "claude_code"},
        system_prompt=system_prompt_config,
        model="claude-opus-4-5-20251101",
        permission_mode="bypassPermissions",
        mcp_servers={},
        continue_conversation=False,
        cwd=config.cwd or str(Path.cwd()),
    )

    async for message in query(prompt=config.prompt, options=options):
        _print_message(message)


def _print_message(message: Message) -> None:
    """Print a message from the agent."""
    if hasattr(message, "result") and message.result:
        print(message.result)
    elif hasattr(message, "content"):
        for block in message.content:
            if hasattr(block, "text"):
                print(block.text)
            elif hasattr(block, "name"):
                print(f"[Tool: {block.name}]")


def main() -> None:
    """CLI entrypoint."""
    chz.entrypoint(lambda cfg: asyncio.run(run_agent(cfg)))


if __name__ == "__main__":
    main()
