import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tinker_agent.agent import run_agent, Config


def main() -> None:
    """CLI entrypoint with interactive prompt."""
    # Create timestamped runs directory (resolve before chdir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = (Path.cwd() / "runs" / timestamp).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Change to the runs directory
    os.chdir(runs_dir)
    print(f"Working directory: {runs_dir}")

    # Interactive CLI loop
    while True:
        try:
            prompt = input("\n> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit", "q"):
                break

            config = Config(prompt=prompt, cwd=str(runs_dir))
            asyncio.run(run_agent(config))

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()
