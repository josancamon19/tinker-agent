#!/usr/bin/env python3
"""Simple deployment script for tinker-agent package to PyPI."""

import subprocess
import sys
from pathlib import Path
from getpass import getpass


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def main():
    """Deploy the package to PyPI."""
    print("=" * 60)
    print("tinker-agent PyPI Deployment")
    print("=" * 60)
    print()

    # Confirm deployment
    confirm = input("Deploy to PyPI? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Deployment cancelled.")
        sys.exit(0)

    # Get PyPI API token
    print()
    print("Enter your PyPI API token:")
    print("(Get it from: https://pypi.org/manage/account/token/)")
    api_token = getpass("Token: ").strip()

    if not api_token:
        print("Error: API token is required")
        sys.exit(1)

    print()
    print("Step 1: Cleaning old builds...")
    dist_dir = Path("dist")
    if dist_dir.exists():
        run_command(["rm", "-rf", "dist"])

    print()
    print("Step 2: Building package...")
    result = run_command(["uv", "build"])
    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        sys.exit(1)
    print("✓ Build successful")

    print()
    print("Step 3: Uploading to PyPI...")
    try:
        result = run_command([
            "uv", "publish",
            "--token", api_token,
        ])
        if result.returncode == 0:
            print()
            print("=" * 60)
            print("✓ Successfully deployed to PyPI!")
            print("=" * 60)
            print()
            print("Install with: pip install tinker-agent")
        else:
            print(f"Upload failed: {result.stderr}")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
