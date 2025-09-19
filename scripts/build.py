#!/usr/bin/env python3
"""
Poetry build script for fastbm25.
This script is designed to work with Poetry's build system.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, cwd=None):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    if cwd:
        print(f"Working directory: {cwd}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)

    if result.returncode != 0:
        print(f"Error: {description} failed")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return False

    print(f"Success: {description}")
    if result.stdout.strip():
        print(f"Output: {result.stdout.strip()}")
    return True


def main():
    """Main build function for Poetry."""
    print("Building fastbm25 with Poetry...")

    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Check if maturin is available
    if not run_command(["maturin", "--version"], "Checking maturin version"):
        print("Installing maturin...")
        if not run_command(
            [sys.executable, "-m", "pip", "install", "maturin"], "Installing maturin"
        ):
            print("Failed to install maturin. Please install it manually.")
            sys.exit(1)

    # Build the wheel using maturin
    if not run_command(
        ["maturin", "build", "--release"],
        "Building wheel with maturin",
        cwd=project_root,
    ):
        print("Build failed!")
        sys.exit(1)

    print("\nBuild completed successfully!")
    print("\nNext steps:")
    print("  poetry install  # Install in development mode")
    print("  poetry build    # Build distribution packages")


if __name__ == "__main__":
    main()
