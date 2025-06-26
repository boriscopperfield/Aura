#!/usr/bin/env python3
"""
Simple test script for the AURA memory system.
"""
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel


console = Console()


def main():
    """Main function."""
    console.print(Panel(
        "[bold]AURA Enhanced Memory System Test[/bold]\n"
        "Simple test to verify the system is working",
        border_style="cyan"
    ))
    
    # Print repository structure
    console.print("\n[bold cyan]Repository Structure:[/bold cyan]")
    
    repo_path = Path("/workspace/Aura")
    
    # List main directories
    console.print("\n[bold]Main Directories:[/bold]")
    for item in repo_path.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            console.print(f"- {item.name}")
    
    # List memory modules
    memory_path = repo_path / "aura" / "memory"
    console.print("\n[bold]Memory Modules:[/bold]")
    for item in memory_path.iterdir():
        if item.is_file() and item.suffix == ".py":
            console.print(f"- {item.name}")
    
    # List documentation
    docs_path = repo_path / "docs"
    console.print("\n[bold]Documentation:[/bold]")
    for item in docs_path.iterdir():
        if item.is_file():
            console.print(f"- {item.name}")
    
    console.print("\n[bold green]Test completed successfully![/bold green]")


if __name__ == "__main__":
    main()