"""
Demo script for AURA system.
"""
import asyncio
import os
from datetime import datetime
from pathlib import Path

from rich.console import Console

from aura.kernel.events import AnalyticalEventType, DirectiveEventType, Event, EventMetadata, EventSource
from aura.kernel.transaction import FileOperation, TransactionProposal, TransactionManager
from aura.memory.learning import LearningExample


async def main():
    """Run the demo."""
    console = Console()
    console.print("[bold blue]AURA System Demo[/bold blue]")
    
    # Create workspace directory
    workspace_path = Path("./aura_workspace")
    os.makedirs(workspace_path, exist_ok=True)
    
    # Initialize transaction manager
    tx_manager = TransactionManager(workspace_path)
    
    # Create a sample event
    event = Event(
        id=f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_sample",
        timestamp=datetime.now(),
        category="directive",
        type=DirectiveEventType.TASK_CREATED,
        source=EventSource(
            type="demo",
            id="demo_script",
            version="1.0.0"
        ),
        payload={
            "task_id": "task_demo",
            "name": "Demo Task",
            "description": "A task created for demonstration purposes"
        },
        metadata=EventMetadata(
            user_id="demo_user",
            session_id="demo_session"
        )
    )
    
    # Create a sample file operation
    file_op = FileOperation(
        type="create_file",
        path="tasks/demo_task/README.md",
        content="# Demo Task\n\nThis task was created by the AURA demo script."
    )
    
    # Create a transaction proposal
    proposal = TransactionProposal(
        events=[event],
        file_operations=[file_op],
        estimated_duration=100.0,
        required_agents=["demo_agent"],
        confidence=0.95
    )
    
    # Execute the transaction
    console.print("[bold]Executing transaction...[/bold]")
    result = await tx_manager.execute(proposal)
    
    if result.success:
        console.print(f"[bold green]Transaction completed successfully![/bold green]")
        console.print(f"Commit hash: {result.commit_hash}")
    else:
        console.print(f"[bold red]Transaction failed: {result.error}[/bold red]")
    
    # Demonstrate learning pipeline
    console.print("\n[bold]Demonstrating Learning Pipeline...[/bold]")
    learning_example = LearningExample()
    await learning_example.demonstrate_preference_learning()


if __name__ == "__main__":
    asyncio.run(main())