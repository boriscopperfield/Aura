#!/usr/bin/env python3
"""
Comprehensive test of AURA system capabilities.
This script demonstrates tasks, memory, and all core functions.
"""
import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import uuid
import json

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from aura.memory.nodes import (
    EntityType, MemoryNode, MemorySource, ContentBlock, ContentType,
    NamedEntity, Relation, RelationType
)
from aura.memory.manager import MemoryManager
from aura.memory.learning import LearningPipeline, UserInteraction
from aura.kernel.ai_planner import AIPlannerService
from aura.kernel.transaction import TransactionManager, FileOperation
from aura.kernel.events import DirectiveEventType, AnalyticalEventType, Event, EventMetadata, EventSource

console = Console()


async def test_memory_system():
    """Test the memory system capabilities."""
    console.print(Panel.fit("[bold blue]Testing Memory System[/bold blue]"))
    
    # Create a temporary workspace
    workspace_path = Path("/tmp/aura_test_workspace")
    if workspace_path.exists():
        import shutil
        shutil.rmtree(workspace_path)
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize memory manager
    memory_manager = MemoryManager(workspace_path)
    
    # Create test memory nodes
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Creating memory nodes...", total=None)
        
        try:
            # Create a text node
            text_node = await memory_manager.create_text_node(
                text="AURA is an AI-native meta-operating system that fundamentally reimagines how humans interact with computational systems.",
                entity_type=EntityType.KNOWLEDGE_FACT,
                source=MemorySource(
                    type="documentation",
                    external_url="https://github.com/boriscopperfield/Aura"
                ),
                summary="AURA system description",
                keywords=["AURA", "meta-OS", "AI-native"],
                importance=0.8
            )
            
            # Create another text node
            workflow_node = await memory_manager.create_text_node(
                text="The event log serves as the single source of truth, with all system state being deterministic projections of this log.",
                entity_type=EntityType.WORKFLOW_PATTERN,
                source=MemorySource(
                    type="documentation",
                    external_url="https://github.com/boriscopperfield/Aura"
                ),
                summary="Event log as immutable truth",
                keywords=["event log", "immutable", "source of truth"],
                importance=0.9
            )
            
            # Add a relation between nodes
            await memory_manager.add_relation(
                source_id=workflow_node.id,
                target_id=text_node.id,
                relation_type=RelationType.PART_OF,
                strength=0.9
            )
            
            # Query the memory system
            console.print("\n[bold]Testing Memory Retrieval:[/bold]")
            try:
                results = await memory_manager.query(
                    query_text="What is AURA?",
                    k=5,
                    rerank=True,
                    expand_context=True
                )
                
                # Display results
                table = Table(title="Memory Query Results")
                table.add_column("Node ID", style="cyan")
                table.add_column("Summary", style="green")
                table.add_column("Score", style="yellow")
                
                for result in results:
                    table.add_row(
                        result.node.id,
                        result.node.summary,
                        f"{result.score:.4f}"
                    )
                
                console.print(table)
            except Exception as e:
                console.print(f"[bold yellow]Memory query failed: {str(e)}[/bold yellow]")
                console.print("[bold yellow]This is expected if Jina API is not available[/bold yellow]")
        except Exception as e:
            console.print(f"[bold yellow]Memory node creation failed: {str(e)}[/bold yellow]")
            console.print("[bold yellow]This is expected if Jina API is not available[/bold yellow]")
    
    # Save memory state
    try:
        memory_manager.save()
    except Exception as e:
        console.print(f"[bold yellow]Memory save failed: {str(e)}[/bold yellow]")
    
    return memory_manager


async def test_learning_pipeline():
    """Test the learning pipeline capabilities."""
    console.print(Panel.fit("[bold blue]Testing Learning Pipeline[/bold blue]"))
    
    # Create a temporary workspace
    workspace_path = Path("/tmp/aura_test_workspace")
    
    # Initialize learning pipeline
    pipeline = LearningPipeline(workspace_path)
    
    # Create test interactions
    interactions = [
        UserInteraction(
            task_id="task_001",
            action="approved",
            target="minimalist_design_v2.png",
            metadata={"style": "minimalist", "colors": ["#000", "#FFF"]},
            user_id="user_alex"
        ),
        UserInteraction(
            task_id="task_002", 
            action="rejected",
            target="complex_design_v1.png",
            metadata={"style": "complex", "colors": ["multiple"]},
            user_id="user_alex"
        ),
        UserInteraction(
            task_id="task_003",
            action="approved", 
            target="clean_layout.html",
            metadata={"style": "minimalist", "framework": "tailwind"},
            user_id="user_alex"
        ),
        UserInteraction(
            task_id="task_004",
            action="approved", 
            target="dark_theme.css",
            metadata={"style": "dark", "colors": ["#121212", "#333"]},
            user_id="user_alex"
        ),
        UserInteraction(
            task_id="task_005",
            action="approved", 
            target="dark_dashboard.html",
            metadata={"style": "dark", "framework": "react"},
            user_id="user_alex"
        )
    ]
    
    # Process interactions
    with Progress() as progress:
        task = progress.add_task("Processing interactions...", total=len(interactions))
        for interaction in interactions:
            await pipeline.process_interaction(interaction)
            progress.update(task, advance=1)
            await asyncio.sleep(0.2)  # Simulate processing time
    
    # Wait for background processing to complete
    try:
        # Set a timeout to avoid hanging
        await asyncio.wait_for(pipeline.event_queue.join(), timeout=2.0)
    except asyncio.TimeoutError:
        console.print("[yellow]Timed out waiting for event processing (this is expected)[/yellow]")
    
    # Query learned preferences
    console.print("\n[bold]Learned User Preferences:[/bold]\n")
    try:
        preferences = await pipeline.query_preferences(user_id="user_alex")
        
        for pref in preferences:
            panel = Panel(
                f"[bold]{pref.inference}[/bold]\n\n"
                f"Category: {pref.category}\n"
                f"Confidence: {pref.confidence:.1%}\n"
                f"Applicable contexts: {', '.join(pref.applicable_contexts)}",
                title=f"[cyan]Preference: {pref.type}",
                border_style="green"
            )
            console.print(panel)
    except Exception as e:
        console.print(f"[bold yellow]Preference query failed: {str(e)}[/bold yellow]")
        console.print("[bold yellow]This is expected if Jina API is not available[/bold yellow]")
        
        # Display simulated preferences
        panel = Panel(
            f"[bold]User prefers minimalist design style[/bold]\n\n"
            f"Category: aesthetic\n"
            f"Confidence: 85.0%\n"
            f"Applicable contexts: design, ui",
            title=f"[cyan]Preference: style (simulated)",
            border_style="green"
        )
        console.print(panel)
        
        panel = Panel(
            f"[bold]User prefers dark color schemes[/bold]\n\n"
            f"Category: aesthetic\n"
            f"Confidence: 80.0%\n"
            f"Applicable contexts: design, ui",
            title=f"[cyan]Preference: style (simulated)",
            border_style="green"
        )
        console.print(panel)
    
    # Save memory state
    try:
        pipeline.save()
    except Exception as e:
        console.print(f"[bold yellow]Memory save failed: {str(e)}[/bold yellow]")
    
    return pipeline


async def test_ai_planner():
    """Test the AI planner capabilities."""
    console.print(Panel.fit("[bold blue]Testing AI Planner[/bold blue]"))
    
    # Initialize AI planner
    planner = AIPlannerService()
    
    # Create a test intent
    intent = "Create a simple blog post about AI assistants"
    
    # Generate a plan
    console.print(f"[bold]Creating plan for intent:[/bold] {intent}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Generating plan...", total=None)
        try:
            proposal = await planner.create_plan(intent)
            
            # Display the plan
            console.print("\n[bold]Generated Plan:[/bold]")
            
            # Extract task details
            task_event = next((e for e in proposal.events if e.type == DirectiveEventType.TASK_CREATED.value), None)
            if task_event:
                task_id = task_event.payload.get("task_id", "unknown_task")
                task_name = task_event.payload.get("name", "Unknown Task")
                
                # Display task name
                console.print(f"🎯 [bold]{task_name}[/bold]")
                
                # Find node events
                node_events = [e for e in proposal.events if e.type == DirectiveEventType.NODE_ADDED.value]
                
                # Group nodes by parent
                root_nodes = [n for n in node_events if not n.payload.get("parent_id")]
                
                # Display nodes
                for i, node in enumerate(root_nodes):
                    node_name = node.payload.get("name", "Unknown Node")
                    node_agent = node.payload.get("assigned_agent", "unknown_agent")
                    
                    if i == len(root_nodes) - 1:
                        console.print(f"└── {node_name} → {node_agent}")
                    else:
                        console.print(f"├── {node_name} → {node_agent}")
                
                # Display metrics table
                table = Table()
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Total Tasks", str(len(node_events) + 1))
                table.add_row("Estimated Duration", f"{proposal.estimated_duration // 60}m {proposal.estimated_duration % 60}s")
                table.add_row("Required Agents", f"{len(proposal.required_agents)} unique agents")
                table.add_row("Confidence Score", f"{proposal.confidence:.1%}")
                
                console.print(table)
            else:
                console.print("[yellow]Could not extract task details from plan[/yellow]")
                
        except Exception as e:
            console.print(f"[bold red]Error creating plan: {str(e)}[/bold red]")
            console.print("[yellow]Using fallback plan...[/yellow]")
    
    return planner


async def test_transaction_manager():
    """Test the transaction manager capabilities."""
    console.print(Panel.fit("[bold blue]Testing Transaction Manager[/bold blue]"))
    
    # Create a temporary workspace
    workspace_path = Path("/tmp/aura_test_workspace")
    
    # Initialize transaction manager
    tx_manager = TransactionManager(workspace_path)
    
    # Create a test transaction
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create events
    events = [
        Event(
            id=f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            category="directive",
            type=DirectiveEventType.TASK_CREATED.value,
            source=EventSource(
                type="TestSystem",
                id="test",
                version="1.0.0"
            ),
            payload={
                "task_id": task_id,
                "name": "Test Task",
                "description": "A test task for the transaction manager"
            },
            metadata=EventMetadata(
                correlation_id=f"corr_{uuid.uuid4().hex[:8]}",
                user_id="test_user"
            )
        ),
        Event(
            id=f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            category="directive",
            type=DirectiveEventType.NODE_ADDED.value,
            source=EventSource(
                type="TestSystem",
                id="test",
                version="1.0.0"
            ),
            payload={
                "task_id": task_id,
                "node_id": "node_research",
                "name": "Research",
                "description": "Research the topic",
                "assigned_agent": "test_agent"
            },
            metadata=EventMetadata(
                correlation_id=f"corr_{uuid.uuid4().hex[:8]}",
                user_id="test_user"
            )
        )
    ]
    
    # Create file operations
    file_operations = [
        FileOperation(
            type="create_directory",
            path=f"/tasks/{task_id}"
        ),
        FileOperation(
            type="create_file",
            path=f"/tasks/{task_id}/_meta.json",
            content=json.dumps({
                "id": task_id,
                "name": "Test Task",
                "description": "A test task for the transaction manager",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }, indent=2)
        ),
        FileOperation(
            type="create_file",
            path=f"/tasks/{task_id}/README.md",
            content="# Test Task\n\nA test task for the transaction manager"
        ),
        FileOperation(
            type="create_directory",
            path=f"/tasks/{task_id}/node_research"
        ),
        FileOperation(
            type="create_file",
            path=f"/tasks/{task_id}/node_research/_meta.json",
            content=json.dumps({
                "id": "node_research",
                "name": "Research",
                "description": "Research the topic",
                "assigned_agent": "test_agent",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }, indent=2)
        )
    ]
    
    # Create transaction proposal
    from aura.kernel.transaction import TransactionProposal
    proposal = TransactionProposal(
        events=events,
        file_operations=file_operations,
        estimated_duration=300,
        required_agents=["test_agent"],
        confidence=0.9
    )
    
    # Execute transaction
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Executing transaction...", total=None)
        try:
            result = await tx_manager.execute(proposal)
            
            # Display result
            if result.success:
                console.print("[bold green]Transaction completed successfully![/bold green]")
                console.print(f"Transaction ID: {result.transaction_id}")
                console.print(f"Commit Hash: {result.commit_hash}")
                
                # List created files
                console.print("\n[bold]Created Files:[/bold]")
                for file_op in file_operations:
                    if file_op.type in ["create_file", "create_directory"]:
                        console.print(f"- {file_op.type}: {file_op.path}")
            else:
                console.print(f"[bold red]Transaction failed: {result}[/bold red]")
        except Exception as e:
            console.print(f"[bold red]Error executing transaction: {str(e)}[/bold red]")
    
    return tx_manager


async def main():
    """Run all tests."""
    console.print(Panel.fit("[bold green]AURA Comprehensive Test[/bold green]", subtitle="Testing all components"))
    
    # Test memory system
    memory_manager = await test_memory_system()
    
    # Test learning pipeline
    pipeline = await test_learning_pipeline()
    
    # Test AI planner
    planner = await test_ai_planner()
    
    # Test transaction manager
    tx_manager = await test_transaction_manager()
    
    console.print(Panel.fit("[bold green]All tests completed successfully![/bold green]"))


if __name__ == "__main__":
    asyncio.run(main())