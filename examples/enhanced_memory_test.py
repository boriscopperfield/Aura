#!/usr/bin/env python3
"""
Test script for the enhanced memory system.

This script demonstrates the hierarchical memory system with relationship modeling
and graph-based retrieval.
"""
import os
import asyncio
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

from aura.memory.models import (
    MemoryNode, ContentBlock, ContentType, EntityType, 
    MemorySource, NamedEntity, Relation, RelationType,
    Query
)
from aura.memory.enhanced_manager import EnhancedMemoryManager
from aura.memory.hierarchy import MemoryLayer
from aura.memory.relationships import Relationship


console = Console()


async def create_test_nodes(manager: EnhancedMemoryManager):
    """Create test memory nodes.
    
    Args:
        manager: Enhanced memory manager
    """
    console.print("\n[bold cyan]Creating test memory nodes...[/bold cyan]")
    
    # Create source
    source = MemorySource(
        type="test",
        user_id="test_user"
    )
    
    # Create cat image node
    cat_node = await manager.create_node(
        content=[
            ContentBlock(
                type=ContentType.TEXT,
                data="A cute cat mascot for our tech company's marketing campaign",
                metadata={"language": "en"}
            ),
            ContentBlock(
                type=ContentType.TEXT,
                data="The cat is blue and white, holding a shopping bag",
                metadata={"language": "en"}
            )
        ],
        entity_type=EntityType.TASK_ARTIFACT,
        source=source,
        summary="Cat mascot image",
        keywords=["cat", "mascot", "marketing", "tech"],
        entities=[
            NamedEntity(type="animal", value="cat", confidence=0.95),
            NamedEntity(type="color", value="blue", confidence=0.9),
            NamedEntity(type="color", value="white", confidence=0.9)
        ],
        importance=0.8
    )
    
    console.print(f"Created cat node: [green]{cat_node.id}[/green]")
    
    # Create banner node
    banner_node = await manager.create_node(
        content=[
            ContentBlock(
                type=ContentType.TEXT,
                data="Cyber Monday sale banner featuring our mascot",
                metadata={"language": "en"}
            ),
            ContentBlock(
                type=ContentType.TEXT,
                data="The banner has a dark blue background with white text and the cat mascot in the center",
                metadata={"language": "en"}
            )
        ],
        entity_type=EntityType.TASK_ARTIFACT,
        source=source,
        summary="Cyber Monday banner",
        keywords=["banner", "cyber monday", "sale", "marketing"],
        entities=[
            NamedEntity(type="event", value="cyber monday", confidence=0.95),
            NamedEntity(type="color", value="dark blue", confidence=0.9),
            NamedEntity(type="color", value="white", confidence=0.9)
        ],
        relations=[
            Relation(
                type=RelationType.DEPENDS_ON,
                target_id=cat_node.id,
                strength=0.9
            )
        ],
        importance=0.85
    )
    
    console.print(f"Created banner node: [green]{banner_node.id}[/green]")
    
    # Create email node
    email_node = await manager.create_node(
        content=[
            ContentBlock(
                type=ContentType.TEXT,
                data="Email template for Cyber Monday campaign",
                metadata={"language": "en"}
            ),
            ContentBlock(
                type=ContentType.TEXT,
                data="The email features the banner at the top and product listings below",
                metadata={"language": "en"}
            )
        ],
        entity_type=EntityType.TASK_ARTIFACT,
        source=source,
        summary="Cyber Monday email template",
        keywords=["email", "template", "cyber monday", "marketing"],
        entities=[
            NamedEntity(type="event", value="cyber monday", confidence=0.95),
            NamedEntity(type="document", value="email", confidence=0.9)
        ],
        relations=[
            Relation(
                type=RelationType.DEPENDS_ON,
                target_id=banner_node.id,
                strength=0.9
            )
        ],
        importance=0.75
    )
    
    console.print(f"Created email node: [green]{email_node.id}[/green]")
    
    # Create pattern node
    pattern_node = await manager.create_node(
        content=[
            ContentBlock(
                type=ContentType.TEXT,
                data="Pattern: Mascot-based marketing campaign",
                metadata={"language": "en"}
            ),
            ContentBlock(
                type=ContentType.TEXT,
                data="1. Create mascot character\n2. Design banner with mascot\n3. Create email template\n4. Create social media posts",
                metadata={"language": "en"}
            )
        ],
        entity_type=EntityType.WORKFLOW_PATTERN,
        source=source,
        summary="Mascot marketing campaign pattern",
        keywords=["pattern", "mascot", "marketing", "workflow"],
        entities=[
            NamedEntity(type="pattern", value="marketing campaign", confidence=0.95)
        ],
        relations=[
            Relation(
                type=RelationType.REFERENCES,
                target_id=cat_node.id,
                strength=0.7
            ),
            Relation(
                type=RelationType.REFERENCES,
                target_id=banner_node.id,
                strength=0.7
            ),
            Relation(
                type=RelationType.REFERENCES,
                target_id=email_node.id,
                strength=0.7
            )
        ],
        importance=0.9
    )
    
    console.print(f"Created pattern node: [green]{pattern_node.id}[/green]")
    
    # Create dog mascot node (for a different campaign)
    dog_node = await manager.create_node(
        content=[
            ContentBlock(
                type=ContentType.TEXT,
                data="A friendly dog mascot for our Easter sale campaign",
                metadata={"language": "en"}
            ),
            ContentBlock(
                type=ContentType.TEXT,
                data="The dog is brown and white, holding an Easter basket",
                metadata={"language": "en"}
            )
        ],
        entity_type=EntityType.TASK_ARTIFACT,
        source=source,
        summary="Dog mascot image",
        keywords=["dog", "mascot", "marketing", "easter"],
        entities=[
            NamedEntity(type="animal", value="dog", confidence=0.95),
            NamedEntity(type="color", value="brown", confidence=0.9),
            NamedEntity(type="color", value="white", confidence=0.9),
            NamedEntity(type="event", value="easter", confidence=0.9)
        ],
        importance=0.75
    )
    
    console.print(f"Created dog node: [green]{dog_node.id}[/green]")
    
    # Add relationship between cat and dog mascots
    relationship = await manager.add_relationship(
        source_id=cat_node.id,
        target_id=dog_node.id,
        relation_type=RelationType.SIMILAR_TO,
        strength=0.8,
        bidirectional=True,
        context={"similarity": "both mascot characters"}
    )
    
    console.print(f"Created relationship: [green]{relationship.id}[/green]")
    
    return {
        "cat_node": cat_node,
        "banner_node": banner_node,
        "email_node": email_node,
        "pattern_node": pattern_node,
        "dog_node": dog_node
    }


async def test_memory_layers(manager: EnhancedMemoryManager, nodes: Dict):
    """Test memory layers.
    
    Args:
        manager: Enhanced memory manager
        nodes: Dictionary of test nodes
    """
    console.print("\n[bold cyan]Testing memory layers...[/bold cyan]")
    
    # Check which layer each node is in
    for name, node in nodes.items():
        l1_node = manager.layered_memory.l1_nodes.get(node.id)
        l2_node = manager.layered_memory.l2_nodes.get(node.id)
        l3_node = manager.layered_memory.l3_nodes.get(node.id)
        
        layers = []
        if l1_node:
            layers.append("L1")
        if l2_node:
            layers.append("L2")
        if l3_node:
            layers.append("L3")
        
        console.print(f"{name}: {', '.join(layers)}")
    
    # Simulate memory lifecycle
    console.print("\n[bold]Simulating memory lifecycle...[/bold]")
    
    # Demote cat node from L1 to L2
    cat_node = nodes["cat_node"]
    manager.layered_memory.demote_node(cat_node.id, MemoryLayer.L1, MemoryLayer.L2)
    console.print(f"Demoted cat node from L1 to L2")
    
    # Demote dog node from L1 to L3
    dog_node = nodes["dog_node"]
    manager.layered_memory.demote_node(dog_node.id, MemoryLayer.L1, MemoryLayer.L2)
    manager.layered_memory.demote_node(dog_node.id, MemoryLayer.L2, MemoryLayer.L3)
    console.print(f"Demoted dog node from L1 to L3")
    
    # Access cat node to promote it back to L1
    await manager.get_node(cat_node.id)
    console.print(f"Accessed cat node to promote it back to L1")
    
    # Check layers again
    console.print("\n[bold]Updated memory layers:[/bold]")
    for name, node in nodes.items():
        l1_node = manager.layered_memory.l1_nodes.get(node.id)
        l2_node = manager.layered_memory.l2_nodes.get(node.id)
        l3_node = manager.layered_memory.l3_nodes.get(node.id)
        
        layers = []
        if l1_node:
            layers.append("L1")
        if l2_node:
            layers.append("L2")
        if l3_node:
            layers.append("L3")
        
        console.print(f"{name}: {', '.join(layers)}")


async def test_graph_retrieval(manager: EnhancedMemoryManager, nodes: Dict):
    """Test graph-based retrieval.
    
    Args:
        manager: Enhanced memory manager
        nodes: Dictionary of test nodes
    """
    console.print("\n[bold cyan]Testing graph-based retrieval...[/bold cyan]")
    
    # Simple query
    console.print("\n[bold]Simple query:[/bold] 'cat mascot'")
    results = await manager.search("cat mascot", k=5)
    
    # Display results
    table = Table(title="Search Results")
    table.add_column("Rank", style="cyan")
    table.add_column("Node ID", style="blue")
    table.add_column("Summary", style="green")
    table.add_column("Score", style="yellow")
    
    for i, node in enumerate(results["nodes"]):
        table.add_row(
            str(i + 1),
            node.node.id,
            node.node.summary,
            f"{node.score:.4f}"
        )
    
    console.print(table)
    
    # Graph expansion query
    console.print("\n[bold]Graph expansion query:[/bold] 'marketing campaign'")
    results = await manager.search("marketing campaign", k=5, max_hops=2)
    
    # Display results
    table = Table(title="Search Results with Graph Expansion")
    table.add_column("Rank", style="cyan")
    table.add_column("Node ID", style="blue")
    table.add_column("Summary", style="green")
    table.add_column("Score", style="yellow")
    table.add_column("Path", style="magenta")
    
    for i, node in enumerate(results["nodes"]):
        path = node.match_details.get("path") if node.match_details else "direct"
        table.add_row(
            str(i + 1),
            node.node.id,
            node.node.summary,
            f"{node.score:.4f}",
            path or "direct"
        )
    
    console.print(table)
    
    # Display closure
    console.print("\n[bold]Closure information:[/bold]")
    
    expansion_table = Table(title="Expansion Paths")
    expansion_table.add_column("Hop", style="cyan")
    expansion_table.add_column("From", style="blue")
    expansion_table.add_column("Relation", style="green")
    expansion_table.add_column("To", style="yellow")
    expansion_table.add_column("Reason", style="magenta")
    
    for path in results["closure"]["expansion_path"]:
        expansion_table.add_row(
            str(path["hop"]),
            path["from"],
            path["relation"],
            path["to"],
            path["reason"]
        )
    
    console.print(expansion_table)


async def test_relationship_modeling(manager: EnhancedMemoryManager, nodes: Dict):
    """Test relationship modeling.
    
    Args:
        manager: Enhanced memory manager
        nodes: Dictionary of test nodes
    """
    console.print("\n[bold cyan]Testing relationship modeling...[/bold cyan]")
    
    # Get cat node
    cat_node = nodes["cat_node"]
    
    # Get outgoing relationships
    outgoing = manager.entity_relationship_model.get_outgoing_relationships(cat_node.id)
    
    # Display outgoing relationships
    console.print(f"\n[bold]Outgoing relationships from {cat_node.summary}:[/bold]")
    
    table = Table()
    table.add_column("Relationship", style="cyan")
    table.add_column("Target", style="green")
    table.add_column("Strength", style="yellow")
    
    for rel in outgoing:
        target = manager.entity_relationship_model.get_node(rel.target_id)
        target_summary = target.summary if target else rel.target_id
        table.add_row(
            rel.type.value,
            target_summary,
            f"{rel.strength:.2f}"
        )
    
    console.print(table)
    
    # Get incoming relationships
    incoming = manager.entity_relationship_model.get_incoming_relationships(cat_node.id)
    
    # Display incoming relationships
    console.print(f"\n[bold]Incoming relationships to {cat_node.summary}:[/bold]")
    
    table = Table()
    table.add_column("Source", style="cyan")
    table.add_column("Relationship", style="green")
    table.add_column("Strength", style="yellow")
    
    for rel in incoming:
        source = manager.entity_relationship_model.get_node(rel.source_id)
        source_summary = source.summary if source else rel.source_id
        table.add_row(
            source_summary,
            rel.type.value,
            f"{rel.strength:.2f}"
        )
    
    console.print(table)
    
    # Get related nodes
    related = manager.entity_relationship_model.get_related_nodes(
        cat_node.id,
        max_hops=2
    )
    
    # Display related nodes
    console.print(f"\n[bold]Related nodes to {cat_node.summary} (up to 2 hops):[/bold]")
    
    table = Table()
    table.add_column("Node", style="cyan")
    table.add_column("Relationship", style="green")
    table.add_column("Hops", style="yellow")
    
    for node_id, info in related.items():
        node = info["node"]
        rel = info["relationship"]
        table.add_row(
            node.summary,
            rel.type.value,
            str(info["hop_count"])
        )
    
    console.print(table)


async def main():
    """Main function."""
    console.print(Panel(
        "[bold]AURA Enhanced Memory System Test[/bold]\n"
        "Demonstrating hierarchical memory with relationship modeling",
        border_style="cyan"
    ))
    
    # Create test directory
    test_dir = Path("/tmp/aura_memory_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize memory manager
    manager = EnhancedMemoryManager(workspace_path=test_dir)
    
    # Create test nodes
    nodes = await create_test_nodes(manager)
    
    # Test memory layers
    await test_memory_layers(manager, nodes)
    
    # Test graph retrieval
    await test_graph_retrieval(manager, nodes)
    
    # Test relationship modeling
    await test_relationship_modeling(manager, nodes)
    
    console.print("\n[bold green]Test completed successfully![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())