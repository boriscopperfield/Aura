#!/usr/bin/env python3
"""
Test script for the memory models.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aura.memory.models import (
    MemoryNode, ContentBlock, ContentType, EntityType, 
    MemorySource, NamedEntity, Relation, RelationType
)


console = Console()


def test_memory_node():
    """Test memory node creation and serialization."""
    console.print("\n[bold cyan]Testing Memory Node Creation:[/bold cyan]")
    
    # Create source
    source = MemorySource(
        type="test",
        user_id="test_user"
    )
    
    # Create content blocks
    content_blocks = [
        ContentBlock(
            type=ContentType.TEXT,
            data="This is a test memory node",
            metadata={"language": "en"}
        ),
        ContentBlock(
            type=ContentType.TEXT,
            data="It contains multiple content blocks",
            metadata={"language": "en"}
        )
    ]
    
    # Create named entities
    entities = [
        NamedEntity(type="concept", value="memory", confidence=0.9),
        NamedEntity(type="concept", value="test", confidence=0.95)
    ]
    
    # Create memory node
    node = MemoryNode(
        id="mem_test_001",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        entity_type=EntityType.TASK_ARTIFACT,
        source=source,
        content=content_blocks,
        summary="Test memory node",
        keywords=["test", "memory", "node"],
        entities=entities,
        relations=[],  # Empty relations list
        importance=0.7,
        access_count=0,
        last_accessed=datetime.now(),
        decay_rate=0.01  # Default decay rate
    )
    
    # Display node
    console.print(f"Created node: [green]{node.id}[/green]")
    console.print(f"Entity type: {node.entity_type}")
    console.print(f"Summary: {node.summary}")
    console.print(f"Keywords: {', '.join(node.keywords)}")
    console.print(f"Importance: {node.importance}")
    
    # Test serialization
    console.print("\n[bold]Testing Serialization:[/bold]")
    node_dict = node.model_dump()
    console.print(f"Serialized to dictionary with {len(node_dict)} fields")
    
    # Test text content extraction
    console.print("\n[bold]Testing Text Content Extraction:[/bold]")
    text_content = node.get_text_content()
    console.print(f"Extracted text content ({len(text_content)} characters):")
    console.print(text_content)
    
    # Test content hash
    console.print("\n[bold]Testing Content Hash:[/bold]")
    # Calculate hash manually
    import hashlib
    content_hash = hashlib.md5(node.get_text_content().encode()).hexdigest()
    console.print(f"Content hash: {content_hash}")
    
    # Test access stats
    console.print("\n[bold]Testing Access Stats:[/bold]")
    console.print(f"Initial access count: {node.access_count}")
    node.update_access_stats()
    console.print(f"Updated access count: {node.access_count}")
    
    return node


def test_relations(node):
    """Test relations between memory nodes."""
    console.print("\n[bold cyan]Testing Relations:[/bold cyan]")
    
    # Create another node
    source = MemorySource(
        type="test",
        user_id="test_user"
    )
    
    related_node = MemoryNode(
        id="mem_test_002",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        entity_type=EntityType.TASK_ARTIFACT,
        source=source,
        content=[
            ContentBlock(
                type=ContentType.TEXT,
                data="This is a related memory node",
                metadata={"language": "en"}
            )
        ],
        summary="Related test node",
        keywords=["related", "test", "node"],
        entities=[],  # Empty entities list
        relations=[],  # Empty relations list
        importance=0.6,
        access_count=0,
        last_accessed=datetime.now(),
        decay_rate=0.01  # Default decay rate
    )
    
    console.print(f"Created related node: [green]{related_node.id}[/green]")
    
    # Create relation
    relation = Relation(
        type=RelationType.SIMILAR_TO,
        target_id=related_node.id,
        strength=0.8
    )
    
    # Add relation to node
    node.relations.append(relation)
    
    # Display relations
    console.print("\n[bold]Node Relations:[/bold]")
    
    table = Table()
    table.add_column("Relation Type", style="cyan")
    table.add_column("Target ID", style="green")
    table.add_column("Strength", style="yellow")
    
    for rel in node.relations:
        table.add_row(
            rel.type.value,
            rel.target_id,
            f"{rel.strength:.2f}"
        )
    
    console.print(table)
    
    return related_node


def main():
    """Main function."""
    console.print(Panel(
        "[bold]AURA Memory Models Test[/bold]\n"
        "Testing memory node models and relations",
        border_style="cyan"
    ))
    
    # Test memory node
    node = test_memory_node()
    
    # Test relations
    related_node = test_relations(node)
    
    console.print("\n[bold green]Test completed successfully![/bold green]")


if __name__ == "__main__":
    main()