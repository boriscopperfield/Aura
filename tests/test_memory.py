"""
Tests for the memory model.
"""
from datetime import datetime

import pytest

from aura.memory.nodes import (
    ContentBlock,
    ContentType,
    EmbeddingSet,
    EntityType,
    MemoryNode,
    MemorySource,
    NamedEntity,
    Relation,
    RelationType,
)


def test_memory_node_serialization():
    """Test memory node serialization to dictionary."""
    # Create a sample memory node
    node = MemoryNode(
        id="mem_20250626_001",
        created_at=datetime.fromisoformat("2025-06-26T11:05:15+00:00"),
        updated_at=datetime.fromisoformat("2025-06-26T11:30:22+00:00"),
        entity_type=EntityType.TASK_ARTIFACT,
        source=MemorySource(
            type="task_execution",
            task_id="task_cyber_monday_banner",
            node_path="/generate_assets/generate_mascot"
        ),
        content=[
            ContentBlock(
                type=ContentType.TEXT,
                data="A friendly robot mascot for Cyber Monday tech sale",
                metadata={"language": "en", "tokens": 10}
            ),
            ContentBlock(
                type=ContentType.IMAGE,
                data=b"image_data",  # Binary data in a real system
                metadata={"width": 1024, "height": 1024, "format": "png"},
                text_description="Blue and white robot with shopping cart"
            )
        ],
        summary="Robot mascot design for Cyber Monday campaign",
        keywords=["robot", "mascot", "cyber_monday", "marketing", "tech_sale"],
        entities=[
            NamedEntity(type="brand", value="TechCorp", confidence=0.95),
            NamedEntity(type="event", value="Cyber Monday 2025", confidence=0.99)
        ],
        relations=[
            Relation(
                type=RelationType.PRODUCED_BY,
                target_id="task_generate_mascot",
                strength=1.0
            ),
            Relation(
                type=RelationType.SIMILAR_TO,
                target_id="mem_previous_mascot",
                strength=0.75,
                metadata={"similarity_type": "style"}
            )
        ],
        importance=0.85,
        access_count=12,
        last_accessed=datetime.fromisoformat("2025-06-26T11:30:22+00:00"),
        decay_rate=0.01,
        embeddings=EmbeddingSet(
            chunk_vector_id=1024,
            token_vector_ids={"start": 50100, "end": 50185}
        )
    )
    
    # Serialize to dictionary
    data = node.to_dict()
    
    # Check fields
    assert data["id"] == "mem_20250626_001"
    assert data["created_at"] == "2025-06-26T11:05:15+00:00"
    assert data["entity_type"] == "task_artifact"
    assert data["source"]["type"] == "task_execution"
    assert data["source"]["task_id"] == "task_cyber_monday_banner"
    assert len(data["content"]) == 2
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["data"] == "A friendly robot mascot for Cyber Monday tech sale"
    assert data["content"][1]["type"] == "image"
    assert data["content"][1]["data"] == "<binary_data>"  # Binary data is replaced
    assert data["summary"] == "Robot mascot design for Cyber Monday campaign"
    assert "robot" in data["keywords"]
    assert len(data["entities"]) == 2
    assert data["entities"][0]["type"] == "brand"
    assert len(data["relations"]) == 2
    assert data["relations"][0]["type"] == "produced_by"
    assert data["importance"] == 0.85
    assert data["embeddings"]["chunk_vector_id"] == 1024


def test_memory_node_deserialization():
    """Test memory node deserialization from dictionary."""
    # Create a dictionary
    data = {
        "id": "mem_20250626_001",
        "created_at": "2025-06-26T11:05:15+00:00",
        "updated_at": "2025-06-26T11:30:22+00:00",
        "entity_type": "task_artifact",
        "source": {
            "type": "task_execution",
            "task_id": "task_cyber_monday_banner",
            "node_path": "/generate_assets/generate_mascot",
            "external_url": None,
            "user_id": None
        },
        "content": [
            {
                "type": "text",
                "data": "A friendly robot mascot for Cyber Monday tech sale",
                "metadata": {"language": "en", "tokens": 10},
                "text_description": None,
                "extracted_features": None
            },
            {
                "type": "image",
                "data": "image_data_placeholder",  # String representation in test
                "metadata": {"width": 1024, "height": 1024, "format": "png"},
                "text_description": "Blue and white robot with shopping cart",
                "extracted_features": None
            }
        ],
        "summary": "Robot mascot design for Cyber Monday campaign",
        "keywords": ["robot", "mascot", "cyber_monday", "marketing", "tech_sale"],
        "entities": [
            {"type": "brand", "value": "TechCorp", "confidence": 0.95},
            {"type": "event", "value": "Cyber Monday 2025", "confidence": 0.99}
        ],
        "relations": [
            {
                "type": "produced_by",
                "target_id": "task_generate_mascot",
                "strength": 1.0,
                "metadata": {}
            },
            {
                "type": "similar_to",
                "target_id": "mem_previous_mascot",
                "strength": 0.75,
                "metadata": {"similarity_type": "style"}
            }
        ],
        "importance": 0.85,
        "access_count": 12,
        "last_accessed": "2025-06-26T11:30:22+00:00",
        "decay_rate": 0.01,
        "embeddings": {
            "chunk_vector_id": 1024,
            "token_vector_ids": {"start": 50100, "end": 50185}
        }
    }
    
    # Deserialize from dictionary
    node = MemoryNode.from_dict(data)
    
    # Check fields
    assert node.id == "mem_20250626_001"
    assert node.created_at.isoformat() == "2025-06-26T11:05:15+00:00"
    assert node.entity_type == EntityType.TASK_ARTIFACT
    assert node.source.type == "task_execution"
    assert node.source.task_id == "task_cyber_monday_banner"
    assert len(node.content) == 2
    assert node.content[0].type == ContentType.TEXT
    assert node.content[0].data == "A friendly robot mascot for Cyber Monday tech sale"
    assert node.content[1].type == ContentType.IMAGE
    assert node.content[1].data == "image_data_placeholder"
    assert node.summary == "Robot mascot design for Cyber Monday campaign"
    assert "robot" in node.keywords
    assert len(node.entities) == 2
    assert node.entities[0].type == "brand"
    assert len(node.relations) == 2
    assert node.relations[0].type == RelationType.PRODUCED_BY
    assert node.importance == 0.85
    assert node.embeddings.chunk_vector_id == 1024