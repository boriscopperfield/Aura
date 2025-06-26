"""
Unit tests for memory models.
"""
import pytest
from datetime import datetime
from aura.memory.models import (
    MemoryNode, ContentBlock, ContentType, EntityType,
    MemorySource, NamedEntity, Relation, RelationType
)


class TestMemoryNode:
    """Test MemoryNode functionality."""
    
    def test_memory_node_creation(self):
        """Test basic memory node creation."""
        source = MemorySource(type="test", user_id="test_user")
        
        node = MemoryNode(
            id="test_node",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            entity_type=EntityType.TASK_ARTIFACT,
            source=source,
            content=[
                ContentBlock(
                    type=ContentType.TEXT,
                    data="Test content",
                    metadata={"language": "en"}
                )
            ],
            summary="Test node",
            keywords=["test"],
            entities=[],
            relations=[],
            importance=0.5,
            access_count=0,
            last_accessed=datetime.now(),
            decay_rate=0.01
        )
        
        assert node.id == "test_node"
        assert node.entity_type == EntityType.TASK_ARTIFACT
        assert len(node.content) == 1
        assert node.content[0].data == "Test content"
        assert node.importance == 0.5
    
    def test_memory_node_serialization(self):
        """Test memory node serialization."""
        source = MemorySource(type="test", user_id="test_user")
        
        node = MemoryNode(
            id="test_node",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            entity_type=EntityType.TASK_ARTIFACT,
            source=source,
            content=[
                ContentBlock(
                    type=ContentType.TEXT,
                    data="Test content",
                    metadata={"language": "en"}
                )
            ],
            summary="Test node",
            keywords=["test"],
            entities=[],
            relations=[],
            importance=0.5,
            access_count=0,
            last_accessed=datetime.now(),
            decay_rate=0.01
        )
        
        # Test serialization
        serialized = node.model_dump()
        assert isinstance(serialized, dict)
        assert serialized["id"] == "test_node"
        assert serialized["entity_type"] == "task_artifact"
    
    def test_text_content_extraction(self):
        """Test extracting text content from memory node."""
        source = MemorySource(type="test", user_id="test_user")
        
        node = MemoryNode(
            id="test_node",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            entity_type=EntityType.TASK_ARTIFACT,
            source=source,
            content=[
                ContentBlock(
                    type=ContentType.TEXT,
                    data="First text block",
                    metadata={"language": "en"}
                ),
                ContentBlock(
                    type=ContentType.TEXT,
                    data="Second text block",
                    metadata={"language": "en"}
                )
            ],
            summary="Test node",
            keywords=["test"],
            entities=[],
            relations=[],
            importance=0.5,
            access_count=0,
            last_accessed=datetime.now(),
            decay_rate=0.01
        )
        
        text_content = node.get_text_content()
        assert "First text block" in text_content
        assert "Second text block" in text_content
    
    def test_access_stats_update(self):
        """Test updating access statistics."""
        source = MemorySource(type="test", user_id="test_user")
        
        node = MemoryNode(
            id="test_node",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            entity_type=EntityType.TASK_ARTIFACT,
            source=source,
            content=[
                ContentBlock(
                    type=ContentType.TEXT,
                    data="Test content",
                    metadata={"language": "en"}
                )
            ],
            summary="Test node",
            keywords=["test"],
            entities=[],
            relations=[],
            importance=0.5,
            access_count=0,
            last_accessed=datetime.now(),
            decay_rate=0.01
        )
        
        initial_count = node.access_count
        initial_time = node.last_accessed
        
        node.update_access_stats()
        
        assert node.access_count == initial_count + 1
        assert node.last_accessed > initial_time
    
    def test_content_hash_calculation(self):
        """Test content hash calculation."""
        source = MemorySource(type="test", user_id="test_user")
        
        node = MemoryNode(
            id="test_node",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            entity_type=EntityType.TASK_ARTIFACT,
            source=source,
            content=[
                ContentBlock(
                    type=ContentType.TEXT,
                    data="Test content",
                    metadata={"language": "en"}
                )
            ],
            summary="Test node",
            keywords=["test"],
            entities=[],
            relations=[],
            importance=0.5,
            access_count=0,
            last_accessed=datetime.now(),
            decay_rate=0.01
        )
        
        hash1 = node.get_content_hash()
        hash2 = node.get_content_hash()
        
        # Same content should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length


class TestContentBlock:
    """Test ContentBlock functionality."""
    
    def test_content_block_creation(self):
        """Test content block creation."""
        block = ContentBlock(
            type=ContentType.TEXT,
            data="Test content",
            metadata={"language": "en", "tokens": 2}
        )
        
        assert block.type == ContentType.TEXT
        assert block.data == "Test content"
        assert block.metadata["language"] == "en"
        assert block.metadata["tokens"] == 2
    
    def test_content_block_types(self):
        """Test different content block types."""
        text_block = ContentBlock(
            type=ContentType.TEXT,
            data="Text content",
            metadata={}
        )
        
        image_block = ContentBlock(
            type=ContentType.IMAGE,
            data="base64_image_data",
            metadata={"width": 1024, "height": 768}
        )
        
        code_block = ContentBlock(
            type=ContentType.CODE,
            data="print('hello world')",
            metadata={"language": "python"}
        )
        
        assert text_block.type == ContentType.TEXT
        assert image_block.type == ContentType.IMAGE
        assert code_block.type == ContentType.CODE


class TestRelation:
    """Test Relation functionality."""
    
    def test_relation_creation(self):
        """Test relation creation."""
        relation = Relation(
            type=RelationType.SIMILAR_TO,
            target_id="target_node",
            strength=0.8
        )
        
        assert relation.type == RelationType.SIMILAR_TO
        assert relation.target_id == "target_node"
        assert relation.strength == 0.8
    
    def test_relation_types(self):
        """Test different relation types."""
        similar_relation = Relation(
            type=RelationType.SIMILAR_TO,
            target_id="node1",
            strength=0.9
        )
        
        part_of_relation = Relation(
            type=RelationType.PART_OF,
            target_id="node2",
            strength=1.0
        )
        
        references_relation = Relation(
            type=RelationType.REFERENCES,
            target_id="node3",
            strength=0.7
        )
        
        assert similar_relation.type == RelationType.SIMILAR_TO
        assert part_of_relation.type == RelationType.PART_OF
        assert references_relation.type == RelationType.REFERENCES


class TestNamedEntity:
    """Test NamedEntity functionality."""
    
    def test_named_entity_creation(self):
        """Test named entity creation."""
        entity = NamedEntity(
            type="person",
            value="John Doe",
            confidence=0.95
        )
        
        assert entity.type == "person"
        assert entity.value == "John Doe"
        assert entity.confidence == 0.95
    
    def test_entity_types(self):
        """Test different entity types."""
        person = NamedEntity(type="person", value="Alice", confidence=0.9)
        organization = NamedEntity(type="organization", value="ACME Corp", confidence=0.85)
        concept = NamedEntity(type="concept", value="artificial intelligence", confidence=0.95)
        
        assert person.type == "person"
        assert organization.type == "organization"
        assert concept.type == "concept"


class TestMemorySource:
    """Test MemorySource functionality."""
    
    def test_memory_source_creation(self):
        """Test memory source creation."""
        source = MemorySource(
            type="user_input",
            user_id="user123",
            session_id="session456"
        )
        
        assert source.type == "user_input"
        assert source.user_id == "user123"
        assert source.session_id == "session456"
    
    def test_different_source_types(self):
        """Test different source types."""
        user_source = MemorySource(type="user_input", user_id="user1")
        system_source = MemorySource(type="system", user_id="system")
        agent_source = MemorySource(type="agent", user_id="agent1")
        
        assert user_source.type == "user_input"
        assert system_source.type == "system"
        assert agent_source.type == "agent"