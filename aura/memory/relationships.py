"""
Relationship modeling for AURA memory system.

This module provides models and utilities for representing and managing
relationships between memory nodes, with support for relationship embeddings.
"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Set

from pydantic import BaseModel, Field

from aura.config import settings
from aura.utils.logging import logger
from aura.memory.models import RelationType, MemoryNode


class RelationshipMetadata(BaseModel):
    """Metadata for relationships."""
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    confidence: float = 1.0
    context: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class Relationship(BaseModel):
    """Relationship between memory nodes."""
    
    id: str
    type: RelationType
    source_id: str
    target_id: str
    strength: float = 1.0
    bidirectional: bool = False
    metadata: RelationshipMetadata = Field(default_factory=RelationshipMetadata)
    embedding: Optional[List[float]] = None
    
    @classmethod
    def create(
        cls,
        type: RelationType,
        source_id: str,
        target_id: str,
        strength: float = 1.0,
        bidirectional: bool = False,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> "Relationship":
        """Create a new relationship.
        
        Args:
            type: Relationship type
            source_id: Source node ID
            target_id: Target node ID
            strength: Relationship strength
            bidirectional: Whether the relationship is bidirectional
            context: Additional context
            tags: Tags for the relationship
            
        Returns:
            New relationship
        """
        # Generate a unique ID
        rel_id = f"rel_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create metadata
        metadata = RelationshipMetadata(
            created_at=datetime.now(),
            updated_at=datetime.now(),
            confidence=1.0,
            context=context or {},
            tags=tags or []
        )
        
        # Create relationship
        return cls(
            id=rel_id,
            type=type,
            source_id=source_id,
            target_id=target_id,
            strength=strength,
            bidirectional=bidirectional,
            metadata=metadata
        )
    
    def to_embedding_input(self) -> Dict[str, Any]:
        """Convert to format for embedding API.
        
        Returns:
            Embedding API input
        """
        # Create a text representation of the relationship
        text = f"{self.source_id} --{self.type.value}--> {self.target_id}"
        
        # Add context if available
        if self.metadata.context:
            context_str = ", ".join(f"{k}: {v}" for k, v in self.metadata.context.items())
            text += f" [Context: {context_str}]"
        
        # Add tags if available
        if self.metadata.tags:
            tags_str = ", ".join(self.metadata.tags)
            text += f" [Tags: {tags_str}]"
        
        return {"text": text}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "type": self.type.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "strength": self.strength,
            "bidirectional": self.bidirectional,
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat(),
                "confidence": self.metadata.confidence,
                "context": self.metadata.context,
                "tags": self.metadata.tags
            },
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        """Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Relationship
        """
        # Convert string type to enum
        rel_type = RelationType(data["type"])
        
        # Convert string timestamps to datetime
        metadata = RelationshipMetadata(
            created_at=datetime.fromisoformat(data["metadata"]["created_at"]),
            updated_at=datetime.fromisoformat(data["metadata"]["updated_at"]),
            confidence=data["metadata"]["confidence"],
            context=data["metadata"]["context"],
            tags=data["metadata"]["tags"]
        )
        
        # Create relationship
        return cls(
            id=data["id"],
            type=rel_type,
            source_id=data["source_id"],
            target_id=data["target_id"],
            strength=data["strength"],
            bidirectional=data["bidirectional"],
            metadata=metadata,
            embedding=data.get("embedding")
        )


class EntityRelationshipModel:
    """Model for entities and their relationships."""
    
    def __init__(self):
        """Initialize the entity-relationship model."""
        self.nodes: Dict[str, MemoryNode] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.outgoing: Dict[str, Set[str]] = {}  # source_id -> relationship_ids
        self.incoming: Dict[str, Set[str]] = {}  # target_id -> relationship_ids
        self.logger = logger
    
    def add_node(self, node: MemoryNode) -> None:
        """Add a node to the model.
        
        Args:
            node: Memory node
        """
        self.nodes[node.id] = node
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the model.
        
        Args:
            relationship: Relationship
        """
        self.relationships[relationship.id] = relationship
        
        # Update indexes
        source_id = relationship.source_id
        if source_id not in self.outgoing:
            self.outgoing[source_id] = set()
        self.outgoing[source_id].add(relationship.id)
        
        target_id = relationship.target_id
        if target_id not in self.incoming:
            self.incoming[target_id] = set()
        self.incoming[target_id].add(relationship.id)
        
        # If bidirectional, add reverse indexes
        if relationship.bidirectional:
            if target_id not in self.outgoing:
                self.outgoing[target_id] = set()
            self.outgoing[target_id].add(relationship.id)
            
            if source_id not in self.incoming:
                self.incoming[source_id] = set()
            self.incoming[source_id].add(relationship.id)
    
    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a node by ID.
        
        Args:
            node_id: Node ID
            
        Returns:
            Memory node or None if not found
        """
        return self.nodes.get(node_id)
    
    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Get a relationship by ID.
        
        Args:
            relationship_id: Relationship ID
            
        Returns:
            Relationship or None if not found
        """
        return self.relationships.get(relationship_id)
    
    def get_outgoing_relationships(self, node_id: str) -> List[Relationship]:
        """Get outgoing relationships for a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            List of outgoing relationships
        """
        if node_id not in self.outgoing:
            return []
        
        return [
            self.relationships[rel_id]
            for rel_id in self.outgoing[node_id]
            if rel_id in self.relationships
        ]
    
    def get_incoming_relationships(self, node_id: str) -> List[Relationship]:
        """Get incoming relationships for a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            List of incoming relationships
        """
        if node_id not in self.incoming:
            return []
        
        return [
            self.relationships[rel_id]
            for rel_id in self.incoming[node_id]
            if rel_id in self.relationships
        ]
    
    def get_related_nodes(
        self,
        node_id: str,
        relation_types: Optional[List[RelationType]] = None,
        max_hops: int = 1,
        min_strength: float = 0.0
    ) -> Dict[str, Dict[str, Any]]:
        """Get related nodes for a node.
        
        Args:
            node_id: Node ID
            relation_types: Optional list of relation types to filter by
            max_hops: Maximum number of hops
            min_strength: Minimum relationship strength
            
        Returns:
            Dictionary of related nodes with relationship info
        """
        if node_id not in self.nodes:
            return {}
        
        # Initialize result
        result: Dict[str, Dict[str, Any]] = {}
        
        # Initialize visited set and queue
        visited = {node_id}
        queue = [(node_id, 0, None)]  # (node_id, hop_count, relationship)
        
        # BFS traversal
        while queue:
            current_id, hop_count, relationship = queue.pop(0)
            
            # Skip if we've reached max hops
            if hop_count >= max_hops:
                continue
            
            # Get outgoing relationships
            outgoing = self.get_outgoing_relationships(current_id)
            
            for rel in outgoing:
                # Skip if relationship type doesn't match filter
                if relation_types and rel.type not in relation_types:
                    continue
                
                # Skip if relationship strength is too low
                if rel.strength < min_strength:
                    continue
                
                target_id = rel.target_id
                
                # Skip if we've already visited this node
                if target_id in visited:
                    continue
                
                # Add to result
                if target_id in self.nodes:
                    result[target_id] = {
                        "node": self.nodes[target_id],
                        "relationship": rel,
                        "hop_count": hop_count + 1,
                        "path": [current_id, target_id]
                    }
                
                # Mark as visited
                visited.add(target_id)
                
                # Add to queue for next hop
                queue.append((target_id, hop_count + 1, rel))
        
        return result
    
    def encode_entity_with_relations(
        self,
        node_id: str,
        embedder: Any
    ) -> Dict[str, Any]:
        """Encode an entity with its relationships.
        
        Args:
            node_id: Node ID
            embedder: Embedding service
            
        Returns:
            Dictionary with entity and relationship embeddings
        """
        node = self.get_node(node_id)
        if not node:
            self.logger.warning(f"Node {node_id} not found")
            return {}
        
        # Get relationships
        outgoing = self.get_outgoing_relationships(node_id)
        incoming = self.get_incoming_relationships(node_id)
        
        # Create context for entity embedding
        entity_context = node.get_text_content()
        
        # Add relationship context
        for rel in outgoing:
            target = self.get_node(rel.target_id)
            if target:
                entity_context += f"\nRelated to: {target.summary} via {rel.type.value}"
        
        for rel in incoming:
            source = self.get_node(rel.source_id)
            if source:
                entity_context += f"\nReferenced by: {source.summary} via {rel.type.value}"
        
        # Encode entity with context
        import asyncio
        entity_embedding = asyncio.run(embedder.embed_text(entity_context))
        
        # Encode relationships
        relation_embeddings = {}
        for rel in outgoing + incoming:
            rel_input = rel.to_embedding_input()
            rel_embedding = asyncio.run(embedder.embed_text(rel_input["text"]))
            relation_embeddings[rel.id] = rel_embedding
            
            # Update relationship with embedding
            rel.embedding = rel_embedding
        
        return {
            "entity_id": node_id,
            "entity_embedding": entity_embedding,
            "relation_embeddings": relation_embeddings
        }


# Global instance
entity_relationship_model = EntityRelationshipModel()