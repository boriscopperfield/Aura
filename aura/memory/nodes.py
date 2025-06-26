"""
Memory node model for AURA system.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import hashlib
import json
import base64
import numpy as np


class EntityType(Enum):
    """Types of entities in the memory graph."""
    
    TASK = "task"
    TASK_ARTIFACT = "task_artifact"
    USER_PREFERENCE = "user_preference"
    WORKFLOW_PATTERN = "workflow_pattern"
    KNOWLEDGE_FACT = "knowledge_fact"
    SYSTEM_INSIGHT = "system_insight"
    EXTERNAL_RESOURCE = "external_resource"


class ContentType(Enum):
    """Types of content in a memory node."""
    
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"
    STRUCTURED_DATA = "structured_data"


class RelationType(Enum):
    """Types of relations between memory nodes."""
    
    PRODUCED_BY = "produced_by"
    DEPENDS_ON = "depends_on"
    SIMILAR_TO = "similar_to"
    REFERENCES = "references"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    PART_OF = "part_of"
    PRECEDES = "precedes"
    FOLLOWS = "follows"


@dataclass
class MemorySource:
    """Source of a memory node."""
    
    type: str
    task_id: Optional[str] = None
    node_path: Optional[str] = None
    external_url: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class NamedEntity:
    """Named entity extracted from content."""
    
    type: str
    value: str
    confidence: float


@dataclass
class ContentBlock:
    """Single piece of content within a memory node."""
    
    type: ContentType
    data: Union[str, bytes]
    metadata: Dict[str, Any]
    text_description: Optional[str] = None  # For non-text content
    extracted_features: Optional[Dict[str, Any]] = None
    
    def get_hash(self) -> str:
        """Generate a hash of the content for caching."""
        if isinstance(self.data, str):
            content = self.data.encode('utf-8')
        else:
            content = self.data
            
        return hashlib.sha256(content).hexdigest()
    
    def to_embedding_input(self) -> Dict[str, Any]:
        """Convert to format for embedding API."""
        if self.type == ContentType.TEXT:
            return {"text": self.data if isinstance(self.data, str) else self.data.decode('utf-8')}
        elif self.type == ContentType.IMAGE:
            # If data is a URL, return as is
            if isinstance(self.data, str) and (self.data.startswith('http://') or self.data.startswith('https://')):
                return {"image": self.data}
            # Otherwise, assume it's bytes and encode as base64
            img_data = self.data if isinstance(self.data, bytes) else self.data.encode('utf-8')
            return {"image": base64.b64encode(img_data).decode('utf-8')}
        elif self.type == ContentType.CODE:
            # For code, use the text data with a special prefix
            code_text = self.data if isinstance(self.data, str) else self.data.decode('utf-8')
            return {"text": f"```\n{code_text}\n```"}
        else:
            # For other types, use text description if available
            if self.text_description:
                return {"text": self.text_description}
            return {"text": f"Content of type {self.type.value}"}


@dataclass
class Relation:
    """Edge in the knowledge graph."""
    
    type: RelationType
    target_id: str
    strength: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingSet:
    """Set of embeddings for a memory node."""
    
    # Chunk-level embeddings (one vector per content block)
    chunk_embeddings: Dict[str, List[float]] = field(default_factory=dict)
    
    # Token-level embeddings (sequence of vectors per content block)
    token_embeddings: Dict[str, List[List[float]]] = field(default_factory=dict)
    
    # Vector store IDs for efficient retrieval
    vector_ids: Dict[str, Any] = field(default_factory=dict)
    
    # Model information
    model_info: Dict[str, str] = field(default_factory=dict)
    
    # Legacy fields for backward compatibility
    chunk_embedding: List[float] = field(default_factory=list)
    chunk_vector_id: Optional[int] = None
    token_vector_ids: Optional[Dict[str, int]] = None
    
    def get_combined_chunk_embedding(self) -> List[float]:
        """Get a combined chunk embedding by averaging all chunk embeddings."""
        if self.chunk_embeddings:
            # Average all chunk embeddings
            embeddings = list(self.chunk_embeddings.values())
            return list(np.mean(embeddings, axis=0))
        elif self.chunk_embedding:
            # Backward compatibility
            return self.chunk_embedding
        return []


@dataclass
class MemoryNode:
    """Multimodal memory node in the knowledge graph."""
    
    id: str
    created_at: datetime
    updated_at: datetime
    entity_type: EntityType
    source: MemorySource
    
    # Multimodal content
    content: List[ContentBlock]
    
    # Semantic information
    summary: str
    keywords: List[str]
    entities: List[NamedEntity]
    
    # Graph relationships
    relations: List[Relation]
    
    # Importance and decay
    importance: float
    access_count: int
    last_accessed: datetime
    decay_rate: float
    
    # Vector embeddings
    embeddings: EmbeddingSet = field(default_factory=EmbeddingSet)
    
    def get_text_content(self) -> str:
        """Get all text content concatenated."""
        texts = []
        for block in self.content:
            if block.type == ContentType.TEXT or block.type == ContentType.CODE:
                if isinstance(block.data, str):
                    texts.append(block.data)
                else:
                    texts.append(block.data.decode('utf-8'))
            elif block.text_description:
                texts.append(block.text_description)
        return "\n".join(texts)
    
    def get_embedding_inputs(self) -> List[Dict[str, Any]]:
        """Get inputs formatted for embedding API."""
        return [block.to_embedding_input() for block in self.content]
    
    def update_access_stats(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def calculate_relevance(self, current_time: datetime) -> float:
        """Calculate relevance score based on importance, recency, and decay."""
        # Time since last access in days
        time_diff = (current_time - self.last_accessed).total_seconds() / (24 * 3600)
        
        # Decay factor based on time
        decay_factor = 1.0 / (1.0 + self.decay_rate * time_diff)
        
        # Combine importance and recency
        return self.importance * decay_factor * (1.0 + 0.1 * self.access_count)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory node to dictionary."""
        # Convert embeddings to serializable format
        embeddings_dict = {
            "chunk_vector_id": self.embeddings.chunk_vector_id,
            "token_vector_ids": self.embeddings.token_vector_ids,
            "vector_ids": self.embeddings.vector_ids,
            "model_info": self.embeddings.model_info
        }
        
        # Add chunk embeddings if they exist (convert numpy arrays to lists)
        if self.embeddings.chunk_embeddings:
            embeddings_dict["chunk_embeddings"] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.embeddings.chunk_embeddings.items()
            }
        
        # Don't include token embeddings in serialization (too large)
        
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "entity_type": self.entity_type.value,
            "source": {
                "type": self.source.type,
                "task_id": self.source.task_id,
                "node_path": self.source.node_path,
                "external_url": self.source.external_url,
                "user_id": self.source.user_id
            },
            "content": [
                {
                    "type": block.type.value,
                    "data": block.data if isinstance(block.data, str) else "<binary_data>",
                    "metadata": block.metadata,
                    "text_description": block.text_description,
                    "extracted_features": block.extracted_features
                }
                for block in self.content
            ],
            "summary": self.summary,
            "keywords": self.keywords,
            "entities": [
                {
                    "type": entity.type,
                    "value": entity.value,
                    "confidence": entity.confidence
                }
                for entity in self.entities
            ],
            "relations": [
                {
                    "type": relation.type.value,
                    "target_id": relation.target_id,
                    "strength": relation.strength,
                    "metadata": relation.metadata
                }
                for relation in self.relations
            ],
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "decay_rate": self.decay_rate,
            "embeddings": embeddings_dict
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNode':
        """Create memory node from dictionary."""
        # Convert string entity_type to enum
        entity_type = EntityType(data["entity_type"])
        
        # Create source
        source = MemorySource(
            type=data["source"]["type"],
            task_id=data["source"].get("task_id"),
            node_path=data["source"].get("node_path"),
            external_url=data["source"].get("external_url"),
            user_id=data["source"].get("user_id")
        )
        
        # Create content blocks
        content = []
        for block_data in data["content"]:
            content_type = ContentType(block_data["type"])
            content.append(ContentBlock(
                type=content_type,
                data=block_data["data"],
                metadata=block_data["metadata"],
                text_description=block_data.get("text_description"),
                extracted_features=block_data.get("extracted_features")
            ))
        
        # Create entities
        entities = [
            NamedEntity(
                type=entity_data["type"],
                value=entity_data["value"],
                confidence=entity_data["confidence"]
            )
            for entity_data in data["entities"]
        ]
        
        # Create relations
        relations = [
            Relation(
                type=RelationType(relation_data["type"]),
                target_id=relation_data["target_id"],
                strength=relation_data["strength"],
                metadata=relation_data.get("metadata", {})
            )
            for relation_data in data["relations"]
        ]
        
        # Create embeddings
        embeddings = EmbeddingSet(
            chunk_vector_id=data["embeddings"].get("chunk_vector_id"),
            token_vector_ids=data["embeddings"].get("token_vector_ids"),
            vector_ids=data["embeddings"].get("vector_ids", {}),
            model_info=data["embeddings"].get("model_info", {})
        )
        
        # Add chunk embeddings if they exist
        if "chunk_embeddings" in data["embeddings"]:
            embeddings.chunk_embeddings = data["embeddings"]["chunk_embeddings"]
        
        # Create memory node
        return cls(
            id=data["id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            entity_type=entity_type,
            source=source,
            content=content,
            summary=data["summary"],
            keywords=data["keywords"],
            entities=entities,
            relations=relations,
            importance=data["importance"],
            access_count=data["access_count"],
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            decay_rate=data["decay_rate"],
            embeddings=embeddings
        )