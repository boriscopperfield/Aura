"""
Memory models for AURA system.

This module defines the data models for the memory system.
"""
import hashlib
import base64
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, ClassVar

import numpy as np
from pydantic import BaseModel, Field, validator

from aura.utils.serialization import json_dumps, json_loads


class EntityType(str, Enum):
    """Types of entities in the memory graph."""
    
    TASK = "task"
    TASK_ARTIFACT = "task_artifact"
    USER_PREFERENCE = "user_preference"
    WORKFLOW_PATTERN = "workflow_pattern"
    KNOWLEDGE_FACT = "knowledge_fact"
    SYSTEM_INSIGHT = "system_insight"
    EXTERNAL_RESOURCE = "external_resource"


class ContentType(str, Enum):
    """Types of content in a memory node."""
    
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"
    STRUCTURED_DATA = "structured_data"


class RelationType(str, Enum):
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


class MemorySource(BaseModel):
    """Source of a memory node."""
    
    type: str
    task_id: Optional[str] = None
    node_path: Optional[str] = None
    external_url: Optional[str] = None
    user_id: Optional[str] = None


class NamedEntity(BaseModel):
    """Named entity extracted from content."""
    
    type: str
    value: str
    confidence: float


class ContentBlock(BaseModel):
    """Single piece of content within a memory node."""
    
    type: ContentType
    data: Union[str, bytes]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    text_description: Optional[str] = None  # For non-text content
    extracted_features: Optional[Dict[str, Any]] = None
    
    def get_hash(self) -> str:
        """Generate a hash of the content for caching.
        
        Returns:
            Content hash
        """
        if isinstance(self.data, str):
            content = self.data.encode('utf-8')
        else:
            content = self.data
            
        return hashlib.sha256(content).hexdigest()
    
    def to_embedding_input(self) -> Dict[str, Any]:
        """Convert to format for embedding API.
        
        Returns:
            Embedding API input
        """
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


class Relation(BaseModel):
    """Edge in the knowledge graph."""
    
    type: RelationType
    target_id: str
    strength: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingSet(BaseModel):
    """Set of embeddings for a memory node."""
    
    # Chunk-level embeddings (one vector per content block)
    chunk_embeddings: Dict[str, List[float]] = Field(default_factory=dict)
    
    # Token-level embeddings (sequence of vectors per content block)
    token_embeddings: Dict[str, List[List[float]]] = Field(default_factory=dict)
    
    # Vector store IDs for efficient retrieval
    vector_ids: Dict[str, Any] = Field(default_factory=dict)
    
    # Model information
    model_info: Dict[str, str] = Field(default_factory=dict)
    
    # Legacy fields for backward compatibility
    chunk_embedding: List[float] = Field(default_factory=list)
    chunk_vector_id: Optional[int] = None
    token_vector_ids: Optional[Dict[str, int]] = None
    
    def get_combined_chunk_embedding(self) -> List[float]:
        """Get a combined chunk embedding by averaging all chunk embeddings.
        
        Returns:
            Combined embedding vector
        """
        if self.chunk_embeddings:
            # Average all chunk embeddings
            embeddings = list(self.chunk_embeddings.values())
            return list(np.mean(embeddings, axis=0))
        elif self.chunk_embedding:
            # Backward compatibility
            return self.chunk_embedding
        return []


class MemoryNode(BaseModel):
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
    embeddings: EmbeddingSet = Field(default_factory=EmbeddingSet)
    
    def get_text_content(self) -> str:
        """Get all text content concatenated.
        
        Returns:
            Concatenated text content
        """
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
        """Get inputs formatted for embedding API.
        
        Returns:
            List of embedding API inputs
        """
        return [block.to_embedding_input() for block in self.content]
    
    def update_access_stats(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def calculate_relevance(self, current_time: datetime) -> float:
        """Calculate relevance score based on importance, recency, and decay.
        
        Args:
            current_time: Current time
            
        Returns:
            Relevance score
        """
        # Time since last access in days
        time_diff = (current_time - self.last_accessed).total_seconds() / (24 * 3600)
        
        # Decay factor based on time
        decay_factor = 1.0 / (1.0 + self.decay_rate * time_diff)
        
        # Combine importance and recency
        return self.importance * decay_factor * (1.0 + 0.1 * self.access_count)


class Query(BaseModel):
    """Query for memory retrieval."""
    
    text: Optional[str] = None
    image: Optional[Union[str, bytes]] = None
    embedding: Optional[List[float]] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    
    @validator("image")
    def validate_image(cls, v: Optional[Union[str, bytes]]) -> Optional[Union[str, bytes]]:
        """Validate image data."""
        if v is None:
            return v
            
        # If it's a URL, return as is
        if isinstance(v, str) and (v.startswith('http://') or v.startswith('https://')):
            return v
            
        # If it's bytes, return as is
        if isinstance(v, bytes):
            return v
            
        # If it's a base64 string, convert to bytes
        if isinstance(v, str) and v.startswith('data:image/'):
            # Extract base64 data from data URL
            base64_data = v.split(',')[1]
            return base64.b64decode(base64_data)
            
        # If it's a plain base64 string
        if isinstance(v, str):
            try:
                return base64.b64decode(v)
            except:
                pass
                
        raise ValueError("Invalid image format. Must be URL, bytes, or base64 string.")


class ScoredMemoryNode(BaseModel):
    """Memory node with relevance score."""
    
    node: MemoryNode
    score: float
    match_details: Optional[Dict[str, Any]] = None