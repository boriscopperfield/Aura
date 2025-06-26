"""
Memory manager for AURA system.
"""
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import asyncio

from rich.console import Console

from aura.memory.nodes import (
    MemoryNode, ContentBlock, ContentType, EntityType, 
    MemorySource, NamedEntity, Relation, RelationType
)
from aura.memory.embeddings import JinaEmbedder, JinaReranker, EmbeddingCache
from aura.memory.retrieval import MemoryRetriever, Query, ScoredMemoryNode

console = Console()


class MemoryManager:
    """Central manager for AURA's memory system."""
    
    def __init__(self, workspace_path: Path):
        """Initialize the memory manager.
        
        Args:
            workspace_path: Path to the AURA workspace.
        """
        self.workspace_path = workspace_path
        self.memory_dir = workspace_path / "memory"
        
        # Create memory directory if it doesn't exist
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding services
        self.embedder = JinaEmbedder()
        self.reranker = JinaReranker()
        self.cache = EmbeddingCache(self.memory_dir / "cache")
        
        # Initialize retriever
        self.retriever = MemoryRetriever(
            memory_dir=self.memory_dir,
            embedder=self.embedder,
            reranker=self.reranker,
            cache=self.cache
        )
    
    async def create_memory_node(self, 
                                entity_type: Union[EntityType, str],
                                source: MemorySource,
                                content: List[Union[ContentBlock, Dict[str, Any]]],
                                summary: str,
                                keywords: List[str] = None,
                                entities: List[NamedEntity] = None,
                                relations: List[Relation] = None,
                                importance: float = 0.5) -> MemoryNode:
        """Create a new memory node.
        
        Args:
            entity_type: Type of entity.
            source: Source information.
            content: List of content blocks or dictionaries.
            summary: Summary of the node.
            keywords: List of keywords.
            entities: List of named entities.
            relations: List of relations to other nodes.
            importance: Importance score (0.0 to 1.0).
            
        Returns:
            Created memory node.
        """
        # Convert string entity type to enum if needed
        if isinstance(entity_type, str):
            entity_type = EntityType(entity_type)
        
        # Convert content dictionaries to ContentBlock objects
        content_blocks = []
        for item in content:
            if isinstance(item, ContentBlock):
                content_blocks.append(item)
            elif isinstance(item, dict):
                content_type = item.get("type")
                if isinstance(content_type, str):
                    content_type = ContentType(content_type)
                
                content_blocks.append(ContentBlock(
                    type=content_type,
                    data=item.get("data", ""),
                    metadata=item.get("metadata", {}),
                    text_description=item.get("text_description"),
                    extracted_features=item.get("extracted_features")
                ))
        
        # Create node ID
        node_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create memory node
        node = MemoryNode(
            id=node_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            entity_type=entity_type,
            source=source,
            content=content_blocks,
            summary=summary,
            keywords=keywords or [],
            entities=entities or [],
            relations=relations or [],
            importance=importance,
            access_count=0,
            last_accessed=datetime.now(),
            decay_rate=0.01
        )
        
        # Generate embeddings
        node.embeddings = await self.retriever._generate_embeddings(node)
        
        # Add to retriever
        await self.retriever.add_node(node)
        
        # Add relations
        for relation in node.relations:
            await self.retriever.add_relation(
                source_id=node.id,
                target_id=relation.target_id,
                relation_type=relation.type.value,
                strength=relation.strength
            )
        
        return node
    
    async def create_text_node(self, 
                              text: str,
                              entity_type: Union[EntityType, str],
                              source: MemorySource,
                              summary: Optional[str] = None,
                              keywords: List[str] = None,
                              importance: float = 0.5) -> MemoryNode:
        """Create a memory node from text.
        
        Args:
            text: Text content.
            entity_type: Type of entity.
            source: Source information.
            summary: Summary of the node (defaults to truncated text).
            keywords: List of keywords.
            importance: Importance score (0.0 to 1.0).
            
        Returns:
            Created memory node.
        """
        # Create content block
        content = [ContentBlock(
            type=ContentType.TEXT,
            data=text,
            metadata={"length": len(text)},
            text_description=None
        )]
        
        # Use truncated text as summary if not provided
        if summary is None:
            summary = text[:100] + "..." if len(text) > 100 else text
        
        # Create node
        return await self.create_memory_node(
            entity_type=entity_type,
            source=source,
            content=content,
            summary=summary,
            keywords=keywords or [],
            importance=importance
        )
    
    async def create_image_node(self,
                               image: Union[str, Path, bytes],
                               text_description: str,
                               entity_type: Union[EntityType, str],
                               source: MemorySource,
                               summary: Optional[str] = None,
                               keywords: List[str] = None,
                               importance: float = 0.5) -> MemoryNode:
        """Create a memory node from an image.
        
        Args:
            image: Image as URL, file path, or bytes.
            text_description: Text description of the image.
            entity_type: Type of entity.
            source: Source information.
            summary: Summary of the node (defaults to text description).
            keywords: List of keywords.
            importance: Importance score (0.0 to 1.0).
            
        Returns:
            Created memory node.
        """
        # Process image based on type
        if isinstance(image, str) and (image.startswith('http://') or image.startswith('https://')):
            # URL - keep as is
            image_data = image
            metadata = {"source": "url", "url": image}
        elif isinstance(image, (str, Path)):
            # File path - read bytes
            with open(image, 'rb') as f:
                image_data = f.read()
            metadata = {"source": "file", "filename": str(image)}
        else:
            # Raw bytes
            image_data = image
            metadata = {"source": "bytes", "size": len(image_data)}
        
        # Create content block
        content = [ContentBlock(
            type=ContentType.IMAGE,
            data=image_data,
            metadata=metadata,
            text_description=text_description
        )]
        
        # Use text description as summary if not provided
        if summary is None:
            summary = text_description
        
        # Create node
        return await self.create_memory_node(
            entity_type=entity_type,
            source=source,
            content=content,
            summary=summary,
            keywords=keywords or [],
            importance=importance
        )
    
    async def query(self, 
                   query_text: str,
                   filters: Dict[str, Any] = None,
                   k: int = 10,
                   rerank: bool = True,
                   expand_context: bool = True) -> List[ScoredMemoryNode]:
        """Query the memory system.
        
        Args:
            query_text: Query text.
            filters: Filters to apply.
            k: Number of results to return.
            rerank: Whether to rerank results.
            expand_context: Whether to expand context through graph relationships.
            
        Returns:
            List of scored memory nodes.
        """
        query = Query(
            text=query_text,
            filters=filters or {},
            k=k,
            rerank=rerank,
            expand_context=expand_context
        )
        
        return await self.retriever.retrieve(query)
    
    async def add_relation(self, 
                          source_id: str, 
                          target_id: str, 
                          relation_type: Union[RelationType, str],
                          strength: float = 1.0) -> None:
        """Add a relation between two nodes.
        
        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            relation_type: Type of relation.
            strength: Strength of the relation.
        """
        # Convert string relation type to enum value if needed
        if isinstance(relation_type, RelationType):
            relation_type = relation_type.value
        
        await self.retriever.add_relation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            strength=strength
        )
    
    def save(self) -> None:
        """Save memory state to disk."""
        self.retriever.save_indexes()
    
    async def learn_from_event(self, event: Dict[str, Any]) -> Optional[MemoryNode]:
        """Learn from an event and create a memory node if relevant.
        
        Args:
            event: Event dictionary.
            
        Returns:
            Created memory node, if any.
        """
        # Skip if not a relevant event type
        event_type = event.get("type")
        if not event_type:
            return None
        
        # Create source
        source = MemorySource(
            type="event",
            task_id=event.get("metadata", {}).get("task_id"),
            user_id=event.get("metadata", {}).get("user_id")
        )
        
        # Process based on event type
        if event_type.startswith("task."):
            # Task event
            return await self._process_task_event(event, source)
        elif event_type.startswith("preference."):
            # Preference event
            return await self._process_preference_event(event, source)
        elif event_type.startswith("behavior."):
            # Behavior event
            return await self._process_behavior_event(event, source)
        elif event_type.startswith("workflow."):
            # Workflow event
            return await self._process_workflow_event(event, source)
        
        return None
    
    async def _process_task_event(self, event: Dict[str, Any], source: MemorySource) -> Optional[MemoryNode]:
        """Process a task event.
        
        Args:
            event: Event dictionary.
            source: Memory source.
            
        Returns:
            Created memory node, if any.
        """
        event_type = event.get("type")
        payload = event.get("payload", {})
        
        if event_type == "task.completed":
            # Create a memory node for the completed task
            task_id = payload.get("task_id")
            task_name = payload.get("name", "Unknown Task")
            task_description = payload.get("description", "")
            
            # Create content
            content = [ContentBlock(
                type=ContentType.TEXT,
                data=f"Task: {task_name}\n\nDescription: {task_description}",
                metadata={"task_id": task_id}
            )]
            
            # Add any artifacts
            artifacts = payload.get("artifacts", [])
            for artifact in artifacts:
                if artifact.get("type") == "text":
                    content.append(ContentBlock(
                        type=ContentType.TEXT,
                        data=artifact.get("data", ""),
                        metadata=artifact.get("metadata", {})
                    ))
                elif artifact.get("type") == "image":
                    content.append(ContentBlock(
                        type=ContentType.IMAGE,
                        data=artifact.get("data", ""),
                        metadata=artifact.get("metadata", {}),
                        text_description=artifact.get("description", "Image artifact")
                    ))
            
            # Create node
            return await self.create_memory_node(
                entity_type=EntityType.TASK_ARTIFACT,
                source=source,
                content=content,
                summary=f"Completed task: {task_name}",
                keywords=payload.get("tags", []),
                importance=0.7
            )
        
        return None
    
    async def _process_preference_event(self, event: Dict[str, Any], source: MemorySource) -> Optional[MemoryNode]:
        """Process a preference event.
        
        Args:
            event: Event dictionary.
            source: Memory source.
            
        Returns:
            Created memory node, if any.
        """
        payload = event.get("payload", {})
        
        # Create content
        preference_type = payload.get("preference_type", "unknown")
        inference = payload.get("inference", "")
        evidence = payload.get("evidence", [])
        
        content_text = f"Preference Type: {preference_type}\n\nInference: {inference}\n\nEvidence:\n"
        for item in evidence:
            content_text += f"- {item.get('signal')}: {item.get('strength')}\n"
        
        # Create node
        return await self.create_text_node(
            text=content_text,
            entity_type=EntityType.USER_PREFERENCE,
            source=source,
            summary=f"User preference: {inference}",
            keywords=[preference_type] + payload.get("applicable_contexts", []),
            importance=0.8
        )
    
    async def _process_behavior_event(self, event: Dict[str, Any], source: MemorySource) -> Optional[MemoryNode]:
        """Process a behavior event.
        
        Args:
            event: Event dictionary.
            source: Memory source.
            
        Returns:
            Created memory node, if any.
        """
        payload = event.get("payload", {})
        
        # Create content
        pattern = payload.get("pattern", "")
        description = payload.get("description", "")
        examples = payload.get("examples", [])
        
        content_text = f"Behavior Pattern: {pattern}\n\nDescription: {description}\n\nExamples:\n"
        for example in examples:
            content_text += f"- {example}\n"
        
        # Create node
        return await self.create_text_node(
            text=content_text,
            entity_type=EntityType.WORKFLOW_PATTERN,
            source=source,
            summary=f"Behavior pattern: {pattern}",
            keywords=payload.get("tags", []),
            importance=0.6
        )
    
    async def _process_workflow_event(self, event: Dict[str, Any], source: MemorySource) -> Optional[MemoryNode]:
        """Process a workflow event.
        
        Args:
            event: Event dictionary.
            source: Memory source.
            
        Returns:
            Created memory node, if any.
        """
        payload = event.get("payload", {})
        
        # Create content
        pattern_name = payload.get("name", "")
        description = payload.get("description", "")
        steps = payload.get("steps", [])
        
        content_text = f"Workflow Pattern: {pattern_name}\n\nDescription: {description}\n\nSteps:\n"
        for i, step in enumerate(steps):
            content_text += f"{i+1}. {step}\n"
        
        # Create node
        return await self.create_text_node(
            text=content_text,
            entity_type=EntityType.WORKFLOW_PATTERN,
            source=source,
            summary=f"Workflow pattern: {pattern_name}",
            keywords=payload.get("tags", []),
            importance=0.7
        )