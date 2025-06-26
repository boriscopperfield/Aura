"""
Enhanced memory manager for AURA system.

This module provides a unified API for memory operations with hierarchical
storage, relationship modeling, and graph-based retrieval.
"""
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from aura.config import settings
from aura.utils.logging import logger
from aura.utils.errors import MemoryError
from aura.memory.models import (
    MemoryNode, ContentBlock, ContentType, EntityType, 
    MemorySource, NamedEntity, Relation, RelationType,
    Query, ScoredMemoryNode
)
from aura.memory.embeddings import jina_embedder, jina_reranker
from aura.memory.hierarchy import (
    MemoryLayer, LayeredMemoryArchitecture, layered_memory
)
from aura.memory.relationships import (
    Relationship, EntityRelationshipModel, entity_relationship_model
)
from aura.memory.graph_retrieval import GraphAwareRetrieval, graph_retrieval
from aura.memory.lifecycle import (
    MemoryLifecycleManager, HierarchicalQueryProcessor
)


class EnhancedMemoryManager:
    """Enhanced memory manager with hierarchical storage and graph-based retrieval."""
    
    def __init__(self, workspace_path: Optional[Path] = None):
        """Initialize the enhanced memory manager.
        
        Args:
            workspace_path: Path to the workspace directory
        """
        self.workspace_path = workspace_path or settings.workspace.path
        self.memory_path = self.workspace_path / settings.workspace.memory_dir
        self.logger = logger
        
        # Create memory directory if it doesn't exist
        self.memory_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        # Load existing memory nodes
        self._load_memory()
    
    def _initialize_components(self) -> None:
        """Initialize memory system components."""
        # Initialize layered memory architecture
        global layered_memory
        layered_memory = LayeredMemoryArchitecture()
        
        # Initialize entity-relationship model
        global entity_relationship_model
        entity_relationship_model = EntityRelationshipModel()
        
        # Initialize graph-aware retrieval
        global graph_retrieval
        graph_retrieval = GraphAwareRetrieval(
            entity_relationship_model=entity_relationship_model
        )
        
        # Initialize lifecycle manager
        self.lifecycle_manager = MemoryLifecycleManager(
            layered_memory=layered_memory,
            entity_relationship_model=entity_relationship_model
        )
        
        # Initialize query processor
        self.query_processor = HierarchicalQueryProcessor(
            layered_memory=layered_memory,
            entity_relationship_model=entity_relationship_model,
            lifecycle_manager=self.lifecycle_manager
        )
    
    async def create_node(
        self,
        content: List[Union[str, bytes, ContentBlock]],
        entity_type: EntityType,
        source: MemorySource,
        summary: str,
        keywords: List[str],
        entities: Optional[List[NamedEntity]] = None,
        relations: Optional[List[Relation]] = None,
        importance: float = 0.5
    ) -> MemoryNode:
        """Create a new memory node.
        
        Args:
            content: Content blocks or raw content
            entity_type: Type of entity
            source: Source of the memory
            summary: Summary of the memory
            keywords: Keywords for the memory
            entities: Named entities in the memory
            relations: Relations to other memory nodes
            importance: Importance of the memory
            
        Returns:
            Created memory node
            
        Raises:
            MemoryError: If the node creation fails
        """
        try:
            # Process content blocks
            content_blocks = []
            for item in content:
                if isinstance(item, ContentBlock):
                    content_blocks.append(item)
                elif isinstance(item, str):
                    content_blocks.append(ContentBlock(
                        type=ContentType.TEXT,
                        data=item,
                        metadata={"length": len(item)}
                    ))
                elif isinstance(item, bytes):
                    # Try to detect content type
                    if item.startswith(b'\xff\xd8\xff'):  # JPEG
                        content_type = ContentType.IMAGE
                    elif item.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
                        content_type = ContentType.IMAGE
                    elif item.startswith(b'GIF8'):  # GIF
                        content_type = ContentType.IMAGE
                    else:
                        # Default to binary data
                        content_type = ContentType.STRUCTURED_DATA
                    
                    content_blocks.append(ContentBlock(
                        type=content_type,
                        data=item,
                        metadata={"size": len(item)}
                    ))
                else:
                    raise MemoryError(f"Unsupported content type: {type(item)}")
            
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
                keywords=keywords,
                entities=entities or [],
                relations=relations or [],
                importance=importance,
                access_count=0,
                last_accessed=datetime.now(),
                decay_rate=settings.memory.default_decay_rate
            )
            
            # Generate embeddings
            embeddings = await jina_embedder.embed_content_blocks(content_blocks)
            node.embeddings = embeddings
            
            # Add to entity-relationship model
            entity_relationship_model.add_node(node)
            
            # Add to layered memory (L1)
            layered_memory.add_node(node, MemoryLayer.L1)
            
            # Process relations
            if relations:
                for relation in relations:
                    # Create relationship
                    rel = Relationship.create(
                        type=relation.type,
                        source_id=node_id,
                        target_id=relation.target_id,
                        strength=relation.strength,
                        bidirectional=False,
                        context={"source": "node_creation"}
                    )
                    
                    # Add to entity-relationship model
                    entity_relationship_model.add_relationship(rel)
                    
                    # Add to layered memory (L1)
                    layered_memory.add_relation(
                        rel.to_dict(),
                        MemoryLayer.L1
                    )
            
            # Save to disk
            self._save_node(node)
            
            return node
        except Exception as e:
            self.logger.error(f"Error creating memory node: {e}")
            raise MemoryError(f"Failed to create memory node: {str(e)}")
    
    async def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a memory node by ID.
        
        Args:
            node_id: Memory node ID
            
        Returns:
            Memory node or None if not found
        """
        # Try to get from entity-relationship model first
        node = entity_relationship_model.get_node(node_id)
        if node:
            # Update access stats
            node.update_access_stats()
            return node
        
        # Try to get from layered memory
        node = layered_memory.get_node(node_id)
        if node:
            # Update access stats
            node.update_access_stats()
            
            # Promote on access
            await self.lifecycle_manager.promote_on_access(node_id)
            
            return node
        
        return None
    
    async def update_node(self, node: MemoryNode) -> MemoryNode:
        """Update a memory node.
        
        Args:
            node: Memory node to update
            
        Returns:
            Updated memory node
            
        Raises:
            MemoryError: If the node update fails
        """
        try:
            # Update timestamp
            node.updated_at = datetime.now()
            
            # Update embeddings if content changed
            if not node.embeddings or not node.embeddings.chunk_embeddings:
                embeddings = await jina_embedder.embed_content_blocks(node.content)
                node.embeddings = embeddings
            
            # Update in entity-relationship model
            entity_relationship_model.add_node(node)
            
            # Update in layered memory (L1)
            layered_memory.add_node(node, MemoryLayer.L1)
            
            # Save to disk
            self._save_node(node)
            
            return node
        except Exception as e:
            self.logger.error(f"Error updating memory node: {e}")
            raise MemoryError(f"Failed to update memory node: {str(e)}")
    
    async def delete_node(self, node_id: str) -> bool:
        """Delete a memory node.
        
        Args:
            node_id: Memory node ID
            
        Returns:
            True if the node was deleted, False otherwise
        """
        try:
            # Remove from entity-relationship model
            node = entity_relationship_model.get_node(node_id)
            if node:
                # Get all relationships
                outgoing = entity_relationship_model.get_outgoing_relationships(node_id)
                incoming = entity_relationship_model.get_incoming_relationships(node_id)
                
                # Remove relationships
                for rel in outgoing + incoming:
                    # Remove from layered memory
                    layered_memory.l1_relations.remove(rel.id)
                    layered_memory.l2_relations.remove(rel.id)
                    layered_memory.l3_relations.remove(rel.id)
            
            # Remove from layered memory
            layered_memory.l1_nodes.remove(node_id)
            layered_memory.l2_nodes.remove(node_id)
            layered_memory.l3_nodes.remove(node_id)
            
            # Remove from disk
            node_path = self.memory_path / "nodes" / f"{node_id}.json"
            if node_path.exists():
                node_path.unlink()
            
            return True
        except Exception as e:
            self.logger.error(f"Error deleting memory node: {e}")
            return False
    
    async def search(
        self,
        query: Union[str, Query],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """Search for memory nodes with graph-based retrieval.
        
        Args:
            query: Search query
            k: Number of results to return
            filters: Filters to apply
            rerank: Whether to rerank results
            max_hops: Maximum number of hops for graph traversal
            
        Returns:
            Search results with closure information
            
        Raises:
            MemoryError: If the search fails
        """
        try:
            # Convert string query to Query object
            if isinstance(query, str):
                query = Query(text=query, filters=filters or {})
            elif filters:
                query.filters.update(filters)
            
            # Search using graph-aware retrieval
            results = await graph_retrieval.retrieve_with_closure(
                query=query,
                max_hops=max_hops,
                max_nodes=k * 2,
                rerank=rerank
            )
            
            # Update access stats for retrieved nodes
            for scored_node in results["nodes"]:
                scored_node.node.update_access_stats()
                
                # Promote on access
                await self.lifecycle_manager.promote_on_access(scored_node.node.id)
            
            return results
        except Exception as e:
            self.logger.error(f"Error searching memory: {e}")
            raise MemoryError(f"Failed to search memory: {str(e)}")
    
    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        strength: float = 1.0,
        bidirectional: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Relationship]:
        """Add a relationship between two memory nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of relation
            strength: Strength of the relation
            bidirectional: Whether the relationship is bidirectional
            context: Additional context
            
        Returns:
            Created relationship or None if failed
        """
        try:
            # Check if source and target nodes exist
            source_node = await self.get_node(source_id)
            if not source_node:
                self.logger.warning(f"Source node {source_id} not found")
                return None
            
            target_node = await self.get_node(target_id)
            if not target_node:
                self.logger.warning(f"Target node {target_id} not found")
                return None
            
            # Create relationship
            relationship = Relationship.create(
                type=relation_type,
                source_id=source_id,
                target_id=target_id,
                strength=strength,
                bidirectional=bidirectional,
                context=context
            )
            
            # Add to entity-relationship model
            entity_relationship_model.add_relationship(relationship)
            
            # Add to layered memory (L1)
            layered_memory.add_relation(
                relationship.to_dict(),
                MemoryLayer.L1
            )
            
            # Update source node's relations
            source_relation = Relation(
                type=relation_type,
                target_id=target_id,
                strength=strength
            )
            
            if not any(r.target_id == target_id and r.type == relation_type for r in source_node.relations):
                source_node.relations.append(source_relation)
                await self.update_node(source_node)
            
            # Update target node's relations if bidirectional
            if bidirectional:
                target_relation = Relation(
                    type=relation_type,
                    target_id=source_id,
                    strength=strength
                )
                
                if not any(r.target_id == source_id and r.type == relation_type for r in target_node.relations):
                    target_node.relations.append(target_relation)
                    await self.update_node(target_node)
            
            return relationship
        except Exception as e:
            self.logger.error(f"Error adding relationship: {e}")
            return None
    
    async def start_lifecycle_management(self) -> None:
        """Start background lifecycle management."""
        await self.lifecycle_manager.start_background_task()
    
    async def stop_lifecycle_management(self) -> None:
        """Stop background lifecycle management."""
        await self.lifecycle_manager.stop_background_task()
    
    def _load_memory(self) -> None:
        """Load memory nodes from disk."""
        # Create nodes directory if it doesn't exist
        nodes_dir = self.memory_path / "nodes"
        nodes_dir.mkdir(parents=True, exist_ok=True)
        
        # Load nodes
        for node_file in nodes_dir.glob("*.json"):
            try:
                with open(node_file, "r") as f:
                    node_data = f.read()
                
                # Parse node data
                import json
                node_dict = json.loads(node_data)
                
                # Create memory node
                node = MemoryNode.parse_obj(node_dict)
                
                # Add to entity-relationship model
                entity_relationship_model.add_node(node)
                
                # Add to layered memory (L3)
                layered_memory.add_node(node, MemoryLayer.L3)
                
                # Process relations
                for relation in node.relations:
                    # Create relationship
                    rel = Relationship.create(
                        type=relation.type,
                        source_id=node.id,
                        target_id=relation.target_id,
                        strength=relation.strength
                    )
                    
                    # Add to entity-relationship model
                    entity_relationship_model.add_relationship(rel)
                    
                    # Add to layered memory (L3)
                    layered_memory.add_relation(
                        rel.to_dict(),
                        MemoryLayer.L3
                    )
            except Exception as e:
                self.logger.error(f"Error loading memory node {node_file}: {e}")
    
    def _save_node(self, node: MemoryNode) -> None:
        """Save a memory node to disk.
        
        Args:
            node: Memory node to save
        """
        # Create nodes directory if it doesn't exist
        nodes_dir = self.memory_path / "nodes"
        nodes_dir.mkdir(parents=True, exist_ok=True)
        
        # Save node
        node_path = nodes_dir / f"{node.id}.json"
        with open(node_path, "w") as f:
            import json
            f.write(json.dumps(node.dict()))


# Global memory manager instance
enhanced_memory_manager = EnhancedMemoryManager()