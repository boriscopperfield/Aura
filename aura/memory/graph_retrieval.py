"""
Graph-based retrieval system for AURA memory.

This module provides advanced retrieval capabilities that leverage
the graph structure of memory nodes and their relationships.
"""
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from aura.config import settings
from aura.utils.logging import logger
from aura.memory.models import MemoryNode, Query, ScoredMemoryNode, RelationType
from aura.memory.relationships import Relationship, EntityRelationshipModel
from aura.memory.embeddings import jina_embedder, jina_reranker


class GraphAwareRetrieval:
    """Graph-aware retrieval system for memory nodes."""
    
    def __init__(
        self,
        entity_relationship_model: EntityRelationshipModel,
        similarity_threshold: float = 0.7
    ):
        """Initialize the graph-aware retrieval system.
        
        Args:
            entity_relationship_model: Entity-relationship model
            similarity_threshold: Minimum similarity threshold
        """
        self.model = entity_relationship_model
        self.similarity_threshold = similarity_threshold
        self.logger = logger
    
    async def find_seed_entities(
        self,
        query: Query,
        k: int = 10
    ) -> List[ScoredMemoryNode]:
        """Find seed entities for a query.
        
        Args:
            query: Query
            k: Number of results to return
            
        Returns:
            List of scored memory nodes
        """
        # Generate query embedding
        if query.embedding:
            query_embedding = query.embedding
        elif query.text:
            query_embedding = await jina_embedder.embed_text(query.text)
        elif query.image:
            query_embedding = await jina_embedder.embed_image(query.image)
        else:
            self.logger.warning("Query has no text, image, or embedding")
            return []
        
        # Find similar nodes
        scored_nodes = []
        for node_id, node in self.model.nodes.items():
            # Skip nodes without embeddings
            if not node.embeddings or not node.embeddings.chunk_embedding:
                continue
            
            # Calculate similarity
            similarity = self._calculate_similarity(
                query_embedding,
                node.embeddings.chunk_embedding
            )
            
            # Add to results if above threshold
            if similarity >= self.similarity_threshold:
                scored_nodes.append(ScoredMemoryNode(
                    node=node,
                    score=similarity
                ))
        
        # Sort by score
        scored_nodes.sort(key=lambda x: x.score, reverse=True)
        
        # Return top k
        return scored_nodes[:k]
    
    async def build_minimal_closure(
        self,
        query: Query,
        seed_nodes: List[ScoredMemoryNode],
        max_hops: int = 3,
        max_nodes: int = 50
    ) -> Dict[str, Any]:
        """Build minimal logical closure for query.
        
        Args:
            query: Query
            seed_nodes: Seed nodes
            max_hops: Maximum number of hops
            max_nodes: Maximum number of nodes in closure
            
        Returns:
            Closure information
        """
        # Generate query embedding if not provided
        if not query.embedding and query.text:
            query_embedding = await jina_embedder.embed_text(query.text)
        else:
            query_embedding = query.embedding
        
        # Initialize closure
        closure = {
            "core_entities": [node.node.id for node in seed_nodes],
            "related_entities": {},
            "key_relations": {},
            "expansion_path": []
        }
        
        # Initialize visited set and expansion queue
        visited = set(node.node.id for node in seed_nodes)
        expansion_queue = [(node.node.id, 0, None) for node in seed_nodes]  # (node_id, depth, relation)
        
        # Expand through relationships
        while expansion_queue and len(closure["related_entities"]) < max_nodes:
            current_id, depth, relation = expansion_queue.pop(0)
            
            # Skip if we've reached max hops
            if depth >= max_hops:
                continue
            
            # Get outgoing relationships
            outgoing = self.model.get_outgoing_relationships(current_id)
            
            for rel in outgoing:
                target_id = rel.target_id
                
                # Skip if we've already visited this node
                if target_id in visited:
                    continue
                
                # Get target node
                target = self.model.get_node(target_id)
                if not target:
                    continue
                
                # Calculate relevance
                relevance = await self._compute_relevance(
                    query_embedding=query_embedding,
                    entity=target,
                    relation=rel,
                    depth=depth
                )
                
                # Add to closure if relevant
                if relevance >= self.similarity_threshold:
                    # Add to related entities
                    closure["related_entities"][target_id] = {
                        "entity_id": target_id,
                        "entity_type": target.entity_type.value,
                        "summary": target.summary,
                        "path": f"{current_id}->{target_id}",
                        "relation": rel.type.value,
                        "relevance": relevance,
                        "depth": depth + 1
                    }
                    
                    # Add to key relations
                    closure["key_relations"][rel.id] = {
                        "relation_id": rel.id,
                        "relation_type": rel.type.value,
                        "source_id": rel.source_id,
                        "target_id": rel.target_id,
                        "strength": rel.strength,
                        "relevance": relevance
                    }
                    
                    # Add to expansion path
                    closure["expansion_path"].append({
                        "hop": depth,
                        "relation": rel.type.value,
                        "from": current_id,
                        "to": target_id,
                        "reason": self._explain_expansion(rel, relevance)
                    })
                    
                    # Mark as visited
                    visited.add(target_id)
                    
                    # Add to expansion queue
                    expansion_queue.append((target_id, depth + 1, rel))
        
        return closure
    
    async def retrieve_with_closure(
        self,
        query: Query,
        max_hops: int = 2,
        max_nodes: int = 50,
        rerank: bool = True
    ) -> Dict[str, Any]:
        """Retrieve nodes with closure building.
        
        Args:
            query: Query
            max_hops: Maximum number of hops
            max_nodes: Maximum number of nodes in closure
            rerank: Whether to rerank results
            
        Returns:
            Retrieval results with closure
        """
        # Find seed entities
        seed_nodes = await self.find_seed_entities(query)
        
        # Build closure
        closure = await self.build_minimal_closure(
            query=query,
            seed_nodes=seed_nodes,
            max_hops=max_hops,
            max_nodes=max_nodes
        )
        
        # Collect all nodes
        all_nodes = []
        
        # Add seed nodes
        for node in seed_nodes:
            all_nodes.append(node)
        
        # Add related nodes
        for entity_id, entity_info in closure["related_entities"].items():
            node = self.model.get_node(entity_id)
            if node:
                all_nodes.append(ScoredMemoryNode(
                    node=node,
                    score=entity_info["relevance"],
                    match_details={"path": entity_info["path"]}
                ))
        
        # Rerank if requested
        if rerank and query.text and len(all_nodes) > 1:
            # Extract text content
            documents = [node.node.get_text_content() for node in all_nodes]
            
            # Rerank
            reranked = await jina_reranker.rerank(query.text, documents)
            
            # Create new scored nodes
            reranked_nodes = []
            for doc, score in reranked:
                # Find the original node
                for node in all_nodes:
                    if node.node.get_text_content() == doc:
                        reranked_nodes.append(ScoredMemoryNode(
                            node=node.node,
                            score=score,
                            match_details={
                                "reranked": True,
                                "original_score": node.score,
                                "path": node.match_details.get("path") if node.match_details else None
                            }
                        ))
                        break
            
            all_nodes = reranked_nodes
        
        # Sort by score
        all_nodes.sort(key=lambda x: x.score, reverse=True)
        
        return {
            "nodes": all_nodes,
            "closure": closure
        }
    
    async def _compute_relevance(
        self,
        query_embedding: List[float],
        entity: MemoryNode,
        relation: Relationship,
        depth: int
    ) -> float:
        """Compute relevance of an entity to a query.
        
        Args:
            query_embedding: Query embedding
            entity: Memory node
            relation: Relationship
            depth: Depth in expansion
            
        Returns:
            Relevance score
        """
        # Entity similarity
        entity_sim = 0.0
        if entity.embeddings and entity.embeddings.chunk_embedding:
            entity_sim = self._calculate_similarity(
                query_embedding,
                entity.embeddings.chunk_embedding
            )
        
        # Relationship similarity
        relation_sim = 0.0
        if relation.embedding:
            relation_sim = self._calculate_similarity(
                query_embedding,
                relation.embedding
            )
        
        # Depth decay
        depth_decay = 1.0 / (1.0 + 0.5 * depth)
        
        # Combined relevance
        relevance = (0.6 * entity_sim + 0.4 * relation_sim) * depth_decay
        
        return relevance
    
    def _calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _explain_expansion(self, relation: Relationship, relevance: float) -> str:
        """Generate explanation for expansion.
        
        Args:
            relation: Relationship
            relevance: Relevance score
            
        Returns:
            Explanation
        """
        # Get source and target nodes
        source = self.model.get_node(relation.source_id)
        target = self.model.get_node(relation.target_id)
        
        if not source or not target:
            return f"Followed {relation.type.value} relationship with relevance {relevance:.2f}"
        
        return (
            f"Expanded from '{source.summary}' to '{target.summary}' "
            f"via {relation.type.value} relationship with relevance {relevance:.2f}"
        )


# Global instance
graph_retrieval = GraphAwareRetrieval(entity_relationship_model=None)  # Will be initialized later