"""
Memory Graph for managing relationships between memory nodes.
"""
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
import asyncio

from .models import MemoryNode, Relation, RelationType
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MemoryGraph:
    """Graph-based memory system for managing node relationships."""
    
    def __init__(self):
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: Dict[str, List[Relation]] = defaultdict(list)
        self.reverse_edges: Dict[str, List[str]] = defaultdict(list)
        
        # Indexes for fast lookup
        self.type_index: Dict[str, Set[str]] = defaultdict(set)
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
    
    async def add_node(self, node: MemoryNode) -> None:
        """Add a node to the graph."""
        try:
            # Store the node
            self.nodes[node.id] = node
            
            # Update indexes
            self.type_index[node.entity_type.value].add(node.id)
            
            for keyword in node.keywords:
                self.keyword_index[keyword.lower()].add(node.id)
            
            for entity in node.entities:
                entity_key = f"{entity.type}:{entity.value}"
                self.entity_index[entity_key].add(node.id)
            
            # Add relationships
            for relation in node.relations:
                self.edges[node.id].append(relation)
                self.reverse_edges[relation.target_id].append(node.id)
            
            logger.debug(f"Added node {node.id} to memory graph")
            
        except Exception as e:
            logger.error(f"Failed to add node {node.id} to graph: {e}")
            raise
    
    async def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    async def remove_node(self, node_id: str) -> bool:
        """Remove a node from the graph."""
        try:
            if node_id not in self.nodes:
                return False
            
            node = self.nodes[node_id]
            
            # Remove from indexes
            self.type_index[node.entity_type.value].discard(node_id)
            
            for keyword in node.keywords:
                self.keyword_index[keyword.lower()].discard(node_id)
            
            for entity in node.entities:
                entity_key = f"{entity.type}:{entity.value}"
                self.entity_index[entity_key].discard(node_id)
            
            # Remove edges
            if node_id in self.edges:
                for relation in self.edges[node_id]:
                    self.reverse_edges[relation.target_id].remove(node_id)
                del self.edges[node_id]
            
            # Remove reverse edges
            if node_id in self.reverse_edges:
                for source_id in self.reverse_edges[node_id]:
                    self.edges[source_id] = [
                        r for r in self.edges[source_id] 
                        if r.target_id != node_id
                    ]
                del self.reverse_edges[node_id]
            
            # Remove the node
            del self.nodes[node_id]
            
            logger.debug(f"Removed node {node_id} from memory graph")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove node {node_id} from graph: {e}")
            return False
    
    async def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        strength: float = 1.0,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Add a relationship between two nodes."""
        try:
            if source_id not in self.nodes or target_id not in self.nodes:
                logger.warning(f"Cannot add relation: missing nodes {source_id} or {target_id}")
                return False
            
            relation = Relation(
                type=relation_type,
                target_id=target_id,
                strength=strength,
                metadata=metadata or {}
            )
            
            self.edges[source_id].append(relation)
            self.reverse_edges[target_id].append(source_id)
            
            # Also update the source node's relations
            source_node = self.nodes[source_id]
            source_node.relations.append(relation)
            
            logger.debug(f"Added relation {relation_type.value} from {source_id} to {target_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add relation: {e}")
            return False
    
    async def get_neighbors(
        self,
        node_id: str,
        relation_types: Optional[List[RelationType]] = None,
        min_strength: float = 0.0
    ) -> List[str]:
        """Get neighboring nodes."""
        neighbors = []
        
        if node_id not in self.edges:
            return neighbors
        
        for relation in self.edges[node_id]:
            # Filter by relation type
            if relation_types and relation.type not in relation_types:
                continue
            
            # Filter by strength
            if relation.strength < min_strength:
                continue
            
            neighbors.append(relation.target_id)
        
        return neighbors
    
    async def expand_context(
        self,
        seed_nodes: List[str],
        max_hops: int = 2,
        max_nodes: int = 100
    ) -> Set[str]:
        """Expand context by following relationships."""
        visited = set(seed_nodes)
        frontier = set(seed_nodes)
        
        for hop in range(max_hops):
            if len(visited) >= max_nodes:
                break
            
            next_frontier = set()
            
            for node_id in frontier:
                # Get outgoing neighbors
                neighbors = await self.get_neighbors(node_id)
                for neighbor_id in neighbors:
                    if neighbor_id not in visited and len(visited) < max_nodes:
                        next_frontier.add(neighbor_id)
                        visited.add(neighbor_id)
                
                # Get incoming neighbors
                if node_id in self.reverse_edges:
                    for source_id in self.reverse_edges[node_id]:
                        if source_id not in visited and len(visited) < max_nodes:
                            next_frontier.add(source_id)
                            visited.add(source_id)
            
            frontier = next_frontier
            if not frontier:
                break
        
        return visited
    
    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> Optional[List[str]]:
        """Find shortest path between two nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        if source_id == target_id:
            return [source_id]
        
        # BFS to find shortest path
        queue = [(source_id, [source_id])]
        visited = {source_id}
        
        while queue:
            current_id, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            neighbors = await self.get_neighbors(current_id)
            for neighbor_id in neighbors:
                if neighbor_id == target_id:
                    return path + [neighbor_id]
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
        
        return None
    
    async def get_similar_nodes(
        self,
        node_id: str,
        similarity_threshold: float = 0.5,
        max_results: int = 10
    ) -> List[str]:
        """Get nodes similar to the given node."""
        if node_id not in self.nodes:
            return []
        
        node = self.nodes[node_id]
        similar_nodes = []
        
        # Find nodes with similar keywords
        keyword_matches = set()
        for keyword in node.keywords:
            keyword_matches.update(self.keyword_index.get(keyword.lower(), set()))
        
        # Find nodes with similar entities
        entity_matches = set()
        for entity in node.entities:
            entity_key = f"{entity.type}:{entity.value}"
            entity_matches.update(self.entity_index.get(entity_key, set()))
        
        # Combine and score
        candidates = keyword_matches.union(entity_matches)
        candidates.discard(node_id)  # Remove self
        
        for candidate_id in candidates:
            if candidate_id in self.nodes:
                candidate = self.nodes[candidate_id]
                similarity = self._calculate_similarity(node, candidate)
                
                if similarity >= similarity_threshold:
                    similar_nodes.append((candidate_id, similarity))
        
        # Sort by similarity and return top results
        similar_nodes.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in similar_nodes[:max_results]]
    
    async def get_clusters(self, min_cluster_size: int = 3) -> List[List[str]]:
        """Get clusters of related nodes."""
        visited = set()
        clusters = []
        
        for node_id in self.nodes:
            if node_id in visited:
                continue
            
            # Find connected component
            cluster = await self._get_connected_component(node_id)
            
            if len(cluster) >= min_cluster_size:
                clusters.append(list(cluster))
            
            visited.update(cluster)
        
        return clusters
    
    async def _get_connected_component(self, start_node: str) -> Set[str]:
        """Get all nodes connected to the start node."""
        component = set()
        queue = [start_node]
        
        while queue:
            current = queue.pop(0)
            if current in component:
                continue
            
            component.add(current)
            
            # Add neighbors
            neighbors = await self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in component:
                    queue.append(neighbor)
            
            # Add reverse neighbors
            if current in self.reverse_edges:
                for source in self.reverse_edges[current]:
                    if source not in component:
                        queue.append(source)
        
        return component
    
    def _calculate_similarity(self, node1: MemoryNode, node2: MemoryNode) -> float:
        """Calculate similarity between two nodes."""
        similarity = 0.0
        
        # Keyword similarity
        keywords1 = set(kw.lower() for kw in node1.keywords)
        keywords2 = set(kw.lower() for kw in node2.keywords)
        
        if keywords1 or keywords2:
            keyword_intersection = len(keywords1.intersection(keywords2))
            keyword_union = len(keywords1.union(keywords2))
            keyword_similarity = keyword_intersection / keyword_union if keyword_union > 0 else 0
            similarity += keyword_similarity * 0.4
        
        # Entity similarity
        entities1 = set(f"{e.type}:{e.value}" for e in node1.entities)
        entities2 = set(f"{e.type}:{e.value}" for e in node2.entities)
        
        if entities1 or entities2:
            entity_intersection = len(entities1.intersection(entities2))
            entity_union = len(entities1.union(entities2))
            entity_similarity = entity_intersection / entity_union if entity_union > 0 else 0
            similarity += entity_similarity * 0.4
        
        # Type similarity
        if node1.entity_type == node2.entity_type:
            similarity += 0.2
        
        return min(similarity, 1.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        total_edges = sum(len(relations) for relations in self.edges.values())
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": total_edges,
            "node_types": {
                node_type: len(nodes) 
                for node_type, nodes in self.type_index.items()
            },
            "avg_degree": total_edges / len(self.nodes) if self.nodes else 0
        }