"""
Hierarchical memory system for AURA.

This module implements a three-layer memory architecture:
- L1: Hot Cache (Active Working Set)
- L2: Session Memory (Session-level Index)
- L3: Persistent Graph (Persistent Knowledge Graph)

Each layer has different storage characteristics, retrieval strategies,
and lifecycle management policies.
"""
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, TypeVar, Generic

from pydantic import BaseModel, Field

from aura.config import settings
from aura.utils.logging import logger
from aura.memory.models import MemoryNode, EntityType, RelationType


class MemoryLayer(str, Enum):
    """Memory storage layers."""
    
    L1 = "L1"  # Hot Cache (Active Working Set)
    L2 = "L2"  # Session Memory (Session-level Index)
    L3 = "L3"  # Persistent Graph (Persistent Knowledge Graph)


class LayerCriteria(BaseModel):
    """Criteria for memory layer management."""
    
    # Content criteria
    current_task_nodes: bool = False
    recent_nodes: bool = False
    active_artifacts: bool = False
    pending_decisions: bool = False
    error_contexts: bool = False
    session_task_trees: bool = False
    user_interactions: bool = False
    executed_nodes: bool = False
    artifact_registry: bool = False
    learned_patterns: bool = False
    completed_projects: bool = False
    reusable_patterns: bool = False
    artifact_templates: bool = False
    user_preferences: bool = False
    best_practices: bool = False
    failure_patterns: bool = False
    
    # Storage criteria
    max_entries: int = 1000
    ttl: str = "session_end"  # Time-to-live before demotion
    compression: str = "none"  # Compression strategy


# Default criteria for each layer
L1_CRITERIA = LayerCriteria(
    current_task_nodes=True,
    recent_nodes=True,
    active_artifacts=True,
    pending_decisions=True,
    error_contexts=True,
    max_entries=50,
    ttl="10_minutes",
    compression="none"
)

L2_CRITERIA = LayerCriteria(
    session_task_trees=True,
    user_interactions=True,
    executed_nodes=True,
    artifact_registry=True,
    learned_patterns=True,
    max_entries=5000,
    ttl="session_end",
    compression="light"
)

L3_CRITERIA = LayerCriteria(
    completed_projects=True,
    reusable_patterns=True,
    artifact_templates=True,
    user_preferences=True,
    best_practices=True,
    failure_patterns=True,
    max_entries=1000000,
    ttl="1_year",
    compression="aggressive"
)


T = TypeVar('T')


class MemoryStore(Generic[T]):
    """Base class for memory stores."""
    
    def __init__(self, layer: MemoryLayer, criteria: LayerCriteria):
        """Initialize the memory store.
        
        Args:
            layer: Memory layer
            criteria: Layer criteria
        """
        self.layer = layer
        self.criteria = criteria
        self.store: Dict[str, T] = {}
        self.access_times: Dict[str, datetime] = {}
        self.logger = logger
    
    def add(self, key: str, value: T) -> None:
        """Add an item to the store.
        
        Args:
            key: Item key
            value: Item value
        """
        self.store[key] = value
        self.access_times[key] = datetime.now()
        
        # Check if we need to evict items
        if len(self.store) > self.criteria.max_entries:
            self._evict_oldest()
    
    def get(self, key: str) -> Optional[T]:
        """Get an item from the store.
        
        Args:
            key: Item key
            
        Returns:
            Item value or None if not found
        """
        if key in self.store:
            # Update access time
            self.access_times[key] = datetime.now()
            return self.store[key]
        return None
    
    def remove(self, key: str) -> None:
        """Remove an item from the store.
        
        Args:
            key: Item key
        """
        if key in self.store:
            del self.store[key]
        if key in self.access_times:
            del self.access_times[key]
    
    def get_all(self) -> List[T]:
        """Get all items in the store.
        
        Returns:
            List of all items
        """
        return list(self.store.values())
    
    def get_keys(self) -> List[str]:
        """Get all keys in the store.
        
        Returns:
            List of all keys
        """
        return list(self.store.keys())
    
    def get_idle(self, minutes: int) -> List[T]:
        """Get items that haven't been accessed for a while.
        
        Args:
            minutes: Number of minutes of inactivity
            
        Returns:
            List of idle items
        """
        now = datetime.now()
        idle_threshold = now - timedelta(minutes=minutes)
        
        idle_items = []
        for key, access_time in self.access_times.items():
            if access_time < idle_threshold and key in self.store:
                idle_items.append(self.store[key])
        
        return idle_items
    
    def clear(self) -> None:
        """Clear the store."""
        self.store.clear()
        self.access_times.clear()
    
    def _evict_oldest(self) -> None:
        """Evict the oldest items from the store."""
        # Sort by access time
        sorted_items = sorted(
            self.access_times.items(),
            key=lambda x: x[1]
        )
        
        # Calculate how many items to evict
        num_to_evict = len(self.store) - self.criteria.max_entries
        
        # Evict oldest items
        for i in range(min(num_to_evict, len(sorted_items))):
            key = sorted_items[i][0]
            if key in self.store:
                self.logger.debug(f"Evicting {key} from {self.layer} store")
                del self.store[key]
                del self.access_times[key]


class NodeStore(MemoryStore[MemoryNode]):
    """Memory store for memory nodes."""
    
    def __init__(self, layer: MemoryLayer, criteria: LayerCriteria):
        """Initialize the node store.
        
        Args:
            layer: Memory layer
            criteria: Layer criteria
        """
        super().__init__(layer, criteria)
        self.entity_index: Dict[EntityType, Set[str]] = {
            entity_type: set() for entity_type in EntityType
        }
    
    def add(self, key: str, node: MemoryNode) -> None:
        """Add a node to the store.
        
        Args:
            key: Node key
            node: Memory node
        """
        super().add(key, node)
        
        # Update entity index
        self.entity_index[node.entity_type].add(key)
    
    def remove(self, key: str) -> None:
        """Remove a node from the store.
        
        Args:
            key: Node key
        """
        if key in self.store:
            node = self.store[key]
            # Update entity index
            if node.entity_type in self.entity_index:
                self.entity_index[node.entity_type].discard(key)
        
        super().remove(key)
    
    def get_by_entity_type(self, entity_type: EntityType) -> List[MemoryNode]:
        """Get nodes by entity type.
        
        Args:
            entity_type: Entity type
            
        Returns:
            List of nodes with the specified entity type
        """
        if entity_type not in self.entity_index:
            return []
        
        return [
            self.store[key] for key in self.entity_index[entity_type]
            if key in self.store
        ]
    
    def clear(self) -> None:
        """Clear the store."""
        super().clear()
        self.entity_index = {entity_type: set() for entity_type in EntityType}


class RelationshipStore(MemoryStore[Dict[str, Any]]):
    """Memory store for relationships."""
    
    def __init__(self, layer: MemoryLayer, criteria: LayerCriteria):
        """Initialize the relationship store.
        
        Args:
            layer: Memory layer
            criteria: Layer criteria
        """
        super().__init__(layer, criteria)
        self.relation_index: Dict[RelationType, Set[str]] = {
            relation_type: set() for relation_type in RelationType
        }
        self.source_index: Dict[str, Set[str]] = {}  # source_id -> relation_ids
        self.target_index: Dict[str, Set[str]] = {}  # target_id -> relation_ids
    
    def add(self, key: str, relation: Dict[str, Any]) -> None:
        """Add a relationship to the store.
        
        Args:
            key: Relationship key
            relation: Relationship data
        """
        super().add(key, relation)
        
        # Update indexes
        relation_type = relation.get("type")
        if relation_type and isinstance(relation_type, RelationType):
            self.relation_index[relation_type].add(key)
        
        source_id = relation.get("source_id")
        if source_id:
            if source_id not in self.source_index:
                self.source_index[source_id] = set()
            self.source_index[source_id].add(key)
        
        target_id = relation.get("target_id")
        if target_id:
            if target_id not in self.target_index:
                self.target_index[target_id] = set()
            self.target_index[target_id].add(key)
    
    def remove(self, key: str) -> None:
        """Remove a relationship from the store.
        
        Args:
            key: Relationship key
        """
        if key in self.store:
            relation = self.store[key]
            
            # Update indexes
            relation_type = relation.get("type")
            if relation_type and isinstance(relation_type, RelationType):
                self.relation_index[relation_type].discard(key)
            
            source_id = relation.get("source_id")
            if source_id and source_id in self.source_index:
                self.source_index[source_id].discard(key)
                if not self.source_index[source_id]:
                    del self.source_index[source_id]
            
            target_id = relation.get("target_id")
            if target_id and target_id in self.target_index:
                self.target_index[target_id].discard(key)
                if not self.target_index[target_id]:
                    del self.target_index[target_id]
        
        super().remove(key)
    
    def get_by_relation_type(self, relation_type: RelationType) -> List[Dict[str, Any]]:
        """Get relationships by relation type.
        
        Args:
            relation_type: Relation type
            
        Returns:
            List of relationships with the specified relation type
        """
        if relation_type not in self.relation_index:
            return []
        
        return [
            self.store[key] for key in self.relation_index[relation_type]
            if key in self.store
        ]
    
    def get_by_source(self, source_id: str) -> List[Dict[str, Any]]:
        """Get relationships by source ID.
        
        Args:
            source_id: Source node ID
            
        Returns:
            List of relationships with the specified source ID
        """
        if source_id not in self.source_index:
            return []
        
        return [
            self.store[key] for key in self.source_index[source_id]
            if key in self.store
        ]
    
    def get_by_target(self, target_id: str) -> List[Dict[str, Any]]:
        """Get relationships by target ID.
        
        Args:
            target_id: Target node ID
            
        Returns:
            List of relationships with the specified target ID
        """
        if target_id not in self.target_index:
            return []
        
        return [
            self.store[key] for key in self.target_index[target_id]
            if key in self.store
        ]
    
    def clear(self) -> None:
        """Clear the store."""
        super().clear()
        self.relation_index = {
            relation_type: set() for relation_type in RelationType
        }
        self.source_index = {}
        self.target_index = {}


class LayeredMemoryArchitecture:
    """Three-layer memory architecture for AURA."""
    
    def __init__(self):
        """Initialize the layered memory architecture."""
        # Initialize layer criteria
        self.l1_criteria = L1_CRITERIA
        self.l2_criteria = L2_CRITERIA
        self.l3_criteria = L3_CRITERIA
        
        # Initialize node stores
        self.l1_nodes = NodeStore(MemoryLayer.L1, self.l1_criteria)
        self.l2_nodes = NodeStore(MemoryLayer.L2, self.l2_criteria)
        self.l3_nodes = NodeStore(MemoryLayer.L3, self.l3_criteria)
        
        # Initialize relationship stores
        self.l1_relations = RelationshipStore(MemoryLayer.L1, self.l1_criteria)
        self.l2_relations = RelationshipStore(MemoryLayer.L2, self.l2_criteria)
        self.l3_relations = RelationshipStore(MemoryLayer.L3, self.l3_criteria)
        
        # Initialize logger
        self.logger = logger
    
    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a node from any layer.
        
        Args:
            node_id: Node ID
            
        Returns:
            Memory node or None if not found
        """
        # Check L1 first
        node = self.l1_nodes.get(node_id)
        if node:
            return node
        
        # Check L2
        node = self.l2_nodes.get(node_id)
        if node:
            # Promote to L1
            self.promote_node(node_id, MemoryLayer.L2, MemoryLayer.L1)
            return node
        
        # Check L3
        node = self.l3_nodes.get(node_id)
        if node:
            # Promote to L2
            self.promote_node(node_id, MemoryLayer.L3, MemoryLayer.L2)
            return node
        
        return None
    
    def add_node(self, node: MemoryNode, layer: MemoryLayer = MemoryLayer.L1) -> None:
        """Add a node to a specific layer.
        
        Args:
            node: Memory node
            layer: Memory layer
        """
        if layer == MemoryLayer.L1:
            self.l1_nodes.add(node.id, node)
        elif layer == MemoryLayer.L2:
            self.l2_nodes.add(node.id, node)
        elif layer == MemoryLayer.L3:
            self.l3_nodes.add(node.id, node)
    
    def add_relation(
        self,
        relation: Dict[str, Any],
        layer: MemoryLayer = MemoryLayer.L1
    ) -> None:
        """Add a relationship to a specific layer.
        
        Args:
            relation: Relationship data
            layer: Memory layer
        """
        relation_id = relation.get("id")
        if not relation_id:
            self.logger.warning("Relationship has no ID, cannot add to store")
            return
        
        if layer == MemoryLayer.L1:
            self.l1_relations.add(relation_id, relation)
        elif layer == MemoryLayer.L2:
            self.l2_relations.add(relation_id, relation)
        elif layer == MemoryLayer.L3:
            self.l3_relations.add(relation_id, relation)
    
    def promote_node(
        self,
        node_id: str,
        from_layer: MemoryLayer,
        to_layer: MemoryLayer
    ) -> None:
        """Promote a node from one layer to another.
        
        Args:
            node_id: Node ID
            from_layer: Source layer
            to_layer: Target layer
        """
        # Get node from source layer
        node = None
        if from_layer == MemoryLayer.L1:
            node = self.l1_nodes.get(node_id)
        elif from_layer == MemoryLayer.L2:
            node = self.l2_nodes.get(node_id)
        elif from_layer == MemoryLayer.L3:
            node = self.l3_nodes.get(node_id)
        
        if not node:
            self.logger.warning(f"Node {node_id} not found in {from_layer}")
            return
        
        # Add to target layer
        if to_layer == MemoryLayer.L1:
            # Expand node for L1
            expanded_node = self._expand_for_l1(node)
            self.l1_nodes.add(node_id, expanded_node)
        elif to_layer == MemoryLayer.L2:
            # Compress node for L2
            compressed_node = self._compress_for_l2(node)
            self.l2_nodes.add(node_id, compressed_node)
        elif to_layer == MemoryLayer.L3:
            # Further compress node for L3
            persistent_node = self._compress_for_l3(node)
            self.l3_nodes.add(node_id, persistent_node)
    
    def demote_node(
        self,
        node_id: str,
        from_layer: MemoryLayer,
        to_layer: MemoryLayer
    ) -> None:
        """Demote a node from one layer to another.
        
        Args:
            node_id: Node ID
            from_layer: Source layer
            to_layer: Target layer
        """
        # Get node from source layer
        node = None
        if from_layer == MemoryLayer.L1:
            node = self.l1_nodes.get(node_id)
        elif from_layer == MemoryLayer.L2:
            node = self.l2_nodes.get(node_id)
        elif from_layer == MemoryLayer.L3:
            node = self.l3_nodes.get(node_id)
        
        if not node:
            self.logger.warning(f"Node {node_id} not found in {from_layer}")
            return
        
        # Add to target layer (with appropriate compression)
        if to_layer == MemoryLayer.L2:
            # Compress node for L2
            compressed_node = self._compress_for_l2(node)
            self.l2_nodes.add(node_id, compressed_node)
        elif to_layer == MemoryLayer.L3:
            # Further compress node for L3
            persistent_node = self._compress_for_l3(node)
            self.l3_nodes.add(node_id, persistent_node)
        
        # Remove from source layer
        if from_layer == MemoryLayer.L1:
            self.l1_nodes.remove(node_id)
        elif from_layer == MemoryLayer.L2:
            self.l2_nodes.remove(node_id)
    
    def process_idle_nodes(self) -> None:
        """Process idle nodes and demote them if necessary."""
        # L1 -> L2 (after 10 minutes)
        idle_l1_nodes = self.l1_nodes.get_idle(minutes=10)
        for node in idle_l1_nodes:
            self.logger.debug(f"Demoting idle node {node.id} from L1 to L2")
            self.demote_node(node.id, MemoryLayer.L1, MemoryLayer.L2)
    
    def _expand_for_l1(self, node: MemoryNode) -> MemoryNode:
        """Expand a node for L1 (add more context).
        
        Args:
            node: Memory node
            
        Returns:
            Expanded memory node
        """
        # For now, just return the node as is
        # In a real implementation, we might add more context
        return node
    
    def _compress_for_l2(self, node: MemoryNode) -> MemoryNode:
        """Compress a node for L2 (remove some details).
        
        Args:
            node: Memory node
            
        Returns:
            Compressed memory node
        """
        # For now, just return the node as is
        # In a real implementation, we might remove some details
        return node
    
    def _compress_for_l3(self, node: MemoryNode) -> MemoryNode:
        """Compress a node for L3 (extract patterns).
        
        Args:
            node: Memory node
            
        Returns:
            Compressed memory node
        """
        # For now, just return the node as is
        # In a real implementation, we might extract patterns
        return node


# Global instance
layered_memory = LayeredMemoryArchitecture()