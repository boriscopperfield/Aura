"""
Memory lifecycle management for AURA.

This module provides functionality for managing the lifecycle of memory nodes,
including promotion, demotion, and compression between memory layers.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union

from aura.config import settings
from aura.utils.logging import logger
from aura.memory.models import MemoryNode, EntityType
from aura.memory.hierarchy import MemoryLayer, LayeredMemoryArchitecture
from aura.memory.relationships import Relationship, EntityRelationshipModel


class MemoryLifecycleManager:
    """Manage memory lifecycle across layers."""
    
    def __init__(
        self,
        layered_memory: LayeredMemoryArchitecture,
        entity_relationship_model: EntityRelationshipModel
    ):
        """Initialize the memory lifecycle manager.
        
        Args:
            layered_memory: Layered memory architecture
            entity_relationship_model: Entity-relationship model
        """
        self.layered_memory = layered_memory
        self.entity_model = entity_relationship_model
        self.logger = logger
        
        # Initialize background task
        self.background_task = None
        self.running = False
    
    async def promote_on_access(self, entity_id: str) -> None:
        """Promote entity to higher layer on access.
        
        Args:
            entity_id: Entity ID
        """
        # Check current level
        node = None
        current_level = None
        
        # Check L3
        node = self.layered_memory.l3_nodes.get(entity_id)
        if node:
            current_level = MemoryLayer.L3
        
        # Check L2
        if not node:
            node = self.layered_memory.l2_nodes.get(entity_id)
            if node:
                current_level = MemoryLayer.L2
        
        # Check L1
        if not node:
            node = self.layered_memory.l1_nodes.get(entity_id)
            if node:
                current_level = MemoryLayer.L1
        
        if not node or not current_level:
            self.logger.warning(f"Entity {entity_id} not found in any layer")
            return
        
        # Promote based on current level
        if current_level == MemoryLayer.L3:
            # Promote from L3 to L2
            self.logger.debug(f"Promoting {entity_id} from L3 to L2")
            enhanced_node = await self._rebuild_working_context(node)
            self.layered_memory.l2_nodes.add(entity_id, enhanced_node)
            
            # Also promote related nodes
            await self._prefetch_related(entity_id, MemoryLayer.L2)
            
        elif current_level == MemoryLayer.L2:
            # Promote from L2 to L1
            self.logger.debug(f"Promoting {entity_id} from L2 to L1")
            enhanced_node = await self._expand_for_l1(node)
            self.layered_memory.l1_nodes.add(entity_id, enhanced_node)
            
            # Also promote related nodes
            await self._prefetch_related(entity_id, MemoryLayer.L1)
    
    async def demote_on_idle(self) -> None:
        """Periodically demote inactive memories."""
        # L1 -> L2 (check every minute)
        idle_l1_nodes = self.layered_memory.l1_nodes.get_idle(minutes=10)
        for node in idle_l1_nodes:
            self.logger.debug(f"Demoting idle node {node.id} from L1 to L2")
            compressed_node = await self._compress_for_l2(node)
            self.layered_memory.l2_nodes.add(node.id, compressed_node)
            self.layered_memory.l1_nodes.remove(node.id)
        
        # L2 -> L3 (at session end or for low importance nodes)
        idle_l2_nodes = self.layered_memory.l2_nodes.get_idle(minutes=60)
        for node in idle_l2_nodes:
            if node.importance < 0.3:  # Only demote low importance nodes
                self.logger.debug(f"Demoting low importance node {node.id} from L2 to L3")
                persistent_node = await self._prepare_for_l3(node)
                self.layered_memory.l3_nodes.add(node.id, persistent_node)
                self.layered_memory.l2_nodes.remove(node.id)
    
    async def start_background_task(self) -> None:
        """Start background task for memory lifecycle management."""
        if self.running:
            return
        
        self.running = True
        self.background_task = asyncio.create_task(self._background_loop())
    
    async def stop_background_task(self) -> None:
        """Stop background task for memory lifecycle management."""
        if not self.running:
            return
        
        self.running = False
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
            self.background_task = None
    
    async def _background_loop(self) -> None:
        """Background loop for memory lifecycle management."""
        while self.running:
            try:
                # Demote idle nodes
                await self.demote_on_idle()
                
                # Sleep for a minute
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in memory lifecycle background task: {e}")
                await asyncio.sleep(60)  # Sleep and retry
    
    async def _rebuild_working_context(self, node: MemoryNode) -> MemoryNode:
        """Rebuild working context for L3 -> L2 promotion.
        
        Args:
            node: Memory node
            
        Returns:
            Enhanced memory node
        """
        # For now, just return the node as is
        # In a real implementation, we would rebuild the working context
        return node
    
    async def _expand_for_l1(self, node: MemoryNode) -> MemoryNode:
        """Expand node for L2 -> L1 promotion.
        
        Args:
            node: Memory node
            
        Returns:
            Expanded memory node
        """
        # For now, just return the node as is
        # In a real implementation, we would add more context
        return node
    
    async def _compress_for_l2(self, node: MemoryNode) -> MemoryNode:
        """Compress node for L1 -> L2 demotion.
        
        Args:
            node: Memory node
            
        Returns:
            Compressed memory node
        """
        # For now, just return the node as is
        # In a real implementation, we would remove some details
        return node
    
    async def _prepare_for_l3(self, node: MemoryNode) -> MemoryNode:
        """Prepare node for L2 -> L3 demotion.
        
        Args:
            node: Memory node
            
        Returns:
            Prepared memory node
        """
        # For now, just return the node as is
        # In a real implementation, we would extract patterns
        return node
    
    async def _prefetch_related(self, entity_id: str, target_layer: MemoryLayer) -> None:
        """Prefetch related entities.
        
        Args:
            entity_id: Entity ID
            target_layer: Target memory layer
        """
        # Get related entities
        related_ids = set()
        
        # Get outgoing relationships
        outgoing = self.entity_model.get_outgoing_relationships(entity_id)
        for rel in outgoing:
            related_ids.add(rel.target_id)
        
        # Get incoming relationships
        incoming = self.entity_model.get_incoming_relationships(entity_id)
        for rel in incoming:
            related_ids.add(rel.source_id)
        
        # Limit to 5 related entities
        related_ids = list(related_ids)[:5]
        
        # Promote related entities
        for related_id in related_ids:
            # Check if entity exists in any layer
            l3_node = self.layered_memory.l3_nodes.get(related_id)
            if l3_node and target_layer in (MemoryLayer.L1, MemoryLayer.L2):
                # Promote from L3 to target layer
                if target_layer == MemoryLayer.L2:
                    enhanced_node = await self._rebuild_working_context(l3_node)
                    self.layered_memory.l2_nodes.add(related_id, enhanced_node)
                else:  # L1
                    enhanced_node = await self._expand_for_l1(l3_node)
                    self.layered_memory.l1_nodes.add(related_id, enhanced_node)
            
            l2_node = self.layered_memory.l2_nodes.get(related_id)
            if l2_node and target_layer == MemoryLayer.L1:
                # Promote from L2 to L1
                enhanced_node = await self._expand_for_l1(l2_node)
                self.layered_memory.l1_nodes.add(related_id, enhanced_node)


class HierarchicalQueryProcessor:
    """Process queries across memory layers."""
    
    def __init__(
        self,
        layered_memory: LayeredMemoryArchitecture,
        entity_relationship_model: EntityRelationshipModel,
        lifecycle_manager: MemoryLifecycleManager
    ):
        """Initialize the hierarchical query processor.
        
        Args:
            layered_memory: Layered memory architecture
            entity_relationship_model: Entity-relationship model
            lifecycle_manager: Memory lifecycle manager
        """
        self.layered_memory = layered_memory
        self.entity_model = entity_relationship_model
        self.lifecycle_manager = lifecycle_manager
        self.logger = logger
    
    async def process_query(self, query: Any) -> Dict[str, Any]:
        """Process a query across memory layers.
        
        Args:
            query: Query object
            
        Returns:
            Query results
        """
        # 1. First check L1 (fastest)
        l1_results = await self._search_l1(query)
        if self._sufficient_results(l1_results):
            return {"results": l1_results, "source": "L1"}
        
        # 2. Expand to L2 (including relationship search)
        l2_results = await self._search_l2(query)
        
        # Combine L1 and L2 results
        combined_results = self._merge_results(l1_results, l2_results)
        
        if self._sufficient_results(combined_results):
            return {"results": combined_results, "source": "L1+L2"}
        
        # 3. Query L3 if historical context needed
        if self._needs_historical_context(query):
            l3_results = await self._search_l3(query)
            
            # Combine all results
            final_results = self._merge_results(combined_results, l3_results)
            return {"results": final_results, "source": "L1+L2+L3"}
        
        return {"results": combined_results, "source": "L1+L2"}
    
    async def _search_l1(self, query: Any) -> List[Any]:
        """Search L1 memory.
        
        Args:
            query: Query object
            
        Returns:
            Search results
        """
        # For now, return empty list
        # In a real implementation, we would search L1 memory
        return []
    
    async def _search_l2(self, query: Any) -> List[Any]:
        """Search L2 memory.
        
        Args:
            query: Query object
            
        Returns:
            Search results
        """
        # For now, return empty list
        # In a real implementation, we would search L2 memory
        return []
    
    async def _search_l3(self, query: Any) -> List[Any]:
        """Search L3 memory.
        
        Args:
            query: Query object
            
        Returns:
            Search results
        """
        # For now, return empty list
        # In a real implementation, we would search L3 memory
        return []
    
    def _sufficient_results(self, results: List[Any]) -> bool:
        """Check if results are sufficient.
        
        Args:
            results: Search results
            
        Returns:
            True if results are sufficient, False otherwise
        """
        # For now, just check if we have at least 5 results
        return len(results) >= 5
    
    def _needs_historical_context(self, query: Any) -> bool:
        """Check if query needs historical context.
        
        Args:
            query: Query object
            
        Returns:
            True if query needs historical context, False otherwise
        """
        # For now, always return True
        # In a real implementation, we would analyze the query
        return True
    
    def _merge_results(self, results1: List[Any], results2: List[Any]) -> List[Any]:
        """Merge search results.
        
        Args:
            results1: First set of results
            results2: Second set of results
            
        Returns:
            Merged results
        """
        # For now, just concatenate and deduplicate
        merged = list(results1)
        
        # Add results from results2 that aren't in results1
        result_ids = {r.get("id") for r in results1}
        for r in results2:
            if r.get("id") not in result_ids:
                merged.append(r)
                result_ids.add(r.get("id"))
        
        return merged