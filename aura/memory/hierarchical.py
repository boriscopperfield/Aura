"""
Hierarchical Memory Manager for AURA.
"""
import asyncio
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from .models import MemoryNode
from .graph import MemoryGraph
from ..utils.logging import get_logger
from ..utils.errors import MemoryError

logger = get_logger(__name__)


class MemoryLayer(Enum):
    """Memory layer enumeration."""
    L1_HOT_CACHE = "l1_hot_cache"
    L2_SESSION_MEMORY = "l2_session_memory"
    L3_PERSISTENT_STORAGE = "l3_persistent_storage"


class SearchResult:
    """Search result container."""
    
    def __init__(self, node: MemoryNode, score: float, metadata: Dict[str, Any] = None):
        self.node = node
        self.score = score
        self.metadata = metadata or {}


class HierarchicalMemoryManager:
    """Hierarchical memory management system."""
    
    def __init__(
        self,
        workspace_path: Path,
        l1_capacity: int = 1000,
        l2_capacity: int = 10000
    ):
        self.workspace_path = workspace_path
        self.l1_capacity = l1_capacity
        self.l2_capacity = l2_capacity
        
        # Memory layers
        self.l1_cache: Dict[str, MemoryNode] = {}  # Hot cache
        self.l2_memory: Dict[str, MemoryNode] = {}  # Session memory
        
        # Memory graph for relationships
        self.memory_graph = MemoryGraph()
        
        # Access tracking
        self.access_times: Dict[str, datetime] = {}
        
        # Persistence paths
        self.memory_dir = workspace_path / "memory"
        self.l3_storage_path = self.memory_dir / "persistent_nodes.jsonl"
        
        # State
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the memory manager."""
        try:
            logger.info("Initializing hierarchical memory manager...")
            
            # Create memory directory
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            
            # Load persistent memory if exists
            await self._load_persistent_memory()
            
            self.is_initialized = True
            logger.info("Memory manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {e}")
            raise MemoryError(f"Memory initialization failed: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup memory manager."""
        try:
            logger.info("Cleaning up memory manager...")
            
            # Persist current memory state
            await self.persist_to_l3()
            
            # Clear caches
            self.l1_cache.clear()
            self.l2_memory.clear()
            self.access_times.clear()
            
            self.is_initialized = False
            logger.info("Memory manager cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
    
    async def add_node(self, node: MemoryNode) -> None:
        """Add a node to memory."""
        try:
            # Add to L1 cache first
            await self._add_to_l1(node)
            
            # Add to memory graph
            await self.memory_graph.add_node(node)
            
            # Update access time
            self.access_times[node.id] = datetime.now()
            
            logger.debug(f"Added node {node.id} to memory")
            
        except Exception as e:
            logger.error(f"Failed to add node {node.id}: {e}")
            raise MemoryError(f"Failed to add node: {e}")
    
    async def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve a node from memory."""
        try:
            # Check L1 first
            if node_id in self.l1_cache:
                node = self.l1_cache[node_id]
                await self._update_access(node)
                return node
            
            # Check L2
            if node_id in self.l2_memory:
                node = self.l2_memory[node_id]
                # Promote to L1
                await self._promote_to_l1(node)
                await self._update_access(node)
                return node
            
            # Check L3 (persistent storage)
            node = await self._load_from_l3(node_id)
            if node:
                # Promote to L1
                await self._promote_to_l1(node)
                await self._update_access(node)
                return node
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            raise MemoryError(f"Failed to retrieve node: {e}")
    
    async def search(
        self,
        query: str,
        k: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """Search memory for relevant nodes."""
        try:
            # For now, implement simple text-based search
            # In a full implementation, this would use vector search
            
            all_nodes = {}
            all_nodes.update(self.l1_cache)
            all_nodes.update(self.l2_memory)
            
            # Load some nodes from L3 for search
            l3_nodes = await self._sample_l3_nodes(100)
            all_nodes.update(l3_nodes)
            
            # Simple text matching
            results = []
            query_lower = query.lower()
            
            for node in all_nodes.values():
                score = 0.0
                
                # Check summary
                if query_lower in node.summary.lower():
                    score += 0.5
                
                # Check keywords
                for keyword in node.keywords:
                    if query_lower in keyword.lower():
                        score += 0.3
                
                # Check content
                text_content = node.get_text_content()
                if query_lower in text_content.lower():
                    score += 0.2
                
                if score > 0:
                    results.append(SearchResult(node, score))
            
            # Sort by score and return top k
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise MemoryError(f"Search failed: {e}")
    
    async def get_layer_nodes(self, layer: MemoryLayer) -> Dict[str, MemoryNode]:
        """Get nodes from a specific layer."""
        if layer == MemoryLayer.L1_HOT_CACHE:
            return self.l1_cache.copy()
        elif layer == MemoryLayer.L2_SESSION_MEMORY:
            return self.l2_memory.copy()
        elif layer == MemoryLayer.L3_PERSISTENT_STORAGE:
            return await self._sample_l3_nodes(1000)
        else:
            return {}
    
    async def persist_to_l3(self) -> None:
        """Persist current memory state to L3 storage."""
        try:
            # Combine all nodes
            all_nodes = {}
            all_nodes.update(self.l1_cache)
            all_nodes.update(self.l2_memory)
            
            # Load existing L3 nodes
            existing_nodes = await self._load_all_l3_nodes()
            all_nodes.update(existing_nodes)
            
            # Write to L3 storage
            with open(self.l3_storage_path, 'w') as f:
                for node in all_nodes.values():
                    f.write(json.dumps(node.model_dump()) + '\n')
            
            logger.debug(f"Persisted {len(all_nodes)} nodes to L3 storage")
            
        except Exception as e:
            logger.error(f"Failed to persist to L3: {e}")
            raise MemoryError(f"Failed to persist memory: {e}")
    
    async def _add_to_l1(self, node: MemoryNode) -> None:
        """Add node to L1 cache."""
        # Check capacity
        if len(self.l1_cache) >= self.l1_capacity:
            await self._evict_from_l1()
        
        self.l1_cache[node.id] = node
    
    async def _promote_to_l1(self, node: MemoryNode) -> None:
        """Promote node from L2 to L1."""
        # Remove from L2 if present
        if node.id in self.l2_memory:
            del self.l2_memory[node.id]
        
        # Add to L1
        await self._add_to_l1(node)
    
    async def _evict_from_l1(self) -> None:
        """Evict least recently used node from L1 to L2."""
        if not self.l1_cache:
            return
        
        # Find least recently used node
        lru_node_id = min(
            self.l1_cache.keys(),
            key=lambda nid: self.access_times.get(nid, datetime.min)
        )
        
        # Move to L2
        node = self.l1_cache.pop(lru_node_id)
        await self._add_to_l2(node)
    
    async def _add_to_l2(self, node: MemoryNode) -> None:
        """Add node to L2 memory."""
        # Check capacity
        if len(self.l2_memory) >= self.l2_capacity:
            await self._evict_from_l2()
        
        self.l2_memory[node.id] = node
    
    async def _evict_from_l2(self) -> None:
        """Evict least recently used node from L2."""
        if not self.l2_memory:
            return
        
        # Find least recently used node
        lru_node_id = min(
            self.l2_memory.keys(),
            key=lambda nid: self.access_times.get(nid, datetime.min)
        )
        
        # Remove from L2 (will be persisted to L3 later)
        del self.l2_memory[lru_node_id]
    
    async def _update_access(self, node: MemoryNode) -> None:
        """Update node access statistics."""
        node.update_access_stats()
        self.access_times[node.id] = datetime.now()
    
    async def _load_persistent_memory(self) -> None:
        """Load persistent memory from L3 storage."""
        if not self.l3_storage_path.exists():
            return
        
        try:
            with open(self.l3_storage_path, 'r') as f:
                for line in f:
                    if line.strip():
                        node_data = json.loads(line)
                        node = MemoryNode(**node_data)
                        
                        # Add to L2 initially
                        if len(self.l2_memory) < self.l2_capacity:
                            self.l2_memory[node.id] = node
                            await self.memory_graph.add_node(node)
            
            logger.info(f"Loaded {len(self.l2_memory)} nodes from persistent storage")
            
        except Exception as e:
            logger.error(f"Failed to load persistent memory: {e}")
    
    async def _load_from_l3(self, node_id: str) -> Optional[MemoryNode]:
        """Load a specific node from L3 storage."""
        if not self.l3_storage_path.exists():
            return None
        
        try:
            with open(self.l3_storage_path, 'r') as f:
                for line in f:
                    if line.strip():
                        node_data = json.loads(line)
                        if node_data.get('id') == node_id:
                            return MemoryNode(**node_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load node {node_id} from L3: {e}")
            return None
    
    async def _sample_l3_nodes(self, max_nodes: int) -> Dict[str, MemoryNode]:
        """Sample nodes from L3 storage."""
        nodes = {}
        
        if not self.l3_storage_path.exists():
            return nodes
        
        try:
            count = 0
            with open(self.l3_storage_path, 'r') as f:
                for line in f:
                    if line.strip() and count < max_nodes:
                        node_data = json.loads(line)
                        node = MemoryNode(**node_data)
                        nodes[node.id] = node
                        count += 1
            
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to sample L3 nodes: {e}")
            return nodes
    
    async def _load_all_l3_nodes(self) -> Dict[str, MemoryNode]:
        """Load all nodes from L3 storage."""
        nodes = {}
        
        if not self.l3_storage_path.exists():
            return nodes
        
        try:
            with open(self.l3_storage_path, 'r') as f:
                for line in f:
                    if line.strip():
                        node_data = json.loads(line)
                        node = MemoryNode(**node_data)
                        nodes[node.id] = node
            
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to load all L3 nodes: {e}")
            return nodes