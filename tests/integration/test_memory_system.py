"""
Integration tests for the memory system.
"""
import pytest
import asyncio
from pathlib import Path
from aura.memory.models import MemoryNode, ContentBlock, ContentType, EntityType, MemorySource
from aura.memory.hierarchical import HierarchicalMemoryManager, MemoryLayer
from aura.memory.graph import MemoryGraph
from datetime import datetime


class TestMemorySystemIntegration:
    """Integration tests for the complete memory system."""
    
    @pytest.mark.asyncio
    async def test_memory_lifecycle(self, temp_dir, sample_memory_nodes):
        """Test complete memory lifecycle from creation to retrieval."""
        # Initialize memory system
        memory_manager = HierarchicalMemoryManager(
            workspace_path=temp_dir,
            l1_capacity=100,
            l2_capacity=500
        )
        
        await memory_manager.initialize()
        
        try:
            # Add nodes to memory
            for node in sample_memory_nodes:
                await memory_manager.add_node(node)
            
            # Verify nodes are in L1 (hot cache)
            l1_nodes = await memory_manager.get_layer_nodes(MemoryLayer.L1_HOT_CACHE)
            assert len(l1_nodes) == len(sample_memory_nodes)
            
            # Search for nodes
            results = await memory_manager.search(
                query="test content",
                k=10
            )
            
            assert len(results) > 0
            assert any("test" in result.node.summary.lower() for result in results)
            
            # Test node retrieval by ID
            node_id = sample_memory_nodes[0].id
            retrieved_node = await memory_manager.get_node(node_id)
            
            assert retrieved_node is not None
            assert retrieved_node.id == node_id
            
        finally:
            await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_memory_layer_promotion(self, temp_dir):
        """Test memory layer promotion and demotion."""
        memory_manager = HierarchicalMemoryManager(
            workspace_path=temp_dir,
            l1_capacity=2,  # Small capacity to force promotion
            l2_capacity=10
        )
        
        await memory_manager.initialize()
        
        try:
            # Create test nodes
            nodes = []
            for i in range(5):
                node = MemoryNode(
                    id=f"test_node_{i}",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    entity_type=EntityType.TASK_ARTIFACT,
                    source=MemorySource(type="test", user_id="test_user"),
                    content=[
                        ContentBlock(
                            type=ContentType.TEXT,
                            data=f"Test content {i}",
                            metadata={"language": "en"}
                        )
                    ],
                    summary=f"Test node {i}",
                    keywords=["test", f"node{i}"],
                    entities=[],
                    relations=[],
                    importance=0.5 + (i * 0.1),
                    access_count=0,
                    last_accessed=datetime.now(),
                    decay_rate=0.01
                )
                nodes.append(node)
                await memory_manager.add_node(node)
            
            # Check that only 2 nodes are in L1 (due to capacity limit)
            l1_nodes = await memory_manager.get_layer_nodes(MemoryLayer.L1_HOT_CACHE)
            assert len(l1_nodes) <= 2
            
            # Check that other nodes are in L2
            l2_nodes = await memory_manager.get_layer_nodes(MemoryLayer.L2_SESSION_MEMORY)
            assert len(l2_nodes) > 0
            
            # Access an L2 node to promote it to L1
            l2_node_id = list(l2_nodes.keys())[0]
            await memory_manager.get_node(l2_node_id)
            
            # Verify promotion occurred
            updated_l1_nodes = await memory_manager.get_layer_nodes(MemoryLayer.L1_HOT_CACHE)
            assert l2_node_id in updated_l1_nodes
            
        finally:
            await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_graph_relationship_traversal(self, temp_dir, sample_memory_nodes):
        """Test graph-based relationship traversal."""
        memory_graph = MemoryGraph()
        
        # Add nodes to graph
        for node in sample_memory_nodes:
            await memory_graph.add_node(node)
        
        # Test relationship traversal
        seed_nodes = [sample_memory_nodes[0].id]
        expanded_nodes = await memory_graph.expand_context(
            seed_nodes=seed_nodes,
            max_hops=2
        )
        
        # Should include original node and related nodes
        assert sample_memory_nodes[0].id in expanded_nodes
        
        # If there are relationships, should include related nodes
        if sample_memory_nodes[1].relations:
            assert len(expanded_nodes) > 1
    
    @pytest.mark.asyncio
    async def test_memory_persistence(self, temp_dir, sample_memory_nodes):
        """Test memory persistence across sessions."""
        # First session - add nodes
        memory_manager1 = HierarchicalMemoryManager(
            workspace_path=temp_dir,
            l1_capacity=100,
            l2_capacity=500
        )
        
        await memory_manager1.initialize()
        
        for node in sample_memory_nodes:
            await memory_manager1.add_node(node)
        
        # Force persistence
        await memory_manager1.persist_to_l3()
        await memory_manager1.cleanup()
        
        # Second session - verify nodes are still available
        memory_manager2 = HierarchicalMemoryManager(
            workspace_path=temp_dir,
            l1_capacity=100,
            l2_capacity=500
        )
        
        await memory_manager2.initialize()
        
        try:
            # Search should find persisted nodes
            results = await memory_manager2.search(
                query="test content",
                k=10
            )
            
            assert len(results) > 0
            
            # Direct retrieval should work
            for node in sample_memory_nodes:
                retrieved = await memory_manager2.get_node(node.id)
                assert retrieved is not None
                assert retrieved.id == node.id
                
        finally:
            await memory_manager2.cleanup()
    
    @pytest.mark.asyncio
    async def test_concurrent_memory_operations(self, temp_dir):
        """Test concurrent memory operations."""
        memory_manager = HierarchicalMemoryManager(
            workspace_path=temp_dir,
            l1_capacity=100,
            l2_capacity=500
        )
        
        await memory_manager.initialize()
        
        try:
            # Create multiple nodes concurrently
            async def create_node(i):
                node = MemoryNode(
                    id=f"concurrent_node_{i}",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    entity_type=EntityType.TASK_ARTIFACT,
                    source=MemorySource(type="test", user_id="test_user"),
                    content=[
                        ContentBlock(
                            type=ContentType.TEXT,
                            data=f"Concurrent test content {i}",
                            metadata={"language": "en"}
                        )
                    ],
                    summary=f"Concurrent test node {i}",
                    keywords=["concurrent", "test"],
                    entities=[],
                    relations=[],
                    importance=0.5,
                    access_count=0,
                    last_accessed=datetime.now(),
                    decay_rate=0.01
                )
                await memory_manager.add_node(node)
                return node.id
            
            # Create 10 nodes concurrently
            tasks = [create_node(i) for i in range(10)]
            node_ids = await asyncio.gather(*tasks)
            
            # Verify all nodes were created
            assert len(node_ids) == 10
            
            # Concurrent searches
            async def search_memory(query):
                return await memory_manager.search(query=query, k=5)
            
            search_tasks = [
                search_memory("concurrent"),
                search_memory("test"),
                search_memory("content")
            ]
            
            search_results = await asyncio.gather(*search_tasks)
            
            # All searches should return results
            for results in search_results:
                assert len(results) > 0
                
        finally:
            await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_memory_consistency(self, temp_dir, sample_memory_nodes):
        """Test memory consistency across operations."""
        memory_manager = HierarchicalMemoryManager(
            workspace_path=temp_dir,
            l1_capacity=100,
            l2_capacity=500
        )
        
        await memory_manager.initialize()
        
        try:
            # Add nodes
            for node in sample_memory_nodes:
                await memory_manager.add_node(node)
            
            # Verify consistency between layers
            all_nodes = {}
            
            # Collect nodes from all layers
            for layer in [MemoryLayer.L1_HOT_CACHE, MemoryLayer.L2_SESSION_MEMORY]:
                layer_nodes = await memory_manager.get_layer_nodes(layer)
                for node_id, node in layer_nodes.items():
                    if node_id in all_nodes:
                        # Same node should be identical across layers
                        assert all_nodes[node_id].id == node.id
                        assert all_nodes[node_id].summary == node.summary
                    else:
                        all_nodes[node_id] = node
            
            # Verify search consistency
            for node_id in all_nodes:
                retrieved = await memory_manager.get_node(node_id)
                assert retrieved is not None
                assert retrieved.id == node_id
                
        finally:
            await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_memory_error_handling(self, temp_dir):
        """Test memory system error handling."""
        memory_manager = HierarchicalMemoryManager(
            workspace_path=temp_dir,
            l1_capacity=100,
            l2_capacity=500
        )
        
        await memory_manager.initialize()
        
        try:
            # Test retrieving non-existent node
            result = await memory_manager.get_node("non_existent_node")
            assert result is None
            
            # Test searching with empty query
            results = await memory_manager.search(query="", k=10)
            assert isinstance(results, list)
            
            # Test adding invalid node (this should be handled gracefully)
            # Note: Depending on implementation, this might raise an exception
            # or handle the error gracefully
            
        finally:
            await memory_manager.cleanup()