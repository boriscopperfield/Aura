"""
Performance tests for the memory system.
"""
import pytest
import asyncio
import time
import statistics
from datetime import datetime
from aura.memory.models import MemoryNode, ContentBlock, ContentType, EntityType, MemorySource
from aura.memory.hierarchical import HierarchicalMemoryManager


class TestMemoryPerformance:
    """Performance tests for memory operations."""
    
    @pytest.mark.asyncio
    async def test_memory_insertion_performance(self, temp_dir, performance_config):
        """Test memory insertion performance."""
        memory_manager = HierarchicalMemoryManager(
            workspace_path=temp_dir,
            l1_capacity=1000,
            l2_capacity=5000
        )
        
        await memory_manager.initialize()
        
        try:
            num_nodes = performance_config["memory_nodes"]
            insertion_times = []
            
            for i in range(num_nodes):
                node = MemoryNode(
                    id=f"perf_node_{i}",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    entity_type=EntityType.TASK_ARTIFACT,
                    source=MemorySource(type="test", user_id="test_user"),
                    content=[
                        ContentBlock(
                            type=ContentType.TEXT,
                            data=f"Performance test content {i} " * 10,  # Longer content
                            metadata={"language": "en"}
                        )
                    ],
                    summary=f"Performance test node {i}",
                    keywords=["performance", "test", f"node{i}"],
                    entities=[],
                    relations=[],
                    importance=0.5,
                    access_count=0,
                    last_accessed=datetime.now(),
                    decay_rate=0.01
                )
                
                start_time = time.time()
                await memory_manager.add_node(node)
                end_time = time.time()
                
                insertion_times.append(end_time - start_time)
            
            # Analyze performance
            avg_insertion_time = statistics.mean(insertion_times)
            max_insertion_time = max(insertion_times)
            p95_insertion_time = statistics.quantiles(insertion_times, n=20)[18]  # 95th percentile
            
            print(f"\nMemory Insertion Performance:")
            print(f"  Nodes inserted: {num_nodes}")
            print(f"  Average insertion time: {avg_insertion_time:.4f}s")
            print(f"  Max insertion time: {max_insertion_time:.4f}s")
            print(f"  95th percentile: {p95_insertion_time:.4f}s")
            
            # Performance assertions
            assert avg_insertion_time < 0.1, f"Average insertion time too slow: {avg_insertion_time:.4f}s"
            assert max_insertion_time < 0.5, f"Max insertion time too slow: {max_insertion_time:.4f}s"
            
        finally:
            await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_memory_search_performance(self, temp_dir, performance_config):
        """Test memory search performance."""
        memory_manager = HierarchicalMemoryManager(
            workspace_path=temp_dir,
            l1_capacity=1000,
            l2_capacity=5000
        )
        
        await memory_manager.initialize()
        
        try:
            # Insert test data
            num_nodes = performance_config["memory_nodes"]
            
            for i in range(num_nodes):
                node = MemoryNode(
                    id=f"search_perf_node_{i}",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    entity_type=EntityType.TASK_ARTIFACT,
                    source=MemorySource(type="test", user_id="test_user"),
                    content=[
                        ContentBlock(
                            type=ContentType.TEXT,
                            data=f"Searchable content about topic {i % 10}",
                            metadata={"language": "en"}
                        )
                    ],
                    summary=f"Search performance node {i}",
                    keywords=["search", "performance", f"topic{i % 10}"],
                    entities=[],
                    relations=[],
                    importance=0.5,
                    access_count=0,
                    last_accessed=datetime.now(),
                    decay_rate=0.01
                )
                await memory_manager.add_node(node)
            
            # Test search performance
            search_queries = [
                "topic 1",
                "search performance",
                "content about",
                "node",
                "searchable"
            ]
            
            search_times = []
            
            for query in search_queries:
                start_time = time.time()
                results = await memory_manager.search(query=query, k=20)
                end_time = time.time()
                
                search_times.append(end_time - start_time)
                assert len(results) > 0, f"No results for query: {query}"
            
            # Analyze search performance
            avg_search_time = statistics.mean(search_times)
            max_search_time = max(search_times)
            
            print(f"\nMemory Search Performance:")
            print(f"  Database size: {num_nodes} nodes")
            print(f"  Queries tested: {len(search_queries)}")
            print(f"  Average search time: {avg_search_time:.4f}s")
            print(f"  Max search time: {max_search_time:.4f}s")
            
            # Performance assertions
            max_response_time = performance_config["max_response_time"]
            assert avg_search_time < max_response_time, f"Average search time too slow: {avg_search_time:.4f}s"
            assert max_search_time < max_response_time * 2, f"Max search time too slow: {max_search_time:.4f}s"
            
        finally:
            await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_concurrent_memory_performance(self, temp_dir, performance_config):
        """Test concurrent memory operations performance."""
        memory_manager = HierarchicalMemoryManager(
            workspace_path=temp_dir,
            l1_capacity=1000,
            l2_capacity=5000
        )
        
        await memory_manager.initialize()
        
        try:
            # Pre-populate with some data
            for i in range(100):
                node = MemoryNode(
                    id=f"concurrent_base_node_{i}",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    entity_type=EntityType.TASK_ARTIFACT,
                    source=MemorySource(type="test", user_id="test_user"),
                    content=[
                        ContentBlock(
                            type=ContentType.TEXT,
                            data=f"Base content {i}",
                            metadata={"language": "en"}
                        )
                    ],
                    summary=f"Base node {i}",
                    keywords=["base", "concurrent"],
                    entities=[],
                    relations=[],
                    importance=0.5,
                    access_count=0,
                    last_accessed=datetime.now(),
                    decay_rate=0.01
                )
                await memory_manager.add_node(node)
            
            # Test concurrent operations
            concurrent_requests = performance_config["concurrent_requests"]
            
            async def concurrent_operation(operation_id):
                """Perform mixed operations concurrently."""
                operations = []
                
                # Mix of insertions, searches, and retrievals
                for i in range(10):
                    if i % 3 == 0:
                        # Insert operation
                        node = MemoryNode(
                            id=f"concurrent_op_{operation_id}_{i}",
                            created_at=datetime.now(),
                            updated_at=datetime.now(),
                            entity_type=EntityType.TASK_ARTIFACT,
                            source=MemorySource(type="test", user_id="test_user"),
                            content=[
                                ContentBlock(
                                    type=ContentType.TEXT,
                                    data=f"Concurrent operation {operation_id} content {i}",
                                    metadata={"language": "en"}
                                )
                            ],
                            summary=f"Concurrent node {operation_id}_{i}",
                            keywords=["concurrent", "operation"],
                            entities=[],
                            relations=[],
                            importance=0.5,
                            access_count=0,
                            last_accessed=datetime.now(),
                            decay_rate=0.01
                        )
                        operations.append(memory_manager.add_node(node))
                    elif i % 3 == 1:
                        # Search operation
                        operations.append(memory_manager.search(query="concurrent", k=5))
                    else:
                        # Retrieval operation
                        operations.append(memory_manager.get_node(f"concurrent_base_node_{i % 100}"))
                
                return await asyncio.gather(*operations)
            
            # Run concurrent operations
            start_time = time.time()
            
            tasks = [concurrent_operation(i) for i in range(concurrent_requests)]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Analyze concurrent performance
            total_operations = concurrent_requests * 10
            operations_per_second = total_operations / total_time
            
            print(f"\nConcurrent Memory Performance:")
            print(f"  Concurrent workers: {concurrent_requests}")
            print(f"  Total operations: {total_operations}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Operations per second: {operations_per_second:.2f}")
            
            # Verify all operations completed successfully
            assert len(results) == concurrent_requests
            
            # Performance assertion
            min_ops_per_second = 50  # Minimum acceptable throughput
            assert operations_per_second > min_ops_per_second, \
                f"Throughput too low: {operations_per_second:.2f} ops/sec"
            
        finally:
            await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_memory_layer_performance(self, temp_dir, performance_config):
        """Test performance across memory layers."""
        memory_manager = HierarchicalMemoryManager(
            workspace_path=temp_dir,
            l1_capacity=50,   # Small L1 to force layer transitions
            l2_capacity=200   # Small L2 to force L3 usage
        )
        
        await memory_manager.initialize()
        
        try:
            # Insert enough nodes to populate all layers
            num_nodes = 300
            
            for i in range(num_nodes):
                node = MemoryNode(
                    id=f"layer_perf_node_{i}",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    entity_type=EntityType.TASK_ARTIFACT,
                    source=MemorySource(type="test", user_id="test_user"),
                    content=[
                        ContentBlock(
                            type=ContentType.TEXT,
                            data=f"Layer performance content {i}",
                            metadata={"language": "en"}
                        )
                    ],
                    summary=f"Layer performance node {i}",
                    keywords=["layer", "performance"],
                    entities=[],
                    relations=[],
                    importance=0.5 + (i % 10) * 0.05,  # Varying importance
                    access_count=0,
                    last_accessed=datetime.now(),
                    decay_rate=0.01
                )
                await memory_manager.add_node(node)
            
            # Test retrieval performance from different layers
            l1_times = []
            l2_times = []
            l3_times = []
            
            # Test L1 retrieval (recent nodes)
            for i in range(num_nodes - 10, num_nodes):
                start_time = time.time()
                node = await memory_manager.get_node(f"layer_perf_node_{i}")
                end_time = time.time()
                
                if node:
                    l1_times.append(end_time - start_time)
            
            # Test L2 retrieval (middle nodes)
            for i in range(100, 110):
                start_time = time.time()
                node = await memory_manager.get_node(f"layer_perf_node_{i}")
                end_time = time.time()
                
                if node:
                    l2_times.append(end_time - start_time)
            
            # Test L3 retrieval (old nodes)
            for i in range(0, 10):
                start_time = time.time()
                node = await memory_manager.get_node(f"layer_perf_node_{i}")
                end_time = time.time()
                
                if node:
                    l3_times.append(end_time - start_time)
            
            # Analyze layer performance
            if l1_times:
                avg_l1_time = statistics.mean(l1_times)
                print(f"\nMemory Layer Performance:")
                print(f"  L1 (Hot Cache) avg retrieval: {avg_l1_time:.4f}s")
            
            if l2_times:
                avg_l2_time = statistics.mean(l2_times)
                print(f"  L2 (Session) avg retrieval: {avg_l2_time:.4f}s")
            
            if l3_times:
                avg_l3_time = statistics.mean(l3_times)
                print(f"  L3 (Persistent) avg retrieval: {avg_l3_time:.4f}s")
            
            # Performance assertions (L1 should be fastest)
            if l1_times and l2_times:
                assert statistics.mean(l1_times) <= statistics.mean(l2_times), \
                    "L1 should be faster than L2"
            
            if l2_times and l3_times:
                assert statistics.mean(l2_times) <= statistics.mean(l3_times), \
                    "L2 should be faster than L3"
            
        finally:
            await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_memory_scalability(self, temp_dir):
        """Test memory system scalability with large datasets."""
        memory_manager = HierarchicalMemoryManager(
            workspace_path=temp_dir,
            l1_capacity=1000,
            l2_capacity=5000
        )
        
        await memory_manager.initialize()
        
        try:
            # Test with increasing dataset sizes
            dataset_sizes = [100, 500, 1000, 2000]
            performance_results = {}
            
            for size in dataset_sizes:
                print(f"\nTesting scalability with {size} nodes...")
                
                # Clear previous data
                await memory_manager.cleanup()
                await memory_manager.initialize()
                
                # Insert data
                insert_start = time.time()
                for i in range(size):
                    node = MemoryNode(
                        id=f"scale_node_{size}_{i}",
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        entity_type=EntityType.TASK_ARTIFACT,
                        source=MemorySource(type="test", user_id="test_user"),
                        content=[
                            ContentBlock(
                                type=ContentType.TEXT,
                                data=f"Scalability test content {i}",
                                metadata={"language": "en"}
                            )
                        ],
                        summary=f"Scalability node {i}",
                        keywords=["scalability", "test"],
                        entities=[],
                        relations=[],
                        importance=0.5,
                        access_count=0,
                        last_accessed=datetime.now(),
                        decay_rate=0.01
                    )
                    await memory_manager.add_node(node)
                
                insert_time = time.time() - insert_start
                
                # Test search performance
                search_start = time.time()
                results = await memory_manager.search(query="scalability", k=20)
                search_time = time.time() - search_start
                
                performance_results[size] = {
                    "insert_time": insert_time,
                    "search_time": search_time,
                    "insert_rate": size / insert_time,
                    "results_found": len(results)
                }
                
                print(f"  Insert time: {insert_time:.2f}s ({size/insert_time:.1f} nodes/sec)")
                print(f"  Search time: {search_time:.4f}s")
                print(f"  Results found: {len(results)}")
            
            # Analyze scalability
            print(f"\nScalability Analysis:")
            for size, metrics in performance_results.items():
                print(f"  {size} nodes: {metrics['insert_rate']:.1f} inserts/sec, "
                      f"{metrics['search_time']:.4f}s search")
            
            # Check that performance doesn't degrade too much with scale
            small_insert_rate = performance_results[dataset_sizes[0]]["insert_rate"]
            large_insert_rate = performance_results[dataset_sizes[-1]]["insert_rate"]
            
            # Allow some degradation but not more than 50%
            assert large_insert_rate > small_insert_rate * 0.5, \
                f"Insert rate degraded too much: {large_insert_rate:.1f} vs {small_insert_rate:.1f}"
            
        finally:
            await memory_manager.cleanup()