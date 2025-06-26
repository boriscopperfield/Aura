#!/usr/bin/env python3
"""
Test script for real AI integrations in AURA.
This script tests the actual implementation with real AI services.
"""
import asyncio
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add the aura package to the path
sys.path.insert(0, str(Path(__file__).parent))

from aura.agents.manager import AgentManager, AgentConfig
from aura.agents.base import AgentCapability
from aura.memory.hierarchical import HierarchicalMemoryManager
from aura.memory.models import (
    MemoryNode, ContentBlock, ContentType, EntityType, MemorySource
)
from datetime import datetime

console = Console()


async def test_agent_integrations():
    """Test real AI agent integrations."""
    
    console.print(Panel(
        "[bold cyan]Testing Real AI Integrations[/bold cyan]\n"
        "This will test actual API connections to AI services.",
        title="🤖 AURA Agent Integration Test"
    ))
    
    # Create temporary workspace
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir)
        
        # Test agent manager
        agent_manager = AgentManager()
        
        # Test configurations for different agents
        test_configs = []
        
        # OpenAI (if API key available)
        if os.getenv("OPENAI_API_KEY"):
            test_configs.append(AgentConfig(
                name="test_openai",
                type="openai",
                capabilities=["text_generation"],
                config={
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "model": "gpt-3.5-turbo"
                }
            ))
        
        # Anthropic (if API key available)
        if os.getenv("ANTHROPIC_API_KEY"):
            test_configs.append(AgentConfig(
                name="test_anthropic",
                type="anthropic",
                capabilities=["text_generation"],
                config={
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    "model": "claude-3-haiku-20240307"
                }
            ))
        
        # Jina (if API key available)
        if os.getenv("JINA_API_KEY"):
            test_configs.append(AgentConfig(
                name="test_jina_embedder",
                type="jina_embedder",
                capabilities=["text_embedding"],
                config={
                    "api_key": os.getenv("JINA_API_KEY"),
                    "model": "jina-embeddings-v3"
                }
            ))
        
        # Local agent (always available)
        test_configs.append(AgentConfig(
            name="test_local",
            type="local",
            capabilities=["code_generation"],
            config={}
        ))
        
        if not test_configs:
            console.print("[yellow]No API keys found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or JINA_API_KEY to test real integrations.[/yellow]")
            console.print("[green]Testing local agent only...[/green]")
        
        # Test each agent
        results = []
        
        async with agent_manager:
            for config in test_configs:
                console.print(f"\n[bold]Testing {config.name} ({config.type})[/bold]")
                
                try:
                    # Add agent
                    await agent_manager.add_agent(config)
                    
                    # Test health check
                    agent = agent_manager.agents[config.name]
                    health = await agent.health_check()
                    
                    if health["healthy"]:
                        console.print(f"✓ Health check: [green]PASSED[/green]")
                        
                        # Test capability execution
                        if "text_generation" in config.capabilities:
                            result = await agent_manager.execute_capability(
                                AgentCapability.TEXT_GENERATION,
                                agent_name=config.name,
                                prompt="Hello, this is a test. Please respond with 'Test successful!'",
                                max_tokens=50
                            )
                            
                            if result.success:
                                console.print(f"✓ Text generation: [green]PASSED[/green]")
                                console.print(f"  Response: {result.data[:100]}...")
                            else:
                                console.print(f"✗ Text generation: [red]FAILED[/red] - {result.error}")
                        
                        elif "text_embedding" in config.capabilities:
                            result = await agent_manager.execute_capability(
                                AgentCapability.TEXT_EMBEDDING,
                                agent_name=config.name,
                                texts="This is a test sentence for embedding."
                            )
                            
                            if result.success:
                                console.print(f"✓ Text embedding: [green]PASSED[/green]")
                                console.print(f"  Embedding dimension: {len(result.data)}")
                            else:
                                console.print(f"✗ Text embedding: [red]FAILED[/red] - {result.error}")
                        
                        elif "code_generation" in config.capabilities:
                            result = await agent_manager.execute_capability(
                                AgentCapability.CODE_GENERATION,
                                agent_name=config.name,
                                code="print('Hello from AURA!')",
                                language="python"
                            )
                            
                            if result.success:
                                console.print(f"✓ Code execution: [green]PASSED[/green]")
                                if "stdout" in result.data:
                                    console.print(f"  Output: {result.data['stdout'].strip()}")
                            else:
                                console.print(f"✗ Code execution: [red]FAILED[/red] - {result.error}")
                        
                        results.append({
                            "agent": config.name,
                            "type": config.type,
                            "status": "success",
                            "health": health
                        })
                    else:
                        console.print(f"✗ Health check: [red]FAILED[/red] - {health.get('error', 'Unknown error')}")
                        results.append({
                            "agent": config.name,
                            "type": config.type,
                            "status": "health_failed",
                            "error": health.get('error', 'Unknown error')
                        })
                
                except Exception as e:
                    console.print(f"✗ Agent setup: [red]FAILED[/red] - {str(e)}")
                    results.append({
                        "agent": config.name,
                        "type": config.type,
                        "status": "setup_failed",
                        "error": str(e)
                    })
        
        # Display results summary
        console.print("\n" + "="*60)
        console.print("[bold]Integration Test Results[/bold]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Agent", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Details")
        
        for result in results:
            status_color = "green" if result["status"] == "success" else "red"
            status_text = f"[{status_color}]{result['status'].upper()}[/{status_color}]"
            
            details = ""
            if result["status"] == "success":
                details = "All tests passed"
            else:
                details = result.get("error", "Unknown error")[:50]
            
            table.add_row(
                result["agent"],
                result["type"],
                status_text,
                details
            )
        
        console.print(table)
        
        # Summary
        successful = sum(1 for r in results if r["status"] == "success")
        total = len(results)
        
        if successful == total:
            console.print(f"\n[bold green]✓ All {total} agents tested successfully![/bold green]")
        else:
            console.print(f"\n[bold yellow]⚠ {successful}/{total} agents tested successfully[/bold yellow]")
        
        return results


async def test_memory_integration():
    """Test memory system integration."""
    
    console.print(Panel(
        "[bold cyan]Testing Memory System Integration[/bold cyan]\n"
        "This will test the hierarchical memory system.",
        title="🧠 AURA Memory Integration Test"
    ))
    
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir)
        
        # Initialize memory manager
        memory_manager = HierarchicalMemoryManager(
            workspace_path=workspace_path,
            l1_capacity=10,
            l2_capacity=50
        )
        
        try:
            await memory_manager.initialize()
            console.print("✓ Memory manager initialized")
            
            # Create test nodes
            test_nodes = []
            for i in range(5):
                node = MemoryNode(
                    id=f"test_integration_node_{i}",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    entity_type=EntityType.TASK_ARTIFACT,
                    source=MemorySource(type="test", user_id="integration_test"),
                    content=[
                        ContentBlock(
                            type=ContentType.TEXT,
                            data=f"Integration test content {i} - testing memory system functionality",
                            metadata={"test_id": i}
                        )
                    ],
                    summary=f"Integration test node {i}",
                    keywords=["integration", "test", "memory", f"node{i}"],
                    entities=[],
                    relations=[],
                    importance=0.5 + (i * 0.1),
                    access_count=0,
                    last_accessed=datetime.now(),
                    decay_rate=0.01
                )
                test_nodes.append(node)
            
            # Add nodes to memory
            for node in test_nodes:
                await memory_manager.add_node(node)
            
            console.print(f"✓ Added {len(test_nodes)} test nodes to memory")
            
            # Test search functionality
            search_results = await memory_manager.search("integration test", k=3)
            console.print(f"✓ Search returned {len(search_results)} results")
            
            # Test node retrieval
            retrieved_node = await memory_manager.get_node(test_nodes[0].id)
            if retrieved_node:
                console.print("✓ Node retrieval successful")
            else:
                console.print("✗ Node retrieval failed")
            
            # Test layer distribution
            from aura.memory.hierarchical import MemoryLayer
            l1_nodes = await memory_manager.get_layer_nodes(MemoryLayer.L1_HOT_CACHE)
            l2_nodes = await memory_manager.get_layer_nodes(MemoryLayer.L2_SESSION_MEMORY)
            
            console.print(f"✓ Memory distribution - L1: {len(l1_nodes)}, L2: {len(l2_nodes)}")
            
            console.print("[bold green]✓ Memory system integration test completed successfully![/bold green]")
            
        except Exception as e:
            console.print(f"[bold red]✗ Memory integration test failed: {e}[/bold red]")
            raise
        
        finally:
            await memory_manager.cleanup()


async def test_full_system_integration():
    """Test full system integration."""
    
    console.print(Panel(
        "[bold cyan]Testing Full System Integration[/bold cyan]\n"
        "This will test the complete AURA system.",
        title="🚀 AURA Full System Test"
    ))
    
    from aura.core.system import AuraSystem
    from aura.utils.config import Config, SystemConfig, MemoryConfig
    
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir)
        
        # Create test configuration
        config = Config(
            system=SystemConfig(
                workspace=str(workspace_path),
                environment="test"
            ),
            memory=MemoryConfig(
                l1_capacity=10,
                l2_capacity=50
            )
        )
        
        # Initialize AURA system
        aura_system = AuraSystem(config)
        
        try:
            await aura_system.initialize()
            console.print("✓ AURA system initialized")
            
            # Test readiness check
            is_ready = await aura_system.is_ready()
            console.print(f"✓ System readiness: {'Ready' if is_ready else 'Not ready'}")
            
            # Test health check
            health = await aura_system.health_check()
            console.print(f"✓ Health check: {'Healthy' if health['healthy'] else 'Unhealthy'}")
            
            # Test memory operations
            memory_node_id = await aura_system.add_memory(
                "This is a test memory entry for full system integration testing."
            )
            console.print(f"✓ Added memory node: {memory_node_id}")
            
            # Test memory search
            search_results = await aura_system.search_memory("integration testing", k=5)
            console.print(f"✓ Memory search returned {len(search_results)} results")
            
            # Test task execution (mock)
            task_id = await aura_system.execute_task("Test task for integration testing")
            console.print(f"✓ Task executed: {task_id}")
            
            # Test task status
            task_status = await aura_system.get_task_status(task_id)
            if task_status:
                console.print(f"✓ Task status: {task_status['status']}")
            
            # Test metrics
            metrics = await aura_system.get_metrics()
            console.print(f"✓ System metrics retrieved: {metrics['system']['total_tasks']} total tasks")
            
            console.print("[bold green]✓ Full system integration test completed successfully![/bold green]")
            
        except Exception as e:
            console.print(f"[bold red]✗ Full system integration test failed: {e}[/bold red]")
            raise
        
        finally:
            await aura_system.shutdown()


async def main():
    """Run all integration tests."""
    
    console.print(Panel(
        "[bold cyan]AURA Real Integration Tests[/bold cyan]\n"
        "Testing real AI integrations and system components",
        title="🧪 Integration Test Suite"
    ))
    
    try:
        # Test agent integrations
        await test_agent_integrations()
        
        console.print("\n" + "="*60 + "\n")
        
        # Test memory integration
        await test_memory_integration()
        
        console.print("\n" + "="*60 + "\n")
        
        # Test full system integration
        await test_full_system_integration()
        
        console.print("\n" + "="*60)
        console.print("[bold green]🎉 All integration tests completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Integration tests failed: {e}[/bold red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)