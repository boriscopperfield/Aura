"""
AURA Core System - Main orchestrator for all components.
"""
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

from ..memory.hierarchical import HierarchicalMemoryManager
from ..agents.manager import AgentManager
from ..utils.logging import get_logger
from ..utils.config import Config
from ..utils.errors import AuraError

logger = get_logger(__name__)


class AuraSystem:
    """Main AURA system orchestrator."""
    
    def __init__(self, config: Config):
        self.config = config
        self.workspace_path = Path(config.system.workspace)
        
        # Core components
        self.memory_manager: Optional[HierarchicalMemoryManager] = None
        self.agent_manager: Optional[AgentManager] = None
        
        # System state
        self.is_initialized = False
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_tasks = 0
        self.successful_tasks = 0
    
    async def initialize(self) -> None:
        """Initialize the AURA system."""
        try:
            logger.info("Initializing AURA system...")
            
            # Ensure workspace exists
            self.workspace_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize memory manager
            logger.info("Initializing memory manager...")
            self.memory_manager = HierarchicalMemoryManager(
                workspace_path=self.workspace_path,
                l1_capacity=self.config.memory.l1_capacity,
                l2_capacity=self.config.memory.l2_capacity
            )
            await self.memory_manager.initialize()
            
            # Initialize agent manager
            logger.info("Initializing agent manager...")
            agents_config_path = self.workspace_path / ".aura" / "agents.yaml"
            self.agent_manager = AgentManager(agents_config_path)
            await self.agent_manager.start()
            
            self.is_initialized = True
            logger.info("AURA system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AURA system: {e}")
            raise AuraError(f"System initialization failed: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the AURA system."""
        try:
            logger.info("Shutting down AURA system...")
            
            # Stop agent manager
            if self.agent_manager:
                await self.agent_manager.stop()
            
            # Cleanup memory manager
            if self.memory_manager:
                await self.memory_manager.cleanup()
            
            self.is_initialized = False
            logger.info("AURA system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def is_ready(self) -> bool:
        """Check if system is ready to handle requests."""
        if not self.is_initialized:
            return False
        
        try:
            # Check memory manager
            if not self.memory_manager:
                return False
            
            # Check agent manager
            if not self.agent_manager:
                return False
            
            # Check if at least one agent is available
            agents = await self.agent_manager.get_agents_for_capability(
                capability=None,  # Any capability
                exclude_unhealthy=True
            )
            
            return len(agents) > 0
            
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            "status": "healthy",
            "healthy": True,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "components": {}
        }
        
        try:
            # Check memory manager
            if self.memory_manager:
                try:
                    # Simple memory operation
                    await self.memory_manager.search("health_check", k=1)
                    health["components"]["memory"] = {"status": "healthy", "healthy": True}
                except Exception as e:
                    health["components"]["memory"] = {"status": "unhealthy", "healthy": False, "error": str(e)}
                    health["healthy"] = False
            else:
                health["components"]["memory"] = {"status": "not_initialized", "healthy": False}
                health["healthy"] = False
            
            # Check agent manager
            if self.agent_manager:
                try:
                    agent_stats = self.agent_manager.get_agent_stats()
                    health["components"]["agents"] = {
                        "status": "healthy",
                        "healthy": True,
                        "total_agents": agent_stats["total_agents"]
                    }
                except Exception as e:
                    health["components"]["agents"] = {"status": "unhealthy", "healthy": False, "error": str(e)}
                    health["healthy"] = False
            else:
                health["components"]["agents"] = {"status": "not_initialized", "healthy": False}
                health["healthy"] = False
            
            if not health["healthy"]:
                health["status"] = "unhealthy"
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health.update({
                "status": "error",
                "healthy": False,
                "error": str(e)
            })
        
        return health
    
    async def get_status(self) -> Dict[str, Any]:
        """Get detailed system status."""
        status = {
            "system": {
                "version": "4.0.0",
                "environment": self.config.system.environment,
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "workspace": str(self.workspace_path),
                "initialized": self.is_initialized
            },
            "performance": {
                "total_tasks": self.total_tasks,
                "successful_tasks": self.successful_tasks,
                "success_rate": self.successful_tasks / max(self.total_tasks, 1),
                "active_tasks": len(self.active_tasks)
            },
            "memory": {},
            "agents": {}
        }
        
        # Memory status
        if self.memory_manager:
            try:
                l1_nodes = await self.memory_manager.get_layer_nodes(self.memory_manager.MemoryLayer.L1_HOT_CACHE)
                l2_nodes = await self.memory_manager.get_layer_nodes(self.memory_manager.MemoryLayer.L2_SESSION_MEMORY)
                
                status["memory"] = {
                    "l1_nodes": len(l1_nodes),
                    "l2_nodes": len(l2_nodes),
                    "total_nodes": len(l1_nodes) + len(l2_nodes)
                }
            except Exception as e:
                status["memory"] = {"error": str(e)}
        
        # Agent status
        if self.agent_manager:
            try:
                status["agents"] = self.agent_manager.get_agent_stats()
            except Exception as e:
                status["agents"] = {"error": str(e)}
        
        return status
    
    async def execute_task(self, task_description: str, metadata: Dict[str, Any] = None) -> str:
        """Execute a task and return task ID."""
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        try:
            logger.info(f"Starting task {task_id}: {task_description}")
            
            # Record task start
            self.total_tasks += 1
            self.active_tasks[task_id] = {
                "id": task_id,
                "description": task_description,
                "status": "running",
                "start_time": datetime.now(),
                "metadata": metadata or {}
            }
            
            # For now, this is a simplified implementation
            # In a full implementation, this would:
            # 1. Parse the task description
            # 2. Create a plan using the planner
            # 3. Execute the plan using agents
            # 4. Store results in memory
            
            # Simulate task execution
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Mark task as completed
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["end_time"] = datetime.now()
            self.active_tasks[task_id]["result"] = "Task completed successfully (mock implementation)"
            
            self.successful_tasks += 1
            
            logger.info(f"Task {task_id} completed successfully")
            return task_id
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            
            # Mark task as failed
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "failed"
                self.active_tasks[task_id]["end_time"] = datetime.now()
                self.active_tasks[task_id]["error"] = str(e)
            
            raise AuraError(f"Task execution failed: {e}")
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        return self.active_tasks.get(task_id)
    
    async def search_memory(
        self,
        query: str,
        k: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search memory and return results."""
        if not self.memory_manager:
            raise AuraError("Memory manager not initialized")
        
        try:
            results = await self.memory_manager.search(query, k)
            
            # Convert results to serializable format
            serialized_results = []
            for result in results:
                serialized_results.append({
                    "node_id": result.node.id,
                    "summary": result.node.summary,
                    "score": result.score,
                    "content": result.node.get_text_content()[:500],  # Truncate content
                    "keywords": result.node.keywords,
                    "importance": result.node.importance
                })
            
            return serialized_results
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            raise AuraError(f"Memory search failed: {e}")
    
    async def add_memory(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add content to memory."""
        if not self.memory_manager:
            raise AuraError("Memory manager not initialized")
        
        try:
            from ..memory.models import (
                MemoryNode, ContentBlock, ContentType, EntityType, MemorySource
            )
            
            # Create memory node
            node = MemoryNode(
                id=f"mem_{uuid.uuid4().hex[:8]}",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                entity_type=EntityType.EXTERNAL_RESOURCE,
                source=MemorySource(type="api", user_id="system"),
                content=[
                    ContentBlock(
                        type=ContentType.TEXT,
                        data=content,
                        metadata=metadata or {}
                    )
                ],
                summary=content[:100],  # Simple summary
                keywords=[],  # Would be extracted in full implementation
                entities=[],
                relations=[],
                importance=0.5,
                access_count=0,
                last_accessed=datetime.now(),
                decay_rate=0.01
            )
            
            await self.memory_manager.add_node(node)
            return node.id
            
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise AuraError(f"Failed to add memory: {e}")
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List available agents."""
        if not self.agent_manager:
            raise AuraError("Agent manager not initialized")
        
        try:
            agent_stats = self.agent_manager.get_agent_stats()
            return [
                {
                    "name": name,
                    "stats": stats
                }
                for name, stats in agent_stats["agents"].items()
            ]
            
        except Exception as e:
            logger.error(f"Failed to list agents: {e}")
            raise AuraError(f"Failed to list agents: {e}")
    
    async def check_agent_health(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Check health of a specific agent."""
        if not self.agent_manager:
            raise AuraError("Agent manager not initialized")
        
        try:
            if agent_name not in self.agent_manager.agents:
                return None
            
            agent = self.agent_manager.agents[agent_name]
            health = await agent.health_check()
            return health
            
        except Exception as e:
            logger.error(f"Agent health check failed: {e}")
            raise AuraError(f"Agent health check failed: {e}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        metrics = {
            "system": {
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "total_tasks": self.total_tasks,
                "successful_tasks": self.successful_tasks,
                "failed_tasks": self.total_tasks - self.successful_tasks,
                "success_rate": self.successful_tasks / max(self.total_tasks, 1),
                "active_tasks": len(self.active_tasks)
            },
            "memory": {},
            "agents": {}
        }
        
        # Memory metrics
        if self.memory_manager:
            try:
                l1_nodes = await self.memory_manager.get_layer_nodes(self.memory_manager.MemoryLayer.L1_HOT_CACHE)
                l2_nodes = await self.memory_manager.get_layer_nodes(self.memory_manager.MemoryLayer.L2_SESSION_MEMORY)
                
                metrics["memory"] = {
                    "l1_nodes": len(l1_nodes),
                    "l2_nodes": len(l2_nodes),
                    "total_nodes": len(l1_nodes) + len(l2_nodes)
                }
            except Exception as e:
                metrics["memory"] = {"error": str(e)}
        
        # Agent metrics
        if self.agent_manager:
            try:
                metrics["agents"] = self.agent_manager.get_agent_stats()
            except Exception as e:
                metrics["agents"] = {"error": str(e)}
        
        return metrics