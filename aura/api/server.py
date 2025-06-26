"""
FastAPI server for AURA.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from ..core.system import AuraSystem
from ..agents.manager import AgentManager
from ..memory.hierarchical import HierarchicalMemoryManager
from ..utils.logging import get_logger
from ..utils.config import Config

logger = get_logger(__name__)

# Global system instance
aura_system: AuraSystem = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global aura_system
    
    logger.info("Starting AURA system...")
    
    # Initialize AURA system
    config = app.state.config
    aura_system = AuraSystem(config)
    await aura_system.initialize()
    
    logger.info("AURA system started successfully")
    
    yield
    
    logger.info("Shutting down AURA system...")
    
    # Cleanup AURA system
    if aura_system:
        await aura_system.shutdown()
    
    logger.info("AURA system shutdown complete")


def create_app(config: Config) -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="AURA - AI-Native Meta-OS",
        description="A unified AI system for intelligent task execution",
        version="4.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Store config in app state
    app.state.config = config
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add routes
    setup_routes(app)
    
    return app


def setup_routes(app: FastAPI):
    """Setup API routes."""
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "AURA",
            "version": "4.0.0",
            "description": "AI-Native Meta-OS",
            "status": "running"
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        try:
            if aura_system:
                health = await aura_system.health_check()
                return health
            else:
                return {"status": "starting", "healthy": False}
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=503, detail="Service unhealthy")
    
    @app.get("/ready")
    async def readiness_check():
        """Readiness check endpoint."""
        try:
            if aura_system and await aura_system.is_ready():
                return {"status": "ready", "ready": True}
            else:
                raise HTTPException(status_code=503, detail="Service not ready")
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            raise HTTPException(status_code=503, detail="Service not ready")
    
    @app.get("/status")
    async def system_status():
        """Get detailed system status."""
        try:
            if not aura_system:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            status = await aura_system.get_status()
            return status
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/execute")
    async def execute_task(
        request: Dict[str, Any],
        background_tasks: BackgroundTasks
    ):
        """Execute a task."""
        try:
            if not aura_system:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            task_description = request.get("task")
            if not task_description:
                raise HTTPException(status_code=400, detail="Task description required")
            
            # Execute task asynchronously
            task_id = await aura_system.execute_task(task_description)
            
            return {
                "task_id": task_id,
                "status": "started",
                "message": "Task execution started"
            }
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/tasks/{task_id}")
    async def get_task_status(task_id: str):
        """Get task status."""
        try:
            if not aura_system:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            task_status = await aura_system.get_task_status(task_id)
            
            if not task_status:
                raise HTTPException(status_code=404, detail="Task not found")
            
            return task_status
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/memory/search")
    async def search_memory(
        query: str,
        k: int = 10,
        filters: str = None
    ):
        """Search memory."""
        try:
            if not aura_system:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            # Parse filters if provided
            filter_dict = {}
            if filters:
                import json
                try:
                    filter_dict = json.loads(filters)
                except json.JSONDecodeError:
                    raise HTTPException(status_code=400, detail="Invalid filters JSON")
            
            results = await aura_system.search_memory(
                query=query,
                k=k,
                filters=filter_dict
            )
            
            return {
                "query": query,
                "results": results,
                "count": len(results)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/agents")
    async def list_agents():
        """List available agents."""
        try:
            if not aura_system:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            agents = await aura_system.list_agents()
            return {"agents": agents}
            
        except Exception as e:
            logger.error(f"Failed to list agents: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/agents/{agent_name}/health")
    async def check_agent_health(agent_name: str):
        """Check agent health."""
        try:
            if not aura_system:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            health = await aura_system.check_agent_health(agent_name)
            
            if health is None:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            return health
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Agent health check failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/metrics")
    async def get_metrics():
        """Get system metrics."""
        try:
            if not aura_system:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            metrics = await aura_system.get_metrics()
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/memory/add")
    async def add_memory(request: Dict[str, Any]):
        """Add content to memory."""
        try:
            if not aura_system:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            content = request.get("content")
            metadata = request.get("metadata", {})
            
            if not content:
                raise HTTPException(status_code=400, detail="Content required")
            
            node_id = await aura_system.add_memory(content, metadata)
            
            return {
                "node_id": node_id,
                "status": "added",
                "message": "Content added to memory"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Error handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )


def get_aura_system() -> AuraSystem:
    """Dependency to get AURA system instance."""
    if not aura_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    return aura_system


if __name__ == "__main__":
    # For development
    from ..utils.config import load_config
    
    config = load_config()
    app = create_app(config)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )