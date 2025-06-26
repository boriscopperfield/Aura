"""
Agent Manager for coordinating multiple AI agents.
"""
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import yaml
from pathlib import Path

from .base import Agent, AgentCapability, AgentResult
from .openai_agent import OpenAIAgent
from .jina_agent import JinaEmbedder, JinaReranker
from .anthropic_agent import AnthropicAgent
from .local_agent import LocalAgent
from ..utils.errors import AgentError, NoAgentError
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentConfig:
    """Agent configuration."""
    name: str
    type: str
    capabilities: List[str]
    config: Dict[str, Any]
    enabled: bool = True
    priority: int = 1


class LoadBalancer:
    """Simple round-robin load balancer for agents."""
    
    def __init__(self):
        self.agent_counters: Dict[str, int] = {}
    
    def select_agent(self, agents: List[Agent]) -> Agent:
        """Select agent using round-robin."""
        if not agents:
            raise NoAgentError("No agents available")
        
        if len(agents) == 1:
            return agents[0]
        
        # Get agent type for load balancing
        agent_type = agents[0].__class__.__name__
        
        # Initialize counter if needed
        if agent_type not in self.agent_counters:
            self.agent_counters[agent_type] = 0
        
        # Select agent
        selected = agents[self.agent_counters[agent_type] % len(agents)]
        self.agent_counters[agent_type] += 1
        
        return selected


class AgentManager:
    """Manages and coordinates multiple AI agents."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.agents: Dict[str, Agent] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.load_balancer = LoadBalancer()
        self.health_check_interval = 300  # 5 minutes
        self._health_check_task: Optional[asyncio.Task] = None
        
        if config_path and config_path.exists():
            self.load_config(config_path)
    
    def load_config(self, config_path: Path) -> None:
        """Load agent configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            for agent_name, agent_data in config_data.get('agents', {}).items():
                config = AgentConfig(
                    name=agent_name,
                    type=agent_data['type'],
                    capabilities=agent_data.get('capabilities', []),
                    config=agent_data.get('config', {}),
                    enabled=agent_data.get('enabled', True),
                    priority=agent_data.get('priority', 1)
                )
                self.agent_configs[agent_name] = config
                
                if config.enabled:
                    self._create_agent(config)
                    
        except Exception as e:
            logger.error(f"Failed to load agent config: {e}")
            raise AgentError(f"Failed to load agent config: {e}")
    
    def _create_agent(self, config: AgentConfig) -> Agent:
        """Create agent instance from configuration."""
        try:
            if config.type == "openai":
                agent = OpenAIAgent(**config.config)
            elif config.type == "jina_embedder":
                agent = JinaEmbedder(**config.config)
            elif config.type == "jina_reranker":
                agent = JinaReranker(**config.config)
            elif config.type == "anthropic":
                agent = AnthropicAgent(**config.config)
            elif config.type == "local":
                agent = LocalAgent(**config.config)
            else:
                raise AgentError(f"Unknown agent type: {config.type}")
            
            self.agents[config.name] = agent
            logger.info(f"Created agent: {config.name} ({config.type})")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create agent {config.name}: {e}")
            raise AgentError(f"Failed to create agent {config.name}: {e}")
    
    async def start(self) -> None:
        """Start the agent manager."""
        logger.info("Starting agent manager...")
        
        # Initialize all agents
        for agent in self.agents.values():
            await agent.__aenter__()
        
        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"Agent manager started with {len(self.agents)} agents")
    
    async def stop(self) -> None:
        """Stop the agent manager."""
        logger.info("Stopping agent manager...")
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all agents
        for agent in self.agents.values():
            await agent.__aexit__(None, None, None)
        
        logger.info("Agent manager stopped")
    
    async def get_agents_for_capability(
        self,
        capability: AgentCapability,
        exclude_unhealthy: bool = True
    ) -> List[Agent]:
        """Get all agents that support a capability."""
        agents = []
        
        for agent in self.agents.values():
            if capability in agent.capabilities:
                if exclude_unhealthy:
                    health = await agent.health_check()
                    if health["healthy"]:
                        agents.append(agent)
                else:
                    agents.append(agent)
        
        # Sort by priority (if available in config)
        def get_priority(agent: Agent) -> int:
            config = self.agent_configs.get(agent.name)
            return config.priority if config else 1
        
        agents.sort(key=get_priority, reverse=True)
        return agents
    
    async def select_best_agent(
        self,
        capability: AgentCapability,
        criteria: Optional[Dict[str, Any]] = None
    ) -> Agent:
        """Select the best agent for a capability based on criteria."""
        agents = await self.get_agents_for_capability(capability)
        
        if not agents:
            raise NoAgentError(f"No agents available for capability: {capability.value}")
        
        # If no criteria, use load balancer
        if not criteria:
            return self.load_balancer.select_agent(agents)
        
        # Score agents based on criteria
        scored_agents = []
        for agent in agents:
            score = await self._score_agent(agent, criteria)
            scored_agents.append((agent, score))
        
        # Return highest scoring agent
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        return scored_agents[0][0]
    
    async def _score_agent(self, agent: Agent, criteria: Dict[str, Any]) -> float:
        """Score an agent based on selection criteria."""
        score = 0.0
        
        # Performance score
        stats = agent.get_performance_stats()
        score += stats["success_rate"] * 0.4
        
        # Speed score (inverse of average duration)
        if stats["average_duration"] > 0:
            speed_score = 1.0 / (1.0 + stats["average_duration"])
            score += speed_score * 0.3
        
        # Cost score (if cost limit specified)
        max_cost = criteria.get("max_cost")
        if max_cost and stats["total_cost"] > 0:
            cost_score = max(0, 1.0 - (stats["total_cost"] / max_cost))
            score += cost_score * 0.2
        
        # Circuit breaker penalty
        if agent.circuit_breaker.state == "OPEN":
            score *= 0.1  # Heavy penalty for open circuit
        elif agent.circuit_breaker.state == "HALF_OPEN":
            score *= 0.7  # Moderate penalty for half-open
        
        # Priority bonus
        config = self.agent_configs.get(agent.name)
        if config:
            score += config.priority * 0.1
        
        return score
    
    async def execute_capability(
        self,
        capability: AgentCapability,
        agent_name: Optional[str] = None,
        criteria: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResult:
        """Execute a capability using the best available agent."""
        try:
            if agent_name:
                # Use specific agent
                if agent_name not in self.agents:
                    raise NoAgentError(f"Agent not found: {agent_name}")
                agent = self.agents[agent_name]
                
                if capability not in agent.capabilities:
                    raise AgentError(f"Agent {agent_name} does not support {capability.value}")
            else:
                # Select best agent
                agent = await self.select_best_agent(capability, criteria)
            
            # Execute capability
            logger.info(f"Executing {capability.value} using agent {agent.name}")
            result = await agent.execute(capability, **kwargs)
            
            if result.success:
                logger.info(f"Successfully executed {capability.value} in {result.duration:.2f}s")
            else:
                logger.error(f"Failed to execute {capability.value}: {result.error}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing capability {capability.value}: {e}")
            return AgentResult.error_result(str(e), 0.0)
    
    async def _health_check_loop(self) -> None:
        """Periodic health check for all agents."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                logger.debug("Performing agent health checks...")
                unhealthy_agents = []
                
                for name, agent in self.agents.items():
                    try:
                        health = await agent.health_check()
                        if not health["healthy"]:
                            unhealthy_agents.append(name)
                            logger.warning(f"Agent {name} is unhealthy: {health.get('error', 'Unknown')}")
                    except Exception as e:
                        unhealthy_agents.append(name)
                        logger.error(f"Health check failed for agent {name}: {e}")
                
                if unhealthy_agents:
                    logger.warning(f"Unhealthy agents: {unhealthy_agents}")
                else:
                    logger.debug("All agents are healthy")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics for all agents."""
        stats = {
            "total_agents": len(self.agents),
            "agents": {}
        }
        
        for name, agent in self.agents.items():
            agent_stats = agent.get_performance_stats()
            stats["agents"][name] = agent_stats
        
        return stats
    
    async def add_agent(self, config: AgentConfig) -> None:
        """Add a new agent at runtime."""
        if config.name in self.agents:
            raise AgentError(f"Agent {config.name} already exists")
        
        agent = self._create_agent(config)
        await agent.__aenter__()
        
        self.agent_configs[config.name] = config
        logger.info(f"Added agent: {config.name}")
    
    async def remove_agent(self, agent_name: str) -> None:
        """Remove an agent at runtime."""
        if agent_name not in self.agents:
            raise NoAgentError(f"Agent not found: {agent_name}")
        
        agent = self.agents[agent_name]
        await agent.__aexit__(None, None, None)
        
        del self.agents[agent_name]
        del self.agent_configs[agent_name]
        
        logger.info(f"Removed agent: {agent_name}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()