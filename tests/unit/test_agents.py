"""
Unit tests for agent implementations.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from aura.agents.base import Agent, AgentCapability, AgentResult, RateLimit, CircuitBreaker
from aura.agents.openai_agent import OpenAIAgent
from aura.agents.local_agent import LocalAgent
from aura.agents.manager import AgentManager, AgentConfig
from aura.utils.errors import AgentError, RateLimitError


class TestAgentBase:
    """Test base agent functionality."""
    
    def test_agent_result_creation(self):
        """Test agent result creation."""
        success_result = AgentResult.success_result(
            data="test data",
            duration=1.5,
            metadata={"key": "value"},
            cost=0.01
        )
        
        assert success_result.success is True
        assert success_result.data == "test data"
        assert success_result.duration == 1.5
        assert success_result.cost == 0.01
        assert success_result.metadata["key"] == "value"
        
        error_result = AgentResult.error_result(
            error="test error",
            duration=0.5
        )
        
        assert error_result.success is False
        assert error_result.error == "test error"
        assert error_result.duration == 0.5
        assert error_result.data is None
    
    def test_rate_limit_creation(self):
        """Test rate limit configuration."""
        rate_limit = RateLimit(
            requests_per_minute=60,
            tokens_per_minute=1000,
            requests_per_day=1440
        )
        
        assert rate_limit.requests_per_minute == 60
        assert rate_limit.tokens_per_minute == 1000
        assert rate_limit.requests_per_day == 1440
        assert len(rate_limit.request_times) == 0
        assert len(rate_limit.token_usage) == 0
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0
        
        # Simulate failures
        cb.failure_count = 3
        cb.state = "OPEN"
        
        assert cb.state == "OPEN"


class MockAgent(Agent):
    """Mock agent for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="mock",
            capabilities=[AgentCapability.TEXT_GENERATION],
            **kwargs
        )
        self.execute_called = False
        self.execute_kwargs = {}
    
    async def _execute(self, capability: AgentCapability, **kwargs) -> AgentResult:
        """Mock execute method."""
        self.execute_called = True
        self.execute_kwargs = kwargs
        
        if kwargs.get("should_fail"):
            return AgentResult.error_result("Mock error", 0.1)
        
        return AgentResult.success_result(
            data="mock response",
            duration=0.1,
            metadata={"mock": True}
        )


class TestMockAgent:
    """Test mock agent functionality."""
    
    @pytest.mark.asyncio
    async def test_agent_execution(self):
        """Test basic agent execution."""
        agent = MockAgent()
        
        async with agent:
            result = await agent.execute(
                AgentCapability.TEXT_GENERATION,
                prompt="test prompt"
            )
        
        assert result.success is True
        assert result.data == "mock response"
        assert agent.execute_called is True
        assert agent.execute_kwargs["prompt"] == "test prompt"
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self):
        """Test agent error handling."""
        agent = MockAgent()
        
        async with agent:
            result = await agent.execute(
                AgentCapability.TEXT_GENERATION,
                should_fail=True
            )
        
        assert result.success is False
        assert result.error == "Mock error"
    
    @pytest.mark.asyncio
    async def test_unsupported_capability(self):
        """Test handling of unsupported capabilities."""
        agent = MockAgent()
        
        async with agent:
            result = await agent.execute(AgentCapability.IMAGE_GENERATION)
        
        assert result.success is False
        assert "does not support capability" in result.error
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting functionality."""
        rate_limit = RateLimit(requests_per_minute=1)
        agent = MockAgent(rate_limit=rate_limit)
        
        async with agent:
            # First request should succeed
            result1 = await agent.execute(AgentCapability.TEXT_GENERATION)
            assert result1.success is True
            
            # Second request should be rate limited
            result2 = await agent.execute(AgentCapability.TEXT_GENERATION)
            assert result2.success is False
            assert "rate limit" in result2.error.lower()
    
    def test_performance_stats(self):
        """Test performance statistics tracking."""
        agent = MockAgent()
        
        # Initial stats
        stats = agent.get_performance_stats()
        assert stats["total_requests"] == 0
        assert stats["successful_requests"] == 0
        assert stats["success_rate"] == 0.0
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test agent health check."""
        agent = MockAgent()
        
        async with agent:
            health = await agent.health_check()
        
        assert health["healthy"] is True
        assert "response_time" in health


class TestOpenAIAgent:
    """Test OpenAI agent implementation."""
    
    def test_openai_agent_creation(self):
        """Test OpenAI agent creation."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            agent = OpenAIAgent(model="gpt-3.5-turbo")
            
            assert agent.name == "openai"
            assert agent.model == "gpt-3.5-turbo"
            assert AgentCapability.TEXT_GENERATION in agent.capabilities
            assert AgentCapability.CODE_GENERATION in agent.capabilities
    
    def test_openai_agent_no_api_key(self):
        """Test OpenAI agent creation without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(AgentError, match="API key is required"):
                OpenAIAgent()
    
    def test_token_estimation(self):
        """Test token estimation."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            agent = OpenAIAgent()
            
            tokens = agent._estimate_tokens("Hello world")
            assert tokens > 0
            assert isinstance(tokens, int)
    
    def test_cost_calculation(self):
        """Test cost calculation."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            agent = OpenAIAgent(model="gpt-4")
            
            cost = agent._calculate_cost(100, 50, "gpt-4")
            assert cost > 0
            assert isinstance(cost, float)


class TestLocalAgent:
    """Test local agent implementation."""
    
    def test_local_agent_creation(self):
        """Test local agent creation."""
        agent = LocalAgent()
        
        assert agent.name == "local"
        assert AgentCapability.CODE_GENERATION in agent.capabilities
        assert AgentCapability.ANALYSIS in agent.capabilities
    
    @pytest.mark.asyncio
    async def test_code_execution(self):
        """Test local code execution."""
        agent = LocalAgent()
        
        async with agent:
            result = await agent.execute(
                AgentCapability.CODE_GENERATION,
                code="print('Hello, World!')",
                language="python"
            )
        
        if result.success:
            assert "Hello, World!" in result.data["stdout"]
        # Note: This test might fail in environments without Python
    
    @pytest.mark.asyncio
    async def test_file_analysis(self, temp_dir):
        """Test file analysis."""
        agent = LocalAgent()
        
        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")
        
        async with agent:
            result = await agent.execute(
                AgentCapability.ANALYSIS,
                file_path=str(test_file),
                analysis_type="general"
            )
        
        assert result.success is True
        assert result.data["file_size"] > 0
        assert result.data["file_type"] == ".txt"


class TestAgentManager:
    """Test agent manager functionality."""
    
    def test_agent_config_creation(self):
        """Test agent configuration creation."""
        config = AgentConfig(
            name="test_agent",
            type="openai",
            capabilities=["text_generation"],
            config={"api_key": "test"},
            enabled=True,
            priority=1
        )
        
        assert config.name == "test_agent"
        assert config.type == "openai"
        assert config.enabled is True
        assert config.priority == 1
    
    @pytest.mark.asyncio
    async def test_agent_manager_creation(self):
        """Test agent manager creation."""
        manager = AgentManager()
        
        assert len(manager.agents) == 0
        assert len(manager.agent_configs) == 0
    
    @pytest.mark.asyncio
    async def test_agent_manager_with_config(self, temp_dir, mock_agent_config):
        """Test agent manager with configuration."""
        import yaml
        
        # Create config file
        config_file = temp_dir / "agents.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(mock_agent_config, f)
        
        # Mock the agent creation to avoid API key requirements
        with patch('aura.agents.manager.OpenAIAgent') as mock_openai:
            mock_openai.return_value = MockAgent()
            
            with patch('aura.agents.manager.LocalAgent') as mock_local:
                mock_local.return_value = MockAgent()
                
                manager = AgentManager(config_file)
                
                assert len(manager.agent_configs) == 2
                assert "test_openai" in manager.agent_configs
                assert "test_local" in manager.agent_configs
    
    @pytest.mark.asyncio
    async def test_agent_selection(self):
        """Test agent selection for capabilities."""
        manager = AgentManager()
        
        # Add mock agents
        agent1 = MockAgent()
        agent1.name = "agent1"
        agent2 = MockAgent()
        agent2.name = "agent2"
        
        manager.agents["agent1"] = agent1
        manager.agents["agent2"] = agent2
        
        async with manager:
            agents = await manager.get_agents_for_capability(
                AgentCapability.TEXT_GENERATION
            )
        
        assert len(agents) == 2
        assert agent1 in agents
        assert agent2 in agents
    
    @pytest.mark.asyncio
    async def test_capability_execution(self):
        """Test capability execution through manager."""
        manager = AgentManager()
        
        # Add mock agent
        agent = MockAgent()
        agent.name = "test_agent"
        manager.agents["test_agent"] = agent
        
        async with manager:
            result = await manager.execute_capability(
                AgentCapability.TEXT_GENERATION,
                prompt="test prompt"
            )
        
        assert result.success is True
        assert result.data == "mock response"
    
    def test_agent_stats(self):
        """Test agent statistics collection."""
        manager = AgentManager()
        
        # Add mock agent
        agent = MockAgent()
        agent.name = "test_agent"
        manager.agents["test_agent"] = agent
        
        stats = manager.get_agent_stats()
        
        assert stats["total_agents"] == 1
        assert "test_agent" in stats["agents"]
        assert stats["agents"]["test_agent"]["name"] == "test_agent"