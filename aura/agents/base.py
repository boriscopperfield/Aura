"""
Base agent classes and interfaces.
"""
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import aiohttp
import time

from ..utils.errors import AgentError, RateLimitError, TimeoutError


class AgentCapability(Enum):
    """Agent capabilities."""
    TEXT_GENERATION = "text_generation"
    TEXT_EMBEDDING = "text_embedding"
    IMAGE_GENERATION = "image_generation"
    IMAGE_ANALYSIS = "image_analysis"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    RERANKING = "reranking"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"


@dataclass
class AgentResult:
    """Result from agent execution."""
    success: bool
    data: Any
    metadata: Dict[str, Any]
    duration: float
    cost: Optional[float] = None
    error: Optional[str] = None
    
    @classmethod
    def success_result(cls, data: Any, duration: float, metadata: Dict[str, Any] = None, cost: float = None):
        """Create a successful result."""
        return cls(
            success=True,
            data=data,
            metadata=metadata or {},
            duration=duration,
            cost=cost
        )
    
    @classmethod
    def error_result(cls, error: str, duration: float, metadata: Dict[str, Any] = None):
        """Create an error result."""
        return cls(
            success=False,
            data=None,
            metadata=metadata or {},
            duration=duration,
            error=error
        )


@dataclass
class RateLimit:
    """Rate limiting configuration."""
    requests_per_minute: int
    tokens_per_minute: Optional[int] = None
    requests_per_day: Optional[int] = None
    
    def __post_init__(self):
        self.request_times: List[datetime] = []
        self.token_usage: List[tuple[datetime, int]] = []


class CircuitBreaker:
    """Circuit breaker for agent reliability."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.state = "HALF_OPEN"
            else:
                raise AgentError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e


class Agent(ABC):
    """Base class for all agents."""
    
    def __init__(
        self,
        name: str,
        capabilities: List[AgentCapability],
        rate_limit: Optional[RateLimit] = None,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        self.name = name
        self.capabilities = capabilities
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.max_retries = max_retries
        self.circuit_breaker = CircuitBreaker()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.total_cost = 0.0
        self.total_duration = 0.0
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not self.session:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def check_rate_limit(self, estimated_tokens: int = 0) -> None:
        """Check if request is within rate limits."""
        if not self.rate_limit:
            return
        
        now = datetime.now()
        
        # Clean old request times
        cutoff = now - timedelta(minutes=1)
        self.rate_limit.request_times = [
            t for t in self.rate_limit.request_times if t > cutoff
        ]
        
        # Check request rate limit
        if len(self.rate_limit.request_times) >= self.rate_limit.requests_per_minute:
            raise RateLimitError("Request rate limit exceeded")
        
        # Check token rate limit
        if self.rate_limit.tokens_per_minute and estimated_tokens > 0:
            recent_tokens = sum(
                tokens for time, tokens in self.rate_limit.token_usage
                if time > cutoff
            )
            if recent_tokens + estimated_tokens > self.rate_limit.tokens_per_minute:
                raise RateLimitError("Token rate limit exceeded")
        
        # Record this request
        self.rate_limit.request_times.append(now)
        if estimated_tokens > 0:
            self.rate_limit.token_usage.append((now, estimated_tokens))
    
    async def execute_with_retry(self, func, *args, **kwargs) -> AgentResult:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                
                # Check rate limits
                await self.check_rate_limit()
                
                # Execute with circuit breaker
                result = await self.circuit_breaker.call(func, *args, **kwargs)
                
                duration = time.time() - start_time
                
                # Update stats
                self.total_requests += 1
                self.successful_requests += 1
                self.total_duration += duration
                
                return result
                
            except (RateLimitError, TimeoutError) as e:
                # Don't retry rate limit or timeout errors
                duration = time.time() - start_time
                self.total_requests += 1
                return AgentResult.error_result(str(e), duration)
                
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
        
        # All retries failed
        duration = time.time() - start_time
        self.total_requests += 1
        return AgentResult.error_result(str(last_exception), duration)
    
    @abstractmethod
    async def _execute(self, capability: AgentCapability, **kwargs) -> AgentResult:
        """Execute agent capability. Must be implemented by subclasses."""
        pass
    
    async def execute(self, capability: AgentCapability, **kwargs) -> AgentResult:
        """Execute agent capability with error handling and retries."""
        if capability not in self.capabilities:
            return AgentResult.error_result(
                f"Agent {self.name} does not support capability {capability.value}",
                0.0
            )
        
        return await self.execute_with_retry(self._execute, capability, **kwargs)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        success_rate = (
            self.successful_requests / self.total_requests 
            if self.total_requests > 0 else 0.0
        )
        avg_duration = (
            self.total_duration / self.total_requests 
            if self.total_requests > 0 else 0.0
        )
        
        return {
            "name": self.name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": success_rate,
            "total_cost": self.total_cost,
            "average_duration": avg_duration,
            "circuit_breaker_state": self.circuit_breaker.state
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Simple ping test
            start_time = time.time()
            await self._health_check()
            duration = time.time() - start_time
            
            return {
                "healthy": True,
                "response_time": duration,
                "circuit_breaker_state": self.circuit_breaker.state
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "circuit_breaker_state": self.circuit_breaker.state
            }
    
    async def _health_check(self) -> None:
        """Perform agent-specific health check. Override in subclasses."""
        pass