"""
Async utilities for AURA system.
"""
import asyncio
import functools
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

from aura.utils.logging import logger
from aura.utils.errors import AuraError, ExternalServiceError


T = TypeVar('T')


async def gather_with_concurrency(
    n: int,
    *tasks: asyncio.Task,
    return_exceptions: bool = False
) -> List[Any]:
    """Run tasks with a concurrency limit.
    
    Args:
        n: Maximum number of concurrent tasks
        *tasks: Tasks to run
        return_exceptions: Whether to return exceptions instead of raising them
        
    Returns:
        List of task results
    """
    semaphore = asyncio.Semaphore(n)
    
    async def sem_task(task: asyncio.Task) -> Any:
        async with semaphore:
            return await task
    
    return await asyncio.gather(
        *(sem_task(task) for task in tasks),
        return_exceptions=return_exceptions
    )


def async_retry(
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """Retry an async function on failure.
    
    Args:
        retries: Maximum number of retries
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier
        exceptions: Exceptions to catch and retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            retry_delay = delay
            last_exception = None
            
            for i in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if i < retries:
                        logger.warning(
                            f"Retry {i+1}/{retries} for {func.__name__} after error: {str(e)}"
                        )
                        await asyncio.sleep(retry_delay)
                        retry_delay *= backoff
                    else:
                        logger.error(
                            f"Failed after {retries} retries: {func.__name__} - {str(e)}"
                        )
            
            if isinstance(last_exception, AuraError):
                raise last_exception
            
            # If we get here, we've exhausted all retries
            raise ExternalServiceError(
                f"Failed after {retries} retries",
                service=func.__name__,
                details={"last_error": str(last_exception)}
            )
        
        return wrapper
    
    return decorator


class AsyncLimiter:
    """Rate limiter for async functions."""
    
    def __init__(self, max_calls: int, period: float):
        """Initialize the rate limiter.
        
        Args:
            max_calls: Maximum number of calls per period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = 0
        self.reset_at = time.monotonic() + period
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire a rate limit token."""
        async with self.lock:
            now = time.monotonic()
            
            # Reset counter if period has elapsed
            if now >= self.reset_at:
                self.calls = 0
                self.reset_at = now + self.period
            
            # Wait if we've reached the limit
            if self.calls >= self.max_calls:
                wait_time = self.reset_at - now
                logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                
                # Reset after waiting
                self.calls = 0
                self.reset_at = time.monotonic() + self.period
            
            # Increment call counter
            self.calls += 1
    
    def __call__(self, func: Callable) -> Callable:
        """Decorate a function with rate limiting."""
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            await self.acquire()
            return await func(*args, **kwargs)
        
        return wrapper


class AsyncCache:
    """Simple async cache with TTL."""
    
    def __init__(self, ttl: float = 60.0):
        """Initialize the cache.
        
        Args:
            ttl: Time-to-live in seconds
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl
        self.lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        async with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            now = time.monotonic()
            
            if now > entry["expires_at"]:
                del self.cache[key]
                return None
            
            return entry["value"]
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional custom TTL in seconds
        """
        expires_at = time.monotonic() + (ttl or self.ttl)
        
        async with self.lock:
            self.cache[key] = {
                "value": value,
                "expires_at": expires_at
            }
    
    async def delete(self, key: str) -> None:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
        """
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
    
    async def clear(self) -> None:
        """Clear the entire cache."""
        async with self.lock:
            self.cache.clear()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorate a function with caching."""
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = ":".join(key_parts)
            
            # Check cache
            cached = await self.get(key)
            if cached is not None:
                return cached
            
            # Call function and cache result
            result = await func(*args, **kwargs)
            await self.set(key, result)
            return result
        
        return wrapper