"""
Error handling utilities for AURA system.
"""
from typing import Any, Dict, List, Optional, Type, Union
import traceback
import sys

from aura.utils.logging import logger


class AuraError(Exception):
    """Base exception for all AURA errors."""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Error message
            code: Error code
            details: Additional error details
        """
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the exception to a dictionary.
        
        Returns:
            Dictionary representation of the exception
        """
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details
        }
    
    def __str__(self) -> str:
        """Get string representation of the exception."""
        if self.details:
            return f"{self.code}: {self.message} - {self.details}"
        return f"{self.code}: {self.message}"


class ConfigurationError(AuraError):
    """Error in system configuration."""
    pass


class ValidationError(AuraError):
    """Error in data validation."""
    pass


class AuthenticationError(AuraError):
    """Error in authentication."""
    pass


class AuthorizationError(AuraError):
    """Error in authorization."""
    pass


class ResourceNotFoundError(AuraError):
    """Resource not found."""
    pass


class ResourceExistsError(AuraError):
    """Resource already exists."""
    pass


class ExternalServiceError(AuraError):
    """Error in external service."""
    
    def __init__(
        self,
        message: str,
        service: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Error message
            service: External service name
            code: Error code
            details: Additional error details
        """
        super().__init__(message, code, details)
        self.service = service
        if "service" not in self.details:
            self.details["service"] = service


class OpenAIError(ExternalServiceError):
    """Error in OpenAI service."""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Error message
            code: Error code
            details: Additional error details
        """
        super().__init__(message, "OpenAI", code, details)


class JinaError(ExternalServiceError):
    """Error in Jina AI service."""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Error message
            code: Error code
            details: Additional error details
        """
        super().__init__(message, "Jina", code, details)


class TransactionError(AuraError):
    """Error in transaction processing."""
    pass


class MemoryError(AuraError):
    """Error in memory system."""
    pass


class ExecutionError(AuraError):
    """Error in task execution."""
    pass


class AgentError(AuraError):
    """Agent related errors."""
    pass


class RateLimitError(AgentError):
    """Rate limit exceeded error."""
    pass


class TimeoutError(AgentError):
    """Timeout error."""
    pass


class NoAgentError(AgentError):
    """No suitable agent found error."""
    pass


class LowConfidenceError(AgentError):
    """Agent confidence too low error."""
    pass


def handle_exception(
    exc: Exception,
    log_level: str = "error",
    reraise: bool = True
) -> Dict[str, Any]:
    """Handle an exception.
    
    Args:
        exc: Exception to handle
        log_level: Log level to use
        reraise: Whether to reraise the exception
        
    Returns:
        Dictionary representation of the exception
        
    Raises:
        Exception: The original exception if reraise is True
    """
    # Get traceback information
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    
    # Convert to AuraError if not already
    if not isinstance(exc, AuraError):
        if isinstance(exc, ValueError):
            error = ValidationError(str(exc))
        elif isinstance(exc, KeyError):
            error = ResourceNotFoundError(str(exc))
        elif isinstance(exc, FileNotFoundError):
            error = ResourceNotFoundError(str(exc))
        elif isinstance(exc, PermissionError):
            error = AuthorizationError(str(exc))
        else:
            error = AuraError(str(exc), code=exc.__class__.__name__)
    else:
        error = exc
    
    # Add traceback to details
    error.details["traceback"] = "".join(tb)
    
    # Log the error
    log_func = getattr(logger, log_level)
    log_func(f"Exception: {error}")
    
    # Reraise if requested
    if reraise:
        raise error
    
    return error.to_dict()


def safe_execute(func: callable, *args: Any, **kwargs: Any) -> Any:
    """Execute a function safely, handling exceptions.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Function result or error dictionary
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return handle_exception(e, reraise=False)