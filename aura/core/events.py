"""
Event system for AURA.

This module defines the event model and event handling for the AURA system.
Events are the primary mechanism for tracking system state and changes.
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, ClassVar

from pydantic import BaseModel, Field, validator

from aura.utils.serialization import json_dumps, json_loads
from aura.utils.logging import logger


class DirectiveEventType(str, Enum):
    """Events that change system state."""
    
    # Task lifecycle
    TASK_CREATED = "task.created"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"
    
    # Node operations
    NODE_ADDED = "node.added"
    NODE_UPDATED = "node.updated"
    NODE_REMOVED = "node.removed"
    
    # Execution control
    EXECUTION_REQUESTED = "execution.requested"
    EXECUTION_STARTED = "execution.started"
    EXECUTION_COMPLETED = "execution.completed"
    EXECUTION_FAILED = "execution.failed"
    
    # State management
    CHECKPOINT_CREATED = "checkpoint.created"
    STATE_REVERTED = "state.reverted"


class AnalyticalEventType(str, Enum):
    """Events representing system insights."""
    
    # User patterns
    PREFERENCE_INFERRED = "preference.inferred"
    BEHAVIOR_PATTERN_DETECTED = "behavior.detected"
    
    # System optimization
    BOTTLENECK_IDENTIFIED = "bottleneck.identified"
    OPTIMIZATION_SUGGESTED = "optimization.suggested"
    
    # Knowledge discovery
    WORKFLOW_PATTERN_FOUND = "workflow.found"
    BEST_PRACTICE_LEARNED = "practice.learned"
    
    # Error patterns
    FAILURE_PATTERN_DETECTED = "failure.detected"
    RECOVERY_STRATEGY_PROPOSED = "recovery.proposed"


# Union type for event types
EventType = Union[DirectiveEventType, AnalyticalEventType]


class EventSource(BaseModel):
    """Source of an event."""
    
    type: str
    id: str
    version: str


class EventMetadata(BaseModel):
    """Metadata for event tracing and debugging."""
    
    correlation_id: Optional[str] = None  # Groups related events
    causation_id: Optional[str] = None    # Event that caused this one
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    git_commit: Optional[str] = None
    confidence: Optional[float] = None
    tags: List[str] = Field(default_factory=list)


class Event(BaseModel):
    """Base class for all events."""
    
    id: str  # evt_20250626_123456_uuid4
    timestamp: datetime
    category: Literal["directive", "analytical"]
    type: str  # Using string to avoid validation issues with Enum
    source: EventSource
    payload: Dict[str, Any]
    metadata: EventMetadata
    
    # Class variables for event type mapping
    directive_types: ClassVar[Dict[str, DirectiveEventType]] = {e.value: e for e in DirectiveEventType}
    analytical_types: ClassVar[Dict[str, AnalyticalEventType]] = {e.value: e for e in AnalyticalEventType}
    
    @validator("type")
    def validate_type(cls, v: str, values: Dict[str, Any]) -> str:
        """Validate that the event type is valid for the category."""
        if "category" not in values:
            return v
            
        category = values["category"]
        
        if category == "directive" and v not in cls.directive_types:
            valid_types = ", ".join(cls.directive_types.keys())
            raise ValueError(f"Invalid directive event type: {v}. Valid types: {valid_types}")
            
        if category == "analytical" and v not in cls.analytical_types:
            valid_types = ", ".join(cls.analytical_types.keys())
            raise ValueError(f"Invalid analytical event type: {v}. Valid types: {valid_types}")
            
        return v
    
    def get_typed_event_type(self) -> EventType:
        """Get the event type as an Enum value."""
        if self.category == "directive":
            return self.directive_types[self.type]
        else:
            return self.analytical_types[self.type]
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json_dumps(self.dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "Event":
        """Create event from JSON string."""
        data = json_loads(json_str)
        return cls.parse_obj(data)
    
    @classmethod
    def create_directive(
        cls,
        type: DirectiveEventType,
        payload: Dict[str, Any],
        source: EventSource,
        metadata: Optional[EventMetadata] = None
    ) -> "Event":
        """Create a directive event."""
        return cls(
            id=f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            category="directive",
            type=type.value,
            source=source,
            payload=payload,
            metadata=metadata or EventMetadata()
        )
    
    @classmethod
    def create_analytical(
        cls,
        type: AnalyticalEventType,
        payload: Dict[str, Any],
        source: EventSource,
        metadata: Optional[EventMetadata] = None
    ) -> "Event":
        """Create an analytical event."""
        return cls(
            id=f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            category="analytical",
            type=type.value,
            source=source,
            payload=payload,
            metadata=metadata or EventMetadata()
        )


class EventBus:
    """Event bus for publishing and subscribing to events."""
    
    def __init__(self):
        """Initialize the event bus."""
        self.subscribers = {}
        self.logger = logger
    
    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers.
        
        Args:
            event: Event to publish
        """
        self.logger.debug(f"Publishing event: {event.id} ({event.type})")
        
        # Get subscribers for this event type
        event_type = event.type
        subscribers = self.subscribers.get(event_type, [])
        
        # Also notify wildcard subscribers
        wildcard_subscribers = self.subscribers.get("*", [])
        
        # Notify all subscribers
        for subscriber in subscribers + wildcard_subscribers:
            try:
                await subscriber(event)
            except Exception as e:
                self.logger.error(f"Error in event subscriber: {e}")
    
    def subscribe(self, event_type: Union[str, EventType], callback: callable) -> callable:
        """Subscribe to events of a specific type.
        
        Args:
            event_type: Event type to subscribe to, or "*" for all events
            callback: Async function to call when an event is published
            
        Returns:
            Unsubscribe function
        """
        # Convert enum to string if needed
        if isinstance(event_type, (DirectiveEventType, AnalyticalEventType)):
            event_type = event_type.value
        
        # Initialize subscriber list if needed
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        # Add subscriber
        self.subscribers[event_type].append(callback)
        self.logger.debug(f"Subscribed to event type: {event_type}")
        
        # Return unsubscribe function
        def unsubscribe():
            if event_type in self.subscribers and callback in self.subscribers[event_type]:
                self.subscribers[event_type].remove(callback)
                self.logger.debug(f"Unsubscribed from event type: {event_type}")
        
        return unsubscribe


# Global event bus instance
event_bus = EventBus()