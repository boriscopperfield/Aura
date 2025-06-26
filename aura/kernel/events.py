"""
Event model for AURA system.
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

import json


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder for datetime objects."""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class DirectiveEventType(Enum):
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


class AnalyticalEventType(Enum):
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


EventType = Union[DirectiveEventType, AnalyticalEventType]


@dataclass
class EventSource:
    """Source of an event."""
    
    type: str
    id: str
    version: str


@dataclass
class EventMetadata:
    """Metadata for event tracing and debugging."""
    
    correlation_id: Optional[str] = None  # Groups related events
    causation_id: Optional[str] = None    # Event that caused this one
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    git_commit: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class Event:
    """Base class for all events."""
    
    id: str  # evt_20250626_123456_uuid4
    timestamp: datetime
    category: Literal["directive", "analytical"]
    type: EventType
    source: EventSource
    payload: Dict[str, Any]
    metadata: EventMetadata
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(asdict(self), cls=DateTimeEncoder)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Event":
        """Create event from JSON string."""
        data = json.loads(json_str)
        
        # Convert string type to enum
        if data["category"] == "directive":
            data["type"] = DirectiveEventType(data["type"])
        else:
            data["type"] = AnalyticalEventType(data["type"])
        
        # Convert string timestamp to datetime
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        # Create EventSource and EventMetadata
        data["source"] = EventSource(**data["source"])
        data["metadata"] = EventMetadata(**data["metadata"])
        
        return cls(**data)