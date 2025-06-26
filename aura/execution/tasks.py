"""
Task model for AURA system.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class TaskType(Enum):
    """Types of tasks in the system."""
    
    ROOT = "root"
    GROUP = "group"
    ACTION = "action"
    DECISION = "decision"
    ANALYSIS = "analysis"
    GENERATION = "generation"


class TaskStatus(Enum):
    """Status of a task."""
    
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


@dataclass
class TaskMetadata:
    """Task metadata stored in _meta.json."""
    
    created_at: datetime
    updated_at: datetime
    created_by: str
    assigned_agent: Optional[str] = None
    priority: int = 0
    estimated_duration: Optional[timedelta] = None
    actual_duration: Optional[timedelta] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class TaskNode:
    """Node in the task execution tree."""
    
    id: str
    parent_id: Optional[str]
    path: str  # Filesystem path
    name: str
    description: str
    type: TaskType
    status: TaskStatus
    dependencies: List[str]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    metadata: TaskMetadata
    children: List['TaskNode'] = field(default_factory=list)
    
    def to_workspace_path(self, base: Path) -> Path:
        """Convert logical node to filesystem path."""
        return base / self.path.lstrip('/')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task node to dictionary."""
        result = {
            "id": self.id,
            "parent_id": self.parent_id,
            "path": self.path,
            "name": self.name,
            "description": self.description,
            "type": self.type.value,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat(),
                "created_by": self.metadata.created_by,
                "assigned_agent": self.metadata.assigned_agent,
                "priority": self.metadata.priority,
                "estimated_duration": (
                    self.metadata.estimated_duration.total_seconds() 
                    if self.metadata.estimated_duration else None
                ),
                "actual_duration": (
                    self.metadata.actual_duration.total_seconds() 
                    if self.metadata.actual_duration else None
                ),
                "resource_usage": self.metadata.resource_usage,
                "tags": self.metadata.tags
            },
            "children": [child.to_dict() for child in self.children]
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskNode':
        """Create task node from dictionary."""
        # Convert string type and status to enums
        task_type = TaskType(data["type"])
        status = TaskStatus(data["status"])
        
        # Create metadata
        metadata = TaskMetadata(
            created_at=datetime.fromisoformat(data["metadata"]["created_at"]),
            updated_at=datetime.fromisoformat(data["metadata"]["updated_at"]),
            created_by=data["metadata"]["created_by"],
            assigned_agent=data["metadata"]["assigned_agent"],
            priority=data["metadata"]["priority"],
            estimated_duration=(
                timedelta(seconds=data["metadata"]["estimated_duration"])
                if data["metadata"]["estimated_duration"] else None
            ),
            actual_duration=(
                timedelta(seconds=data["metadata"]["actual_duration"])
                if data["metadata"]["actual_duration"] else None
            ),
            resource_usage=data["metadata"]["resource_usage"],
            tags=data["metadata"]["tags"]
        )
        
        # Create children recursively
        children = [cls.from_dict(child) for child in data.get("children", [])]
        
        # Create task node
        return cls(
            id=data["id"],
            parent_id=data["parent_id"],
            path=data["path"],
            name=data["name"],
            description=data["description"],
            type=task_type,
            status=status,
            dependencies=data["dependencies"],
            inputs=data["inputs"],
            outputs=data["outputs"],
            metadata=metadata,
            children=children
        )