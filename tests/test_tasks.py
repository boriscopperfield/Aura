"""
Tests for the task model.
"""
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from aura.execution.tasks import TaskMetadata, TaskNode, TaskStatus, TaskType


def test_task_node_serialization():
    """Test task node serialization to dictionary."""
    # Create a sample task node
    task = TaskNode(
        id="task_test",
        parent_id=None,
        path="/tasks/test_task",
        name="Test Task",
        description="A task created for testing purposes",
        type=TaskType.ROOT,
        status=TaskStatus.PENDING,
        dependencies=[],
        inputs={},
        outputs={},
        metadata=TaskMetadata(
            created_at=datetime.fromisoformat("2025-06-26T12:34:56+00:00"),
            updated_at=datetime.fromisoformat("2025-06-26T12:34:56+00:00"),
            created_by="test_user",
            priority=1,
            tags=["test", "example"]
        ),
        children=[]
    )
    
    # Add a child task
    child_task = TaskNode(
        id="task_test_child",
        parent_id="task_test",
        path="/tasks/test_task/child",
        name="Child Task",
        description="A child task for testing purposes",
        type=TaskType.ACTION,
        status=TaskStatus.PENDING,
        dependencies=[],
        inputs={},
        outputs={},
        metadata=TaskMetadata(
            created_at=datetime.fromisoformat("2025-06-26T12:35:00+00:00"),
            updated_at=datetime.fromisoformat("2025-06-26T12:35:00+00:00"),
            created_by="test_user",
            assigned_agent="test_agent",
            priority=2,
            estimated_duration=timedelta(minutes=30),
            tags=["test", "child"]
        ),
        children=[]
    )
    
    task.children.append(child_task)
    
    # Serialize to dictionary
    data = task.to_dict()
    
    # Check fields
    assert data["id"] == "task_test"
    assert data["parent_id"] is None
    assert data["path"] == "/tasks/test_task"
    assert data["name"] == "Test Task"
    assert data["type"] == "root"
    assert data["status"] == "pending"
    assert len(data["children"]) == 1
    assert data["children"][0]["id"] == "task_test_child"
    assert data["children"][0]["metadata"]["estimated_duration"] == 1800.0  # 30 minutes in seconds


def test_task_node_deserialization():
    """Test task node deserialization from dictionary."""
    # Create a dictionary
    data = {
        "id": "task_test",
        "parent_id": None,
        "path": "/tasks/test_task",
        "name": "Test Task",
        "description": "A task created for testing purposes",
        "type": "root",
        "status": "pending",
        "dependencies": [],
        "inputs": {},
        "outputs": {},
        "metadata": {
            "created_at": "2025-06-26T12:34:56+00:00",
            "updated_at": "2025-06-26T12:34:56+00:00",
            "created_by": "test_user",
            "assigned_agent": None,
            "priority": 1,
            "estimated_duration": None,
            "actual_duration": None,
            "resource_usage": {},
            "tags": ["test", "example"]
        },
        "children": [
            {
                "id": "task_test_child",
                "parent_id": "task_test",
                "path": "/tasks/test_task/child",
                "name": "Child Task",
                "description": "A child task for testing purposes",
                "type": "action",
                "status": "pending",
                "dependencies": [],
                "inputs": {},
                "outputs": {},
                "metadata": {
                    "created_at": "2025-06-26T12:35:00+00:00",
                    "updated_at": "2025-06-26T12:35:00+00:00",
                    "created_by": "test_user",
                    "assigned_agent": "test_agent",
                    "priority": 2,
                    "estimated_duration": 1800.0,  # 30 minutes in seconds
                    "actual_duration": None,
                    "resource_usage": {},
                    "tags": ["test", "child"]
                },
                "children": []
            }
        ]
    }
    
    # Deserialize from dictionary
    task = TaskNode.from_dict(data)
    
    # Check fields
    assert task.id == "task_test"
    assert task.parent_id is None
    assert task.path == "/tasks/test_task"
    assert task.name == "Test Task"
    assert task.type == TaskType.ROOT
    assert task.status == TaskStatus.PENDING
    assert len(task.children) == 1
    assert task.children[0].id == "task_test_child"
    assert task.children[0].metadata.estimated_duration == timedelta(minutes=30)


def test_workspace_path_conversion():
    """Test conversion of logical path to filesystem path."""
    # Create a sample task node
    task = TaskNode(
        id="task_test",
        parent_id=None,
        path="/tasks/test_task",
        name="Test Task",
        description="A task created for testing purposes",
        type=TaskType.ROOT,
        status=TaskStatus.PENDING,
        dependencies=[],
        inputs={},
        outputs={},
        metadata=TaskMetadata(
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="test_user"
        ),
        children=[]
    )
    
    # Convert to workspace path
    base_path = Path("/workspace")
    workspace_path = task.to_workspace_path(base_path)
    
    # Check path
    assert workspace_path == Path("/workspace/tasks/test_task")