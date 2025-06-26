"""
Tests for the transaction manager.
"""
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from aura.kernel.events import DirectiveEventType, Event, EventMetadata, EventSource
from aura.kernel.transaction import FileOperation, TransactionManager, TransactionProposal


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.mark.asyncio
async def test_transaction_execution(temp_workspace):
    """Test executing a transaction."""
    # Initialize transaction manager
    tx_manager = TransactionManager(temp_workspace)
    
    # Create a sample event
    event = Event(
        id="evt_test",
        timestamp=datetime.now(),
        category="directive",
        type=DirectiveEventType.TASK_CREATED,
        source=EventSource(
            type="test",
            id="test_script",
            version="1.0.0"
        ),
        payload={
            "task_id": "task_test",
            "name": "Test Task",
            "description": "A task created for testing purposes"
        },
        metadata=EventMetadata(
            user_id="test_user",
            session_id="test_session"
        )
    )
    
    # Create a sample file operation
    file_op = FileOperation(
        type="create_file",
        path="tasks/test_task/README.md",
        content="# Test Task\n\nThis task was created for testing purposes."
    )
    
    # Create a transaction proposal
    proposal = TransactionProposal(
        events=[event],
        file_operations=[file_op],
        estimated_duration=100.0,
        required_agents=["test_agent"],
        confidence=0.95
    )
    
    # Execute the transaction
    result = await tx_manager.execute(proposal)
    
    # Check result
    assert result.success
    assert result.transaction_id is not None
    assert result.commit_hash is not None
    
    # Check that the event log was created
    event_log_path = temp_workspace / "events.jsonl"
    assert event_log_path.exists()
    
    # Check that the file was created
    file_path = temp_workspace / "tasks" / "test_task" / "README.md"
    assert file_path.exists()
    
    # Check file content
    with open(file_path, "r") as f:
        content = f.read()
        assert content == "# Test Task\n\nThis task was created for testing purposes."


@pytest.mark.asyncio
async def test_transaction_rollback(temp_workspace):
    """Test transaction rollback on failure."""
    # Initialize transaction manager
    tx_manager = TransactionManager(temp_workspace)
    
    # Create a sample event
    event = Event(
        id="evt_test",
        timestamp=datetime.now(),
        category="directive",
        type=DirectiveEventType.TASK_CREATED,
        source=EventSource(
            type="test",
            id="test_script",
            version="1.0.0"
        ),
        payload={
            "task_id": "task_test",
            "name": "Test Task",
            "description": "A task created for testing purposes"
        },
        metadata=EventMetadata(
            user_id="test_user",
            session_id="test_session"
        )
    )
    
    # Create a valid file operation
    file_op1 = FileOperation(
        type="create_file",
        path="tasks/test_task/README.md",
        content="# Test Task\n\nThis task was created for testing purposes."
    )
    
    # Create an invalid file operation (modify non-existent file)
    file_op2 = FileOperation(
        type="modify_file",
        path="tasks/test_task/non_existent.md",
        content="This file doesn't exist."
    )
    
    # Create a transaction proposal with the invalid operation
    proposal = TransactionProposal(
        events=[event],
        file_operations=[file_op1, file_op2],
        estimated_duration=100.0,
        required_agents=["test_agent"],
        confidence=0.95
    )
    
    # Execute the transaction (should fail)
    result = await tx_manager.execute(proposal)
    
    # Check result
    assert not result.success
    assert result.error is not None
    
    # Check that the file was not created (rollback)
    file_path = temp_workspace / "tasks" / "test_task" / "README.md"
    assert not file_path.exists()