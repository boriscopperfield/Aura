"""
System-level tests for AURA.
"""
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from aura.kernel.events import DirectiveEventType, Event, EventMetadata, EventSource
from aura.kernel.transaction import FileOperation, TransactionManager, TransactionProposal
from aura.memory.learning import LearningPipeline, UserInteraction


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.mark.asyncio
async def test_end_to_end_workflow(temp_workspace):
    """Test an end-to-end workflow with AURA."""
    # Initialize transaction manager
    tx_manager = TransactionManager(temp_workspace)
    
    # Initialize learning pipeline
    learning_pipeline = LearningPipeline()
    
    # Step 1: Create a task
    task_event = Event(
        id=f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_task",
        timestamp=datetime.now(),
        category="directive",
        type=DirectiveEventType.TASK_CREATED,
        source=EventSource(
            type="test",
            id="test_system",
            version="1.0.0"
        ),
        payload={
            "task_id": "task_marketing",
            "name": "Marketing Campaign",
            "description": "Create a marketing campaign for our new product"
        },
        metadata=EventMetadata(
            user_id="test_user",
            session_id="test_session"
        )
    )
    
    # Create task directory and README
    task_file_op = FileOperation(
        type="create_file",
        path="tasks/marketing/README.md",
        content="# Marketing Campaign\n\nCreate a marketing campaign for our new product."
    )
    
    # Create transaction proposal
    task_proposal = TransactionProposal(
        events=[task_event],
        file_operations=[task_file_op],
        estimated_duration=100.0,
        required_agents=["test_agent"],
        confidence=0.95
    )
    
    # Execute transaction
    task_result = await tx_manager.execute(task_proposal)
    assert task_result.success
    
    # Step 2: Add a subtask
    subtask_event = Event(
        id=f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_subtask",
        timestamp=datetime.now(),
        category="directive",
        type=DirectiveEventType.NODE_ADDED,
        source=EventSource(
            type="test",
            id="test_system",
            version="1.0.0"
        ),
        payload={
            "task_id": "task_marketing",
            "node_id": "node_design",
            "parent_id": "task_marketing",
            "path": "/tasks/marketing/design",
            "name": "Design Assets",
            "assigned_agent": "design_agent"
        },
        metadata=EventMetadata(
            user_id="test_user",
            session_id="test_session"
        )
    )
    
    # Create subtask directory and files
    subtask_file_ops = [
        FileOperation(
            type="create_file",
            path="tasks/marketing/design/README.md",
            content="# Design Assets\n\nCreate design assets for the marketing campaign."
        ),
        FileOperation(
            type="create_file",
            path="tasks/marketing/design/requirements.md",
            content="## Requirements\n\n- Logo\n- Banner\n- Social media images"
        )
    ]
    
    # Create transaction proposal
    subtask_proposal = TransactionProposal(
        events=[subtask_event],
        file_operations=subtask_file_ops,
        estimated_duration=200.0,
        required_agents=["design_agent"],
        confidence=0.9
    )
    
    # Execute transaction
    subtask_result = await tx_manager.execute(subtask_proposal)
    assert subtask_result.success
    
    # Step 3: Process user interactions
    interactions = [
        UserInteraction(
            task_id="task_marketing",
            action="approved",
            target="minimalist_logo.png",
            metadata={"style": "minimalist", "colors": ["#000", "#FFF"]}
        ),
        UserInteraction(
            task_id="task_marketing",
            action="rejected",
            target="complex_banner.png",
            metadata={"style": "complex", "colors": ["multiple"]}
        )
    ]
    
    # Process interactions
    for interaction in interactions:
        await learning_pipeline.process_interaction(interaction)
    
    # Query learned preferences
    preferences = await learning_pipeline.query_preferences()
    
    # Check that a preference was learned
    assert len(preferences) > 0
    
    # Step 4: Complete the task
    complete_event = Event(
        id=f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_complete",
        timestamp=datetime.now(),
        category="directive",
        type=DirectiveEventType.TASK_COMPLETED,
        source=EventSource(
            type="test",
            id="test_system",
            version="1.0.0"
        ),
        payload={
            "task_id": "task_marketing",
            "completion_time": datetime.now().isoformat(),
            "artifacts": ["logo.png", "banner.png", "social_media.png"]
        },
        metadata=EventMetadata(
            user_id="test_user",
            session_id="test_session"
        )
    )
    
    # Create final files
    complete_file_ops = [
        FileOperation(
            type="create_file",
            path="tasks/marketing/design/logo.png",
            content="<simulated binary data>"
        ),
        FileOperation(
            type="create_file",
            path="tasks/marketing/design/banner.png",
            content="<simulated binary data>"
        ),
        FileOperation(
            type="create_file",
            path="tasks/marketing/design/social_media.png",
            content="<simulated binary data>"
        ),
        FileOperation(
            type="modify_file",
            path="tasks/marketing/README.md",
            content="# Marketing Campaign\n\nCreate a marketing campaign for our new product.\n\n## Status\n\nCompleted on " + datetime.now().isoformat()
        )
    ]
    
    # Create transaction proposal
    complete_proposal = TransactionProposal(
        events=[complete_event],
        file_operations=complete_file_ops,
        estimated_duration=50.0,
        required_agents=["test_agent"],
        confidence=0.98
    )
    
    # Execute transaction
    complete_result = await tx_manager.execute(complete_proposal)
    assert complete_result.success
    
    # Check final state
    assert (temp_workspace / "tasks" / "marketing" / "design" / "logo.png").exists()
    assert (temp_workspace / "tasks" / "marketing" / "design" / "banner.png").exists()
    assert (temp_workspace / "tasks" / "marketing" / "design" / "social_media.png").exists()
    
    # Check README content
    with open(temp_workspace / "tasks" / "marketing" / "README.md", "r") as f:
        content = f.read()
        assert "Completed on " in content