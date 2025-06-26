"""
Tests for the example scripts.
"""
import os
import sys
from pathlib import Path

import pytest

# Add the examples directory to the path
sys.path.append(str(Path(__file__).parent.parent / "examples"))


@pytest.mark.asyncio
@pytest.mark.skip(reason="Integration test that requires a full environment")
async def test_demo_script():
    """Test the demo script."""
    # Import the demo script
    from demo import main
    
    # Run the demo
    await main()
    
    # Check that the workspace was created
    workspace_path = Path("./aura_workspace")
    assert workspace_path.exists()
    
    # Check that the event log was created
    event_log_path = workspace_path / "events.jsonl"
    assert event_log_path.exists()
    
    # Check that the task directory was created
    task_path = workspace_path / "tasks" / "demo_task"
    assert task_path.exists()
    
    # Check that the README.md file was created
    readme_path = task_path / "README.md"
    assert readme_path.exists()
    
    # Check file content
    with open(readme_path, "r") as f:
        content = f.read()
        assert content == "# Demo Task\n\nThis task was created by the AURA demo script."
    
    # Clean up
    import shutil
    shutil.rmtree(workspace_path)