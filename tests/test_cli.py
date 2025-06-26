"""
Tests for the CLI.
"""
from typer.testing import CliRunner

import pytest

from aura.cli.main import app


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


def test_run_command(runner):
    """Test the run command."""
    result = runner.invoke(
        app,
        ["run", "Create a test task"],
        input="n\n"  # Answer "no" to the confirmation prompt
    )
    assert result.exit_code == 0
    assert "AURA - AI-Native Meta-OS" in result.stdout
    assert "Understanding your request" in result.stdout
    assert "Execution Plan Generated" in result.stdout
    assert "Execution cancelled" in result.stdout


def test_status_command(runner):
    """Test the status command."""
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "AURA System Status" in result.stdout
    assert "System Health" in result.stdout
    assert "Active Tasks" in result.stdout
    assert "Recent Events" in result.stdout
    assert "Agent Pool" in result.stdout
    assert "Memory Graph" in result.stdout


def test_memory_command(runner):
    """Test the memory command."""
    result = runner.invoke(
        app,
        ["memory", "What marketing strategies worked best?"],
        input="n\n"  # Answer "no" to the confirmation prompt
    )
    assert result.exit_code == 0
    assert "Searching memory graph" in result.stdout
    assert "Key Insights from Previous Launches" in result.stdout
    assert "Most Effective Strategies" in result.stdout
    assert "Success Patterns" in result.stdout
    assert "Lessons Learned" in result.stdout
    assert "Strategy creation cancelled" in result.stdout


def test_log_command(runner):
    """Test the log command."""
    result = runner.invoke(app, ["log", "--graph"])
    assert result.exit_code == 0
    assert "commit e7f8g9h0" in result.stdout
    assert "TASK: marketing_campaign" in result.stdout
    assert "Events: 8 events" in result.stdout
    assert "Files changed:" in result.stdout


def test_revert_command(runner):
    """Test the revert command."""
    result = runner.invoke(
        app,
        ["revert", "v2.0-campaign-init"],
        input="n\n"  # Answer "no" to the confirmation prompt
    )
    assert result.exit_code == 0
    assert "Time Travel Warning" in result.stdout
    assert "This will revert your workspace to" in result.stdout
    assert "Time travel cancelled" in result.stdout


def test_config_command(runner):
    """Test the config command."""
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "AURA Configuration" in result.stdout
    assert "system:" in result.stdout
    assert "kernel:" in result.stdout
    assert "memory:" in result.stdout
    assert "agents:" in result.stdout
    assert "git:" in result.stdout