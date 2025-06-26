"""
Tests for the configuration system.
"""
import os
from pathlib import Path

import pytest

from aura.config import settings


def test_settings_defaults():
    """Test that settings have default values."""
    assert settings.app_name == "AURA"
    assert settings.app_version == "4.0.0"
    assert isinstance(settings.workspace.path, Path)
    assert settings.memory.vector_dimensions == 768
    assert settings.memory.similarity_threshold == 0.7


def test_settings_override():
    """Test that settings can be overridden."""
    # Save original value
    original_path = settings.workspace.path
    
    try:
        # Override setting
        test_path = Path("/tmp/test_workspace")
        settings.workspace.path = test_path
        
        # Check that it was updated
        assert settings.workspace.path == test_path
    finally:
        # Restore original value
        settings.workspace.path = original_path


def test_settings_from_env(monkeypatch):
    """Test that settings can be loaded from environment variables."""
    # Set environment variable
    monkeypatch.setenv("AURA_DEBUG", "true")
    
    # Reload settings
    from importlib import reload
    from aura.config import settings as settings_module
    reload(settings_module)
    
    # Check that it was updated
    from aura.config import settings
    assert settings.debug is True
    
    # Clean up
    monkeypatch.setenv("AURA_DEBUG", "false")
    reload(settings_module)