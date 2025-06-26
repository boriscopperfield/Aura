"""
Tests for module imports.
"""
import pytest


def test_kernel_imports():
    """Test importing kernel modules."""
    from aura.kernel import events
    from aura.kernel import transaction
    
    assert hasattr(events, 'Event')
    assert hasattr(events, 'DirectiveEventType')
    assert hasattr(events, 'AnalyticalEventType')
    
    assert hasattr(transaction, 'TransactionManager')
    assert hasattr(transaction, 'TransactionProposal')
    assert hasattr(transaction, 'FileOperation')


def test_execution_imports():
    """Test importing execution modules."""
    from aura.execution import tasks
    
    assert hasattr(tasks, 'TaskNode')
    assert hasattr(tasks, 'TaskType')
    assert hasattr(tasks, 'TaskStatus')
    assert hasattr(tasks, 'TaskMetadata')


def test_memory_imports():
    """Test importing memory modules."""
    from aura.memory import nodes
    from aura.memory import learning
    
    assert hasattr(nodes, 'MemoryNode')
    assert hasattr(nodes, 'ContentBlock')
    assert hasattr(nodes, 'EntityType')
    assert hasattr(nodes, 'RelationType')
    
    assert hasattr(learning, 'LearningPipeline')
    assert hasattr(learning, 'UserInteraction')
    assert hasattr(learning, 'UserPreference')


def test_cli_imports():
    """Test importing CLI modules."""
    from aura.cli import main
    
    assert hasattr(main, 'app')
    assert hasattr(main, 'run')
    assert hasattr(main, 'status')
    assert hasattr(main, 'memory')
    assert hasattr(main, 'log')
    assert hasattr(main, 'revert')
    assert hasattr(main, 'config')