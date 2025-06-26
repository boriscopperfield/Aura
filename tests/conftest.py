"""
Pytest configuration and fixtures for AURA tests.
"""
import asyncio
import pytest
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
import os

# Test environment setup
os.environ["AURA_ENV"] = "test"
os.environ["AURA_LOG_LEVEL"] = "DEBUG"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
async def mock_workspace(temp_dir: Path) -> AsyncGenerator[Path, None]:
    """Create a mock AURA workspace."""
    workspace = temp_dir / "aura_workspace"
    workspace.mkdir()
    
    # Create basic structure
    (workspace / ".aura").mkdir()
    (workspace / "tasks").mkdir()
    (workspace / "memory").mkdir()
    (workspace / "memory" / "graph").mkdir()
    (workspace / "memory" / "indexes").mkdir()
    
    # Create config file
    config_content = """
system:
  version: "4.0.0"
  workspace: "{workspace}"

memory:
  cache_size: "1GB"
  retention_days: 30

agents:
  default_timeout: 30
  max_retries: 2
""".format(workspace=workspace)
    
    with open(workspace / ".aura" / "config.yaml", "w") as f:
        f.write(config_content)
    
    # Create empty event log
    with open(workspace / "events.jsonl", "w") as f:
        pass
    
    yield workspace


@pytest.fixture
def sample_memory_nodes():
    """Sample memory nodes for testing."""
    from aura.memory.models import (
        MemoryNode, ContentBlock, ContentType, EntityType,
        MemorySource, NamedEntity, Relation, RelationType
    )
    from datetime import datetime
    
    source = MemorySource(type="test", user_id="test_user")
    
    node1 = MemoryNode(
        id="test_node_1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        entity_type=EntityType.TASK_ARTIFACT,
        source=source,
        content=[
            ContentBlock(
                type=ContentType.TEXT,
                data="This is test content for node 1",
                metadata={"language": "en"}
            )
        ],
        summary="Test node 1",
        keywords=["test", "node", "content"],
        entities=[
            NamedEntity(type="concept", value="testing", confidence=0.9)
        ],
        relations=[],
        importance=0.8,
        access_count=0,
        last_accessed=datetime.now(),
        decay_rate=0.01
    )
    
    node2 = MemoryNode(
        id="test_node_2",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        entity_type=EntityType.KNOWLEDGE_FACT,
        source=source,
        content=[
            ContentBlock(
                type=ContentType.TEXT,
                data="This is test content for node 2",
                metadata={"language": "en"}
            )
        ],
        summary="Test node 2",
        keywords=["test", "knowledge", "fact"],
        entities=[
            NamedEntity(type="concept", value="knowledge", confidence=0.95)
        ],
        relations=[
            Relation(
                type=RelationType.SIMILAR_TO,
                target_id="test_node_1",
                strength=0.7
            )
        ],
        importance=0.9,
        access_count=0,
        last_accessed=datetime.now(),
        decay_rate=0.01
    )
    
    return [node1, node2]


@pytest.fixture
def mock_agent_config():
    """Mock agent configuration for testing."""
    return {
        "agents": {
            "test_openai": {
                "type": "openai",
                "enabled": True,
                "priority": 1,
                "config": {
                    "api_key": "test_key",
                    "model": "gpt-3.5-turbo"
                },
                "capabilities": ["text_generation", "analysis"]
            },
            "test_local": {
                "type": "local",
                "enabled": True,
                "priority": 2,
                "config": {},
                "capabilities": ["code_generation", "analysis"]
            }
        }
    }


@pytest.fixture
def sample_events():
    """Sample events for testing."""
    from aura.core.events import Event, EventType, EventSource
    from datetime import datetime
    
    return [
        Event(
            id="test_event_1",
            timestamp=datetime.now(),
            category="directive",
            type=EventType.TASK_CREATED,
            source=EventSource(type="user", id="test_user"),
            payload={"task_id": "test_task", "name": "Test Task"},
            metadata={}
        ),
        Event(
            id="test_event_2",
            timestamp=datetime.now(),
            category="analytical",
            type=EventType.PATTERN_DETECTED,
            source=EventSource(type="cognitive_service", id="pattern_detector"),
            payload={"pattern": "user_preference", "confidence": 0.85},
            metadata={}
        )
    ]


# Mock external services for testing
@pytest.fixture
def mock_openai_responses():
    """Mock OpenAI API responses."""
    return {
        "chat_completion": {
            "choices": [
                {
                    "message": {
                        "content": "This is a test response from OpenAI"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            }
        },
        "embedding": {
            "data": [
                {
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 200  # 1000-dim vector
                }
            ],
            "usage": {
                "total_tokens": 5
            }
        }
    }


@pytest.fixture
def mock_jina_responses():
    """Mock Jina AI API responses."""
    return {
        "embedding": {
            "data": [
                {
                    "embedding": [0.2, 0.3, 0.4, 0.5, 0.6] * 200  # 1000-dim vector
                }
            ],
            "usage": {
                "total_tokens": 5
            }
        },
        "rerank": {
            "results": [
                {
                    "index": 0,
                    "relevance_score": 0.95,
                    "document": "Most relevant document"
                },
                {
                    "index": 1,
                    "relevance_score": 0.75,
                    "document": "Less relevant document"
                }
            ],
            "usage": {
                "total_tokens": 20
            }
        }
    }


# Performance testing fixtures
@pytest.fixture
def performance_config():
    """Configuration for performance tests."""
    return {
        "memory_nodes": 1000,
        "concurrent_requests": 10,
        "test_duration": 30,  # seconds
        "max_response_time": 1.0,  # seconds
        "min_success_rate": 0.95
    }