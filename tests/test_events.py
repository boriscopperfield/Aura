"""
Tests for the event model.
"""
import json
from datetime import datetime

import pytest

from aura.kernel.events import (
    AnalyticalEventType,
    DirectiveEventType,
    Event,
    EventMetadata,
    EventSource,
)


def test_event_serialization():
    """Test event serialization to JSON."""
    # Create a sample event
    event = Event(
        id="evt_20250626_123456_test",
        timestamp=datetime.fromisoformat("2025-06-26T12:34:56+00:00"),
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
    
    # Serialize to JSON
    json_str = event.to_json()
    
    # Parse JSON
    data = json.loads(json_str)
    
    # Check fields
    assert data["id"] == "evt_20250626_123456_test"
    assert data["timestamp"] == "2025-06-26T12:34:56+00:00"
    assert data["category"] == "directive"
    assert data["type"] == "task.created"
    assert data["source"]["type"] == "test"
    assert data["payload"]["task_id"] == "task_test"
    assert data["metadata"]["user_id"] == "test_user"


def test_event_deserialization():
    """Test event deserialization from JSON."""
    # Create a JSON string
    json_str = """
    {
        "id": "evt_20250626_123456_test",
        "timestamp": "2025-06-26T12:34:56+00:00",
        "category": "directive",
        "type": "task.created",
        "source": {
            "type": "test",
            "id": "test_script",
            "version": "1.0.0"
        },
        "payload": {
            "task_id": "task_test",
            "name": "Test Task",
            "description": "A task created for testing purposes"
        },
        "metadata": {
            "user_id": "test_user",
            "session_id": "test_session",
            "correlation_id": null,
            "causation_id": null,
            "git_commit": null,
            "confidence": null
        }
    }
    """
    
    # Deserialize from JSON
    event = Event.from_json(json_str)
    
    # Check fields
    assert event.id == "evt_20250626_123456_test"
    assert event.timestamp.isoformat() == "2025-06-26T12:34:56+00:00"
    assert event.category == "directive"
    assert event.type == DirectiveEventType.TASK_CREATED
    assert event.source.type == "test"
    assert event.payload["task_id"] == "task_test"
    assert event.metadata.user_id == "test_user"


def test_analytical_event():
    """Test creating an analytical event."""
    # Create a sample analytical event
    event = Event(
        id="evt_20250626_123456_analysis",
        timestamp=datetime.fromisoformat("2025-06-26T12:34:56+00:00"),
        category="analytical",
        type=AnalyticalEventType.PREFERENCE_INFERRED,
        source=EventSource(
            type="preference_learner",
            id="learner_v1",
            version="1.0.0"
        ),
        payload={
            "user_id": "user_alex",
            "preference_type": "aesthetic",
            "inference": "User prefers minimalist designs with high contrast",
            "evidence": [
                {
                    "event_id": "evt_20250626_105230_abc123",
                    "signal": "approved_design",
                    "strength": 0.9
                },
                {
                    "event_id": "evt_20250626_105845_def456",
                    "signal": "rejected_complex_layout",
                    "strength": 0.85
                }
            ],
            "confidence": 0.88,
            "applicable_contexts": ["ui_design", "marketing_materials"]
        },
        metadata=EventMetadata(
            user_id="system",
            confidence=0.88
        )
    )
    
    # Serialize to JSON
    json_str = event.to_json()
    
    # Parse JSON
    data = json.loads(json_str)
    
    # Check fields
    assert data["category"] == "analytical"
    assert data["type"] == "preference.inferred"
    assert data["payload"]["inference"] == "User prefers minimalist designs with high contrast"
    assert len(data["payload"]["evidence"]) == 2
    assert data["payload"]["confidence"] == 0.88