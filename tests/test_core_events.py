"""
Tests for the core events system.
"""
import asyncio
from datetime import datetime

import pytest

from aura.core.events import (
    DirectiveEventType, AnalyticalEventType, Event, EventSource, EventMetadata, event_bus
)


def test_event_creation():
    """Test that events can be created."""
    # Create event source
    source = EventSource(
        type="TestSource",
        id="test_source_1",
        version="1.0.0"
    )
    
    # Create event metadata
    metadata = EventMetadata(
        correlation_id="corr_123",
        user_id="user_test",
        session_id="sess_456",
        confidence=0.95
    )
    
    # Create directive event
    directive_event = Event(
        id="evt_20250626_123456_abcdef",
        timestamp=datetime.now(),
        category="directive",
        type=DirectiveEventType.TASK_CREATED.value,
        source=source,
        payload={"task_id": "task_123", "name": "Test Task"},
        metadata=metadata
    )
    
    # Create analytical event
    analytical_event = Event(
        id="evt_20250626_123457_ghijkl",
        timestamp=datetime.now(),
        category="analytical",
        type=AnalyticalEventType.PREFERENCE_INFERRED.value,
        source=source,
        payload={"preference": "dark_mode", "confidence": 0.9},
        metadata=metadata
    )
    
    # Check event properties
    assert directive_event.category == "directive"
    assert directive_event.type == DirectiveEventType.TASK_CREATED.value
    assert directive_event.payload["task_id"] == "task_123"
    
    assert analytical_event.category == "analytical"
    assert analytical_event.type == AnalyticalEventType.PREFERENCE_INFERRED.value
    assert analytical_event.payload["preference"] == "dark_mode"


def test_event_serialization():
    """Test that events can be serialized and deserialized."""
    # Create event
    event = Event(
        id="evt_20250626_123456_abcdef",
        timestamp=datetime.now(),
        category="directive",
        type=DirectiveEventType.TASK_CREATED.value,
        source=EventSource(
            type="TestSource",
            id="test_source_1",
            version="1.0.0"
        ),
        payload={"task_id": "task_123", "name": "Test Task"},
        metadata=EventMetadata(
            correlation_id="corr_123",
            user_id="user_test",
            session_id="sess_456",
            confidence=0.95
        )
    )
    
    # Serialize to JSON
    json_str = event.to_json()
    
    # Deserialize from JSON
    event2 = Event.from_json(json_str)
    
    # Check that deserialized event matches original
    assert event2.id == event.id
    assert event2.category == event.category
    assert event2.type == event.type
    assert event2.payload == event.payload
    assert event2.metadata.correlation_id == event.metadata.correlation_id


@pytest.mark.asyncio
async def test_event_bus():
    """Test that events can be published and received."""
    # Create event
    event = Event(
        id="evt_20250626_123456_abcdef",
        timestamp=datetime.now(),
        category="directive",
        type=DirectiveEventType.TASK_CREATED.value,
        source=EventSource(
            type="TestSource",
            id="test_source_1",
            version="1.0.0"
        ),
        payload={"task_id": "task_123", "name": "Test Task"},
        metadata=EventMetadata(
            correlation_id="corr_123",
            user_id="user_test",
            session_id="sess_456",
            confidence=0.95
        )
    )
    
    # Create event receiver
    received_events = []
    
    async def event_receiver(evt):
        received_events.append(evt)
    
    # Subscribe to events
    unsubscribe = event_bus.subscribe(DirectiveEventType.TASK_CREATED.value, event_receiver)
    
    # Publish event
    await event_bus.publish(event)
    
    # Wait for event to be processed
    await asyncio.sleep(0.1)
    
    # Check that event was received
    assert len(received_events) == 1
    assert received_events[0].id == event.id
    
    # Unsubscribe
    unsubscribe()
    
    # Publish another event
    event2 = Event(
        id="evt_20250626_123457_ghijkl",
        timestamp=datetime.now(),
        category="directive",
        type=DirectiveEventType.TASK_CREATED.value,
        source=event.source,
        payload={"task_id": "task_456", "name": "Test Task 2"},
        metadata=event.metadata
    )
    
    await event_bus.publish(event2)
    
    # Wait for event to be processed
    await asyncio.sleep(0.1)
    
    # Check that event was not received (unsubscribed)
    assert len(received_events) == 1