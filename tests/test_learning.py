"""
Tests for the learning pipeline.
"""
import pytest

from aura.memory.learning import LearningPipeline, UserInteraction


@pytest.mark.asyncio
async def test_preference_learning():
    """Test learning user preferences from interactions."""
    # Initialize learning pipeline
    pipeline = LearningPipeline()
    
    # Create sample interactions
    interactions = [
        UserInteraction(
            task_id="task_001",
            action="approved",
            target="minimalist_design_v1.png",
            metadata={"style": "minimalist", "colors": ["#000", "#FFF"]}
        ),
        UserInteraction(
            task_id="task_002",
            action="rejected",
            target="complex_design_v1.png",
            metadata={"style": "complex", "colors": ["multiple"]}
        ),
        UserInteraction(
            task_id="task_003",
            action="approved",
            target="minimalist_design_v2.png",
            metadata={"style": "minimalist", "colors": ["#000", "#FFF"]}
        )
    ]
    
    # Process interactions
    for interaction in interactions:
        await pipeline.process_interaction(interaction)
    
    # Query learned preferences
    preferences = await pipeline.query_preferences()
    
    # Check that a preference was learned
    assert len(preferences) > 0
    
    # Check preference details
    preference = preferences[0]
    assert preference.type == "design"
    assert "minimalist" in preference.inference
    assert preference.confidence > 0.5
    assert len(preference.supporting_events) == 2


@pytest.mark.asyncio
async def test_preference_node_creation():
    """Test creating a memory node for a preference."""
    # Initialize learning pipeline
    pipeline = LearningPipeline()
    
    # Create sample interactions
    interactions = [
        UserInteraction(
            task_id="task_001",
            action="approved",
            target="minimalist_design_v1.png",
            metadata={"style": "minimalist", "colors": ["#000", "#FFF"]}
        ),
        UserInteraction(
            task_id="task_002",
            action="rejected",
            target="complex_design_v1.png",
            metadata={"style": "complex", "colors": ["multiple"]}
        ),
        UserInteraction(
            task_id="task_003",
            action="approved",
            target="minimalist_design_v2.png",
            metadata={"style": "minimalist", "colors": ["#000", "#FFF"]}
        )
    ]
    
    # Process interactions
    for interaction in interactions:
        await pipeline.process_interaction(interaction)
    
    # Query learned preferences
    preferences = await pipeline.query_preferences()
    
    # Create a memory node for the preference
    node = await pipeline.create_preference_node(preferences[0])
    
    # Check node details
    assert node.entity_type.value == "user_preference"
    assert node.summary == preferences[0].inference
    assert "preference" in node.keywords
    assert node.source.type == "preference_learner"