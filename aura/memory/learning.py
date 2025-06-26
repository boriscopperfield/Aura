"""
Learning pipeline for AURA memory system.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel

from aura.memory.nodes import EntityType, MemoryNode, MemorySource, NamedEntity, Relation, RelationType


@dataclass
class UserInteraction:
    """User interaction with the system."""
    
    task_id: str
    action: str  # approved, rejected, modified, etc.
    target: str  # ID or path of the target
    metadata: Dict[str, any]


@dataclass
class UserPreference:
    """Inferred user preference."""
    
    type: str  # design, content, workflow, etc.
    category: str  # visual, textual, structural, etc.
    inference: str  # Description of the preference
    confidence: float
    supporting_events: List[str]  # Event IDs that support this inference


class LearningPipeline:
    """Pipeline for learning from user interactions."""
    
    def __init__(self):
        self.console = Console()
        self.interactions = []
        self.preferences = []
    
    async def process_interaction(self, interaction: UserInteraction) -> None:
        """Process a user interaction."""
        self.interactions.append(interaction)
        
        # Log the interaction
        self.console.log(f"[bold cyan]Processing interaction:[/bold cyan] {interaction.action} on {interaction.target}")
        
        # Analyze for patterns
        await self.analyze_patterns()
    
    async def analyze_patterns(self) -> None:
        """Analyze interactions for patterns."""
        # This would be a complex analysis in a real system
        # Here we just simulate finding patterns
        
        # Check if we have enough interactions
        if len(self.interactions) < 3:
            return
        
        # Look for style preferences
        style_counts = {}
        for interaction in self.interactions:
            style = interaction.metadata.get("style")
            if not style:
                continue
            
            if style not in style_counts:
                style_counts[style] = {"approved": 0, "rejected": 0}
            
            if interaction.action == "approved":
                style_counts[style]["approved"] += 1
            elif interaction.action == "rejected":
                style_counts[style]["rejected"] += 1
        
        # Find preferred styles
        for style, counts in style_counts.items():
            if counts["approved"] >= 2 and counts["rejected"] == 0:
                # User seems to prefer this style
                preference = UserPreference(
                    type="design",
                    category="visual",
                    inference=f"User prefers {style} design style",
                    confidence=min(0.5 + 0.1 * counts["approved"], 0.9),
                    supporting_events=[interaction.task_id for interaction in self.interactions 
                                      if interaction.metadata.get("style") == style and interaction.action == "approved"]
                )
                
                # Check if we already have this preference
                if not any(p.inference == preference.inference for p in self.preferences):
                    self.preferences.append(preference)
                    self.console.log(f"[bold green]New preference detected:[/bold green] {preference.inference} (confidence: {preference.confidence:.1%})")
    
    async def create_preference_node(self, preference: UserPreference) -> MemoryNode:
        """Create a memory node for a user preference."""
        node_id = f"mem_{datetime.now().strftime('%Y%m%d')}_{len(self.preferences):03d}"
        
        # Create memory node
        node = MemoryNode(
            id=node_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            entity_type=EntityType.USER_PREFERENCE,
            source=MemorySource(
                type="preference_learner",
                user_id="user_alex"
            ),
            content=[],  # No content for preference nodes
            summary=preference.inference,
            keywords=[preference.type, preference.category, "preference"],
            entities=[
                NamedEntity(
                    type="preference_type",
                    value=preference.type,
                    confidence=1.0
                ),
                NamedEntity(
                    type="preference_category",
                    value=preference.category,
                    confidence=1.0
                )
            ],
            relations=[
                Relation(
                    type=RelationType.PRODUCED_BY,
                    target_id="preference_learner",
                    strength=1.0
                )
            ],
            importance=0.8,
            access_count=1,
            last_accessed=datetime.now(),
            decay_rate=0.001
        )
        
        return node
    
    async def query_preferences(self) -> List[UserPreference]:
        """Query learned preferences."""
        return self.preferences


class LearningExample:
    """Demonstrates AURA's learning capabilities."""
    
    async def demonstrate_preference_learning(self):
        """Demonstrate preference learning."""
        pipeline = LearningPipeline()
        console = Console()
        
        # Track user interactions
        interactions = [
            UserInteraction(
                task_id="task_001",
                action="approved",
                target="minimalist_design_v2.png",
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
                target="clean_layout.html",
                metadata={"style": "minimalist", "framework": "tailwind"}
            )
        ]
        
        # Process interactions
        for interaction in interactions:
            await pipeline.process_interaction(interaction)
        
        # Query learned preferences
        preferences = await pipeline.query_preferences()
        
        # Display results
        console.print("\n[bold]Learned User Preferences:[/bold]\n")
        
        for pref in preferences:
            panel = Panel(
                f"[bold]{pref.inference}[/bold]\n\n"
                f"Category: {pref.category}\n"
                f"Confidence: {pref.confidence:.1%}\n"
                f"Evidence: {len(pref.supporting_events)} interactions",
                title=f"[cyan]Preference: {pref.type}",
                border_style="green"
            )
            console.print(panel)
        
        # Show how preferences affect planning
        console.print("\n[bold]Applied to Next Task:[/bold]\n")
        
        console.print("✓ Automatically selected minimalist template")
        console.print("✓ Chose monochrome color scheme")  
        console.print("✓ Selected Tailwind CSS framework")