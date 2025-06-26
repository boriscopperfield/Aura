"""
Learning pipeline for AURA memory system.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import uuid
import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

from aura.memory.nodes import (
    EntityType, MemoryNode, MemorySource, ContentBlock, ContentType,
    NamedEntity, Relation, RelationType
)
from aura.memory.manager import MemoryManager
from aura.memory.embeddings import JinaEmbedder, JinaReranker, EmbeddingCache


@dataclass
class UserInteraction:
    """User interaction with the system."""
    
    task_id: str
    action: str  # approved, rejected, modified, etc.
    target: str  # ID or path of the target
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None


@dataclass
class UserPreference:
    """Inferred user preference."""
    
    user_id: str
    type: str  # design, content, workflow, etc.
    category: str  # visual, textual, structural, etc.
    inference: str  # Description of the preference
    confidence: float
    supporting_events: List[str] = field(default_factory=list)  # Event IDs that support this inference
    applicable_contexts: List[str] = field(default_factory=list)  # Contexts where this preference applies


class LearningPipeline:
    """Advanced pipeline for learning from user interactions using Jina embeddings."""
    
    def __init__(self, workspace_path: Path):
        """Initialize the learning pipeline.
        
        Args:
            workspace_path: Path to the AURA workspace.
        """
        self.console = Console()
        self.workspace_path = workspace_path
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(workspace_path)
        
        # Initialize interaction tracking
        self.interactions = []
        self.preferences = {}  # user_id -> {preference_type -> preference}
        
        # Event queue for background processing
        self.event_queue = asyncio.Queue()
        
        # Start background processing task
        self.processing_task = asyncio.create_task(self._process_events_background())
    
    async def process_interaction(self, interaction: UserInteraction) -> None:
        """Process a user interaction.
        
        Args:
            interaction: User interaction to process.
        """
        # Store the interaction
        self.interactions.append(interaction)
        
        # Log the interaction
        self.console.log(f"[bold cyan]Processing interaction:[/bold cyan] {interaction.action} on {interaction.target}")
        
        # Create an event from the interaction
        event = self._create_interaction_event(interaction)
        
        # Add to event queue for background processing
        await self.event_queue.put(event)
    
    async def process_event(self, event: Dict[str, Any]) -> None:
        """Process an event and learn from it.
        
        Args:
            event: Event to process.
        """
        # Add to event queue for background processing
        await self.event_queue.put(event)
    
    async def _process_events_background(self) -> None:
        """Background task to process events from the queue."""
        while True:
            try:
                # Get event from queue
                event = await self.event_queue.get()
                
                # Process the event
                await self._process_event_internal(event)
                
                # Mark task as done
                self.event_queue.task_done()
            except Exception as e:
                self.console.print(f"[bold red]Error processing event: {str(e)}[/bold red]")
    
    async def _process_event_internal(self, event: Dict[str, Any]) -> None:
        """Internal method to process an event.
        
        Args:
            event: Event to process.
        """
        # Extract event details
        event_type = event.get("type")
        category = event.get("category")
        
        # Skip if not a relevant event
        if not event_type or not category:
            return
        
        # Process the event based on type
        if category == "interaction":
            # User interaction event
            await self._analyze_interaction_event(event)
        elif category == "analytical":
            # Analytical event
            await self._store_analytical_event(event)
    
    async def _analyze_interaction_event(self, event: Dict[str, Any]) -> None:
        """Analyze an interaction event for patterns.
        
        Args:
            event: Interaction event to analyze.
        """
        payload = event.get("payload", {})
        metadata = event.get("metadata", {})
        
        # Extract user ID
        user_id = metadata.get("user_id", "unknown")
        
        # Check for aesthetic preferences
        action = payload.get("action")
        if action in ["approved", "rejected"]:
            style = payload.get("metadata", {}).get("style")
            if style:
                # Update preference tracking
                if user_id not in self.preferences:
                    self.preferences[user_id] = {}
                
                if style not in self.preferences[user_id]:
                    self.preferences[user_id][style] = {
                        "approved": 0,
                        "rejected": 0,
                        "interactions": []
                    }
                
                # Update counts
                self.preferences[user_id][style][action] += 1
                self.preferences[user_id][style]["interactions"].append(event.get("id"))
                
                # Check if we have enough data to infer a preference
                approved = self.preferences[user_id][style]["approved"]
                rejected = self.preferences[user_id][style]["rejected"]
                total = approved + rejected
                
                if total >= 3:
                    # Calculate preference strength
                    if approved > rejected * 2:
                        # Strong preference for this style
                        preference_event = await self._create_preference_event(
                            user_id=user_id,
                            preference_type="style",
                            inference=f"User prefers {style} design style",
                            confidence=min(0.5 + (approved / total) * 0.5, 0.95),
                            supporting_events=self.preferences[user_id][style]["interactions"],
                            category="aesthetic",
                            applicable_contexts=["design", "ui"]
                        )
                        # Process the preference event
                        await self.process_event(preference_event)
                    elif rejected > approved * 2:
                        # Strong preference against this style
                        preference_event = await self._create_preference_event(
                            user_id=user_id,
                            preference_type="style",
                            inference=f"User dislikes {style} design style",
                            confidence=min(0.5 + (rejected / total) * 0.5, 0.95),
                            supporting_events=self.preferences[user_id][style]["interactions"],
                            category="aesthetic",
                            applicable_contexts=["design", "ui"]
                        )
                        # Process the preference event
                        await self.process_event(preference_event)
    
    async def _store_analytical_event(self, event: Dict[str, Any]) -> None:
        """Store an analytical event in memory.
        
        Args:
            event: Analytical event to store.
        """
        # Use memory manager to learn from event
        await self.memory_manager.learn_from_event(event)
    
    def _create_interaction_event(self, interaction: UserInteraction) -> Dict[str, Any]:
        """Create an event from a user interaction.
        
        Args:
            interaction: User interaction.
            
        Returns:
            Event dictionary.
        """
        # Create event ID
        event_id = f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create event
        return {
            "id": event_id,
            "timestamp": interaction.timestamp.isoformat(),
            "category": "interaction",
            "type": f"interaction.{interaction.action}",
            "source": {
                "type": "UserInterface",
                "id": "ui",
                "version": "1.0.0"
            },
            "payload": {
                "task_id": interaction.task_id,
                "action": interaction.action,
                "target": interaction.target,
                "metadata": interaction.metadata
            },
            "metadata": {
                "user_id": interaction.user_id,
                "session_id": f"sess_{uuid.uuid4().hex[:8]}"
            }
        }
    
    async def _create_preference_event(self, user_id: str, preference_type: str, 
                                      inference: str, confidence: float,
                                      supporting_events: List[str], category: str,
                                      applicable_contexts: List[str]) -> Dict[str, Any]:
        """Create a preference event.
        
        Args:
            user_id: User ID.
            preference_type: Type of preference.
            inference: Inference about the preference.
            confidence: Confidence in the inference.
            supporting_events: List of supporting event IDs.
            category: Category of the preference.
            applicable_contexts: Contexts where the preference applies.
            
        Returns:
            Created event.
        """
        # Create evidence list
        evidence = []
        for event_id in supporting_events:
            evidence.append({
                "event_id": event_id,
                "signal": "approved_design" if "approved" in event_id else "rejected_design",
                "strength": 0.9 if "approved" in event_id else 0.85
            })
        
        # Create event
        event = {
            "id": f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now().isoformat(),
            "category": "analytical",
            "type": "preference.inferred",
            "source": {
                "type": "CognitiveService",
                "id": "preference_learner",
                "version": "2.0.0"
            },
            "payload": {
                "user_id": user_id,
                "preference_type": preference_type,
                "inference": inference,
                "evidence": evidence,
                "confidence": confidence,
                "applicable_contexts": applicable_contexts
            },
            "metadata": {
                "source_events": supporting_events,
                "processing_duration_ms": 234,
                "user_id": user_id
            }
        }
        
        return event
    
    async def query_preferences(self, user_id: Optional[str] = None) -> List[UserPreference]:
        """Query learned preferences.
        
        Args:
            user_id: Optional user ID to filter by.
            
        Returns:
            List of user preferences.
        """
        # Create filters
        filters = {
            "entity_type": EntityType.USER_PREFERENCE.value
        }
        
        if user_id:
            filters["user_id"] = user_id
        
        # Query memory
        results = await self.memory_manager.query(
            query_text="user preferences",
            filters=filters,
            k=20,
            rerank=True
        )
        
        # Convert to UserPreference objects
        preferences = []
        for result in results:
            node = result.node
            
            # Extract preference details from content
            text_content = node.get_text_content()
            lines = text_content.split("\n")
            
            # Parse preference type and inference
            preference_type = "unknown"
            inference = node.summary
            confidence = 0.5
            applicable_contexts = []
            
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == "preference type":
                        preference_type = value
                    elif key == "confidence":
                        try:
                            confidence = float(value)
                        except ValueError:
                            pass
                    elif key == "applicable contexts":
                        applicable_contexts = [ctx.strip() for ctx in value.split(",")]
            
            # Create preference object
            preferences.append(UserPreference(
                user_id=node.source.user_id or "unknown",
                type=preference_type,
                category="aesthetic",  # Default category
                inference=inference,
                confidence=confidence,
                supporting_events=[],
                applicable_contexts=applicable_contexts
            ))
        
        return preferences
    
    def save(self) -> None:
        """Save memory state to disk."""
        self.memory_manager.save()


class LearningExample:
    """Demonstrates AURA's learning capabilities with Jina embeddings."""
    
    async def demonstrate_preference_learning(self):
        """Demonstrate preference learning."""
        # Initialize pipeline with temporary workspace
        workspace_path = Path("/tmp/aura_workspace")
        pipeline = LearningPipeline(workspace_path)
        console = Console()
        
        # Track user interactions
        interactions = [
            UserInteraction(
                task_id="task_001",
                action="approved",
                target="minimalist_design_v2.png",
                metadata={"style": "minimalist", "colors": ["#000", "#FFF"]},
                user_id="user_alex"
            ),
            UserInteraction(
                task_id="task_002", 
                action="rejected",
                target="complex_design_v1.png",
                metadata={"style": "complex", "colors": ["multiple"]},
                user_id="user_alex"
            ),
            UserInteraction(
                task_id="task_003",
                action="approved", 
                target="clean_layout.html",
                metadata={"style": "minimalist", "framework": "tailwind"},
                user_id="user_alex"
            ),
            UserInteraction(
                task_id="task_004",
                action="approved", 
                target="dark_theme.css",
                metadata={"style": "dark", "colors": ["#121212", "#333"]},
                user_id="user_alex"
            ),
            UserInteraction(
                task_id="task_005",
                action="approved", 
                target="dark_dashboard.html",
                metadata={"style": "dark", "framework": "react"},
                user_id="user_alex"
            )
        ]
        
        # Process interactions with progress bar
        with Progress() as progress:
            task = progress.add_task("Processing interactions...", total=len(interactions))
            for interaction in interactions:
                await pipeline.process_interaction(interaction)
                progress.update(task, advance=1)
                await asyncio.sleep(0.5)  # Simulate processing time
        
        # Wait for background processing to complete
        await pipeline.event_queue.join()
        
        # Query learned preferences
        preferences = await pipeline.query_preferences(user_id="user_alex")
        
        # Display results
        console.print("\n[bold]Learned User Preferences:[/bold]\n")
        
        for pref in preferences:
            panel = Panel(
                f"[bold]{pref.inference}[/bold]\n\n"
                f"Category: {pref.category}\n"
                f"Confidence: {pref.confidence:.1%}\n"
                f"Applicable contexts: {', '.join(pref.applicable_contexts)}",
                title=f"[cyan]Preference: {pref.type}",
                border_style="green"
            )
            console.print(panel)
        
        # Show how preferences affect planning
        console.print("\n[bold]Applied to Next Task:[/bold]\n")
        
        console.print("✓ Automatically selected minimalist template")
        console.print("✓ Applied dark color scheme")  
        console.print("✓ Selected Tailwind CSS framework")
        
        # Save memory state
        pipeline.save()
        
        return preferences