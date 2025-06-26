"""
AI-powered planner for AURA system.
"""
import os
import json
import dotenv
from typing import Dict, List, Any, Optional
from datetime import datetime

import openai
from rich.console import Console

from aura.kernel.events import DirectiveEventType, AnalyticalEventType, Event, EventMetadata, EventSource
from aura.kernel.transaction import FileOperation, TransactionProposal

# Load environment variables
dotenv.load_dotenv("/workspace/.env")

# Configure OpenAI client
openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Debug info
print(f"OpenAI API Key: {openai.api_key[:5]}...{openai.api_key[-4:] if openai.api_key else 'None'}")
print(f"OpenAI Base URL: {openai.base_url}")

console = Console()

class AIPlannerService:
    """AI-powered planner service for AURA."""
    
    def __init__(self):
        """Initialize the AI planner service."""
        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        self.model = "gpt-4o"
        
    async def create_plan(self, intent: str) -> TransactionProposal:
        """Create a plan based on user intent."""
        console.print(f"[bold blue]Creating plan for intent:[/bold blue] {intent}")
        
        # Generate a plan using the AI model
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": f"Create a plan for the following intent: {intent}"}
            ],
            response_format={"type": "json_object"}
        )
        
        # Extract the plan from the response
        plan_json = response.choices[0].message.content
        plan = json.loads(plan_json)
        
        # Create events from the plan
        events = self._create_events_from_plan(plan, intent)
        
        # Create file operations from the plan
        file_operations = self._create_file_operations_from_plan(plan)
        
        # Create transaction proposal
        proposal = TransactionProposal(
            events=events,
            file_operations=file_operations,
            estimated_duration=plan.get("estimated_duration_seconds", 3600),
            required_agents=plan.get("required_agents", ["gpt-4o"]),
            confidence=plan.get("confidence", 0.9)
        )
        
        return proposal
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the AI model."""
        return """
        You are the AI Planner for the AURA system, an AI-native meta-operating system.
        Your job is to analyze user intent and create a detailed execution plan.
        
        The plan should include:
        1. A hierarchical task breakdown
        2. Required agents for each task
        3. Dependencies between tasks
        4. File operations needed
        5. Estimated duration
        6. Confidence score
        
        Respond with a JSON object that follows this structure:
        {
            "task_id": "unique_task_id",
            "name": "Task Name",
            "description": "Detailed task description",
            "subtasks": [
                {
                    "node_id": "unique_node_id",
                    "name": "Subtask Name",
                    "description": "Subtask description",
                    "assigned_agent": "agent_name",
                    "estimated_duration_seconds": 300,
                    "dependencies": ["other_node_id"],
                    "file_operations": [
                        {
                            "type": "create_file",
                            "path": "/tasks/task_id/subtask/file.txt",
                            "content": "File content"
                        }
                    ]
                }
            ],
            "estimated_duration_seconds": 1800,
            "required_agents": ["gpt-4o", "dall-e-3"],
            "confidence": 0.95
        }
        """
    
    def _create_events_from_plan(self, plan: Dict[str, Any], intent: str) -> List[Event]:
        """Create events from the plan."""
        events = []
        
        # Create task creation event
        task_created_event = Event(
            id=f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}",
            timestamp=datetime.now(),
            category="directive",
            type=DirectiveEventType.TASK_CREATED.value,
            source=EventSource(
                type="AIPlannerService",
                id="planner_v1",
                version="1.0.0"
            ),
            payload={
                "task_id": plan["task_id"],
                "name": plan["name"],
                "description": plan["description"],
                "intent": intent
            },
            metadata=EventMetadata(
                correlation_id=f"req_{os.urandom(4).hex()}",
                user_id="user_current",
                session_id=f"sess_{os.urandom(4).hex()}",
                git_commit=None,
                confidence=plan.get("confidence", 0.9)
            )
        )
        events.append(task_created_event)
        
        # Create node added events for each subtask
        for subtask in plan.get("subtasks", []):
            node_added_event = Event(
                id=f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}",
                timestamp=datetime.now(),
                category="directive",
                type=DirectiveEventType.NODE_ADDED.value,
                source=EventSource(
                    type="AIPlannerService",
                    id="planner_v1",
                    version="1.0.0"
                ),
                payload={
                    "task_id": plan["task_id"],
                    "node_id": subtask["node_id"],
                    "name": subtask["name"],
                    "description": subtask["description"],
                    "assigned_agent": subtask.get("assigned_agent", "gpt-4o"),
                    "dependencies": subtask.get("dependencies", [])
                },
                metadata=EventMetadata(
                    correlation_id=task_created_event.metadata.correlation_id,
                    causation_id=task_created_event.id,
                    user_id="user_current",
                    session_id=task_created_event.metadata.session_id,
                    git_commit=None,
                    confidence=plan.get("confidence", 0.9)
                )
            )
            events.append(node_added_event)
        
        return events
    
    def _create_file_operations_from_plan(self, plan: Dict[str, Any]) -> List[FileOperation]:
        """Create file operations from the plan."""
        file_operations = []
        
        # Create task directory
        task_dir_op = FileOperation(
            type="create_directory",
            path=f"/tasks/{plan['task_id']}"
        )
        file_operations.append(task_dir_op)
        
        # Create task metadata file
        task_meta_op = FileOperation(
            type="create_file",
            path=f"/tasks/{plan['task_id']}/_meta.json",
            content=json.dumps({
                "id": plan["task_id"],
                "name": plan["name"],
                "description": plan["description"],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "estimated_duration": plan.get("estimated_duration_seconds", 3600),
                "required_agents": plan.get("required_agents", ["gpt-4o"]),
                "confidence": plan.get("confidence", 0.9)
            }, indent=2)
        )
        file_operations.append(task_meta_op)
        
        # Create README.md file
        readme_op = FileOperation(
            type="create_file",
            path=f"/tasks/{plan['task_id']}/README.md",
            content=f"# {plan['name']}\n\n{plan['description']}"
        )
        file_operations.append(readme_op)
        
        # Create file operations for each subtask
        for subtask in plan.get("subtasks", []):
            # Create subtask directory
            subtask_dir_op = FileOperation(
                type="create_directory",
                path=f"/tasks/{plan['task_id']}/{subtask['node_id']}"
            )
            file_operations.append(subtask_dir_op)
            
            # Create subtask metadata file
            subtask_meta_op = FileOperation(
                type="create_file",
                path=f"/tasks/{plan['task_id']}/{subtask['node_id']}/_meta.json",
                content=json.dumps({
                    "id": subtask["node_id"],
                    "name": subtask["name"],
                    "description": subtask["description"],
                    "assigned_agent": subtask.get("assigned_agent", "gpt-4o"),
                    "dependencies": subtask.get("dependencies", []),
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "estimated_duration": subtask.get("estimated_duration_seconds", 300)
                }, indent=2)
            )
            file_operations.append(subtask_meta_op)
            
            # Add any specific file operations from the subtask
            for file_op in subtask.get("file_operations", []):
                operation = FileOperation(
                    type=file_op["type"],
                    path=file_op["path"],
                    content=file_op.get("content", "")
                )
                file_operations.append(operation)
        
        return file_operations