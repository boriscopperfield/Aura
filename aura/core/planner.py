"""
AI-powered planner for AURA system.

This module provides AI planning capabilities for the AURA system,
translating high-level user intents into executable plans.
"""
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

import openai
from pydantic import BaseModel, Field

from aura.config import settings
from aura.core.events import (
    DirectiveEventType, AnalyticalEventType, Event, EventMetadata, EventSource
)
from aura.core.transaction import FileOperation, FileOperationType, TransactionProposal
from aura.utils.logging import logger
from aura.utils.errors import OpenAIError
from aura.utils.async_utils import async_retry


class PlanRequest(BaseModel):
    """Request for creating a plan."""
    
    intent: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)


class PlannerService:
    """Base class for planner services."""
    
    async def create_plan(self, request: PlanRequest) -> TransactionProposal:
        """Create a plan based on user intent.
        
        Args:
            request: Plan request
            
        Returns:
            Transaction proposal
        """
        raise NotImplementedError("Subclasses must implement create_plan")


class AIPlannerService(PlannerService):
    """AI-powered planner service for AURA."""
    
    def __init__(self):
        """Initialize the AI planner service."""
        self.client = openai.OpenAI(
            api_key=settings.openai.api_key,
            base_url=settings.openai.base_url
        )
        self.model = settings.openai.model
        self.logger = logger
    
    @async_retry(retries=3, exceptions=(Exception,))
    async def create_plan(self, request: PlanRequest) -> TransactionProposal:
        """Create a plan based on user intent.
        
        Args:
            request: Plan request
            
        Returns:
            Transaction proposal
            
        Raises:
            OpenAIError: If the OpenAI API call fails
        """
        self.logger.info(f"Creating plan for intent: {request.intent}")
        
        try:
            # Generate a plan using the AI model
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": f"Create a plan for the following intent: {request.intent}"}
                ],
                response_format={"type": "json_object"},
                timeout=settings.openai.timeout
            )
            
            # Extract the plan from the response
            plan_json = response.choices[0].message.content
            self.logger.debug(f"Response received: {plan_json[:100]}...")
            
            # Clean up the response if it contains markdown code blocks
            if plan_json.startswith("```json"):
                plan_json = plan_json.replace("```json", "", 1)
                if plan_json.endswith("```"):
                    plan_json = plan_json[:-3]
                plan_json = plan_json.strip()
            
            # Parse JSON response
            plan = json.loads(plan_json)
        except Exception as e:
            self.logger.error(f"Error with OpenAI API: {str(e)}")
            raise OpenAIError(f"Failed to generate plan: {str(e)}")
        
        # Create events from the plan
        events = self._create_events_from_plan(plan, request)
        
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
    
    def _create_events_from_plan(self, plan: Dict[str, Any], request: PlanRequest) -> List[Event]:
        """Create events from the plan.
        
        Args:
            plan: Plan data
            request: Plan request
            
        Returns:
            List of events
        """
        events = []
        
        # Create correlation ID for all events
        correlation_id = f"req_{uuid.uuid4().hex[:8]}"
        
        # Create task creation event
        task_created_event = Event(
            id=f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
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
                "intent": request.intent
            },
            metadata=EventMetadata(
                correlation_id=correlation_id,
                user_id=request.user_id or "user_current",
                session_id=request.session_id or f"sess_{uuid.uuid4().hex[:8]}",
                git_commit=None,
                confidence=plan.get("confidence", 0.9)
            )
        )
        events.append(task_created_event)
        
        # Create node added events for each subtask
        for subtask in plan.get("subtasks", []):
            node_added_event = Event(
                id=f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
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
                    user_id=request.user_id or "user_current",
                    session_id=task_created_event.metadata.session_id,
                    git_commit=None,
                    confidence=plan.get("confidence", 0.9)
                )
            )
            events.append(node_added_event)
        
        return events
    
    def _create_file_operations_from_plan(self, plan: Dict[str, Any]) -> List[FileOperation]:
        """Create file operations from the plan.
        
        Args:
            plan: Plan data
            
        Returns:
            List of file operations
        """
        file_operations = []
        
        # Create task directory
        task_dir_op = FileOperation(
            type=FileOperationType.CREATE_DIRECTORY,
            path=f"/tasks/{plan['task_id']}"
        )
        file_operations.append(task_dir_op)
        
        # Create task metadata file
        task_meta_op = FileOperation(
            type=FileOperationType.CREATE_FILE,
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
            type=FileOperationType.CREATE_FILE,
            path=f"/tasks/{plan['task_id']}/README.md",
            content=f"# {plan['name']}\n\n{plan['description']}"
        )
        file_operations.append(readme_op)
        
        # Create file operations for each subtask
        for subtask in plan.get("subtasks", []):
            # Create subtask directory
            subtask_dir_op = FileOperation(
                type=FileOperationType.CREATE_DIRECTORY,
                path=f"/tasks/{plan['task_id']}/{subtask['node_id']}"
            )
            file_operations.append(subtask_dir_op)
            
            # Create subtask metadata file
            subtask_meta_op = FileOperation(
                type=FileOperationType.CREATE_FILE,
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
                    type=FileOperationType(file_op["type"]),
                    path=file_op["path"],
                    content=file_op.get("content", "")
                )
                file_operations.append(operation)
        
        return file_operations


class SimulatedPlannerService(PlannerService):
    """Simulated planner service for testing."""
    
    async def create_plan(self, request: PlanRequest) -> TransactionProposal:
        """Create a simulated plan.
        
        Args:
            request: Plan request
            
        Returns:
            Transaction proposal
        """
        logger.info(f"Creating simulated plan for intent: {request.intent}")
        
        # Create a simple task ID
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create a simple plan based on the intent
        plan = {
            "task_id": task_id,
            "name": f"Blog Post: {request.intent[:30]}...",
            "description": f"Create a blog post about {request.intent}",
            "subtasks": [
                {
                    "node_id": "research",
                    "name": "Research",
                    "description": "Research the topic",
                    "assigned_agent": "gpt-4o",
                    "estimated_duration_seconds": 300,
                    "dependencies": [],
                    "file_operations": [
                        {
                            "type": "create_file",
                            "path": f"/tasks/{task_id}/research/notes.md",
                            "content": f"# Research Notes\n\nTopic: {request.intent}"
                        }
                    ]
                },
                {
                    "node_id": "writing",
                    "name": "Writing",
                    "description": "Write the blog post",
                    "assigned_agent": "gpt-4o",
                    "estimated_duration_seconds": 600,
                    "dependencies": ["research"],
                    "file_operations": [
                        {
                            "type": "create_file",
                            "path": f"/tasks/{task_id}/writing/blog_post.md",
                            "content": f"# Blog Post\n\nTopic: {request.intent}\n\n## Introduction\n\n## Main Content\n\n## Conclusion"
                        }
                    ]
                },
                {
                    "node_id": "editing",
                    "name": "Editing",
                    "description": "Edit and finalize the blog post",
                    "assigned_agent": "gpt-4o",
                    "estimated_duration_seconds": 300,
                    "dependencies": ["writing"],
                    "file_operations": []
                }
            ],
            "estimated_duration_seconds": 1200,
            "required_agents": ["gpt-4o"],
            "confidence": 0.8
        }
        
        # Create events
        events = []
        
        # Create correlation ID for all events
        correlation_id = f"req_{uuid.uuid4().hex[:8]}"
        
        # Create task creation event
        task_created_event = Event(
            id=f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            category="directive",
            type=DirectiveEventType.TASK_CREATED.value,
            source=EventSource(
                type="SimulatedPlannerService",
                id="planner_sim",
                version="1.0.0"
            ),
            payload={
                "task_id": plan["task_id"],
                "name": plan["name"],
                "description": plan["description"],
                "intent": request.intent
            },
            metadata=EventMetadata(
                correlation_id=correlation_id,
                user_id=request.user_id or "user_current",
                session_id=request.session_id or f"sess_{uuid.uuid4().hex[:8]}",
                git_commit=None,
                confidence=plan.get("confidence", 0.9)
            )
        )
        events.append(task_created_event)
        
        # Create node added events for each subtask
        for subtask in plan.get("subtasks", []):
            node_added_event = Event(
                id=f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
                timestamp=datetime.now(),
                category="directive",
                type=DirectiveEventType.NODE_ADDED.value,
                source=EventSource(
                    type="SimulatedPlannerService",
                    id="planner_sim",
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
                    user_id=request.user_id or "user_current",
                    session_id=task_created_event.metadata.session_id,
                    git_commit=None,
                    confidence=plan.get("confidence", 0.9)
                )
            )
            events.append(node_added_event)
        
        # Create file operations
        file_operations = []
        
        # Create task directory
        task_dir_op = FileOperation(
            type=FileOperationType.CREATE_DIRECTORY,
            path=f"/tasks/{plan['task_id']}"
        )
        file_operations.append(task_dir_op)
        
        # Create task metadata file
        task_meta_op = FileOperation(
            type=FileOperationType.CREATE_FILE,
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
            type=FileOperationType.CREATE_FILE,
            path=f"/tasks/{plan['task_id']}/README.md",
            content=f"# {plan['name']}\n\n{plan['description']}"
        )
        file_operations.append(readme_op)
        
        # Create file operations for each subtask
        for subtask in plan.get("subtasks", []):
            # Create subtask directory
            subtask_dir_op = FileOperation(
                type=FileOperationType.CREATE_DIRECTORY,
                path=f"/tasks/{plan['task_id']}/{subtask['node_id']}"
            )
            file_operations.append(subtask_dir_op)
            
            # Create subtask metadata file
            subtask_meta_op = FileOperation(
                type=FileOperationType.CREATE_FILE,
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
                    type=FileOperationType(file_op["type"]),
                    path=file_op["path"],
                    content=file_op.get("content", "")
                )
                file_operations.append(operation)
        
        # Create transaction proposal
        proposal = TransactionProposal(
            events=events,
            file_operations=file_operations,
            estimated_duration=plan.get("estimated_duration_seconds", 3600),
            required_agents=plan.get("required_agents", ["gpt-4o"]),
            confidence=plan.get("confidence", 0.9)
        )
        
        return proposal


# Factory function to get the appropriate planner service
def get_planner_service(use_ai: bool = True) -> PlannerService:
    """Get a planner service.
    
    Args:
        use_ai: Whether to use the AI planner service
        
    Returns:
        Planner service
    """
    if use_ai and settings.openai.api_key:
        return AIPlannerService()
    else:
        return SimulatedPlannerService()