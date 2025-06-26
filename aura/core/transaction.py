"""
Transaction system for AURA.

This module defines the transaction model and transaction handling for the AURA system.
Transactions ensure ACID properties for all state changes in the system.
"""
import os
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import git
from pydantic import BaseModel, Field, validator

from aura.config import settings
from aura.core.events import Event, event_bus
from aura.utils.logging import logger
from aura.utils.errors import TransactionError


class FileOperationType(str, Enum):
    """Types of file operations."""
    
    CREATE_FILE = "create_file"
    MODIFY_FILE = "modify_file"
    DELETE_FILE = "delete_file"
    CREATE_DIRECTORY = "create_directory"
    DELETE_DIRECTORY = "delete_directory"


class FileOperation(BaseModel):
    """File operation in a transaction."""
    
    type: FileOperationType
    path: str
    content: Optional[str] = None
    
    @validator("content")
    def validate_content(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Validate that content is provided for file creation/modification."""
        if "type" not in values:
            return v
            
        op_type = values["type"]
        
        if op_type in (FileOperationType.CREATE_FILE, FileOperationType.MODIFY_FILE) and v is None:
            raise ValueError(f"Content is required for {op_type}")
            
        return v


class TransactionProposal(BaseModel):
    """Proposal for a transaction."""
    
    events: List[Event]
    file_operations: List[FileOperation]
    estimated_duration: float
    required_agents: List[str]
    confidence: float


class TransactionResult(BaseModel):
    """Result of a transaction."""
    
    success: bool
    transaction_id: str
    event_ids: Optional[List[str]] = None
    commit_hash: Optional[str] = None
    error: Optional[str] = None


class TransactionManager:
    """Ensures ACID properties for all state changes."""
    
    def __init__(self, workspace_path: Optional[Path] = None):
        """Initialize the transaction manager.
        
        Args:
            workspace_path: Path to the workspace directory
        """
        self.workspace_path = workspace_path or settings.workspace.path
        self.event_log_path = self.workspace_path / settings.workspace.event_log_path
        self.logger = logger
        
        # Initialize Git repository if it doesn't exist
        if not (self.workspace_path / ".git").exists():
            self.logger.info(f"Initializing Git repository at {self.workspace_path}")
            self.repo = git.Repo.init(self.workspace_path)
            
            # Create initial commit to establish HEAD
            if not os.path.exists(self.workspace_path / "README.md"):
                with open(self.workspace_path / "README.md", "w") as f:
                    f.write("# AURA Workspace\n\nThis is an AURA workspace.")
                    
            self.repo.git.add("README.md")
            self.repo.git.commit("-m", "Initial commit")
        else:
            self.repo = git.Repo(self.workspace_path)
    
    async def execute(self, proposal: TransactionProposal) -> TransactionResult:
        """Execute a transaction with ACID guarantees.
        
        Args:
            proposal: Transaction proposal
            
        Returns:
            Transaction result
        """
        tx_id = f"tx_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        try:
            self.logger.info(f"Beginning transaction {tx_id}")
            
            # Phase 1: Append to event log
            event_ids = await self.append_events(proposal.events)
            
            # Phase 2: Apply filesystem changes
            await self.apply_file_operations(proposal.file_operations)
            
            # Phase 3: Git commit
            commit_hash = await self.git_commit(tx_id, proposal)
            
            # Phase 4: Publish events
            for event in proposal.events:
                await event_bus.publish(event)
            
            self.logger.info(f"Transaction {tx_id} committed successfully")
            
            return TransactionResult(
                success=True,
                transaction_id=tx_id,
                event_ids=event_ids,
                commit_hash=commit_hash
            )
            
        except Exception as e:
            self.logger.error(f"Transaction {tx_id} failed: {e}")
            
            # Rollback all changes
            await self.rollback_transaction(tx_id)
            
            return TransactionResult(
                success=False,
                transaction_id=tx_id,
                error=str(e)
            )
    
    async def append_events(self, events: List[Event]) -> List[str]:
        """Append events to the event log.
        
        Args:
            events: Events to append
            
        Returns:
            List of event IDs
        """
        event_ids = []
        
        # Create event log directory if it doesn't exist
        os.makedirs(os.path.dirname(self.event_log_path), exist_ok=True)
        
        # Append events to the log
        with open(self.event_log_path, "a") as f:
            for event in events:
                f.write(event.to_json() + "\n")
                event_ids.append(event.id)
                f.flush()  # Ensure data is written to disk
                os.fsync(f.fileno())  # Force OS to write to physical storage
        
        return event_ids
    
    async def apply_file_operations(self, operations: List[FileOperation]) -> None:
        """Apply file operations to the workspace.
        
        Args:
            operations: File operations to apply
            
        Raises:
            TransactionError: If a file operation fails
        """
        for op in operations:
            path = self.workspace_path / op.path.lstrip("/")
            
            if op.type == FileOperationType.CREATE_FILE:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write(op.content or "")
            
            elif op.type == FileOperationType.MODIFY_FILE:
                if not path.exists():
                    raise TransactionError(f"Cannot modify non-existent file: {path}")
                with open(path, "w") as f:
                    f.write(op.content or "")
            
            elif op.type == FileOperationType.DELETE_FILE:
                if not path.exists():
                    raise TransactionError(f"Cannot delete non-existent file: {path}")
                os.remove(path)
            
            elif op.type == FileOperationType.CREATE_DIRECTORY:
                os.makedirs(path, exist_ok=True)
            
            elif op.type == FileOperationType.DELETE_DIRECTORY:
                if not path.exists():
                    raise TransactionError(f"Cannot delete non-existent directory: {path}")
                os.rmdir(path)
    
    async def git_commit(self, tx_id: str, proposal: TransactionProposal) -> str:
        """Commit changes to Git.
        
        Args:
            tx_id: Transaction ID
            proposal: Transaction proposal
            
        Returns:
            Commit hash
        """
        # Stage all changes
        self.repo.git.add("-A")
        
        # Create commit message
        event_summaries = "\n".join([
            f"- {event.type}: {event.payload.get('summary', 'No summary')}"
            for event in proposal.events[:5]
        ])
        
        if len(proposal.events) > 5:
            event_summaries += f"\n- ... and {len(proposal.events) - 5} more events"
        
        file_changes = []
        for op in proposal.file_operations:
            if op.type == FileOperationType.CREATE_FILE:
                file_changes.append(f"- Added: {op.path}")
            elif op.type == FileOperationType.MODIFY_FILE:
                file_changes.append(f"- Modified: {op.path}")
            elif op.type == FileOperationType.DELETE_FILE:
                file_changes.append(f"- Deleted: {op.path}")
        
        file_summary = "\n".join(file_changes[:5])
        if len(file_changes) > 5:
            file_summary += f"\n- ... and {len(file_changes) - 5} more files"
        
        commit_message = f"""TRANSACTION: {tx_id}

Events: {len(proposal.events)} events
{event_summaries}

Files changed:
{file_summary}

Confidence: {proposal.confidence:.2f}
Duration: {proposal.estimated_duration:.0f}ms
"""
        
        # Commit changes
        commit = self.repo.index.commit(commit_message)
        
        return commit.hexsha
    
    async def rollback_transaction(self, tx_id: str) -> None:
        """Rollback a failed transaction.
        
        Args:
            tx_id: Transaction ID
        """
        self.logger.warning(f"Rolling back transaction {tx_id}")
        
        # Reset Git repository to last commit
        self.repo.git.reset("--hard", "HEAD")
        
        # Clean untracked files
        self.repo.git.clean("-fd")
        
        self.logger.warning(f"Transaction {tx_id} rolled back successfully")


# Global transaction manager instance
transaction_manager = TransactionManager()