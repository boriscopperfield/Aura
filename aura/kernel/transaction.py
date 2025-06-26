"""
Transaction manager for AURA system.
"""
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import git
from rich.console import Console

from aura.kernel.events import Event


@dataclass
class FileOperation:
    """File operation in a transaction."""
    
    type: str  # create_file, modify_file, delete_file, create_dir, delete_dir
    path: str
    content: Optional[str] = None


@dataclass
class TransactionProposal:
    """Proposal for a transaction."""
    
    events: List[Event]
    file_operations: List[FileOperation]
    estimated_duration: float
    required_agents: List[str]
    confidence: float


@dataclass
class TransactionResult:
    """Result of a transaction."""
    
    success: bool
    transaction_id: str
    event_ids: List[str] = None
    commit_hash: Optional[str] = None
    error: Optional[str] = None


class TransactionError(Exception):
    """Error during transaction execution."""
    pass


class TransactionManager:
    """Ensures ACID properties for all state changes."""
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.event_log_path = workspace_path / "events.jsonl"
        self.console = Console()
        
        # Initialize Git repository if it doesn't exist
        if not (workspace_path / ".git").exists():
            self.repo = git.Repo.init(workspace_path)
            # Create initial commit to establish HEAD
            if not os.path.exists(workspace_path / "README.md"):
                with open(workspace_path / "README.md", "w") as f:
                    f.write("# AURA Workspace\n\nThis is an AURA workspace.")
            self.repo.git.add("README.md")
            self.repo.git.commit("-m", "Initial commit")
        else:
            self.repo = git.Repo(workspace_path)
    
    async def execute(self, proposal: TransactionProposal) -> TransactionResult:
        """Execute a transaction with ACID guarantees."""
        tx_id = f"tx_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        try:
            self.console.log(f"[bold green]Beginning transaction {tx_id}[/bold green]")
            
            # Phase 1: Append to event log
            event_ids = await self.append_events(proposal.events)
            
            # Phase 2: Apply filesystem changes
            await self.apply_file_operations(proposal.file_operations)
            
            # Phase 3: Git commit
            commit_hash = await self.git_commit(tx_id, proposal)
            
            self.console.log(f"[bold green]Transaction {tx_id} committed successfully[/bold green]")
            
            return TransactionResult(
                success=True,
                transaction_id=tx_id,
                event_ids=event_ids,
                commit_hash=commit_hash
            )
            
        except Exception as e:
            self.console.log(f"[bold red]Transaction {tx_id} failed: {e}[/bold red]")
            
            # Rollback all changes
            await self.rollback_transaction(tx_id)
            
            return TransactionResult(
                success=False,
                transaction_id=tx_id,
                error=str(e)
            )
    
    async def append_events(self, events: List[Event]) -> List[str]:
        """Append events to the event log."""
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
        """Apply file operations to the workspace."""
        for op in operations:
            path = self.workspace_path / op.path.lstrip("/")
            
            if op.type == "create_file":
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write(op.content or "")
            
            elif op.type == "modify_file":
                if not path.exists():
                    raise TransactionError(f"Cannot modify non-existent file: {path}")
                with open(path, "w") as f:
                    f.write(op.content or "")
            
            elif op.type == "delete_file":
                if not path.exists():
                    raise TransactionError(f"Cannot delete non-existent file: {path}")
                os.remove(path)
            
            elif op.type == "create_dir":
                os.makedirs(path, exist_ok=True)
            
            elif op.type == "delete_dir":
                if not path.exists():
                    raise TransactionError(f"Cannot delete non-existent directory: {path}")
                os.rmdir(path)
    
    async def git_commit(self, tx_id: str, proposal: TransactionProposal) -> str:
        """Commit changes to Git."""
        # Stage all changes
        self.repo.git.add("-A")
        
        # Create commit message
        event_summaries = "\n".join([
            f"- {event.type.value}: {event.payload.get('summary', 'No summary')}"
            for event in proposal.events[:5]
        ])
        
        if len(proposal.events) > 5:
            event_summaries += f"\n- ... and {len(proposal.events) - 5} more events"
        
        file_changes = []
        for op in proposal.file_operations:
            if op.type == "create_file":
                file_changes.append(f"- Added: {op.path}")
            elif op.type == "modify_file":
                file_changes.append(f"- Modified: {op.path}")
            elif op.type == "delete_file":
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
        """Rollback a failed transaction."""
        self.console.log(f"[bold yellow]Rolling back transaction {tx_id}[/bold yellow]")
        
        # Reset Git repository to last commit
        self.repo.git.reset("--hard", "HEAD")
        
        # Clean untracked files
        self.repo.git.clean("-fd")
        
        self.console.log(f"[bold yellow]Transaction {tx_id} rolled back successfully[/bold yellow]")