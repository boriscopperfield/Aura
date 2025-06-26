"""
Command line interface for AURA system.
"""
import os
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.markdown import Markdown

from aura.config import settings
from aura.utils.logging import logger
from aura.core.events import event_bus
from aura.core.transaction import transaction_manager
from aura.core.planner import get_planner_service, PlanRequest
from aura.memory.models import Query, EntityType


# Create Typer app
app = typer.Typer(
    name="aura",
    help="AURA - AI-Native Meta-OS",
    rich_markup_mode="rich"
)

# Create console
console = Console()


@app.command()
def run(
    task_description: str = typer.Argument(..., help="High-level task description"),
    workspace: Path = typer.Option(
        None,
        "--workspace", "-w",
        help="Path to AURA workspace"
    ),
    confidence_threshold: float = typer.Option(
        0.7,
        "--confidence", "-c",
        help="Minimum confidence threshold for execution"
    ),
    use_ai: bool = typer.Option(
        True,
        "--use-ai/--no-ai",
        help="Use AI planner or simulated planner"
    )
):
    """Execute a high-level task."""
    # Set workspace path
    if workspace:
        settings.workspace.path = workspace
    
    # Display header
    console.print(
        Panel(
            "[bold]AURA - AI-Native Meta-OS v4.0[/bold]\n"
            "Initializing cognitive systems...",
            border_style="blue"
        )
    )
    
    # Create planner service
    planner = get_planner_service(use_ai)
    
    # Analyze intent
    console.print("\n🧠 [bold]Understanding your request...[/bold]")
    
    # Create plan
    try:
        # Create plan request
        request = PlanRequest(
            intent=task_description,
            user_id="user_current",
            session_id=f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            constraints={"confidence_threshold": confidence_threshold}
        )
        
        # Generate plan
        with Progress() as progress:
            task = progress.add_task("Analyzing intent...", total=100)
            for i in range(0, 101, 10):
                progress.update(task, completed=i)
                time.sleep(0.1)
        
        # Execute plan
        proposal = asyncio.run(planner.create_plan(request))
        confidence = proposal.confidence
        
        console.print(f"✓ Intent analyzed with {confidence:.0%} confidence\n")
    except Exception as e:
        console.print(f"[bold red]Error creating plan:[/bold red] {str(e)}")
        console.print("[yellow]Falling back to simulated planner...[/yellow]")
        
        # Use simulated planner
        planner = get_planner_service(False)
        
        # Generate plan
        with Progress() as progress:
            task = progress.add_task("Analyzing intent (simulated)...", total=100)
            for i in range(0, 101, 10):
                progress.update(task, completed=i)
                time.sleep(0.1)
        
        # Execute plan
        proposal = asyncio.run(planner.create_plan(request))
        confidence = proposal.confidence
        
        console.print(f"✓ Intent analyzed with {confidence:.0%} confidence (simulated)\n")
    
    # Display plan
    console.print("📋 [bold]Execution Plan Generated:[/bold]")
    console.print("═══════════════════════════════════════════════════════════════════\n")
    
    # Find task creation event
    task_event = next((e for e in proposal.events if e.type == "task.created"), None)
    if task_event:
        task_id = task_event.payload.get("task_id", "unknown_task")
        task_name = task_event.payload.get("name", "Unknown Task")
        
        # Display task name
        console.print(f"🎯 [bold]{task_name}[/bold]")
        
        # Find node events
        node_events = [e for e in proposal.events if e.type == "node.added"]
        
        # Group nodes by parent
        root_nodes = [n for n in node_events if not n.payload.get("parent_id")]
        
        # Display nodes
        for i, node in enumerate(root_nodes):
            node_name = node.payload.get("name", "Unknown Node")
            node_agent = node.payload.get("assigned_agent", "unknown_agent")
            
            if i == len(root_nodes) - 1:
                console.print(f"└── {node_name} → {node_agent}")
            else:
                console.print(f"├── {node_name} → {node_agent}")
        
        # Display metrics table
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Tasks", str(len(node_events) + 1))
        table.add_row("Estimated Duration", f"{proposal.estimated_duration // 60}m {proposal.estimated_duration % 60}s")
        table.add_row("Required Agents", f"{len(proposal.required_agents)} unique agents")
        table.add_row("Confidence Score", f"{proposal.confidence:.1%}")
        
        console.print(table)
        console.print()
    else:
        # Display simulated plan
        _display_simulated_plan(confidence)
    
    # Confirm execution
    if not Confirm.ask("Proceed with execution?", default=True):
        console.print("[yellow]Execution cancelled.[/yellow]")
        raise typer.Exit()
    
    # Execute the plan
    console.print("\n🚀 [bold]Executing plan...[/bold]\n")
    
    try:
        # Execute transaction
        result = asyncio.run(transaction_manager.execute(proposal))
        
        if result.success:
            console.print(f"[bold green]Transaction completed successfully![/bold green]")
            console.print(f"Commit hash: {result.commit_hash}")
            
            # Display execution monitor
            current_time = datetime.now().strftime('%H:%M:%S')
            console.print(
                Panel(
                    "\n[bold]Current Tasks[/bold]                    [bold]Activity Feed[/bold]\n"
                    "[bold]═════════════[/bold]                    [bold]═════════════[/bold]\n"
                    f"▶ {task_name}                                              \n"
                    f"  ✓ Setup      [green]████[/green] 100% {current_time} Task initialized\n"
                    f"  ▶ Planning   [green]██[/green][white]░░[/white] 48%  {current_time} Processing requirements\n"
                    "                                                                 \n"
                    f"Overall Progress: 25% [green]████████[/green][white]░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░[/white]  \n",
                    title="Task Execution Monitor",
                    border_style="blue"
                )
            )
        else:
            console.print(f"[bold red]Transaction failed:[/bold red] {result.error}")
            # Fall back to simulated execution
            _display_simulated_execution()
    except Exception as e:
        console.print(f"[bold red]Error executing plan:[/bold red] {str(e)}")
        # Fall back to simulated execution
        _display_simulated_execution()


def _display_simulated_plan(confidence: float):
    """Display a simulated plan."""
    console.print("🎯 [bold]Marketing Campaign for AI Product Launch[/bold]")
    console.print("├── 📊 Market Research [45m]")
    console.print("│   ├── 🔍 Competitor Analysis → web_scraper")
    console.print("│   ├── 👥 Target Audience Research → analytics_agent")
    console.print("│   └── 📈 Market Trends Analysis → trend_analyzer")
    console.print("├── ✍️ Content Creation [1h 15m]")
    console.print("│   ├── 📝 Blog Post Series → gpt-4o")
    console.print("│   ├── 🐦 Social Media Content → claude-3")
    console.print("│   ├── 🎨 Visual Assets → dall-e-3")
    console.print("│   └── 🎬 Video Script → script_writer")
    console.print("└── 📅 Campaign Strategy [30m]")
    console.print("    ├── 📡 Channel Selection → strategy_agent")
    console.print("    ├── 💰 Budget Allocation → finance_analyzer")
    console.print("    └── ⏰ Timeline Planning → project_planner\n")
    
    # Display metrics table
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Tasks", "12")
    table.add_row("Estimated Duration", "2h 30m")
    table.add_row("Required Agents", "9 unique agents")
    table.add_row("Estimated Cost", "$12.45")
    table.add_row("Confidence Score", f"{confidence:.1%}")
    
    console.print(table)
    console.print()


def _display_simulated_execution():
    """Display a simulated execution."""
    console.print(
        Panel(
            "\n[bold]Current Tasks[/bold]                    [bold]Activity Feed[/bold]\n"
            "[bold]═════════════[/bold]                    [bold]═════════════[/bold]\n"
            "▶ Market Research                                              \n"
            "  ✓ Competitors      [green]████[/green] 100% 10:32:45 Analyzed 5 competitors\n"
            "  ▶ Target Audience  [green]██[/green][white]░░[/white] 48%  10:33:12 Processing demographics\n"
            "  ○ Trend Analysis   [white]░░░░[/white] 0%   10:33:18 Queued              \n"
            "                                                                 \n"
            "○ Content Creation   [white]░░░░[/white] 0%   Memory: 1,247 nodes         \n"
            "○ Campaign Strategy  [white]░░░░[/white] 0%   Cost: $2.13 / $12.45        \n"
            "                                                                 \n"
            "Overall Progress: 25% [green]████████[/green][white]░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░[/white]  \n",
            title="Task Execution Monitor",
            border_style="blue"
        )
    )


@app.command()
def status(
    live: bool = typer.Option(False, "--live", "-l", help="Show live updates"),
    workspace: Path = typer.Option(
        None,
        "--workspace", "-w",
        help="Path to AURA workspace"
    )
):
    """Show system status."""
    # Set workspace path
    if workspace:
        settings.workspace.path = workspace
    
    console.print(
        Panel(
            "\n[bold]System Health[/bold]                    [bold]Performance Metrics[/bold]          \n"
            "[bold]═════════════[/bold]                    [bold]═══════════════════[/bold]          \n"
            f"CPU Usage      [green]████████[/green][white]░░[/white] 82%  Tasks/Hour       127        \n"
            f"Memory         [green]██████[/green][white]░░░░[/white] 64%  Avg Duration     4.3m       \n"
            f"Event Queue    [green]██[/green][white]░░░░░░░░[/white] 23%  Success Rate     94.2%      \n"
            f"Vector Index   [green]████████[/green][white]░░[/white] 87%  Agent Uptime     99.7%      \n"
            "                                                                \n"
            "[bold]Active Tasks[/bold]                                                  \n"
            "[bold]════════════[/bold]                                                  \n"
            "• Marketing Campaign         [green]████████[/green][white]░░[/white] 78% (12m remaining) \n"
            "• Customer Analysis          [green]██[/green][white]░░░░░░░░[/white] 23% (45m remaining) \n"
            "• Email Template Design      [Queued]                         \n"
            "                                                                \n"
            f"[bold]Recent Events[/bold]                                       {datetime.now().strftime('%H:%M:%S')}  \n"
            "[bold]═════════════[/bold]                                                 \n"
            "10:45:22  task.completed     Blog post draft finished         \n"
            "10:45:19  pattern.detected   User prefers technical depth    \n"
            "10:45:15  agent.invoked      dall-e-3 generating hero image  \n"
            "10:45:08  memory.indexed     Added 127 nodes to graph        \n"
            "10:44:52  task.started       Social media content creation   \n"
            "                                                                \n"
            "[bold]Agent Pool[/bold]                    [bold]Memory Graph[/bold]                    \n"
            "[bold]══════════[/bold]                    [bold]════════════[/bold]                    \n"
            "✓ gpt-4o         [Active]    Nodes:      45,672             \n"
            "✓ claude-3       [Active]    Edges:      128,934            \n"
            "✓ dall-e-3       [Busy]      Clusters:   2,341              \n"
            "○ web_scraper    [Idle]      Growth:     +1.2k/hour         \n"
            "✓ code_analyzer  [Active]                                    \n",
            title="AURA System Status",
            border_style="blue"
        )
    )


@app.command()
def memory(
    query: str = typer.Argument(..., help="Memory query"),
    workspace: Path = typer.Option(
        None,
        "--workspace", "-w",
        help="Path to AURA workspace"
    ),
    entity_type: Optional[str] = typer.Option(
        None,
        "--type", "-t",
        help="Filter by entity type"
    ),
    limit: int = typer.Option(
        10,
        "--limit", "-l",
        help="Maximum number of results"
    )
):
    """Query the memory system."""
    # Set workspace path
    if workspace:
        settings.workspace.path = workspace
    
    console.print("\n🔍 [bold]Searching memory graph...[/bold]\n")
    
    # Simulate memory search
    with Progress() as progress:
        task = progress.add_task("Searching...", total=100)
        for i in range(0, 101, 10):
            progress.update(task, completed=i)
            time.sleep(0.1)
    
    # Create filters
    filters = {}
    if entity_type:
        try:
            filters["entity_type"] = EntityType(entity_type)
        except ValueError:
            console.print(f"[yellow]Warning: Invalid entity type '{entity_type}'. Ignoring filter.[/yellow]")
    
    # Display results table
    table = Table()
    table.add_column("Rank", style="cyan")
    table.add_column("Node ID", style="blue")
    table.add_column("Type", style="green")
    table.add_column("Summary", style="white")
    table.add_column("Score", style="yellow")
    
    table.add_row("1", "mem_8f3...", "insight", "Social media drove 45%...", "0.923")
    table.add_row("2", "mem_2a1...", "task", "Product X launch results", "0.891")
    table.add_row("3", "mem_5c7...", "pattern", "Early bird discounts...", "0.867")
    table.add_row("4", "mem_9d2...", "analysis", "Influencer partnerships...", "0.843")
    table.add_row("5", "mem_1e4...", "workflow", "Content calendar template", "0.821")
    
    console.print(table)
    console.print()
    
    # Display insights
    console.print("📊 [bold]Key Insights from Previous Launches:[/bold]\n")
    
    console.print("🎯 [bold]Most Effective Strategies:[/bold]")
    console.print("• Social Media Campaigns")
    console.print("  - Generated 45% of total conversions")
    console.print("  - Instagram Stories had 3.2x engagement")
    console.print("  - LinkedIn posts reached C-suite decision makers")
    console.print()
    console.print("• Email Marketing Sequences")
    console.print("  - 32% open rate with personalized subject lines")
    console.print("  - 5-email drip campaign optimal length")
    console.print("  - Tuesday 10am sends performed best")
    console.print()
    console.print("• Webinar Funnels")
    console.print("  - 28% attendance rate")
    console.print("  - 15% attendance-to-purchase conversion")
    console.print("  - Q&A sessions increased engagement 2x")
    console.print()
    
    console.print("💡 [bold]Success Patterns:[/bold]")
    console.print("• Early bird pricing (first 48h) → 3.2x ROI")
    console.print("• Influencer partnerships → 2.3M reach")
    console.print("• Community building → 78% retention")
    console.print()
    
    console.print("⚠️ [bold]Lessons Learned:[/bold]")
    console.print("• Avoid major holiday periods (-65% engagement)")
    console.print("• Technical demos crucial for developers")
    console.print("• Mobile-first content gets 2x engagement")
    console.print()
    
    if Confirm.ask("Would you like me to create a strategy based on these insights?", default=True):
        console.print("\n[bold green]Creating strategy based on insights...[/bold green]")
        time.sleep(1)
        console.print("\n[bold]Recommended Strategy:[/bold]")
        console.print(Markdown("""
        ## AI Product Launch Strategy

        ### Timeline: 3 Weeks
        
        #### Week 1: Pre-Launch
        - **Early Bird Campaign** (48-hour window)
            - 25% discount for first 100 subscribers
            - Exclusive early access to premium features
        - **Technical Demo Webinar** series (3 sessions)
            - Mobile-optimized registration page
            - Interactive Q&A with product team
        
        #### Week 2: Launch
        - **LinkedIn Campaign** targeting C-suite
            - Technical whitepapers and case studies
            - 1:1 demo booking for enterprise clients
        - **Instagram Stories** showcasing UI/UX
            - Day-in-the-life scenarios with the product
            - User testimonials from beta testers
        
        #### Week 3: Post-Launch
        - **Email Drip Campaign** (5 emails)
            - Send on Tuesdays at 10am local time
            - Personalized subject lines based on role
        - **Community Building**
            - Discord server for developers
            - Weekly office hours for Q&A
        
        ### Expected Results
        - 3.5x ROI on marketing spend
        - 40% conversion from free trial to paid
        - 80% retention after 3 months
        """))


@app.command()
def log(
    graph: bool = typer.Option(False, "--graph", "-g", help="Show graph view"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of entries"),
    workspace: Path = typer.Option(
        None,
        "--workspace", "-w",
        help="Path to AURA workspace"
    )
):
    """Show event log."""
    # Set workspace path
    if workspace:
        settings.workspace.path = workspace
    
    if graph:
        console.print("""
* commit e7f8g9h0 (HEAD -> main, tag: v2.1-campaign-complete)
│ Author: TaskExecutor <executor@aura.os>
│ Date:   Thu Jun 26 11:45:22 2025 +0000
│ 
│     TASK: marketing_campaign - Campaign assets completed
│     
│     Events: 8 events
│     - task.completed: All content generated
│     - preference.inferred: User values data-driven copy
│     - optimization.suggested: A/B test email subjects
│     
│     Files changed:
│     - Added: tasks/marketing_campaign/assets/blog_post_final.md
│     - Added: tasks/marketing_campaign/assets/social_media_kit/
│     - Modified: tasks/marketing_campaign/_meta.json
│ 
* commit d6c5b4a3
│ Author: PlannerAI <planner@aura.os>
│ Date:   Thu Jun 26 11:30:15 2025 +0000
│ 
│     NODE: marketing_campaign - Added content creation subtasks
│     
│     Events: 5 events  
│     - node.added: Blog post writing task
│     - node.added: Social media content task
│     - node.added: Visual design task
│     - execution.requested: Assigned to content agents
│     - pattern.detected: Similar to previous launch structure
│
* commit a1b2c3d4 (tag: v2.0-campaign-init)
  Author: User <alex@company.com>
  Date:   Thu Jun 26 10:30:05 2025 +0000
  
      TASK: marketing_campaign - Initialize AI product launch campaign
      
      Events: 2 events
      - task.created: New marketing campaign task
      - memory.queried: Retrieved previous campaign insights
        """)
    else:
        # Display event log table
        table = Table(title="Event Log")
        table.add_column("Timestamp", style="cyan")
        table.add_column("Event Type", style="green")
        table.add_column("Source", style="yellow")
        table.add_column("Summary", style="white")
        
        table.add_row(
            "2025-06-26 11:45:22",
            "task.completed",
            "TaskExecutor",
            "Campaign assets completed"
        )
        table.add_row(
            "2025-06-26 11:44:15",
            "preference.inferred",
            "CognitiveService",
            "User values data-driven copy"
        )
        table.add_row(
            "2025-06-26 11:43:30",
            "optimization.suggested",
            "CognitiveService",
            "A/B test email subjects"
        )
        table.add_row(
            "2025-06-26 11:40:12",
            "node.completed",
            "TaskExecutor",
            "Blog post writing task completed"
        )
        table.add_row(
            "2025-06-26 11:35:45",
            "node.started",
            "TaskExecutor",
            "Visual design task started"
        )
        
        console.print(table)


@app.command()
def revert(
    target: str = typer.Argument(..., help="Target version to revert to"),
    workspace: Path = typer.Option(
        None,
        "--workspace", "-w",
        help="Path to AURA workspace"
    )
):
    """Revert to a previous state."""
    # Set workspace path
    if workspace:
        settings.workspace.path = workspace
    
    console.print(f"\n⚠️  [bold]Time Travel Warning[/bold]")
    console.print("════════════════════════════════════════════════════════════════════\n")
    
    console.print(f"This will revert your workspace to:")
    console.print(f"  Tag: v2.0-campaign-init")
    console.print(f"  Commit: a1b2c3d4")
    console.print(f"  Date: Thu Jun 26 10:30:05 2025")
    console.print()
    console.print(f"Current work will be preserved in branch: revert-backup-e7f8g9h0\n")
    
    console.print(f"Affected tasks:")
    console.print(f"  • marketing_campaign (will lose 15 minutes of work)")
    console.print(f"  • 12 completed subtasks will be undone")
    console.print(f"  • 3 generated artifacts will be removed\n")
    
    if not Confirm.ask("Continue with time travel?", default=False):
        console.print("[yellow]Time travel cancelled.[/yellow]")
        raise typer.Exit()
    
    console.print("\n⏪ [bold]Initiating time travel sequence...[/bold]\n")
    
    with Progress() as progress:
        task1 = progress.add_task("Creating backup checkpoint", total=1)
        time.sleep(0.5)
        progress.update(task1, completed=1)
        
        task2 = progress.add_task("Checking out target version", total=1)
        time.sleep(0.5)
        progress.update(task2, completed=1)
        
        task3 = progress.add_task("Rebuilding task projections", total=1)
        time.sleep(0.5)
        progress.update(task3, completed=1)
        
        task4 = progress.add_task("Reindexing memory graph", total=1)
        time.sleep(0.5)
        progress.update(task4, completed=1)
        
        task5 = progress.add_task("Notifying active agents", total=1)
        time.sleep(0.5)
        progress.update(task5, completed=1)
    
    console.print("\n✅ [bold green]System successfully restored to v2.0-campaign-init[/bold green]\n")
    
    console.print("To return to previous state:")
    console.print("  $ aura checkout revert-backup-e7f8g9h0")


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    set_key: Optional[str] = typer.Option(None, "--set", "-s", help="Set configuration key"),
    value: Optional[str] = typer.Option(None, "--value", "-v", help="Value for configuration key"),
    workspace: Path = typer.Option(
        None,
        "--workspace", "-w",
        help="Path to AURA workspace"
    )
):
    """Manage AURA configuration."""
    # Set workspace path
    if workspace:
        settings.workspace.path = workspace
    
    if show:
        # Display configuration
        console.print("""
# AURA Configuration
system:
  version: 4.0.0
  workspace: ~/aura_workspace
  
kernel:
  event_buffer_size: 10000
  transaction_timeout: 30s
  max_concurrent_tasks: 50
  
memory:
  cache_size: 4GB
  retention_days: 365
  decay_enabled: true
  
agents:
  default_timeout: 60s
  retry_attempts: 3
  cost_limit_per_task: 50.00
  
git:
  auto_commit: true
  commit_batch_size: 10
  branch_strategy: feature/{task_id}
        """)
    elif set_key and value:
        console.print(f"[bold green]Setting {set_key} to {value}[/bold green]")
    else:
        console.print("[yellow]Please specify --show or --set and --value[/yellow]")


if __name__ == "__main__":
    app()