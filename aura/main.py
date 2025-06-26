"""
AURA Main Application Entry Point
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional
import uvicorn
import typer
from rich.console import Console
from rich.panel import Panel

from .core.system import AuraSystem
from .api.server import create_app
from .utils.logging import setup_logging, get_logger
from .utils.config import load_config

console = Console()
logger = get_logger(__name__)

app = typer.Typer(
    name="aura",
    help="AURA - AI-Native Meta-OS",
    rich_markup_mode="rich"
)


@app.command()
def run(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    workers: int = typer.Option(1, help="Number of worker processes"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
    dev: bool = typer.Option(False, help="Run in development mode"),
    prod: bool = typer.Option(False, help="Run in production mode"),
    config_path: Optional[str] = typer.Option(None, help="Path to configuration file"),
):
    """Run the AURA server."""
    
    # Determine environment
    if dev:
        env = "development"
        reload = True
        workers = 1
    elif prod:
        env = "production"
        reload = False
        workers = workers or 4
    else:
        env = os.getenv("AURA_ENV", "development")
    
    # Setup logging
    setup_logging(env)
    
    # Load configuration
    config = load_config(config_path)
    
    # Display startup banner
    display_banner(env, host, port, workers)
    
    # Create FastAPI app
    fastapi_app = create_app(config)
    
    # Run server
    uvicorn.run(
        fastapi_app,
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_config=None,  # Use our custom logging
        access_log=False  # We handle access logs ourselves
    )


@app.command()
def init(
    workspace: str = typer.Option("./aura_workspace", help="Workspace directory"),
    force: bool = typer.Option(False, help="Force initialization even if workspace exists")
):
    """Initialize a new AURA workspace."""
    
    workspace_path = Path(workspace).resolve()
    
    if workspace_path.exists() and not force:
        console.print(f"[red]Workspace already exists at {workspace_path}[/red]")
        console.print("Use --force to reinitialize")
        raise typer.Exit(1)
    
    console.print(f"[green]Initializing AURA workspace at {workspace_path}[/green]")
    
    # Create workspace structure
    workspace_path.mkdir(parents=True, exist_ok=True)
    (workspace_path / ".aura").mkdir(exist_ok=True)
    (workspace_path / "tasks").mkdir(exist_ok=True)
    (workspace_path / "memory").mkdir(exist_ok=True)
    (workspace_path / "memory" / "graph").mkdir(exist_ok=True)
    (workspace_path / "memory" / "indexes").mkdir(exist_ok=True)
    
    # Create configuration files
    create_default_config(workspace_path)
    create_default_agents_config(workspace_path)
    
    # Initialize git repository
    import subprocess
    try:
        subprocess.run(["git", "init"], cwd=workspace_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "AURA System"], cwd=workspace_path, check=True)
        subprocess.run(["git", "config", "user.email", "aura@system.local"], cwd=workspace_path, check=True)
        
        # Create initial commit
        (workspace_path / "events.jsonl").touch()
        subprocess.run(["git", "add", "."], cwd=workspace_path, check=True)
        subprocess.run(["git", "commit", "-m", "Initial AURA workspace"], cwd=workspace_path, check=True)
        
        console.print("[green]✓[/green] Git repository initialized")
    except subprocess.CalledProcessError:
        console.print("[yellow]⚠[/yellow] Git initialization failed (git not available)")
    
    console.print(f"[green]✓[/green] AURA workspace initialized successfully")
    console.print(f"[dim]Workspace location: {workspace_path}[/dim]")


@app.command()
def status():
    """Show AURA system status."""
    
    console.print("[bold]AURA System Status[/bold]")
    
    # This would connect to a running AURA instance
    # For now, just show a placeholder
    console.print("[yellow]Not implemented yet - requires running AURA instance[/yellow]")


@app.command()
def test(
    pattern: str = typer.Option("", help="Test pattern to match"),
    verbose: bool = typer.Option(False, help="Verbose output"),
    coverage: bool = typer.Option(False, help="Run with coverage"),
):
    """Run AURA tests."""
    
    import subprocess
    
    cmd = ["python", "-m", "pytest"]
    
    if pattern:
        cmd.extend(["-k", pattern])
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=aura", "--cov-report=html", "--cov-report=term"])
    
    cmd.append("tests/")
    
    console.print(f"[green]Running tests: {' '.join(cmd)}[/green]")
    
    try:
        result = subprocess.run(cmd, check=True)
        console.print("[green]✓[/green] All tests passed")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗[/red] Tests failed with exit code {e.returncode}")
        raise typer.Exit(e.returncode)


def display_banner(env: str, host: str, port: int, workers: int):
    """Display startup banner."""
    
    banner_text = f"""
[bold cyan]AURA - AI-Native Meta-OS[/bold cyan]
[dim]Version 4.0.0[/dim]

Environment: [yellow]{env}[/yellow]
Host: [green]{host}[/green]
Port: [green]{port}[/green]
Workers: [green]{workers}[/green]

[dim]Starting cognitive systems...[/dim]
"""
    
    console.print(Panel(banner_text, title="🧠 AURA System", border_style="cyan"))


def create_default_config(workspace_path: Path):
    """Create default configuration file."""
    
    config_content = f"""# AURA Configuration
system:
  version: "4.0.0"
  workspace: "{workspace_path}"
  environment: "development"
  log_level: "INFO"

server:
  host: "0.0.0.0"
  port: 8000
  workers: 1

memory:
  l1_capacity: 1000
  l2_capacity: 10000
  cache_size: "2GB"
  retention_days: 90

agents:
  default_timeout: 60
  max_retries: 3
  cost_limit_per_task: 10.00

security:
  secret_key: "dev-secret-key-change-in-production"
  algorithm: "HS256"
  access_token_expire_minutes: 30

monitoring:
  prometheus_enabled: false
  metrics_port: 9090
  health_check_interval: 30
"""
    
    config_file = workspace_path / ".aura" / "config.yaml"
    with open(config_file, "w") as f:
        f.write(config_content)


def create_default_agents_config(workspace_path: Path):
    """Create default agents configuration file."""
    
    agents_content = """# AURA Agents Configuration
agents:
  # OpenAI GPT models
  openai:
    type: "openai"
    enabled: false  # Set to true and add API key to enable
    priority: 1
    config:
      api_key: ""  # Add your OpenAI API key here
      model: "gpt-4"
      base_url: "https://api.openai.com/v1"
    capabilities:
      - "text_generation"
      - "code_generation"
      - "analysis"
      - "image_generation"

  # Anthropic Claude models
  anthropic:
    type: "anthropic"
    enabled: false  # Set to true and add API key to enable
    priority: 2
    config:
      api_key: ""  # Add your Anthropic API key here
      model: "claude-3-sonnet-20240229"
    capabilities:
      - "text_generation"
      - "analysis"
      - "summarization"

  # Jina AI embeddings
  jina_embedder:
    type: "jina_embedder"
    enabled: false  # Set to true and add API key to enable
    priority: 1
    config:
      api_key: ""  # Add your Jina API key here
      model: "jina-embeddings-v3"
    capabilities:
      - "text_embedding"

  # Jina AI reranking
  jina_reranker:
    type: "jina_reranker"
    enabled: false  # Set to true and add API key to enable
    priority: 1
    config:
      api_key: ""  # Add your Jina API key here
      model: "jina-reranker-v2-base-multilingual"
    capabilities:
      - "reranking"

  # Local tools
  local:
    type: "local"
    enabled: true
    priority: 3
    config: {}
    capabilities:
      - "code_generation"
      - "analysis"
      - "image_analysis"
"""
    
    agents_file = workspace_path / ".aura" / "agents.yaml"
    with open(agents_file, "w") as f:
        f.write(agents_content)


if __name__ == "__main__":
    app()