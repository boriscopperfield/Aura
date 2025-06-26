"""
Settings management for AURA system.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseSettings, Field, validator
import dotenv


# Load environment variables from .env file
dotenv.load_dotenv()


class OpenAISettings(BaseSettings):
    """OpenAI API settings."""
    
    api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    base_url: str = Field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    model: str = "gpt-4o"
    timeout: int = 60
    
    class Config:
        env_prefix = "OPENAI_"


class JinaSettings(BaseSettings):
    """Jina AI API settings."""
    
    api_key: str = Field(default_factory=lambda: os.environ.get("JINA_API_KEY", ""))
    base_url: str = Field(default_factory=lambda: os.environ.get("JINA_BASE_URL", "https://api.jina.ai/v1"))
    embedding_model: str = "jina-embeddings-v2"
    reranker_model: str = "jina-reranker-v1"
    timeout: int = 30
    
    class Config:
        env_prefix = "JINA_"


class MemorySettings(BaseSettings):
    """Memory system settings."""
    
    vector_dimensions: int = 768
    cache_size: int = 10000
    similarity_threshold: float = 0.7
    max_results: int = 50
    rerank_ratio: float = 2.0  # Retrieve 2x more results than needed for reranking
    default_importance: float = 0.5
    default_decay_rate: float = 0.01
    
    class Config:
        env_prefix = "MEMORY_"


class WorkspaceSettings(BaseSettings):
    """Workspace settings."""
    
    path: Path = Field(default_factory=lambda: Path(os.environ.get("AURA_WORKSPACE", str(Path.home() / "aura_workspace"))))
    event_log_path: str = "events.jsonl"
    memory_dir: str = "memory"
    tasks_dir: str = "tasks"
    
    @validator("path")
    def validate_path(cls, v: Path) -> Path:
        """Validate and create workspace path if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_prefix = "WORKSPACE_"


class LoggingSettings(BaseSettings):
    """Logging settings."""
    
    level: str = Field(default_factory=lambda: os.environ.get("AURA_LOG_LEVEL", "INFO"))
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = Field(default_factory=lambda: os.environ.get("AURA_LOG_FILE", None))
    
    class Config:
        env_prefix = "LOGGING_"


class AgentSettings(BaseSettings):
    """Agent settings."""
    
    max_concurrent: int = 10
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: int = 5
    
    class Config:
        env_prefix = "AGENT_"


class Settings(BaseSettings):
    """Global settings for AURA system."""
    
    # Application settings
    app_name: str = "AURA"
    app_version: str = "4.0.0"
    debug: bool = Field(default_factory=lambda: os.environ.get("AURA_DEBUG", "False").lower() == "true")
    
    # API settings
    openai: OpenAISettings = OpenAISettings()
    jina: JinaSettings = JinaSettings()
    
    # System settings
    memory: MemorySettings = MemorySettings()
    workspace: WorkspaceSettings = WorkspaceSettings()
    logging: LoggingSettings = LoggingSettings()
    agents: AgentSettings = AgentSettings()
    
    # Performance settings
    max_workers: int = 10
    
    class Config:
        env_prefix = "AURA_"


# Create global settings instance
settings = Settings()