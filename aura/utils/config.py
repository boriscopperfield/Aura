"""
Configuration management for AURA.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class SystemConfig(BaseModel):
    """System configuration."""
    version: str = "4.0.0"
    workspace: str = "./aura_workspace"
    environment: str = "development"
    log_level: str = "INFO"


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


class MemoryConfig(BaseModel):
    """Memory configuration."""
    l1_capacity: int = 1000
    l2_capacity: int = 10000
    cache_size: str = "2GB"
    retention_days: int = 90


class AgentsConfig(BaseModel):
    """Agents configuration."""
    default_timeout: int = 60
    max_retries: int = 3
    cost_limit_per_task: float = 10.00


class SecurityConfig(BaseModel):
    """Security configuration."""
    secret_key: str = "dev-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    prometheus_enabled: bool = False
    metrics_port: int = 9090
    health_check_interval: int = 30


class StorageConfig(BaseModel):
    """Storage configuration."""
    redis_url: Optional[str] = None
    postgres_url: Optional[str] = None


class Config(BaseSettings):
    """Main configuration class."""
    
    system: SystemConfig = Field(default_factory=SystemConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    class Config:
        env_prefix = "AURA_"
        env_nested_delimiter = "__"


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file and environment variables."""
    
    # Default config
    config_data = {}
    
    # Load from file if provided
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
    else:
        # Try to find config file in common locations
        possible_paths = [
            Path.cwd() / "config.yaml",
            Path.cwd() / ".aura" / "config.yaml",
            Path.cwd() / "aura_workspace" / ".aura" / "config.yaml",
            Path.home() / ".aura" / "config.yaml"
        ]
        
        for path in possible_paths:
            if path.exists():
                with open(path, 'r') as f:
                    config_data = yaml.safe_load(f)
                break
    
    # Create config object
    config = Config(**config_data)
    
    # Override with environment variables
    config = Config(
        system=SystemConfig(
            version=config.system.version,
            workspace=os.getenv("AURA_WORKSPACE", config.system.workspace),
            environment=os.getenv("AURA_ENV", config.system.environment),
            log_level=os.getenv("AURA_LOG_LEVEL", config.system.log_level)
        ),
        server=ServerConfig(
            host=os.getenv("AURA_HOST", config.server.host),
            port=int(os.getenv("AURA_PORT", config.server.port)),
            workers=int(os.getenv("AURA_WORKERS", config.server.workers))
        ),
        memory=config.memory,
        agents=config.agents,
        security=SecurityConfig(
            secret_key=os.getenv("SECRET_KEY", config.security.secret_key),
            algorithm=config.security.algorithm,
            access_token_expire_minutes=config.security.access_token_expire_minutes
        ),
        monitoring=config.monitoring,
        storage=StorageConfig(
            redis_url=os.getenv("REDIS_URL", config.storage.redis_url),
            postgres_url=os.getenv("POSTGRES_URL", config.storage.postgres_url)
        )
    )
    
    return config


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to file."""
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict
    config_dict = {
        "system": config.system.model_dump(),
        "server": config.server.model_dump(),
        "memory": config.memory.model_dump(),
        "agents": config.agents.model_dump(),
        "security": config.security.model_dump(),
        "monitoring": config.monitoring.model_dump(),
        "storage": config.storage.model_dump()
    }
    
    # Remove sensitive data
    if "secret_key" in config_dict["security"]:
        config_dict["security"]["secret_key"] = "***REDACTED***"
    
    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def validate_config(config: Config) -> Dict[str, Any]:
    """Validate configuration and return validation results."""
    issues = []
    warnings = []
    
    # Check workspace path
    workspace_path = Path(config.system.workspace)
    if not workspace_path.exists():
        warnings.append(f"Workspace directory does not exist: {workspace_path}")
    
    # Check security settings
    if config.system.environment == "production":
        if config.security.secret_key == "dev-secret-key-change-in-production":
            issues.append("Secret key must be changed in production")
        
        if config.system.log_level == "DEBUG":
            warnings.append("Debug logging enabled in production")
    
    # Check memory settings
    if config.memory.l1_capacity > config.memory.l2_capacity:
        issues.append("L1 capacity cannot be larger than L2 capacity")
    
    # Check agent settings
    if config.agents.cost_limit_per_task <= 0:
        issues.append("Cost limit per task must be positive")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings
    }