# Project AURA: An AI-Native Meta-OS

<div align="center">
  <img src="https://img.shields.io/badge/version-4.0.0-blue.svg" alt="Version 4.0.0">
  <img src="https://img.shields.io/badge/python-3.10+-green.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="License: MIT">
</div>

AURA is an AI-native meta-operating system that fundamentally reimagines how humans interact with computational systems. Rather than forcing users to decompose their intentions into primitive operations, AURA accepts high-level, ambiguous goals and autonomously orchestrates complex workflows to achieve them.

## Core Philosophy

AURA is built on three pillars:

1. **Event Log as Immutable Truth**: All system state is a deterministic projection of the event log
2. **Workspace as Tangible State**: File structure mirrors logical organization for transparency
3. **Decoupled Cognition**: Task execution is cleanly separated from learning

## Features

- **Intent-Based Execution**: Translate natural language requests into complex workflows
- **Multimodal Memory**: Store and retrieve text, images, code, and structured data
- **Versioned Workspace**: Git-based versioning for all system state
- **Cognitive Services**: Background learning from system events
- **Agent Orchestration**: Coordinate specialized AI agents for different tasks

## Getting Started

### Installation

```bash
# Install from PyPI
pip install aura-os

# Or install from source
git clone https://github.com/boriscopperfield/Aura.git
cd Aura
pip install -e .
```

### Configuration

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_key
JINA_API_KEY=your_jina_key
AURA_WORKSPACE=~/aura_workspace
```

### Basic Usage

```bash
# Show help
aura --help

# Create a task
aura run "Create a blog post about AI assistants"

# Check system status
aura status

# Query memory
aura memory "What marketing strategies worked best for our previous launches?"

# View event log
aura log --graph

# Revert to previous state
aura revert v2.0-campaign-init
```

## System Architecture

AURA consists of several key components:

- **Core System**:
  - **Command Handler**: Validates and routes user input
  - **Planner AI**: Translates intent into executable plans
  - **Transaction Manager**: Ensures ACID properties for state changes
  - **Event Bus**: Distributes events to subsystems

- **Storage Layer**:
  - **Event Log**: Immutable record of all system events
  - **Task Workspace**: File structure mirroring logical organization
  - **Git Repository**: Version control for workspace

- **Memory System**:
  - **Memory Graph**: Knowledge base with multimodal nodes
  - **Vector Indexes**: Efficient similarity search
  - **Cognitive Services**: Background learning and pattern recognition

- **Execution Layer**:
  - **Task Executor**: Orchestrates workflow execution
  - **Agent Manager**: Coordinates specialized AI agents
  - **Agent Pool**: Collection of AI models and tools

## Advanced Memory System

AURA features a sophisticated memory system with multimodal embedding capabilities:

### Retrieval Models (Embedders)
- **Multimodal Chunk Embedding**: Encodes content blocks into vectors for coarse-grained retrieval
- **Multimodal Token Embedding**: Encodes elements into sequence vectors for fine-grained search
- **Context-Aware Embeddings**: Understands intra-document context for more relevant results

### Refinement Models (Rerankers)
- **Multimodal Reranker**: Takes a query and candidate chunks to precisely score relevance
- **Graph-Based Expansion**: Follows relationships to discover related content

### Memory Learning
- **Preference Learning**: Automatically detects user preferences from interactions
- **Pattern Recognition**: Identifies workflow patterns and best practices
- **Continuous Improvement**: Memory evolves over time with usage

## AI Integration

AURA integrates with modern AI models:

- **OpenAI GPT Models**: For planning and reasoning
- **Jina AI Embeddings**: For efficient memory retrieval
- **Custom Agent Pool**: Extensible framework for specialized AI agents

## Development

### Prerequisites

- Python 3.10+
- Poetry for dependency management
- OpenAI API key (for AI planning)
- Jina AI API key (for memory embeddings)

### Setup Development Environment

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Format code
poetry run black .
poetry run isort .

# Lint code
poetry run ruff .
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.