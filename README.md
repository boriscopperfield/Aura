# Project AURA: An AI-Native Meta-OS

AURA is an AI-native meta-operating system that fundamentally reimagines how humans interact with computational systems. Rather than forcing users to decompose their intentions into primitive operations, AURA accepts high-level, ambiguous goals and autonomously orchestrates complex workflows to achieve them.

## Core Philosophy

AURA is built on three pillars:
1. **Event Log as Immutable Truth**: All system state is a deterministic projection of the event log
2. **Workspace as Tangible State**: File structure mirrors logical organization for transparency
3. **Decoupled Cognition**: Task execution is cleanly separated from learning

## Getting Started

```bash
# Install the package
pip install -e .

# Run AURA
aura --help

# Create a task
aura run "Create a blog post about AI assistants"
```

## System Architecture

AURA consists of several key components:
- **AURA Kernel**: Command Handler, Planner AI, Transaction Manager
- **Storage Layer**: Event Log, Task Workspace, Git Repository
- **Memory System**: Memory Graph, Vector Indexes, Cognitive Services
- **Execution Layer**: Task Executor, Agent Manager, Agent Pool
- **Event Distribution**: Event Bus

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

## Development Status

This project is currently in active development based on the v4.0 specification.

## Requirements

- Python 3.10+
- OpenAI API key (for AI planning)
- Jina AI API key (for memory embeddings)

## License

MIT