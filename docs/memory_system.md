# AURA Enhanced Memory System

## Overview

AURA's enhanced memory system is designed to provide efficient, context-aware storage and retrieval of information. It implements a hierarchical storage architecture with relationship modeling and graph-based retrieval.

## Key Features

### 1. Three-Layer Storage Architecture

The memory system is organized into three layers, each with different characteristics:

- **L1: Hot Cache (Active Working Set)**
  - Contains currently active nodes and their immediate context
  - Optimized for fast access and frequent updates
  - Limited capacity with automatic demotion of idle nodes

- **L2: Session Memory (Session-level Index)**
  - Contains all nodes relevant to the current session
  - Balances access speed and storage capacity
  - Persists throughout the session

- **L3: Persistent Graph (Persistent Knowledge Graph)**
  - Contains all historical knowledge
  - Optimized for storage efficiency and pattern extraction
  - Persists across sessions

### 2. Entity-Relationship Modeling

The memory system models both entities (nodes) and their relationships:

- **Entities**: Represent discrete pieces of information (tasks, artifacts, patterns, etc.)
- **Relationships**: Connect entities with typed, directional links
- **Embeddings**: Both entities and relationships have vector embeddings

Key relationship types include:
- `PRODUCES`: Node produces an artifact
- `DEPENDS_ON`: Node depends on another node
- `DERIVES_FROM`: Artifact derives from another artifact
- `SIMILAR_TO`: Entity is similar to another entity
- `REFERENCES`: Entity references another entity
- `PART_OF`: Entity is part of another entity

### 3. Graph-Based Retrieval

The memory system uses graph traversal for comprehensive retrieval:

- **Seed Entity Search**: Find initial entities matching the query
- **Graph Expansion**: Follow relationships to discover related entities
- **Relevance Scoring**: Score entities based on direct relevance and relationship context
- **Minimal Closure Building**: Construct a minimal set of entities that provide complete context

### 4. Memory Lifecycle Management

The memory system automatically manages the lifecycle of entities:

- **Promotion**: Move entities to higher layers when accessed
- **Demotion**: Move idle entities to lower layers
- **Prefetching**: Proactively load related entities
- **Compression**: Apply different compression strategies for each layer

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Enhanced Memory Manager                   │
└───────────────────────────────┬─────────────────────────────┘
                                │
    ┌───────────────────────────┼───────────────────────────┐
    │                           │                           │
┌───▼───────────────┐   ┌───────▼─────────────┐   ┌─────────▼───────────┐
│ Layered Memory    │   │ Entity-Relationship │   │ Graph-Based         │
│ Architecture      │   │ Model               │   │ Retrieval           │
└───────────────────┘   └───────────────────────┘   └─────────────────────┘
    │       │               │           │               │
    │       │               │           │               │
┌───▼───┐ ┌─▼────┐     ┌────▼────┐ ┌────▼─────┐     ┌───▼────────────────┐
│  L1   │ │  L2  │     │ Entities│ │Relations │     │ Hierarchical Query │
│ Cache │ │ Index│     │         │ │          │     │ Processor          │
└───────┘ └──────┘     └─────────┘ └──────────┘     └────────────────────┘
    │       │               │           │               │
    │       │               │           │               │
    └───────┴───────────────┴───────────┴───────────────┘
                            │
                    ┌───────▼────────┐
                    │ Memory         │
                    │ Lifecycle      │
                    │ Manager        │
                    └────────────────┘
```

## Usage Examples

### Creating a Memory Node

```python
node = await memory_manager.create_node(
    content=[
        ContentBlock(
            type=ContentType.TEXT,
            data="A cute cat mascot for our tech company's marketing campaign",
            metadata={"language": "en"}
        ),
        ContentBlock(
            type=ContentType.IMAGE,
            data=image_bytes,
            metadata={"format": "png", "size": len(image_bytes)}
        )
    ],
    entity_type=EntityType.TASK_ARTIFACT,
    source=MemorySource(type="task", task_id="task_123"),
    summary="Cat mascot image",
    keywords=["cat", "mascot", "marketing", "tech"],
    entities=[
        NamedEntity(type="animal", value="cat", confidence=0.95),
        NamedEntity(type="color", value="blue", confidence=0.9)
    ],
    importance=0.8
)
```

### Adding a Relationship

```python
relationship = await memory_manager.add_relationship(
    source_id="node_123",
    target_id="node_456",
    relation_type=RelationType.PRODUCES,
    strength=0.9,
    bidirectional=False,
    context={"created_at": "2025-06-26T10:30:00Z"}
)
```

### Searching with Graph Expansion

```python
results = await memory_manager.search(
    query="marketing campaign with mascot",
    k=10,
    filters={"entity_type": EntityType.TASK_ARTIFACT},
    rerank=True,
    max_hops=2
)

# Access results
for scored_node in results["nodes"]:
    print(f"Node: {scored_node.node.summary}, Score: {scored_node.score}")

# Access closure information
for path in results["closure"]["expansion_path"]:
    print(f"Expansion: {path['from']} --{path['relation']}--> {path['to']}")
```

## Implementation Details

### Memory Node Structure

Memory nodes contain:
- **Content**: Multimodal content blocks (text, images, code, etc.)
- **Metadata**: Entity type, source, creation time, etc.
- **Semantic Information**: Summary, keywords, named entities
- **Relationships**: Links to other nodes
- **Embeddings**: Vector representations for retrieval

### Relationship Structure

Relationships contain:
- **Type**: The type of relationship
- **Source and Target**: The connected nodes
- **Strength**: The strength of the relationship (0.0 to 1.0)
- **Metadata**: Additional context about the relationship
- **Embedding**: Vector representation of the relationship

### Retrieval Process

1. **Query Embedding**: Convert query to vector representation
2. **Seed Search**: Find initial nodes matching the query
3. **Graph Expansion**: Follow relationships to find related nodes
4. **Relevance Scoring**: Score nodes based on query relevance and relationship context
5. **Reranking**: Rerank results for final ordering
6. **Closure Building**: Construct a minimal closure of related nodes

## Performance Considerations

- **Caching**: Embeddings are cached to avoid redundant computation
- **Lazy Loading**: L3 nodes are loaded on demand
- **Prefetching**: Related nodes are prefetched based on access patterns
- **Background Processing**: Lifecycle management runs in the background
- **Compression**: Different compression strategies are applied at each layer