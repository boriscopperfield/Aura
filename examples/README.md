# AURA Examples

This directory contains example scripts that demonstrate AURA's capabilities.

## Comprehensive Test

The `comprehensive_test.py` script demonstrates all major components of AURA:

- Memory System: Creating and querying memory nodes with Jina embeddings
- Learning Pipeline: Processing user interactions and learning preferences
- AI Planner: Generating plans from natural language intents
- Transaction Manager: Executing transactions with ACID guarantees

To run the comprehensive test:

```bash
python examples/comprehensive_test.py
```

## Demo Script

The `demo.py` script provides a simpler demonstration of AURA's learning capabilities:

```bash
python examples/demo.py
```

## Requirements

These examples require:
- Python 3.10+
- All AURA dependencies installed
- OpenAI API key (for AI planning)
- Jina AI API key (for memory embeddings)

Set the following environment variables:
```bash
export OPENAI_API_KEY=your_openai_key
export JINA_API_KEY=your_jina_key
```

Or create a `.env` file in the project root with these variables.