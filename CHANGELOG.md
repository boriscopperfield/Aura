# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Enhanced memory system with Jina AI embeddings
- Multimodal content support (text, images, code)
- Context-aware embeddings for more relevant results
- FAISS-based vector search with reranking
- Memory learning pipeline with background processing
- Comprehensive test script demonstrating all capabilities
- Examples directory with documentation

### Changed
- Updated README with memory system documentation
- Added new dependencies in pyproject.toml
- Improved error handling in AI planner

### Fixed
- Fixed JSON parsing for markdown code blocks in API responses
- Added fallback plan mechanism for when the AI planner API fails
- Fixed task_name variable reference in CLI