"""
Memory retrieval system for AURA.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import asyncio
import numpy as np
from pathlib import Path
import json
import faiss
import hashlib

from rich.console import Console
from rich.progress import Progress

from aura.memory.nodes import MemoryNode, ContentBlock, ContentType
from aura.memory.embeddings import JinaEmbedder, JinaReranker, EmbeddingCache

console = Console()


@dataclass
class Query:
    """Query for memory retrieval."""
    text: str
    filters: Dict[str, Any] = field(default_factory=dict)
    modalities: List[Dict[str, Any]] = field(default_factory=list)
    k: int = 10
    rerank: bool = True
    expand_context: bool = True
    max_hops: int = 2


@dataclass
class ScoredMemoryNode:
    """Memory node with relevance score."""
    node: MemoryNode
    score: float
    match_details: Dict[str, Any] = field(default_factory=dict)


class MemoryRetriever:
    """Intelligent retrieval system for memory nodes."""
    
    def __init__(self, 
                 memory_dir: Path,
                 embedder: Optional[JinaEmbedder] = None,
                 reranker: Optional[JinaReranker] = None,
                 cache: Optional[EmbeddingCache] = None):
        """Initialize the memory retriever.
        
        Args:
            memory_dir: Directory containing memory files.
            embedder: Jina embedder for generating embeddings.
            reranker: Jina reranker for reranking results.
            cache: Cache for embeddings.
        """
        self.memory_dir = memory_dir
        self.nodes_path = memory_dir / "graph" / "nodes.jsonl"
        self.edges_path = memory_dir / "graph" / "edges.jsonl"
        self.index_dir = memory_dir / "indexes"
        
        # Create directories if they don't exist
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        (self.memory_dir / "graph").mkdir(exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)
        
        # Initialize embedder and reranker
        self.embedder = embedder or JinaEmbedder()
        self.reranker = reranker or JinaReranker()
        self.cache = cache or EmbeddingCache()
        
        # Initialize in-memory graph
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: Dict[str, List[str]] = {}  # node_id -> [related_node_ids]
        
        # Initialize vector indexes
        self.chunk_index = None
        self.chunk_index_to_node: Dict[int, str] = {}  # index -> node_id
        
        # Load existing data
        self._load_data()
    
    async def retrieve(self, query: Query) -> List[ScoredMemoryNode]:
        """Retrieve memory nodes based on query.
        
        Args:
            query: Query object with text and filters.
            
        Returns:
            List of scored memory nodes.
        """
        with Progress() as progress:
            # Stage 1: Query embedding
            embed_task = progress.add_task("Embedding query...", total=1)
            query_embedding = await self._embed_query(query.text)
            progress.update(embed_task, completed=1)
            
            # Stage 2: Vector search
            search_task = progress.add_task("Searching vector index...", total=1)
            vector_results = await self._vector_search(query_embedding, k=query.k*2)
            progress.update(search_task, completed=1)
            
            # Stage 3: Filter results
            filter_task = progress.add_task("Applying filters...", total=1)
            filtered_results = self._apply_filters(vector_results, query.filters)
            progress.update(filter_task, completed=1)
            
            # Stage 4: Graph expansion (if requested)
            if query.expand_context:
                expand_task = progress.add_task("Expanding context...", total=1)
                expanded_results = await self._expand_context(filtered_results, max_hops=query.max_hops)
                progress.update(expand_task, completed=1)
            else:
                expanded_results = filtered_results
            
            # Stage 5: Reranking (if requested)
            if query.rerank and len(expanded_results) > 1:
                rerank_task = progress.add_task("Reranking results...", total=1)
                final_results = await self._rerank_results(query.text, expanded_results, top_k=query.k)
                progress.update(rerank_task, completed=1)
            else:
                # Just truncate to k results
                final_results = expanded_results[:query.k]
        
        return final_results
    
    async def add_node(self, node: MemoryNode) -> str:
        """Add a memory node to the graph and index.
        
        Args:
            node: Memory node to add.
            
        Returns:
            ID of the added node.
        """
        # Generate embeddings if not already present
        if not node.embeddings or not node.embeddings.chunk_embeddings:
            node.embeddings = await self._generate_embeddings(node)
        
        # Add to in-memory graph
        self.nodes[node.id] = node
        
        # Add to vector index
        await self._index_node(node)
        
        # Append to nodes file
        with open(self.nodes_path, 'a') as f:
            f.write(json.dumps(node.to_dict()) + '\n')
        
        return node.id
    
    async def add_relation(self, source_id: str, target_id: str, relation_type: str, strength: float = 1.0) -> None:
        """Add a relation between two nodes.
        
        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            relation_type: Type of relation.
            strength: Strength of the relation.
        """
        # Add to in-memory graph
        if source_id not in self.edges:
            self.edges[source_id] = []
        if target_id not in self.edges[source_id]:
            self.edges[source_id].append(target_id)
        
        # Add to source node's relations
        if source_id in self.nodes:
            source_node = self.nodes[source_id]
            # Check if relation already exists
            for relation in source_node.relations:
                if relation.target_id == target_id and relation.type.value == relation_type:
                    relation.strength = strength
                    break
            else:
                # Add new relation
                from aura.memory.nodes import Relation, RelationType
                source_node.relations.append(Relation(
                    type=RelationType(relation_type),
                    target_id=target_id,
                    strength=strength
                ))
        
        # Append to edges file
        with open(self.edges_path, 'a') as f:
            f.write(json.dumps({
                "source_id": source_id,
                "target_id": target_id,
                "type": relation_type,
                "strength": strength
            }) + '\n')
    
    async def _embed_query(self, query_text: str) -> np.ndarray:
        """Embed query text.
        
        Args:
            query_text: Query text to embed.
            
        Returns:
            Query embedding.
        """
        # Check cache first
        query_hash = hashlib.sha256(query_text.encode('utf-8')).hexdigest()
        cached = self.cache.get(f"query_{query_hash}")
        if cached:
            return np.array(cached["embedding"])
        
        # Generate embedding
        response = await self.embedder.embed_text(
            [query_text],
            return_multivector=False,
            task="retrieval.query",
            late_chunking=True
        )
        
        if "error" in response:
            console.print(f"[bold red]Error embedding query: {response['error']}[/bold red]")
            # Return zero vector as fallback
            return np.zeros(512)
        
        embedding = np.array(response["data"][0]["embedding"])
        
        # Cache the result
        self.cache.set(f"query_{query_hash}", {"embedding": embedding.tolist()})
        
        return embedding
    
    async def _generate_embeddings(self, node: MemoryNode) -> Dict[str, Any]:
        """Generate embeddings for a memory node.
        
        Args:
            node: Memory node to embed.
            
        Returns:
            Embedding set for the node.
        """
        from aura.memory.nodes import EmbeddingSet
        
        # Initialize embedding set
        embedding_set = EmbeddingSet(
            model_info={"model": "jina-embeddings-v4"}
        )
        
        # Process each content block
        for block in node.content:
            # Generate hash for caching
            block_hash = block.get_hash()
            
            # Check cache first
            cached = self.cache.get(f"block_{block_hash}")
            if cached:
                if "chunk_embedding" in cached:
                    embedding_set.chunk_embeddings[block_hash] = cached["chunk_embedding"]
                if "token_embeddings" in cached:
                    embedding_set.token_embeddings[block_hash] = cached["token_embeddings"]
                continue
            
            # Convert to embedding input format
            embedding_input = block.to_embedding_input()
            
            # Generate chunk embedding
            chunk_response = await self.embedder.embed_multimodal(
                [embedding_input],
                return_multivector=False,
                task="retrieval.passage",
                late_chunking=True
            )
            
            if "error" not in chunk_response:
                chunk_embedding = np.array(chunk_response["data"][0]["embedding"])
                embedding_set.chunk_embeddings[block_hash] = chunk_embedding.tolist()
                
                # Cache the result
                self.cache.set(f"block_{block_hash}", {
                    "chunk_embedding": chunk_embedding.tolist()
                })
            
            # Generate token embeddings for text content only (optional)
            if block.type == ContentType.TEXT or block.type == ContentType.CODE:
                token_response = await self.embedder.embed_multimodal(
                    [embedding_input],
                    return_multivector=True,
                    task="retrieval.passage",
                    late_chunking=True
                )
                
                if "error" not in token_response and "multivector" in token_response["data"][0]:
                    token_embeddings = token_response["data"][0]["multivector"]
                    embedding_set.token_embeddings[block_hash] = token_embeddings
                    
                    # Update cache
                    cached_data = self.cache.get(f"block_{block_hash}") or {}
                    cached_data["token_embeddings"] = token_embeddings
                    self.cache.set(f"block_{block_hash}", cached_data)
        
        return embedding_set
    
    async def _vector_search(self, query_embedding: np.ndarray, k: int = 10) -> List[ScoredMemoryNode]:
        """Search vector index for similar nodes.
        
        Args:
            query_embedding: Query embedding.
            k: Number of results to return.
            
        Returns:
            List of scored memory nodes.
        """
        if self.chunk_index is None or len(self.nodes) == 0:
            return []
        
        # Search the index
        distances, indices = self.chunk_index.search(np.array([query_embedding]), k)
        
        # Convert to scored nodes
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS returns -1 for padding when there are fewer results than k
                continue
                
            node_id = self.chunk_index_to_node.get(idx)
            if node_id and node_id in self.nodes:
                node = self.nodes[node_id]
                score = 1.0 - distances[0][i]  # Convert distance to similarity score
                results.append(ScoredMemoryNode(
                    node=node,
                    score=score,
                    match_details={"vector_score": score}
                ))
        
        return results
    
    def _apply_filters(self, results: List[ScoredMemoryNode], filters: Dict[str, Any]) -> List[ScoredMemoryNode]:
        """Apply filters to search results.
        
        Args:
            results: List of scored memory nodes.
            filters: Filters to apply.
            
        Returns:
            Filtered list of scored memory nodes.
        """
        if not filters:
            return results
            
        filtered_results = []
        for result in results:
            node = result.node
            
            # Apply entity type filter
            if "entity_type" in filters and node.entity_type.value != filters["entity_type"]:
                continue
                
            # Apply source type filter
            if "source_type" in filters and node.source.type != filters["source_type"]:
                continue
                
            # Apply user filter
            if "user_id" in filters and node.source.user_id != filters["user_id"]:
                continue
                
            # Apply task filter
            if "task_id" in filters and node.source.task_id != filters["task_id"]:
                continue
                
            # Apply keyword filter
            if "keywords" in filters:
                keywords = filters["keywords"]
                if not any(kw in node.keywords for kw in keywords):
                    continue
                    
            # Apply date range filter
            if "date_range" in filters:
                date_range = filters["date_range"]
                if "start" in date_range and node.created_at < date_range["start"]:
                    continue
                if "end" in date_range and node.created_at > date_range["end"]:
                    continue
            
            # Node passed all filters
            filtered_results.append(result)
            
        return filtered_results
    
    async def _expand_context(self, results: List[ScoredMemoryNode], max_hops: int = 2) -> List[ScoredMemoryNode]:
        """Expand results by following graph relationships.
        
        Args:
            results: Initial results.
            max_hops: Maximum number of hops to follow.
            
        Returns:
            Expanded list of scored memory nodes.
        """
        if not results or max_hops <= 0:
            return results
            
        # Get initial node IDs
        initial_ids = {result.node.id for result in results}
        expanded_ids = set(initial_ids)
        
        # Expand by following edges
        frontier = list(initial_ids)
        for hop in range(max_hops):
            next_frontier = []
            
            for node_id in frontier:
                # Follow outgoing edges
                for target_id in self.edges.get(node_id, []):
                    if target_id not in expanded_ids:
                        expanded_ids.add(target_id)
                        next_frontier.append(target_id)
            
            frontier = next_frontier
            if not frontier:
                break
        
        # Create expanded results
        expanded_results = list(results)  # Start with original results
        
        # Add new nodes with lower scores
        for node_id in expanded_ids:
            if node_id not in initial_ids and node_id in self.nodes:
                # Calculate score based on distance from initial results
                # The further away, the lower the score
                score = 0.5 / (hop + 1)
                
                expanded_results.append(ScoredMemoryNode(
                    node=self.nodes[node_id],
                    score=score,
                    match_details={"expanded_from_graph": True, "hops": hop + 1}
                ))
        
        return expanded_results
    
    async def _rerank_results(self, query: str, results: List[ScoredMemoryNode], top_k: int = 10) -> List[ScoredMemoryNode]:
        """Rerank results using the reranker.
        
        Args:
            query: Original query text.
            results: Results to rerank.
            top_k: Number of top results to return.
            
        Returns:
            Reranked list of scored memory nodes.
        """
        if not results:
            return []
            
        # Extract text content from nodes
        documents = []
        for result in results:
            documents.append(result.node.get_text_content())
        
        # Rerank using Jina reranker
        rerank_response = await self.reranker.rerank(query, documents, top_k=top_k)
        
        if "error" in rerank_response:
            console.print(f"[bold yellow]Error reranking results: {rerank_response['error']}[/bold yellow]")
            # Fall back to original ranking, but limit to top_k
            return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
        
        # Create reranked results
        reranked_results = []
        for item in rerank_response["results"]:
            idx = item["index"]
            score = item["score"]
            
            # Update the score in the original result
            result = results[idx]
            result.score = score
            result.match_details["reranker_score"] = score
            
            reranked_results.append(result)
        
        return reranked_results
    
    async def _index_node(self, node: MemoryNode) -> None:
        """Add a node to the vector index.
        
        Args:
            node: Node to index.
        """
        # Skip if no embeddings
        if not node.embeddings or not node.embeddings.chunk_embeddings:
            return
            
        # Get combined embedding
        embedding = node.embeddings.get_combined_chunk_embedding()
        if not embedding:
            return
            
        # Initialize index if needed
        if self.chunk_index is None:
            dimension = len(embedding)
            self.chunk_index = faiss.IndexFlatL2(dimension)
        
        # Add to index
        embedding_np = np.array([embedding], dtype=np.float32)
        idx = len(self.chunk_index_to_node)
        self.chunk_index.add(embedding_np)
        self.chunk_index_to_node[idx] = node.id
        
        # Update node's vector ID
        node.embeddings.vector_ids["chunk_index"] = idx
    
    def _load_data(self) -> None:
        """Load existing data from files."""
        # Load nodes
        if self.nodes_path.exists():
            with open(self.nodes_path, 'r') as f:
                for line in f:
                    try:
                        node_dict = json.loads(line.strip())
                        node = MemoryNode.from_dict(node_dict)
                        self.nodes[node.id] = node
                    except Exception as e:
                        console.print(f"[bold yellow]Error loading node: {e}[/bold yellow]")
        
        # Load edges
        if self.edges_path.exists():
            with open(self.edges_path, 'r') as f:
                for line in f:
                    try:
                        edge = json.loads(line.strip())
                        source_id = edge["source_id"]
                        target_id = edge["target_id"]
                        
                        if source_id not in self.edges:
                            self.edges[source_id] = []
                        if target_id not in self.edges[source_id]:
                            self.edges[source_id].append(target_id)
                    except Exception as e:
                        console.print(f"[bold yellow]Error loading edge: {e}[/bold yellow]")
        
        # Build vector index
        self._build_vector_index()
    
    def _build_vector_index(self) -> None:
        """Build vector index from loaded nodes."""
        if not self.nodes:
            return
            
        # Get dimension from first node with embeddings
        dimension = None
        for node in self.nodes.values():
            if node.embeddings:
                embedding = node.embeddings.get_combined_chunk_embedding()
                if embedding:
                    dimension = len(embedding)
                    break
        
        if dimension is None:
            return
            
        # Initialize index
        self.chunk_index = faiss.IndexFlatL2(dimension)
        
        # Add nodes to index
        for idx, (node_id, node) in enumerate(self.nodes.items()):
            if node.embeddings:
                embedding = node.embeddings.get_combined_chunk_embedding()
                if embedding:
                    embedding_np = np.array([embedding], dtype=np.float32)
                    self.chunk_index.add(embedding_np)
                    self.chunk_index_to_node[idx] = node_id
                    
                    # Update node's vector ID
                    node.embeddings.vector_ids["chunk_index"] = idx
    
    def save_indexes(self) -> None:
        """Save vector indexes to disk."""
        if self.chunk_index is not None:
            index_path = self.index_dir / "chunk_vectors.faiss"
            faiss.write_index(self.chunk_index, str(index_path))
            
            # Save mapping
            mapping_path = self.index_dir / "chunk_mapping.json"
            with open(mapping_path, 'w') as f:
                # Convert keys to strings for JSON
                mapping = {str(k): v for k, v in self.chunk_index_to_node.items()}
                json.dump(mapping, f)
    
    def load_indexes(self) -> None:
        """Load vector indexes from disk."""
        index_path = self.index_dir / "chunk_vectors.faiss"
        mapping_path = self.index_dir / "chunk_mapping.json"
        
        if index_path.exists() and mapping_path.exists():
            try:
                # Load index
                self.chunk_index = faiss.read_index(str(index_path))
                
                # Load mapping
                with open(mapping_path, 'r') as f:
                    mapping = json.load(f)
                    # Convert keys back to integers
                    self.chunk_index_to_node = {int(k): v for k, v in mapping.items()}
            except Exception as e:
                console.print(f"[bold red]Error loading indexes: {e}[/bold red]")
                # Initialize new index
                self._build_vector_index()