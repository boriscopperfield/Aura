"""
Embedding models for the AURA memory system.
"""
import os
import json
import base64
from typing import List, Dict, Any, Optional, Union
import requests
from pathlib import Path
import numpy as np
from rich.console import Console

console = Console()

class JinaEmbedder:
    """Jina AI embedding service for multimodal content."""
    
    def __init__(self, model: str = "jina-embeddings-v4"):
        """Initialize the Jina embedder.
        
        Args:
            model: The model to use for embeddings.
        """
        self.api_key = os.environ.get("JINA_API_KEY")
        self.base_url = os.environ.get("JINA_BASE_URL", "https://api.jina.ai/v1/embeddings")
        self.model = model
        
        if not self.api_key:
            console.print("[bold red]Warning: JINA_API_KEY not found in environment variables[/bold red]")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    async def embed_text(self, 
                         texts: List[str], 
                         return_multivector: bool = False,
                         task: str = "retrieval.query",
                         late_chunking: bool = True) -> Dict[str, Any]:
        """Embed text content.
        
        Args:
            texts: List of text strings to embed.
            return_multivector: If True, return token-level embeddings, otherwise chunk-level.
            task: The embedding task type (retrieval.query or retrieval.passage).
            late_chunking: If True, use context-aware embeddings.
            
        Returns:
            Dictionary containing the embeddings.
        """
        input_data = [{"text": text} for text in texts]
        return await self._embed(input_data, return_multivector, task, late_chunking)
    
    async def embed_images(self, 
                          images: List[Union[str, Path, bytes]], 
                          return_multivector: bool = False,
                          task: str = "retrieval.query",
                          late_chunking: bool = True) -> Dict[str, Any]:
        """Embed image content.
        
        Args:
            images: List of images as URLs, file paths, or bytes.
            return_multivector: If True, return token-level embeddings, otherwise chunk-level.
            task: The embedding task type (retrieval.query or retrieval.passage).
            late_chunking: If True, use context-aware embeddings.
            
        Returns:
            Dictionary containing the embeddings.
        """
        input_data = []
        for img in images:
            if isinstance(img, str) and (img.startswith('http://') or img.startswith('https://')):
                # URL
                input_data.append({"image": img})
            elif isinstance(img, (str, Path)):
                # File path
                with open(img, 'rb') as f:
                    img_bytes = f.read()
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    input_data.append({"image": img_base64})
            elif isinstance(img, bytes):
                # Raw bytes
                img_base64 = base64.b64encode(img).decode('utf-8')
                input_data.append({"image": img_base64})
            else:
                raise ValueError(f"Unsupported image format: {type(img)}")
        
        return await self._embed(input_data, return_multivector, task, late_chunking)
    
    async def embed_multimodal(self, 
                              content: List[Dict[str, Any]], 
                              return_multivector: bool = False,
                              task: str = "retrieval.query",
                              late_chunking: bool = True) -> Dict[str, Any]:
        """Embed multimodal content.
        
        Args:
            content: List of content items, each with 'text' or 'image' key.
            return_multivector: If True, return token-level embeddings, otherwise chunk-level.
            task: The embedding task type (retrieval.query or retrieval.passage).
            late_chunking: If True, use context-aware embeddings.
            
        Returns:
            Dictionary containing the embeddings.
        """
        # Process any image paths to base64
        for item in content:
            if 'image' in item and isinstance(item['image'], (str, Path)) and not (
                    item['image'].startswith('http://') or item['image'].startswith('https://')):
                with open(item['image'], 'rb') as f:
                    img_bytes = f.read()
                    item['image'] = base64.b64encode(img_bytes).decode('utf-8')
        
        return await self._embed(content, return_multivector, task, late_chunking)
    
    async def _embed(self, 
                    input_data: List[Dict[str, Any]], 
                    return_multivector: bool = False,
                    task: str = "retrieval.query",
                    late_chunking: bool = True) -> Dict[str, Any]:
        """Internal method to call the Jina API.
        
        Args:
            input_data: List of content items to embed.
            return_multivector: If True, return token-level embeddings, otherwise chunk-level.
            task: The embedding task type (retrieval.query or retrieval.passage).
            late_chunking: If True, use context-aware embeddings.
            
        Returns:
            Dictionary containing the embeddings.
        """
        data = {
            "model": self.model,
            "return_multivector": return_multivector,
            "task": task,
            "late_chunking": late_chunking,
            "input": input_data
        }
        
        try:
            # Using requests directly since we need to handle the async in the caller
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(self.base_url, headers=self.headers, json=data)
            )
            
            if response.status_code != 200:
                console.print(f"[bold red]Error from Jina API: {response.status_code}[/bold red]")
                console.print(f"[bold red]Response: {response.text}[/bold red]")
                return {"error": response.text}
            
            return response.json()
        except Exception as e:
            console.print(f"[bold red]Error calling Jina API: {str(e)}[/bold red]")
            return {"error": str(e)}


class JinaReranker:
    """Jina AI reranker service for multimodal content."""
    
    def __init__(self, model: str = "jina-reranker-v1-base-en"):
        """Initialize the Jina reranker.
        
        Args:
            model: The model to use for reranking.
        """
        self.api_key = os.environ.get("JINA_API_KEY")
        self.base_url = os.environ.get("JINA_BASE_URL", "https://api.jina.ai/v1/rerank")
        self.model = model
        
        if not self.api_key:
            console.print("[bold red]Warning: JINA_API_KEY not found in environment variables[/bold red]")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    async def rerank(self, 
                    query: str, 
                    documents: List[str], 
                    top_k: int = 5) -> Dict[str, Any]:
        """Rerank documents based on relevance to query.
        
        Args:
            query: The query string.
            documents: List of document strings to rerank.
            top_k: Number of top results to return.
            
        Returns:
            Dictionary containing the reranked results.
        """
        data = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_k": top_k
        }
        
        try:
            # Using requests directly since we need to handle the async in the caller
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(self.base_url, headers=self.headers, json=data)
            )
            
            if response.status_code != 200:
                console.print(f"[bold red]Error from Jina API: {response.status_code}[/bold red]")
                console.print(f"[bold red]Response: {response.text}[/bold red]")
                return {"error": response.text}
            
            return response.json()
        except Exception as e:
            console.print(f"[bold red]Error calling Jina API: {str(e)}[/bold red]")
            return {"error": str(e)}


class EmbeddingCache:
    """Cache for embeddings to avoid redundant API calls."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the embedding cache.
        
        Args:
            cache_dir: Directory to store cache files. If None, uses a temporary directory.
        """
        if cache_dir is None:
            cache_dir = Path("/tmp/aura_embedding_cache")
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for faster lookups
        self.memory_cache = {}
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get embedding from cache.
        
        Args:
            key: Cache key (usually content hash).
            
        Returns:
            Cached embedding or None if not found.
        """
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    embedding = json.load(f)
                # Update memory cache
                self.memory_cache[key] = embedding
                return embedding
            except Exception as e:
                console.print(f"[bold yellow]Error reading cache file: {str(e)}[/bold yellow]")
        
        return None
    
    def set(self, key: str, embedding: Dict[str, Any]) -> None:
        """Store embedding in cache.
        
        Args:
            key: Cache key (usually content hash).
            embedding: Embedding data to cache.
        """
        # Update memory cache
        self.memory_cache[key] = embedding
        
        # Update disk cache
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(embedding, f)
        except Exception as e:
            console.print(f"[bold yellow]Error writing cache file: {str(e)}[/bold yellow]")
    
    def clear(self) -> None:
        """Clear the cache."""
        # Clear memory cache
        self.memory_cache = {}
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception as e:
                console.print(f"[bold yellow]Error deleting cache file: {str(e)}[/bold yellow]")