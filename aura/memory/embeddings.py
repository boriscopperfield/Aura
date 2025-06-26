"""
Embedding system for AURA memory.

This module provides embedding capabilities for the memory system,
converting content into vector representations for efficient retrieval.
"""
import os
import json
import base64
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

import numpy as np
import requests
from pydantic import BaseModel

from aura.config import settings
from aura.utils.logging import logger
from aura.utils.errors import JinaError
from aura.utils.async_utils import AsyncCache, async_retry
from aura.memory.models import ContentBlock, EmbeddingSet


class EmbeddingRequest(BaseModel):
    """Request for embedding generation."""
    
    texts: Optional[List[str]] = None
    images: Optional[List[str]] = None
    model: str


class EmbeddingResponse(BaseModel):
    """Response from embedding API."""
    
    embeddings: List[List[float]]
    model: str
    usage: Dict[str, Any]


class EmbeddingCache:
    """Cache for embeddings to avoid redundant API calls."""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """Initialize the embedding cache.
        
        Args:
            cache_dir: Directory for persistent cache storage
        """
        if cache_dir is None:
            cache_dir = Path(os.environ.get("AURA_CACHE_DIR", "/tmp/aura_embedding_cache"))
        elif isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
            
        self.cache_dir = cache_dir
        self.memory_cache = {}
        self.logger = logger
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def get(self, content_hash: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache.
        
        Args:
            content_hash: Hash of the content
            model: Embedding model name
            
        Returns:
            Cached embedding or None if not found
        """
        # Check memory cache first
        cache_key = f"{content_hash}:{model}"
        if cache_key in self.memory_cache:
            self.logger.debug(f"Cache hit for {cache_key}")
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    embedding = json.load(f)
                
                # Add to memory cache
                self.memory_cache[cache_key] = embedding
                self.logger.debug(f"Disk cache hit for {cache_key}")
                return embedding
            except Exception as e:
                self.logger.warning(f"Error reading cache file {cache_path}: {e}")
        
        return None
    
    async def set(self, content_hash: str, model: str, embedding: List[float]) -> None:
        """Set embedding in cache.
        
        Args:
            content_hash: Hash of the content
            model: Embedding model name
            embedding: Embedding vector
        """
        # Add to memory cache
        cache_key = f"{content_hash}:{model}"
        self.memory_cache[cache_key] = embedding
        
        # Add to disk cache
        cache_path = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_path, 'w') as f:
                json.dump(embedding, f)
        except Exception as e:
            self.logger.warning(f"Error writing cache file {cache_path}: {e}")
    
    async def clear(self) -> None:
        """Clear the cache."""
        self.memory_cache.clear()
        
        # Clear disk cache
        for file in self.cache_dir.glob("*.json"):
            try:
                file.unlink()
            except Exception as e:
                self.logger.warning(f"Error removing cache file {file}: {e}")


class JinaEmbedder:
    """Embedding service using Jina AI API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        cache: Optional[EmbeddingCache] = None
    ):
        """Initialize the embedder.
        
        Args:
            api_key: Jina AI API key
            base_url: Jina AI API base URL
            model: Embedding model name
            cache: Embedding cache
        """
        self.api_key = api_key or settings.jina.api_key
        self.base_url = base_url or settings.jina.base_url
        self.model = model or settings.jina.embedding_model
        self.cache = cache or EmbeddingCache()
        self.logger = logger
        
        # Create headers
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    @async_retry(retries=3, exceptions=(Exception,))
    async def embed_text(
        self, 
        text: str,
        return_multivector: bool = False,
        task: str = "retrieval.query",
        late_chunking: bool = True
    ) -> List[float]:
        """Generate embedding for text.
        
        Args:
            text: Text to embed
            return_multivector: Whether to return token-level embeddings
            task: Embedding task type
            late_chunking: Whether to use context-aware embeddings
            
        Returns:
            Embedding vector
            
        Raises:
            JinaError: If the API call fails
        """
        # Generate content hash for caching
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        
        # Check cache
        cached = await self.cache.get(content_hash, self.model)
        if cached:
            return cached
        
        # Call API
        try:
            # Using requests directly since we need to handle the async in the caller
            import asyncio
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(
                    f"{self.base_url}/embeddings",
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "return_multivector": return_multivector,
                        "task": task,
                        "late_chunking": late_chunking,
                        "input": [{"text": text}]
                    },
                    timeout=settings.jina.timeout
                )
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract embedding
            embedding = result["data"][0]["embedding"]
            
            # Cache embedding
            await self.cache.set(content_hash, self.model, embedding)
            
            return embedding
        except Exception as e:
            self.logger.error(f"Error embedding text with Jina AI: {e}")
            raise JinaError(f"Failed to embed text: {str(e)}")
    
    @async_retry(retries=3, exceptions=(Exception,))
    async def embed_texts(
        self, 
        texts: List[str],
        return_multivector: bool = False,
        task: str = "retrieval.query",
        late_chunking: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: Texts to embed
            return_multivector: Whether to return token-level embeddings
            task: Embedding task type
            late_chunking: Whether to use context-aware embeddings
            
        Returns:
            List of embedding vectors
            
        Raises:
            JinaError: If the API call fails
        """
        # Check for empty input
        if not texts:
            return []
        
        # Check cache for all texts
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            cached = await self.cache.get(content_hash, self.model)
            
            if cached:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # If all embeddings are cached, return them
        if not uncached_texts:
            return embeddings
        
        # Call API for uncached texts
        try:
            # Using requests directly since we need to handle the async in the caller
            import asyncio
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(
                    f"{self.base_url}/embeddings",
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "return_multivector": return_multivector,
                        "task": task,
                        "late_chunking": late_chunking,
                        "input": [{"text": text} for text in uncached_texts]
                    },
                    timeout=settings.jina.timeout
                )
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract embeddings
            api_embeddings = [item["embedding"] for item in result["data"]]
            
            # Cache embeddings and update results
            for i, (text, embedding) in enumerate(zip(uncached_texts, api_embeddings)):
                content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
                await self.cache.set(content_hash, self.model, embedding)
                
                # Update embeddings list
                embeddings[uncached_indices[i]] = embedding
            
            return embeddings
        except Exception as e:
            self.logger.error(f"Error embedding texts with Jina AI: {e}")
            raise JinaError(f"Failed to embed texts: {str(e)}")
    
    @async_retry(retries=3, exceptions=(Exception,))
    async def embed_image(
        self, 
        image: Union[str, bytes],
        return_multivector: bool = False,
        task: str = "retrieval.query",
        late_chunking: bool = True
    ) -> List[float]:
        """Generate embedding for image.
        
        Args:
            image: Image URL or bytes
            return_multivector: Whether to return token-level embeddings
            task: Embedding task type
            late_chunking: Whether to use context-aware embeddings
            
        Returns:
            Embedding vector
            
        Raises:
            JinaError: If the API call fails
        """
        # Generate content hash for caching
        if isinstance(image, str):
            # If it's a URL, use the URL as the hash
            if image.startswith('http://') or image.startswith('https://'):
                content_hash = hashlib.sha256(image.encode('utf-8')).hexdigest()
            else:
                # Assume it's base64
                content_hash = hashlib.sha256(image.encode('utf-8')).hexdigest()
        else:
            # It's bytes
            content_hash = hashlib.sha256(image).hexdigest()
        
        # Check cache
        cached = await self.cache.get(content_hash, self.model)
        if cached:
            return cached
        
        # Prepare image data
        if isinstance(image, bytes):
            image_data = base64.b64encode(image).decode('utf-8')
        elif isinstance(image, str) and (image.startswith('http://') or image.startswith('https://')):
            # URL
            image_data = image
        else:
            # Assume it's already base64
            image_data = image
        
        # Call API
        try:
            # Using requests directly since we need to handle the async in the caller
            import asyncio
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(
                    f"{self.base_url}/embeddings",
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "return_multivector": return_multivector,
                        "task": task,
                        "late_chunking": late_chunking,
                        "input": [{"image": image_data}]
                    },
                    timeout=settings.jina.timeout
                )
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract embedding
            embedding = result["data"][0]["embedding"]
            
            # Cache embedding
            await self.cache.set(content_hash, self.model, embedding)
            
            return embedding
        except Exception as e:
            self.logger.error(f"Error embedding image with Jina AI: {e}")
            raise JinaError(f"Failed to embed image: {str(e)}")
    
    @async_retry(retries=3, exceptions=(Exception,))
    async def embed_content_blocks(
        self, 
        blocks: List[ContentBlock],
        return_multivector: bool = False,
        task: str = "retrieval.query",
        late_chunking: bool = True
    ) -> EmbeddingSet:
        """Generate embeddings for content blocks.
        
        Args:
            blocks: Content blocks to embed
            return_multivector: Whether to return token-level embeddings
            task: Embedding task type
            late_chunking: Whether to use context-aware embeddings
            
        Returns:
            Embedding set
            
        Raises:
            JinaError: If the API call fails
        """
        # Initialize embedding set
        embedding_set = EmbeddingSet(
            model_info={
                "chunk": self.model,
                "token": self.model
            }
        )
        
        # Process each content block
        for block in blocks:
            # Get embedding input
            embedding_input = block.to_embedding_input()
            
            # Generate content hash
            content_hash = block.get_hash()
            
            # Generate embedding based on content type
            if "text" in embedding_input:
                text = embedding_input["text"]
                embedding = await self.embed_text(
                    text, 
                    return_multivector=return_multivector,
                    task=task,
                    late_chunking=late_chunking
                )
                embedding_set.chunk_embeddings[content_hash] = embedding
            elif "image" in embedding_input:
                image = embedding_input["image"]
                embedding = await self.embed_image(
                    image,
                    return_multivector=return_multivector,
                    task=task,
                    late_chunking=late_chunking
                )
                embedding_set.chunk_embeddings[content_hash] = embedding
        
        # For backward compatibility
        if embedding_set.chunk_embeddings:
            # Use the first embedding as the legacy chunk_embedding
            embedding_set.chunk_embedding = list(embedding_set.chunk_embeddings.values())[0]
        
        return embedding_set


class JinaReranker:
    """Reranking service using Jina AI API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """Initialize the reranker.
        
        Args:
            api_key: Jina AI API key
            base_url: Jina AI API base URL
            model: Reranker model name
        """
        self.api_key = api_key or settings.jina.api_key
        self.base_url = base_url or settings.jina.base_url
        self.model = model or settings.jina.reranker_model
        self.logger = logger
        
        # Create headers
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    @async_retry(retries=3, exceptions=(Exception,))
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """Rerank documents based on relevance to query.
        
        Args:
            query: Query text
            documents: Documents to rerank
            top_k: Number of top results to return
            
        Returns:
            List of (document, score) tuples
            
        Raises:
            JinaError: If the API call fails
        """
        # Check for empty input
        if not documents:
            return []
        
        # Call API
        try:
            # Using requests directly since we need to handle the async in the caller
            import asyncio
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(
                    f"{self.base_url}/rerank",
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "query": query,
                        "documents": documents,
                        "top_k": top_k
                    },
                    timeout=settings.jina.timeout
                )
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract results
            reranked = [(documents[item["index"]], item["score"]) for item in result["data"]]
            
            return reranked
        except Exception as e:
            self.logger.error(f"Error reranking with Jina AI: {e}")
            raise JinaError(f"Failed to rerank documents: {str(e)}")


# Create global instances
embedding_cache = EmbeddingCache()
jina_embedder = JinaEmbedder(cache=embedding_cache)
jina_reranker = JinaReranker()