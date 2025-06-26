"""
Jina AI Agent implementation for embeddings and reranking.
"""
import os
from typing import Any, Dict, List, Optional, Union
import aiohttp

from .base import Agent, AgentCapability, AgentResult, RateLimit
from ..utils.errors import AgentError


class JinaEmbedder(Agent):
    """Jina AI Embeddings integration."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "jina-embeddings-v3",
        base_url: str = "https://api.jina.ai/v1",
        **kwargs
    ):
        super().__init__(
            name="jina-embedder",
            capabilities=[AgentCapability.TEXT_EMBEDDING],
            rate_limit=RateLimit(
                requests_per_minute=200,
                tokens_per_minute=1000000
            ),
            **kwargs
        )
        
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise AgentError("Jina AI API key is required")
        
        self.model = model
        self.base_url = base_url
        
        # Pricing per 1M tokens
        self.pricing = {
            "jina-embeddings-v3": 0.02,
            "jina-embeddings-v2": 0.02
        }
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _estimate_tokens(self, texts: List[str]) -> int:
        """Estimate tokens for a list of texts."""
        total_chars = sum(len(text) for text in texts)
        return total_chars // 4  # Rough estimation
    
    def _calculate_cost(self, tokens: int) -> float:
        """Calculate embedding cost."""
        price_per_million = self.pricing.get(self.model, 0.02)
        return (tokens / 1_000_000) * price_per_million
    
    async def _execute(self, capability: AgentCapability, **kwargs) -> AgentResult:
        """Execute Jina capability."""
        if capability == AgentCapability.TEXT_EMBEDDING:
            return await self._embed_texts(**kwargs)
        else:
            return AgentResult.error_result(
                f"Capability {capability.value} not implemented",
                0.0
            )
    
    async def _embed_texts(
        self,
        texts: Union[str, List[str]],
        task: str = "retrieval.passage",
        dimensions: Optional[int] = None,
        **kwargs
    ) -> AgentResult:
        """Generate embeddings for texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Check rate limits
        estimated_tokens = self._estimate_tokens(texts)
        await self.check_rate_limit(estimated_tokens)
        
        payload = {
            "model": self.model,
            "input": texts,
            "task": task
        }
        
        if dimensions:
            payload["dimensions"] = dimensions
        
        try:
            async with self.session.post(
                f"{self.base_url}/embeddings",
                headers=self._get_headers(),
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return AgentResult.error_result(
                        f"Jina API error: {response.status} - {error_text}",
                        0.0
                    )
                
                data = await response.json()
                
                # Extract embeddings
                embeddings = [item["embedding"] for item in data["data"]]
                usage = data.get("usage", {})
                
                # Calculate cost
                total_tokens = usage.get("total_tokens", estimated_tokens)
                cost = self._calculate_cost(total_tokens)
                self.total_cost += cost
                
                return AgentResult.success_result(
                    data=embeddings[0] if len(embeddings) == 1 else embeddings,
                    duration=0.0,
                    cost=cost,
                    metadata={
                        "model": self.model,
                        "task": task,
                        "dimensions": dimensions,
                        "total_tokens": total_tokens,
                        "num_texts": len(texts)
                    }
                )
                
        except Exception as e:
            return AgentResult.error_result(str(e), 0.0)
    
    async def embed_single(self, text: str, **kwargs) -> List[float]:
        """Convenience method to embed a single text."""
        result = await self.execute(AgentCapability.TEXT_EMBEDDING, texts=text, **kwargs)
        if result.success:
            return result.data
        else:
            raise AgentError(f"Embedding failed: {result.error}")
    
    async def embed_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Convenience method to embed multiple texts."""
        result = await self.execute(AgentCapability.TEXT_EMBEDDING, texts=texts, **kwargs)
        if result.success:
            return result.data
        else:
            raise AgentError(f"Embedding failed: {result.error}")
    
    async def _health_check(self) -> None:
        """Perform Jina health check."""
        try:
            # Test with a simple embedding
            result = await self._embed_texts(["health check"])
            if not result.success:
                raise AgentError(f"Health check failed: {result.error}")
        except Exception as e:
            raise AgentError(f"Health check failed: {e}")


class JinaReranker(Agent):
    """Jina AI Reranker integration."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "jina-reranker-v2-base-multilingual",
        base_url: str = "https://api.jina.ai/v1",
        **kwargs
    ):
        super().__init__(
            name="jina-reranker",
            capabilities=[AgentCapability.RERANKING],
            rate_limit=RateLimit(
                requests_per_minute=200,
                tokens_per_minute=1000000
            ),
            **kwargs
        )
        
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise AgentError("Jina AI API key is required")
        
        self.model = model
        self.base_url = base_url
        
        # Pricing per 1M tokens
        self.pricing = {
            "jina-reranker-v2-base-multilingual": 0.02,
            "jina-reranker-v1-base-en": 0.02
        }
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _estimate_tokens(self, query: str, documents: List[str]) -> int:
        """Estimate tokens for reranking."""
        total_chars = len(query) + sum(len(doc) for doc in documents)
        return total_chars // 4
    
    def _calculate_cost(self, tokens: int) -> float:
        """Calculate reranking cost."""
        price_per_million = self.pricing.get(self.model, 0.02)
        return (tokens / 1_000_000) * price_per_million
    
    async def _execute(self, capability: AgentCapability, **kwargs) -> AgentResult:
        """Execute Jina reranking capability."""
        if capability == AgentCapability.RERANKING:
            return await self._rerank_documents(**kwargs)
        else:
            return AgentResult.error_result(
                f"Capability {capability.value} not implemented",
                0.0
            )
    
    async def _rerank_documents(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_documents: bool = True,
        **kwargs
    ) -> AgentResult:
        """Rerank documents based on query relevance."""
        # Check rate limits
        estimated_tokens = self._estimate_tokens(query, documents)
        await self.check_rate_limit(estimated_tokens)
        
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "return_documents": return_documents
        }
        
        if top_k:
            payload["top_k"] = top_k
        
        try:
            async with self.session.post(
                f"{self.base_url}/rerank",
                headers=self._get_headers(),
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return AgentResult.error_result(
                        f"Jina API error: {response.status} - {error_text}",
                        0.0
                    )
                
                data = await response.json()
                
                # Extract results
                results = data["results"]
                usage = data.get("usage", {})
                
                # Calculate cost
                total_tokens = usage.get("total_tokens", estimated_tokens)
                cost = self._calculate_cost(total_tokens)
                self.total_cost += cost
                
                return AgentResult.success_result(
                    data=results,
                    duration=0.0,
                    cost=cost,
                    metadata={
                        "model": self.model,
                        "query": query,
                        "num_documents": len(documents),
                        "top_k": top_k,
                        "total_tokens": total_tokens
                    }
                )
                
        except Exception as e:
            return AgentResult.error_result(str(e), 0.0)
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Convenience method for reranking."""
        result = await self.execute(
            AgentCapability.RERANKING,
            query=query,
            documents=documents,
            top_k=top_k
        )
        if result.success:
            return result.data
        else:
            raise AgentError(f"Reranking failed: {result.error}")
    
    async def _health_check(self) -> None:
        """Perform Jina reranker health check."""
        try:
            # Test with simple reranking
            result = await self._rerank_documents(
                query="test",
                documents=["test document", "another document"]
            )
            if not result.success:
                raise AgentError(f"Health check failed: {result.error}")
        except Exception as e:
            raise AgentError(f"Health check failed: {e}")