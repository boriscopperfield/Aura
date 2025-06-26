"""
Anthropic Claude Agent implementation.
"""
import os
from typing import Any, Dict, List, Optional
import aiohttp

from .base import Agent, AgentCapability, AgentResult, RateLimit
from ..utils.errors import AgentError


class AnthropicAgent(Agent):
    """Anthropic Claude API integration."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-sonnet-20240229",
        base_url: str = "https://api.anthropic.com/v1",
        **kwargs
    ):
        super().__init__(
            name="anthropic",
            capabilities=[
                AgentCapability.TEXT_GENERATION,
                AgentCapability.CODE_GENERATION,
                AgentCapability.ANALYSIS,
                AgentCapability.SUMMARIZATION
            ],
            rate_limit=RateLimit(
                requests_per_minute=50,
                tokens_per_minute=100000
            ),
            **kwargs
        )
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise AgentError("Anthropic API key is required")
        
        self.model = model
        self.base_url = base_url
        
        # Pricing per 1M tokens (as of 2024)
        self.pricing = {
            "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25}
        }
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars = 1 token)."""
        return len(text) // 4
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate API cost."""
        if model not in self.pricing:
            return 0.0
        
        pricing = self.pricing[model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    async def _execute(self, capability: AgentCapability, **kwargs) -> AgentResult:
        """Execute Anthropic capability."""
        if capability == AgentCapability.TEXT_GENERATION:
            return await self._generate_text(**kwargs)
        elif capability == AgentCapability.CODE_GENERATION:
            return await self._generate_code(**kwargs)
        elif capability == AgentCapability.ANALYSIS:
            return await self._analyze_text(**kwargs)
        elif capability == AgentCapability.SUMMARIZATION:
            return await self._summarize_text(**kwargs)
        else:
            return AgentResult.error_result(
                f"Capability {capability.value} not implemented",
                0.0
            )
    
    async def _generate_text(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AgentResult:
        """Generate text using Claude."""
        messages = [{"role": "user", "content": prompt}]
        
        # Estimate tokens for rate limiting
        total_text = (system_prompt or "") + prompt
        estimated_tokens = self._estimate_tokens(total_text) + max_tokens
        await self.check_rate_limit(estimated_tokens)
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            async with self.session.post(
                f"{self.base_url}/messages",
                headers=self._get_headers(),
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return AgentResult.error_result(
                        f"Anthropic API error: {response.status} - {error_text}",
                        0.0
                    )
                
                data = await response.json()
                
                # Extract response
                content = data["content"][0]["text"]
                usage = data.get("usage", {})
                
                # Calculate cost
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                cost = self._calculate_cost(input_tokens, output_tokens, self.model)
                
                self.total_cost += cost
                
                return AgentResult.success_result(
                    data=content,
                    duration=0.0,
                    cost=cost,
                    metadata={
                        "model": self.model,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "stop_reason": data.get("stop_reason")
                    }
                )
                
        except Exception as e:
            return AgentResult.error_result(str(e), 0.0)
    
    async def _generate_code(
        self,
        prompt: str,
        language: str = "python",
        **kwargs
    ) -> AgentResult:
        """Generate code using Claude."""
        system_prompt = f"""You are an expert {language} programmer. 
Generate clean, efficient, well-documented code that follows best practices.
Include appropriate error handling and type hints where applicable.
Only return the code, no explanations unless specifically requested."""
        
        return await self._generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs
        )
    
    async def _analyze_text(
        self,
        text: str,
        analysis_type: str = "general",
        **kwargs
    ) -> AgentResult:
        """Analyze text using Claude."""
        system_prompt = f"""You are an expert analyst with deep knowledge across multiple domains.
Perform a comprehensive {analysis_type} analysis of the provided text.
Be thorough, objective, and provide actionable insights.
Structure your analysis clearly with headings and bullet points."""
        
        return await self._generate_text(
            prompt=f"Analyze this text:\n\n{text}",
            system_prompt=system_prompt,
            **kwargs
        )
    
    async def _summarize_text(
        self,
        text: str,
        max_length: int = 200,
        style: str = "concise",
        **kwargs
    ) -> AgentResult:
        """Summarize text using Claude."""
        system_prompt = f"""You are an expert at creating {style}, accurate summaries.
Create a summary of no more than {max_length} words that captures the essential points.
Maintain the original tone and key insights while being concise."""
        
        return await self._generate_text(
            prompt=f"Summarize this text:\n\n{text}",
            system_prompt=system_prompt,
            max_tokens=max_length * 2,
            **kwargs
        )
    
    async def _health_check(self) -> None:
        """Perform Anthropic health check."""
        try:
            # Simple test message
            result = await self._generate_text(
                prompt="Hello, this is a health check.",
                max_tokens=10
            )
            if not result.success:
                raise AgentError(f"Health check failed: {result.error}")
        except Exception as e:
            raise AgentError(f"Health check failed: {e}")