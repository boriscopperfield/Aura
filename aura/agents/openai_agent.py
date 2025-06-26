"""
OpenAI Agent implementation.
"""
import json
import os
from typing import Any, Dict, List, Optional
import aiohttp

from .base import Agent, AgentCapability, AgentResult, RateLimit
from ..utils.errors import AgentError


class OpenAIAgent(Agent):
    """OpenAI API integration."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        base_url: str = "https://api.openai.com/v1",
        **kwargs
    ):
        super().__init__(
            name="openai",
            capabilities=[
                AgentCapability.TEXT_GENERATION,
                AgentCapability.CODE_GENERATION,
                AgentCapability.ANALYSIS,
                AgentCapability.SUMMARIZATION,
                AgentCapability.IMAGE_GENERATION,
                AgentCapability.IMAGE_ANALYSIS
            ],
            rate_limit=RateLimit(
                requests_per_minute=60,
                tokens_per_minute=150000
            ),
            **kwargs
        )
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise AgentError("OpenAI API key is required")
        
        self.model = model
        self.base_url = base_url
        
        # Pricing per 1K tokens (as of 2024)
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "dall-e-3": {"1024x1024": 0.04, "1792x1024": 0.08, "1024x1792": 0.08}
        }
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars = 1 token)."""
        return len(text) // 4
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate API cost."""
        if model not in self.pricing:
            return 0.0
        
        pricing = self.pricing[model]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    async def _execute(self, capability: AgentCapability, **kwargs) -> AgentResult:
        """Execute OpenAI capability."""
        if capability == AgentCapability.TEXT_GENERATION:
            return await self._generate_text(**kwargs)
        elif capability == AgentCapability.CODE_GENERATION:
            return await self._generate_code(**kwargs)
        elif capability == AgentCapability.ANALYSIS:
            return await self._analyze_text(**kwargs)
        elif capability == AgentCapability.SUMMARIZATION:
            return await self._summarize_text(**kwargs)
        elif capability == AgentCapability.IMAGE_GENERATION:
            return await self._generate_image(**kwargs)
        elif capability == AgentCapability.IMAGE_ANALYSIS:
            return await self._analyze_image(**kwargs)
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
        """Generate text using OpenAI."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
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
        
        try:
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return AgentResult.error_result(
                        f"OpenAI API error: {response.status} - {error_text}",
                        0.0
                    )
                
                data = await response.json()
                
                # Extract response
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                
                # Calculate cost
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                cost = self._calculate_cost(input_tokens, output_tokens, self.model)
                
                self.total_cost += cost
                
                return AgentResult.success_result(
                    data=content,
                    duration=0.0,  # Will be set by execute_with_retry
                    cost=cost,
                    metadata={
                        "model": self.model,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": usage.get("total_tokens", 0)
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
        """Generate code using OpenAI."""
        system_prompt = f"""You are an expert {language} programmer. 
Generate clean, efficient, well-documented code that follows best practices.
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
        """Analyze text using OpenAI."""
        system_prompt = f"""You are an expert analyst. 
Perform a {analysis_type} analysis of the provided text.
Be thorough, objective, and provide actionable insights."""
        
        return await self._generate_text(
            prompt=f"Analyze this text:\n\n{text}",
            system_prompt=system_prompt,
            **kwargs
        )
    
    async def _summarize_text(
        self,
        text: str,
        max_length: int = 200,
        **kwargs
    ) -> AgentResult:
        """Summarize text using OpenAI."""
        system_prompt = f"""You are an expert at creating concise, accurate summaries.
Create a summary of no more than {max_length} words that captures the key points."""
        
        return await self._generate_text(
            prompt=f"Summarize this text:\n\n{text}",
            system_prompt=system_prompt,
            max_tokens=max_length * 2,  # Rough token estimate
            **kwargs
        )
    
    async def _generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        **kwargs
    ) -> AgentResult:
        """Generate image using DALL-E."""
        payload = {
            "model": "dall-e-3",
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "n": 1
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/images/generations",
                headers=self._get_headers(),
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return AgentResult.error_result(
                        f"OpenAI API error: {response.status} - {error_text}",
                        0.0
                    )
                
                data = await response.json()
                image_url = data["data"][0]["url"]
                
                # Calculate cost
                cost = self.pricing["dall-e-3"].get(size, 0.04)
                self.total_cost += cost
                
                return AgentResult.success_result(
                    data=image_url,
                    duration=0.0,
                    cost=cost,
                    metadata={
                        "model": "dall-e-3",
                        "size": size,
                        "quality": quality,
                        "prompt": prompt
                    }
                )
                
        except Exception as e:
            return AgentResult.error_result(str(e), 0.0)
    
    async def _analyze_image(
        self,
        image_url: str,
        prompt: str = "Describe this image in detail",
        **kwargs
    ) -> AgentResult:
        """Analyze image using GPT-4 Vision."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
        
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": messages,
            "max_tokens": 1000
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return AgentResult.error_result(
                        f"OpenAI API error: {response.status} - {error_text}",
                        0.0
                    )
                
                data = await response.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                
                # Calculate cost (vision model pricing)
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                cost = self._calculate_cost(input_tokens, output_tokens, "gpt-4")
                
                self.total_cost += cost
                
                return AgentResult.success_result(
                    data=content,
                    duration=0.0,
                    cost=cost,
                    metadata={
                        "model": "gpt-4-vision-preview",
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "image_url": image_url
                    }
                )
                
        except Exception as e:
            return AgentResult.error_result(str(e), 0.0)
    
    async def _health_check(self) -> None:
        """Perform OpenAI health check."""
        try:
            async with self.session.get(
                f"{self.base_url}/models",
                headers=self._get_headers()
            ) as response:
                if response.status != 200:
                    raise AgentError(f"Health check failed: {response.status}")
        except Exception as e:
            raise AgentError(f"Health check failed: {e}")