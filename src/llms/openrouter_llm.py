# OpenRouter LLM wrapper
# src/llms/openrouter_llm.py

"""
OpenRouter LLM integration for LangChain with Gemma 3 12B free model.
"""

import logging
import requests
from typing import Any, List, Mapping, Optional
import json

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import Field

from config.settings import settings

logger = logging.getLogger(__name__)

class OpenRouterLLM(LLM):
    """
    OpenRouter LLM wrapper for LangChain integration.
    Provides access to OpenRouter models through LangChain interface.
    """
    
    openrouter_api_key: str = Field(default="")
    model_name: str = Field(default="google/gemma-3-12b-it:free")
    base_url: str = Field(default="https://openrouter.ai/api/v1")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=4000)
    
    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize OpenRouter LLM.
        
        Args:
            openrouter_api_key: OpenRouter API key
            model_name: Model to use (default: Gemma 3 12B free)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__(**kwargs)
        
        self.openrouter_api_key = openrouter_api_key or settings.OPENROUTER_API_KEY
        self.model_name = model_name or settings.OPENROUTER_MODEL
        self.temperature = temperature if temperature is not None else settings.TEMPERATURE
        self.max_tokens = max_tokens or settings.MAX_TOKENS
        
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key is required")
        
        logger.info(f"Initialized OpenRouter LLM with model: {self.model_name}")
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM."""
        return "openrouter"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url,
        }
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call the OpenRouter API.
        
        Args:
            prompt: The prompt to generate from
            stop: Stop sequences
            run_manager: Callback manager
            **kwargs: Additional keyword arguments
            
        Returns:
            Generated text response
        """
        try:
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/langchain-ai/langchain",
                "X-Title": "LangChain PDF Chunker"
            }
            
            # Prepare payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            # Add stop sequences if provided
            if stop:
                payload["stop"] = stop
            
            # Add any additional parameters
            payload.update(kwargs)
            
            # Make the API call
            url = f"{self.base_url}/chat/completions"
            
            logger.debug(f"Making request to {url} with model {self.model_name}")
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            if "choices" not in response_data or len(response_data["choices"]) == 0:
                raise ValueError("No choices returned from OpenRouter API")
            
            content = response_data["choices"][0]["message"]["content"]
            
            # Log usage if available
            if "usage" in response_data:
                usage = response_data["usage"]
                logger.debug(
                    f"OpenRouter usage - "
                    f"Prompt tokens: {usage.get('prompt_tokens', 'N/A')}, "
                    f"Completion tokens: {usage.get('completion_tokens', 'N/A')}, "
                    f"Total tokens: {usage.get('total_tokens', 'N/A')}"
                )
            
            return content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error calling OpenRouter API: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling OpenRouter API: {str(e)}")
            raise
    
    def generate_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """
        Generate text with retry logic.
        
        Args:
            prompt: The prompt to generate from
            max_retries: Maximum number of retries
            **kwargs: Additional arguments
            
        Returns:
            Generated text
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return self._call(prompt, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        raise last_error
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Async version of _call.
        
        Note: This is a simple wrapper around the sync version.
        For true async, you'd want to use aiohttp or similar.
        """
        import asyncio
        
        # Run the sync version in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._call, 
            prompt, 
            stop, 
            run_manager, 
            **kwargs
        )
    
    def validate_environment(self) -> bool:
        """
        Validate that the environment is properly configured.
        
        Returns:
            True if valid, raises exception otherwise
        """
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        
        # Test API connectivity
        try:
            test_response = self._call("Hello", max_tokens=10)
            logger.info("OpenRouter API validation successful")
            return True
        except Exception as e:
            logger.error(f"OpenRouter API validation failed: {str(e)}")
            raise

class OpenRouterChatLLM(OpenRouterLLM):
    """
    Chat-optimized version of OpenRouter LLM.
    Handles conversation history and system messages.
    """
    
    def __init__(self, system_message: Optional[str] = None, **kwargs):
        """
        Initialize chat LLM.
        
        Args:
            system_message: System message to prepend to conversations
            **kwargs: Additional arguments for OpenRouterLLM
        """
        super().__init__(**kwargs)
        self.system_message = system_message
        self.conversation_history = []
    
    def chat(
        self,
        message: str,
        reset_history: bool = False,
        **kwargs
    ) -> str:
        """
        Have a conversation with the model.
        
        Args:
            message: User message
            reset_history: Whether to reset conversation history
            **kwargs: Additional arguments
            
        Returns:
            Model response
        """
        if reset_history:
            self.conversation_history = []
        
        # Build conversation prompt
        prompt_parts = []
        
        if self.system_message:
            prompt_parts.append(f"System: {self.system_message}")
        
        # Add conversation history
        for entry in self.conversation_history:
            prompt_parts.append(f"User: {entry['user']}")
            prompt_parts.append(f"Assistant: {entry['assistant']}")
        
        # Add current message
        prompt_parts.append(f"User: {message}")
        prompt_parts.append("Assistant:")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        # Generate response
        response = self._call(full_prompt, **kwargs)
        
        # Update conversation history
        self.conversation_history.append({
            "user": message,
            "assistant": response
        })
        
        return response

# Convenience function for creating OpenRouter LLM
def create_openrouter_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs
) -> OpenRouterLLM:
    """
    Create an OpenRouter LLM instance with default settings.
    
    Args:
        model_name: Model to use
        temperature: Sampling temperature
        **kwargs: Additional arguments
        
    Returns:
        OpenRouterLLM instance
    """
    return OpenRouterLLM(
        model_name=model_name,
        temperature=temperature,
        **kwargs
    )