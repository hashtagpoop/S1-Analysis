"""
Usage:
1. Make sure you have an openai or anthropic account
2. Create a new API key: https://platform.openai.com/api-keys or https://console.anthropic.com/settings/keys
3. Add the API key to the .env file (create if not exists) as OPENAI_API_KEY or ANTHROPIC_API_KEY
4. Run the script: `python llm.py`

Import:
1. `from llm import LLM`
2. `llm = LLM.create(provider_type="openai", api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")`
3. `response = llm.complete("Break down this share count in CSV form and timelines (by day) of expiry: 1000 shares of common stock locked up for 3 months, 5000 shares expiring for directors in 90 days, 5000 shares being offered to employees in 1 year")`
"""

import os
from abc import ABC, abstractmethod
from typing import Optional
from openai import OpenAI
from openai.types.chat import ChatCompletion
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion for the given prompt"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI implementation of LLM provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.client: OpenAI = OpenAI(api_key=api_key)
        self.model: str = model

    def complete(self, prompt: str, **kwargs) -> str:
        response: ChatCompletion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

class AnthropicProvider(LLMProvider):
    """Anthropic implementation of LLM provider"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet"):
        self.client: Anthropic = Anthropic(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content[0].text

"""
LLM is implemented as a factory method so that you can create various
LLM instances based on providers and different parameters you care about.

In the future, this can be extended to making each type of LLM call more
configurable as well.

It's a conscious decision not to use Langchain or related libraries because
of the cruft and weird abstractions. Let's keep this simple and add the various
feature support as needed using the raw APIs from OpenAI and Anthropic.
"""
class LLM:
    """Main LLM class that handles different providers"""
    
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate a completion for the given prompt
        
        Args:
            prompt (str): The input prompt
            **kwargs: Additional arguments to pass to the provider
            
        Returns:
            str: The generated completion
        """
        return self.provider.complete(prompt, **kwargs)

    @classmethod
    def create(cls, 
               provider_type: str,
               api_key: str,
               model: Optional[str] = None) -> 'LLM':
        """
        Factory method to create an LLM instance with the specified provider
        
        Args:
            provider_type (str): Type of provider ("openai" or "anthropic")
            api_key (str): API key for the provider
            model (Optional[str]): Model to use, defaults to provider's default
            
        Returns:
            LLM: Configured LLM instance
        """
        if provider_type.lower() == "openai":
            provider = OpenAIProvider(api_key, model) if model else OpenAIProvider(api_key)
        elif provider_type.lower() == "anthropic":
            provider = AnthropicProvider(api_key, model) if model else AnthropicProvider(api_key)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
        
        return cls(provider)

if __name__ == "__main__":
    openai_llm = LLM.create(provider_type="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini")
    response = openai_llm.complete("Break down this share count in CSV form and timelines (by day) of expiry: 1000 shares of common stock locked up for 3 months, 5000 shares expiring for directors in 90 days, 5000 shares being offered to employees in 1 year")
    print(response)