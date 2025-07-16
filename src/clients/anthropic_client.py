import os
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Claude API client for LLM interactions"""

    def __init__(self, model: Optional[str] = None, temperature: float = 0.2, max_tokens: int = 4000):
        """
        Initialize the LLM client with Claude API

        Args:
            model: Claude model to use (defaults to CLAUDE_MODEL env var or claude-3-opus-20240229)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please set it in your .env file.")

        self.model = model or os.getenv("CLAUDE_MODEL", "claude-3-opus-20240229")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key)

    def complete(self, prompt: str) -> str:
        """Send prompt to Claude API and return response"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Error calling Claude API: {e!s}")

    def complete_with_system_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Send prompt to Claude API with a system prompt and return response"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Error calling Claude API: {e!s}")
