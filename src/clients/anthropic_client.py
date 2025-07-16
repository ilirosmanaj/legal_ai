import os
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()


class AnthropicClient:
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

        self.model = model or os.getenv("CLAUDE_MODEL_NAME", "claude-opus-4-20250514")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key)

    def complete(self, prompt: str, temperature: float = 0) -> str:
        """Send prompt to Claude API and return response"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an AI assistant that is an expert in legal documents and legal analysis. You must always respond in JSON loadable format.
                                Return your findings as **valid, raw JSON** using this schema:
                                    - Do **not** include any escaped characters (`\\`)
                                    - Use plain double quotes where needed
                                    - Ensure all keys and values are properly quoted
                                    - No Python-style booleans or enums (use `true`/`false`)
                                    - `null` instead of `None`
                            """,
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Error calling Claude API: {e!s}")

    def complete_with_system_prompt(
        self, system_prompt: str, user_prompt: str, temperature: float = 0, max_tokens: int = 4000
    ) -> str:
        """Send prompt to Claude API with a system prompt and return response"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Error calling Claude API: {e!s}")
