import os
from typing import Optional

import openai
from dotenv import load_dotenv

load_dotenv()


class GPTClient:
    """OpenAI GPT API client for LLM interactions"""

    def __init__(self, model: Optional[str] = None, temperature: float = 0, max_tokens: int = 20000):
        """
        Initialize the GPT client with OpenAI API

        Args:
            model: GPT model to use (defaults to GPT_MODEL env var or gpt-4o)
            max_tokens: Maximum tokens in response
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

        self.model = model or os.getenv("GPT_MODEL", "gpt-4o")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(api_key=api_key)

    def complete(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Send prompt to GPT API and return response"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": """
                            You are an AI assistant that is an expert in legal documents and legal analysis. You must always respond in JSON loadable format.
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
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error calling GPT API: {e!s}")

    def complete_with_system_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send prompt to GPT API with a system prompt and return response"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error calling GPT API: {e!s}")

    def complete_with_messages(
        self, messages: list, temperature: Optional[float] = None, max_tokens: Optional[int] = None
    ) -> str:
        """Send a list of messages to GPT API and return response"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error calling GPT API: {e!s}")
