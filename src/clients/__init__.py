import os

from .anthropic_client import AnthropicClient
from .gpt_client import GPTClient


class LLMClient:
    def __init__(self, temperature: float = 0):
        self.temperature = temperature
        self.model_family = os.getenv("MODEL_FAMILY", "gpt")
        self.client = self._initialize_client()

    def _initialize_client(self):
        if "gpt" in self.model_family.lower():
            self.model = os.getenv("OPENAI_MODEL_NAME", "gpt-4.5")
            return GPTClient(model=self.model, temperature=self.temperature)
        elif "claude" in self.model_family.lower():
            self.model = os.getenv("CLAUDE_MODEL_NAME", "claude-4-opus")
            return AnthropicClient(model=self.model, temperature=self.temperature)
        else:
            raise ValueError(f"Model {self.model_family} not supported")

    def complete(self, prompt: str, temperature: float = 0) -> str:
        return self.client.complete(prompt, temperature)

    def complete_with_system_prompt(self, system_prompt: str, user_prompt: str, temperature: float = 0) -> str:
        return self.client.complete_with_system_prompt(system_prompt, user_prompt, temperature)

    def complete_with_messages(self, prompt: str, temperature: float = 0) -> str:
        return self.client.complete_with_messages(prompt, temperature)


__all__ = ["LLMClient"]
