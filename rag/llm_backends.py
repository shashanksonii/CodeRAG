import os
from abc import ABC, abstractmethod
import ollama
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

class LLMBackend(ABC):
    """
    Abstract interface for LLM backends.
    """
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generates a text completion given a system prompt and a user prompt.
        """
        pass

class OllamaBackend(LLMBackend):
    """
    Local Ollama backend wrapper. Uses the official python 'ollama' client for stability
    and error handling.
    """
    def __init__(self, model: str = "llama3.1", base_url: str = None):
        self.model = model
        self.base_url = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = ollama.Client(host=self.base_url)

    def _handle_error(self, e: Exception):
        raise RuntimeError(
            "Ollama connection failed. Please ensure Ollama is running and the models are pulled.\n\n"
            "Instructions:\n"
            "1. Run `ollama serve` in a terminal window.\n"
            "2. Pull the required models:\n"
            "   `ollama pull nomic-embed-text`\n"
            "   `ollama pull llama3.1`"
        ) from e

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            
            response = self.client.chat(model=self.model, messages=messages)
            return response["message"]["content"]
        except Exception as e:
            self._handle_error(e)

class ClaudeBackend(LLMBackend):
    """
    Anthropic Claude backend wrapper using ChatAnthropic.
    """
    def __init__(self, model: str = "claude-3-5-sonnet-latest"):
        self.model = model
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set.\n\n"
                "Instructions:\n"
                "Please configure ANTHROPIC_API_KEY in your .env file or export it in your environment."
            )
        self.llm = ChatAnthropic(
            model=self.model, 
            anthropic_api_key=self.api_key,
            temperature=0.0
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            raise RuntimeError(f"Claude API call failed: {e}") from e
