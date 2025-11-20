from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI


class LLMClient:
    """
    Cliente simples para interagir com um LLM via LangChain Google GenAI.
    """
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: int = 2,
        api_key: Optional[str] = None,
        **extra,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            os.environ.setdefault("GEMINI_API_KEY", self.api_key)

        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            **extra,
        )

    def send_prompt(self, prompt: str, **kwargs) -> str:
        response = self.llm.invoke(prompt, **kwargs)
        # Se for AIMessage, pega o conteúdo de texto
        if hasattr(response, 'content'):
            return response.content
        return response
