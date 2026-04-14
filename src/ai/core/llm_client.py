import os
from typing import Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from ai.core.token_usage import extract_token_usage, TokenUsage

load_dotenv()


class LLMClient:
    """Simple client for LLM calls through LangChain Google GenAI."""

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: int = 2,
        api_key: Optional[str] = None,
        **extra,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.last_token_usage: TokenUsage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
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

    def get_last_token_usage(self) -> TokenUsage:
        return dict(self.last_token_usage)

    def send_prompt(self, prompt: str, **kwargs) -> str:
        prompt_preview = prompt[:180].replace("\n", " ")

        response = self.llm.invoke(prompt, **kwargs)
        try:
            self.last_token_usage = extract_token_usage(response)
        except Exception:
            self.last_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if hasattr(response, "content"):
            content = response.content
            resp_preview = str(content)[:180].replace("\n", " ")
            return content

        resp_preview = str(response)[:180].replace("\n", " ")
        return response
