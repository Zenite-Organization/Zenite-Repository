# src/core/llm_client.py
from typing import Dict, Any, Optional

class LLMClient:
    """
    Abstração para chamadas ao seu modelo GenAI.
    Substitua send_prompt por integração com OpenAI, Google GenAI, Anthropic, etc.
    """

    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        # TODO: inicializar SDK (chaves, config) aqui

    def send_prompt(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        """
        Envia o prompt e retorna o texto bruto da resposta.
        Substitua o conteúdo deste método para usar a API que você escolher.
        """
        # TODO: integrar com a API do provedor
        raise NotImplementedError("Implemente a ligação com seu provedor de GenAI aqui.")
