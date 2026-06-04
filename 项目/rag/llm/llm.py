from __future__ import annotations

from config.config import (
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)


class LLMService:
    def __init__(self):
        self.llm = None
        if OPENAI_API_KEY:
            from langchain_openai import ChatOpenAI

            kwargs = {
                "api_key": OPENAI_API_KEY,
                "model": LLM_MODEL,
                "temperature": LLM_TEMPERATURE,
                "timeout": LLM_TIMEOUT,
                "max_retries": LLM_MAX_RETRIES,
            }
            if OPENAI_BASE_URL:
                kwargs["base_url"] = OPENAI_BASE_URL
            self.llm = ChatOpenAI(**kwargs)

    @property
    def available(self) -> bool:
        return self.llm is not None

    def generate(self, prompt):
        if self.llm is None:
            raise RuntimeError(
                "OPENAI_API_KEY is empty. Retrieval is available, but answer "
                "generation needs an LLM key or a local LLM adapter."
            )
        response = self.llm.invoke(prompt)
        return response.content
