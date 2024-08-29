from typing import Protocol


class LLM(Protocol):
    def infer_completion(self, prompt: str) -> str: ...
