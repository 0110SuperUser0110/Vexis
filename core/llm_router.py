from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class LLMResult:
    success: bool
    text: str
    model_name: str
    prompt: str
    raw_output: str = ""
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMRouter:
    def __init__(
        self,
        model_name: str = "qwen3-8b-local",
        base_url: str = "http://127.0.0.1:8080",
        timeout_seconds: int = 20,
        temperature: float = 0.05,
        max_tokens: int = 80,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens

    def is_available(self) -> bool:
        try:
            req = Request(
                url=f"{self.base_url}/v1/models",
                method="GET",
            )
            with urlopen(req, timeout=2) as response:
                return response.status == 200
        except Exception:
            return False

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResult:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        raw_text = ""
        try:
            req = Request(
                url=f"{self.base_url}/v1/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urlopen(req, timeout=self.timeout_seconds) as response:
                raw_text = response.read().decode("utf-8", errors="replace")

            payload = json.loads(raw_text)
            text = (
                payload.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            cleaned = self._clean_text(text)

            return LLMResult(
                success=True,
                text=cleaned,
                model_name=self.model_name,
                prompt=prompt,
                raw_output=raw_text,
            )

        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            return LLMResult(
                success=False,
                text="",
                model_name=self.model_name,
                prompt=prompt,
                raw_output=detail,
                error=f"http error {exc.code}",
            )
        except URLError as exc:
            return LLMResult(
                success=False,
                text="",
                model_name=self.model_name,
                prompt=prompt,
                raw_output=raw_text,
                error=f"connection error: {exc}",
            )
        except Exception as exc:
            return LLMResult(
                success=False,
                text="",
                model_name=self.model_name,
                prompt=prompt,
                raw_output=raw_text,
                error=str(exc),
            )

    def _clean_text(self, text: str) -> str:
        cleaned = (text or "").strip()

        banned_prefixes = (
            "[system]",
            "[user]",
            "[final]",
            "answer:",
            "response:",
            "final answer:",
            "social subtype:",
            "base reply meaning:",
        )
        lowered = cleaned.lower()
        if any(lowered.startswith(prefix) for prefix in banned_prefixes):
            return ""

        return cleaned