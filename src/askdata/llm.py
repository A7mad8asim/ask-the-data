"""LLM providers behind one small interface.

- OllamaLLM (default): a local model such as Qwen3-8B. No health data leaves the machine.
- AnthropicLLM (optional): Claude via the API, used to measure the quality gap.
- OracleLLM: replays the gold SQL for known evaluation questions. It lets the tests and
  the evaluation harness run end to end without any model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import requests

from .config import Settings


class LLMError(Exception):
    """The model could not be reached or did not return a usable response."""


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class LLM(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict], purpose: str = "sql") -> LLMResponse:
        """`purpose` is "sql" (write a query) or "answer" (write the reply)."""
        ...


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    return _THINK.sub("", text or "").strip()


# ---------------------------------------------------------------------------- Ollama (local)


class OllamaLLM:
    THINKING_FAMILIES = ("qwen3", "deepseek-r1", "qwq", "magistral")

    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout_s: float = 300, num_ctx: int = 8192):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.num_ctx = num_ctx
        self.name = f"ollama:{model}"

    def _post(self, path: str, payload: dict) -> dict:
        try:
            r = requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout_s)
        except requests.ConnectionError as e:
            raise LLMError(
                f"Cannot reach Ollama at {self.base_url}. Install it from https://ollama.com, "
                f"start it, then run:  ollama pull {payload.get('model')}"
            ) from e
        except requests.Timeout as e:
            raise LLMError(f"Ollama did not answer within {self.timeout_s:g} seconds.") from e
        if r.status_code == 404:
            raise LLMError(f"Model '{payload.get('model')}' is not installed in Ollama. Run:  ollama pull {payload.get('model')}")
        if r.status_code >= 400:
            raise LLMError(f"Ollama error {r.status_code}: {r.text[:300]}")
        return r.json()

    def complete(self, system: str, messages: list[dict], purpose: str = "sql") -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "seed": 7, "num_ctx": self.num_ctx},
        }
        if self.model.lower().startswith(self.THINKING_FAMILIES):
            payload["think"] = False  # faster, and the SQL does not need visible reasoning
        data = self._post("/api/chat", payload)
        return LLMResponse(
            text=strip_think(data.get("message", {}).get("content", "")),
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
        )

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        data = self._post("/api/embed", {"model": model, "input": texts})
        return data["embeddings"]


# ---------------------------------------------------------------------------- Claude API (optional)

# USD per million tokens: (input, output, cache read). Cache writes cost 1.25x input.
CLAUDE_PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
    "claude-fable-5-1": (10.00, 50.00, 0.25),
}


class AnthropicLLM:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", max_tokens: int = 16000):
        import anthropic  # imported here so the local-only setup never needs it at import time

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()  # credentials from ANTHROPIC_API_KEY or `ant auth login`
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"

    def complete(self, system: str, messages: list[dict], purpose: str = "sql") -> LLMResponse:
        anthropic = self._anthropic
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",  # if the model declines, the API retries on its recommended fallback
                output_config={"effort": self.effort},
                # The system prompt (schema + glossary + rules) is identical for every question: cache it.
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=messages,
            )
        except anthropic.AuthenticationError as e:
            raise LLMError("Claude API: authentication failed. Set ANTHROPIC_API_KEY or run `ant auth login`.") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Claude API: rate limited. Wait a moment and try again.") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Claude API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Cannot reach the Claude API. Check the network connection.") from e
        except anthropic.AnthropicError as e:
            raise LLMError(f"Claude API: {e}") from e

        if resp.stop_reason == "refusal":
            raise LLMError("Claude declined this request.")
        if resp.stop_reason == "max_tokens":
            raise LLMError("Claude's response was cut off before it finished.")
        text = "".join(block.text for block in resp.content if block.type == "text")
        u = resp.usage
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        price_in, price_out, price_cache = CLAUDE_PRICES.get(resp.model, CLAUDE_PRICES.get(self.model, (0, 0, 0)))
        cost = (
            u.input_tokens * price_in + cache_write * price_in * 1.25 + cache_read * price_cache + u.output_tokens * price_out
        ) / 1e6
        return LLMResponse(
            text=strip_think(text),
            input_tokens=u.input_tokens + cache_write + cache_read,
            output_tokens=u.output_tokens,
            cost_usd=cost,
        )


# ---------------------------------------------------------------------------- Oracle (testing)


def normalize_question(q: str) -> str:
    return " ".join((q or "").split()).casefold()


class OracleLLM:
    """Answers evaluation questions with their gold SQL. For testing the pipeline and the harness."""

    name = "oracle:gold-sql"

    def __init__(self, gold_sql: dict[str, str]):
        self.gold = {normalize_question(q): sql for q, sql in gold_sql.items()}

    def complete(self, system: str, messages: list[dict], purpose: str = "sql") -> LLMResponse:
        if purpose != "sql":
            return LLMResponse("")  # the pipeline falls back to its plain, number-safe summary
        question = None
        for m in messages:
            if m["role"] == "user":
                for line in str(m["content"]).splitlines():
                    if line.startswith("Question:"):
                        question = line[len("Question:"):].strip()
                break
        sql = self.gold.get(normalize_question(question or ""))
        if sql is None:
            return LLMResponse("NO_SQL: this question is not in the gold set.")
        return LLMResponse(f"```sql\n{sql}\n```")


def make_llm(settings: Settings) -> LLM:
    provider = settings.llm_provider
    if provider == "ollama":
        return OllamaLLM(settings.llm_model, settings.ollama_base_url)
    if provider == "anthropic":
        return AnthropicLLM(settings.llm_model, settings.anthropic_effort)
    if provider == "oracle":
        from .resources import load_gold

        return OracleLLM({item["question"]: item["sql"] for item in load_gold()})
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use ollama, anthropic or oracle.")
