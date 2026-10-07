"""Provider clients, tested with the network mocked: request shape, parsing, errors and cost."""

from types import SimpleNamespace

import pytest
import requests

from askdata.llm import AnthropicLLM, LLMError, OllamaLLM, OracleLLM, strip_think


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status_code, payload or {}, text

    def json(self):
        return self._payload


def test_ollama_request_and_parsing(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(url=url, json=json)
        return FakeResponse(payload={
            "message": {"content": "<think>hmm</think>```sql\nSELECT 1\n```"},
            "prompt_eval_count": 120,
            "eval_count": 9,
        })

    monkeypatch.setattr("askdata.llm.requests.post", fake_post)
    out = OllamaLLM("qwen3:8b", "http://localhost:11434").complete("system text", [{"role": "user", "content": "q"}])
    assert sent["url"] == "http://localhost:11434/api/chat"
    assert sent["json"]["messages"][0] == {"role": "system", "content": "system text"}
    assert sent["json"]["think"] is False and sent["json"]["options"]["temperature"] == 0
    assert out.text == "```sql\nSELECT 1\n```" and out.input_tokens == 120 and out.cost_usd == 0


def test_ollama_non_thinking_model_gets_no_think_flag(monkeypatch):
    sent = {}
    monkeypatch.setattr(
        "askdata.llm.requests.post",
        lambda url, json, timeout: sent.update(json=json) or FakeResponse(payload={"message": {"content": "x"}}),
    )
    OllamaLLM("qwen2.5-coder:7b").complete("s", [])
    assert "think" not in sent["json"]


def test_ollama_errors_are_actionable(monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr("askdata.llm.requests.post", down)
    with pytest.raises(LLMError, match="ollama pull qwen3:8b"):
        OllamaLLM("qwen3:8b").complete("s", [])

    monkeypatch.setattr("askdata.llm.requests.post", lambda *a, **k: FakeResponse(404, text="not found"))
    with pytest.raises(LLMError, match="not installed"):
        OllamaLLM("qwen3:8b").complete("s", [])


def _claude_reply(stop_reason="end_turn", text="```sql\nSELECT 1\n```"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        model="claude-opus-5-5",
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200, cache_creation_input_tokens=0, cache_read_input_tokens=5000),
    )


def test_claude_request_shape_and_cost(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm = AnthropicLLM("claude-opus-5-5", effort="medium")
    sent = {}
    monkeypatch.setattr(llm.client.beta.messages, "create", lambda **kw: sent.update(kw) or _claude_reply())
    out = llm.complete("system text", [{"role": "user", "content": "q"}])
    assert sent["model"] == "claude-opus-5-5"
    assert sent["fallbacks"] == "default" and sent["betas"] == ["server-side-fallback-2026-07-01"]
    assert sent["output_config"] == {"effort": "medium"}
    assert sent["system"] == [{"type": "text", "text": "system text", "cache_control": {"type": "ephemeral"}}]
    assert out.text == "```sql\nSELECT 1\n```"
    assert out.cost_usd == pytest.approx((1000 * 4.00 + 5000 * 0.20 + 200 * 20.00) / 1e6)


def test_claude_refusal_is_an_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm = AnthropicLLM()
    monkeypatch.setattr(llm.client.beta.messages, "create", lambda **kw: _claude_reply("refusal", ""))
    with pytest.raises(LLMError, match="declined"):
        llm.complete("s", [{"role": "user", "content": "q"}])


def test_oracle_reads_the_question_line():
    oracle = OracleLLM({"How many patients?": "SELECT COUNT(*) FROM patients"})
    prompt = "Examples...\nQ: other\nSQL: SELECT 1\n\nQuestion: how  many patients?"
    assert "SELECT COUNT(*) FROM patients" in oracle.complete("s", [{"role": "user", "content": prompt}]).text
    assert oracle.complete("s", [{"role": "user", "content": "Question: unknown"}]).text.startswith("NO_SQL")
    assert oracle.complete("s", [], purpose="answer").text == ""


def test_strip_think():
    assert strip_think("<think>\nlong reasoning\n</think>\nanswer") == "answer"
