"""The question -> SQL -> checked result -> answer pipeline.

    question
      -> retrieve similar examples
      -> LLM writes SQL                      (the model never calculates numbers)
      -> sandbox checks the SQL              (rules enforced in code)
      -> DuckDB runs it, read-only           (timeout, row cap)
      -> small-group suppression             (groups under 10 patients masked)
      -> LLM words the answer from the rows  (every number checked against the table)
    If the SQL is rejected or fails, the error goes back to the model, at most `max_repairs` times.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from .answers import (
    check_grounding,
    detect_language,
    extract_sql,
    is_no_sql,
    markdown_table,
    plain_summary,
    suggest_chart,
)
from .config import Settings
from .db import Database, QueryError
from .llm import LLM, LLMError, LLMResponse, OllamaLLM, make_llm
from .privacy import display_frame, run_protected
from .prompts import answer_system_prompt, answer_user_prompt, repair_prompt, sql_system_prompt, sql_user_prompt
from .resources import load_examples
from .retrieval import ExampleRetriever
from .sandbox import check_sql


@dataclass
class Attempt:
    sql: str | None
    error: str | None
    stage: str  # "ok", "model" (no usable SQL), "sandbox" or "database"


@dataclass
class Answer:
    question: str
    language: str
    status: str = "error"  # "ok", "refused" or "error"
    sql: str | None = None
    attempts: list[Attempt] = field(default_factory=list)
    table: pd.DataFrame | None = None  # result with masked cells as NA
    masked: pd.DataFrame | None = None
    raw: pd.DataFrame | None = None  # unmasked result: used only by the evaluation harness
    suppressed_rows: int = 0
    suppression_method: str = "none"
    truncated: bool = False
    text: str = ""
    grounded: bool | None = None  # None when the model gave no wording to check
    unmatched_numbers: list[float] = field(default_factory=list)
    chart: dict | None = None
    message: str = ""
    examples_used: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def display_table(self) -> pd.DataFrame | None:
        if self.table is None:
            return None
        return display_frame(self.table, self.masked)

    def add_usage(self, resp: LLMResponse) -> None:
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        self.cost_usd += resp.cost_usd


REFUSAL = {
    "en": "This question can't be answered within the privacy and safety rules.",
    "ar": "لا يمكن الإجابة عن هذا السؤال ضمن قواعد الخصوصية والأمان.",
}
FAILURE = {
    "en": "I couldn't build a working query for this question.",
    "ar": "لم أتمكن من بناء استعلام صحيح لهذا السؤال.",
}


def build_retriever(settings: Settings) -> ExampleRetriever:
    examples = load_examples()
    if settings.embeddings in ("bge-m3", "ollama"):
        ollama = OllamaLLM(settings.llm_model, settings.ollama_base_url, timeout_s=60)
        return ExampleRetriever(examples, "bge-m3", lambda texts: ollama.embed(texts, settings.embedding_model))
    return ExampleRetriever(examples, "tfidf")


class Assistant:
    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLM | None = None,
        db: Database | None = None,
        retriever: ExampleRetriever | None = None,
        use_glossary: bool = True,
        use_examples: bool = True,
        max_repairs: int | None = None,
        write_answers: bool = True,
    ):
        self.settings = settings or Settings.from_env()
        self.db = db or Database(self.settings.db_path, self.settings.query_timeout_s, self.settings.max_rows)
        self.llm = llm or make_llm(self.settings)
        self.use_examples = use_examples
        self.retriever = (retriever or build_retriever(self.settings)) if use_examples else None
        self.max_repairs = self.settings.max_repairs if max_repairs is None else max_repairs
        self.write_answers = write_answers
        self.system_prompt = sql_system_prompt(include_glossary=use_glossary)

    # ------------------------------------------------------------------ public

    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        ans = Answer(question=question, language=detect_language(question))
        try:
            self._generate_and_run(ans)
        except LLMError as e:
            ans.status, ans.message = "error", str(e)
        if ans.status == "ok":
            self._word_answer(ans)
        ans.latency_s = time.perf_counter() - start
        return ans

    # ------------------------------------------------------------------ SQL

    def _generate_and_run(self, ans: Answer) -> None:
        examples = self.retriever.top_k(ans.question, self.settings.few_shot_k) if self.retriever else []
        ans.examples_used = [e["id"] for e in examples]
        messages = [{"role": "user", "content": sql_user_prompt(ans.question, examples)}]

        for _ in range(1 + self.max_repairs):
            resp = self.llm.complete(self.system_prompt, messages, purpose="sql")
            ans.add_usage(resp)
            reason = is_no_sql(resp.text)
            if reason:
                ans.attempts.append(Attempt(None, reason, "model"))
                ans.status = "refused"
                ans.message = f"{REFUSAL[ans.language]} ({reason})"
                return

            sql = extract_sql(resp.text)
            if not sql:
                error, stage = "No SQL query was found in the reply.", "model"
            else:
                verdict = check_sql(sql)
                if not verdict.ok:
                    error, stage = verdict.message, "sandbox"
                else:
                    try:
                        result = run_protected(self.db, sql, self.settings.min_group_size)
                    except QueryError as e:
                        error, stage = str(e), "database"
                    else:
                        ans.attempts.append(Attempt(sql, None, "ok"))
                        ans.status, ans.sql = "ok", sql
                        ans.table, ans.masked, ans.raw = result.df, result.masked, result.raw
                        ans.suppressed_rows, ans.suppression_method = result.suppressed_rows, result.method
                        ans.truncated = result.truncated
                        ans.chart = suggest_chart(result.df)
                        return
            ans.attempts.append(Attempt(sql, error, stage))
            messages += [
                {"role": "assistant", "content": resp.text or "(empty reply)"},
                {"role": "user", "content": repair_prompt(error)},
            ]

        last = ans.attempts[-1]
        if last.stage == "sandbox":
            ans.status = "refused"
            ans.message = f"{REFUSAL[ans.language]} {last.error}"
        else:
            ans.status = "error"
            ans.message = f"{FAILURE[ans.language]} {last.error}"

    # ------------------------------------------------------------------ answer

    def _word_answer(self, ans: Answer) -> None:
        shown = ans.display_table
        fallback = plain_summary(shown, ans.language, ans.suppressed_rows, ans.truncated)
        if not self.write_answers:
            ans.text = fallback
            return
        prompt = answer_user_prompt(
            ans.question, ans.sql or "", markdown_table(shown), len(shown), ans.truncated, ans.suppressed_rows
        )
        try:
            resp = self.llm.complete(
                answer_system_prompt(ans.language), [{"role": "user", "content": prompt}], purpose="answer"
            )
        except LLMError:
            ans.text = fallback
            return
        ans.add_usage(resp)
        text = (resp.text or "").strip()
        if not text:
            ans.text = fallback
            return
        grounded, unmatched = check_grounding(
            text, ans.table, ans.question, ans.sql or "", extra=(self.settings.min_group_size,)
        )
        ans.grounded, ans.unmatched_numbers = grounded, unmatched
        ans.text = text if grounded else fallback
