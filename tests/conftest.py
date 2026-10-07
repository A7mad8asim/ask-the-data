import pytest

from askdata.config import Settings
from askdata.db import Database
from askdata.generator import generate
from askdata.llm import LLMResponse


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """A small, seeded copy of the data mart, independent of the user's data/ folder."""
    d = tmp_path_factory.mktemp("data")
    generate(n_patients=4000, seed=42, data_dir=d, verbose=False)
    return d


@pytest.fixture(scope="session")
def settings(data_dir):
    return Settings.from_env().with_overrides(
        data_dir=data_dir,
        llm_provider="oracle",
        embeddings="tfidf",
        few_shot_k=4,
        max_repairs=2,
        query_timeout_s=10.0,
        max_rows=500,
        min_group_size=10,
    )


@pytest.fixture(scope="session")
def db(settings):
    database = Database(settings.db_path, settings.query_timeout_s, settings.max_rows)
    yield database
    database.close()


class ScriptedLLM:
    """Returns pre-written replies in order: one list for SQL requests, one for answers."""

    name = "scripted"

    def __init__(self, sql_replies=(), answer_replies=()):
        self.sql_replies = list(sql_replies)
        self.answer_replies = list(answer_replies)
        self.calls = []

    def complete(self, system, messages, purpose="sql"):
        self.calls.append((purpose, messages))
        queue = self.sql_replies if purpose == "sql" else self.answer_replies
        reply = queue.pop(0) if queue else ""
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(reply, input_tokens=100, output_tokens=20)


@pytest.fixture
def scripted():
    return ScriptedLLM
