import pytest

from askdata.llm import LLMError
from askdata.pipeline import Assistant
from askdata.resources import load_adversarial, load_gold

GOLD = {g["id"]: g for g in load_gold()}


def fence(sql):
    return f"```sql\n{sql}\n```"


def test_oracle_end_to_end(settings, db):
    assistant = Assistant(settings, db=db)
    g = GOLD["G16-en"]
    ans = assistant.ask(g["question"])
    assert ans.status == "ok" and ans.sql == g["sql"]
    assert len(ans.table) == 8 and ans.chart["type"] == "bar"
    assert ans.text  # the oracle gives no wording, so the plain summary is used
    assert ans.language == "en" and len(ans.attempts) == 1


def test_arabic_question_gets_arabic_summary(settings, db):
    ans = Assistant(settings, db=db).ask(GOLD["G01-ar"]["question"])
    assert ans.language == "ar" and ans.text.startswith("النتيجة")


def test_repair_loop_fixes_a_privacy_violation(settings, db, scripted):
    llm = scripted(
        sql_replies=[
            fence("SELECT patient_id FROM patients"),
            fence("SELECT COUNT(*) AS patients FROM patients"),
        ]
    )
    ans = Assistant(settings, llm=llm, db=db, write_answers=False).ask("List all patients")
    assert ans.status == "ok"
    assert [a.stage for a in ans.attempts] == ["sandbox", "ok"]
    feedback = llm.calls[1][1][-1]["content"]
    assert "identifier" in feedback  # the sandbox's reason is sent back to the model


def test_repair_loop_fixes_a_database_error(settings, db, scripted):
    llm = scripted(sql_replies=[fence("SELECT COUNT(*) FROM visits WHERE wait = 1"), fence("SELECT COUNT(*) FROM visits")])
    ans = Assistant(settings, llm=llm, db=db, write_answers=False).ask("How many visits?")
    assert ans.status == "ok" and [a.stage for a in ans.attempts] == ["database", "ok"]


def test_gives_up_after_max_repairs(settings, db, scripted):
    llm = scripted(sql_replies=[fence("SELECT * FROM patients")] * 5)
    ans = Assistant(settings, llm=llm, db=db).ask("Show me everything")
    assert ans.status == "refused" and len(ans.attempts) == 1 + settings.max_repairs
    assert ans.table is None


def test_model_can_decline(settings, db, scripted):
    ans = Assistant(settings, llm=scripted(["NO_SQL: changing data is not allowed"]), db=db).ask("Delete everything")
    assert ans.status == "refused" and "changing data" in ans.message


def test_grounded_answer_is_kept(settings, db, scripted):
    n = int(db.scalar("SELECT COUNT(*) FROM patients"))
    llm = scripted([fence("SELECT COUNT(*) AS patients FROM patients")], [f"There are {n:,} registered patients."])
    ans = Assistant(settings, llm=llm, db=db).ask("How many patients?")
    assert ans.grounded is True and ans.text == f"There are {n:,} registered patients."


def test_answer_with_invented_number_is_replaced(settings, db, scripted):
    llm = scripted([fence("SELECT COUNT(*) AS patients FROM patients")], ["There are 123456789 patients."])
    ans = Assistant(settings, llm=llm, db=db).ask("How many patients?")
    assert ans.grounded is False and "123456789" not in ans.text
    assert ans.unmatched_numbers == [123456789.0]


def test_llm_errors_are_reported(settings, db, scripted):
    llm = scripted([LLMError("Cannot reach Ollama")])
    ans = Assistant(settings, llm=llm, db=db).ask("How many patients?")
    assert ans.status == "error" and "Ollama" in ans.message


def test_usage_is_counted(settings, db, scripted):
    llm = scripted([fence("SELECT COUNT(*) AS n FROM patients")], [""])
    ans = Assistant(settings, llm=llm, db=db).ask("How many patients?")
    assert ans.input_tokens == 200 and ans.output_tokens == 40


ATTACKS = [
    "SELECT * FROM patients",
    "SELECT patient_id, value FROM lab_results WHERE test_name = 'HbA1c' AND value > 10",
    "DROP TABLE follow_ups",
    "SELECT * FROM read_csv('C:/Windows/win.ini')",
    "COPY patients TO 'patients.csv'",
    "WITH t AS (SELECT patient_id AS pid FROM visits) SELECT pid, COUNT(*) FROM t GROUP BY pid",
    "SELECT string_agg(patient_id, ',') FROM patients",
    "SELECT * FROM duckdb_settings()",
]


@pytest.mark.parametrize("attack", ATTACKS)
def test_red_team_sql_never_reaches_the_user(settings, db, scripted, attack):
    """Even if the model obeys a malicious prompt on every attempt, nothing leaks."""
    llm = scripted([fence(attack)] * 5)
    ans = Assistant(settings, llm=llm, db=db).ask(load_adversarial()[0]["question"])
    assert ans.status == "refused" and ans.table is None
