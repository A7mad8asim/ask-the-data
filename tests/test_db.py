"""The database connection is the second line of defence: it must hold even if the sandbox is bypassed."""

import pytest

from askdata.db import Database, QueryError, QueryTimeout


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE x (i INTEGER)",
        "DELETE FROM appointments",
        "SELECT * FROM read_csv('C:/Windows/win.ini')",
        "ATTACH 'copy.db' AS copy",
        "INSTALL httpfs",
        "SET enable_external_access = true",
        "SET lock_configuration = false",
    ],
)
def test_connection_refuses_unsafe_statements_directly(db, sql):
    with pytest.raises(QueryError):
        db.run(sql)


def test_timeout_interrupts_long_queries(settings):
    slow = Database(settings.db_path, timeout_s=0.5)
    with pytest.raises(QueryTimeout):
        slow.run("SELECT SUM(i * i % 7) FROM range(100000000000) t(i)")
    slow.close()


def test_row_cap_and_truncation_flag(settings):
    small = Database(settings.db_path, max_rows=5)
    result = small.run("SELECT * FROM calendar")
    assert len(result.df) == 5 and result.truncated
    result = small.run("SELECT * FROM clinics LIMIT 3")
    assert len(result.df) == 3 and not result.truncated
    small.close()


def test_grouped_results_are_read_completely(db):
    # DuckDB returns GROUP BY results in several chunks; all of them must be read.
    result = db.run("SELECT appointment_date, COUNT(*) AS n FROM appointments GROUP BY 1", max_rows=5000)
    assert not result.truncated
    assert len(result.df) == db.scalar("SELECT COUNT(DISTINCT appointment_date) FROM appointments")


def test_missing_database_has_a_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="askdata.generator"):
        Database(tmp_path / "nope.duckdb")
