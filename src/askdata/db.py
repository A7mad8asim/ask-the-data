"""Read-only DuckDB access with a timeout and a row cap.

This is the second line of defence after the SQL sandbox: even a query that slipped
through the sandbox cannot write, read files, install extensions or change settings,
because the connection itself does not allow it.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

VECTOR_SIZE = 2048  # DuckDB returns results in chunks of this many rows


class QueryError(Exception):
    """The database rejected or failed to run the query."""


class QueryTimeout(QueryError):
    """The query ran longer than the time limit and was interrupted."""


@dataclass
class QueryResult:
    df: pd.DataFrame
    truncated: bool  # True when the result had more rows than the row cap
    seconds: float


class Database:
    def __init__(self, db_path: Path, timeout_s: float = 10.0, max_rows: int = 500):
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(
                f"No database at {db_path}. Generate it first:  python -m askdata.generator"
            )
        self.db_path = db_path
        self.timeout_s = timeout_s
        self.max_rows = max_rows
        self._con = duckdb.connect(
            str(db_path),
            read_only=True,
            config={
                "enable_external_access": False,  # no files, URLs, ATTACH or extension downloads
                "autoinstall_known_extensions": False,
                "autoload_known_extensions": False,
            },
        )
        try:
            self._con.execute("SET lock_configuration = true")  # settings can no longer be changed
        except duckdb.InvalidInputException:
            pass  # another connection in this process already locked the same database instance

    def run(self, sql: str, max_rows: int | None = None) -> QueryResult:
        """Run one query. Returns at most `max_rows` rows and flags truncation."""
        limit = self.max_rows if max_rows is None else max_rows
        cur = self._con.cursor()
        timer = threading.Timer(self.timeout_s, cur.interrupt)
        start = time.perf_counter()
        timer.start()
        try:
            cur.execute(sql)
            # Results arrive in chunks (often small ones after a GROUP BY): read until the cap is passed.
            frames = [cur.fetch_df_chunk(max(1, math.ceil((limit + 1) / VECTOR_SIZE)))]
            rows = len(frames[0])
            while rows <= limit and len(frames[-1]) > 0:
                frames.append(cur.fetch_df_chunk(1))
                rows += len(frames[-1])
            df = pd.concat([f for f in frames if len(f)] or frames[:1], ignore_index=True)
        except duckdb.InterruptException as e:
            raise QueryTimeout(f"The query took longer than {self.timeout_s:g} seconds and was stopped.") from e
        except duckdb.Error as e:
            raise QueryError(str(e).splitlines()[0]) from e
        finally:
            timer.cancel()
            cur.close()
        truncated = len(df) > limit
        return QueryResult(df.head(limit).reset_index(drop=True), truncated, time.perf_counter() - start)

    def scalar(self, sql: str):
        return self.run(sql, max_rows=1).df.iat[0, 0]

    def close(self) -> None:
        self._con.close()
