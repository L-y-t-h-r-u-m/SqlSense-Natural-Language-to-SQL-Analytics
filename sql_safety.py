"""SQL safety validation for SqlSense analytics.

Enforces read-only querying against DuckDB:
- Permits single-statement SELECT queries (including CTEs).
- Blocks destructive operations (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, etc.).
- Rejects multiple statements in a single request.
- Provides descriptive, user-friendly error messages when blocked.
"""
import re
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import duckdb
from langchain_community.utilities import SQLDatabase
from sqlalchemy.sql.base import Executable

FORBIDDEN_STATEMENT_TYPES = {
    duckdb.StatementType.INSERT: "INSERT",
    duckdb.StatementType.UPDATE: "UPDATE",
    duckdb.StatementType.DELETE: "DELETE",
    duckdb.StatementType.DROP: "DROP",
    duckdb.StatementType.ALTER: "ALTER",
    duckdb.StatementType.CREATE: "CREATE",
    duckdb.StatementType.CREATE_FUNC: "CREATE",
    duckdb.StatementType.COPY: "COPY",
    duckdb.StatementType.COPY_DATABASE: "COPY",
    duckdb.StatementType.ATTACH: "ATTACH",
    duckdb.StatementType.DETACH: "DETACH",
    duckdb.StatementType.EXPORT: "EXPORT",
    duckdb.StatementType.LOAD: "LOAD",
    duckdb.StatementType.EXTENSION: "EXTENSION",
    duckdb.StatementType.PRAGMA: "PRAGMA",
    duckdb.StatementType.SET: "SET",
    duckdb.StatementType.TRANSACTION: "TRANSACTION",
    duckdb.StatementType.VACUUM: "VACUUM",
    duckdb.StatementType.CALL: "CALL",
    duckdb.StatementType.EXECUTE: "EXECUTE",
}

FORBIDDEN_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "ATTACH",
    "DETACH",
    "EXPORT",
    "IMPORT",
    "VACUUM",
}

_CODE_FENCE_PATTERN = re.compile(
    r"^```(?:sql)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE
)


def clean_sql_text(sql: str) -> str:
    """Strip markdown code fence blocks and leading/trailing whitespace."""
    if not isinstance(sql, str):
        return ""
    cleaned = sql.strip()
    match = _CODE_FENCE_PATTERN.match(cleaned)
    if match:
        cleaned = match.group(1).strip()
    return cleaned


def validate_sql(sql: str) -> Tuple[bool, str]:
    """Validate that a SQL string is a single, safe, read-only SELECT query.

    Returns:
        (is_safe, message):
            (True, "Query is safe.") if validation succeeds,
            (False, user_friendly_error_message) otherwise.
    """
    clean_sql = clean_sql_text(sql)
    if not clean_sql:
        return False, "Query is empty. Please provide a valid SQL query."

    # 1. Statement extraction using DuckDB's native parser
    try:
        stmts = duckdb.extract_statements(clean_sql)
    except Exception as exc:
        return False, f"Invalid SQL syntax: {exc}"

    if len(stmts) == 0:
        return False, "Query contains no executable SQL statements."

    if len(stmts) > 1:
        return (
            False,
            "Multiple SQL statements are not permitted in a single request. "
            "Only a single read-only query is allowed.",
        )

    stmt = stmts[0]

    # 2. Check statement type: Only SELECT statements are permitted
    if stmt.type != duckdb.StatementType.SELECT:
        if stmt.type == duckdb.StatementType.DELETE and re.search(
            r"\bTRUNCATE\b", clean_sql, re.IGNORECASE
        ):
            op_name = "TRUNCATE"
        else:
            op_name = FORBIDDEN_STATEMENT_TYPES.get(stmt.type, stmt.type.name)
        return (
            False,
            f"Destructive or write operation '{op_name}' is not permitted. "
            f"Only read-only SELECT queries are allowed.",
        )

    # 3. Defensive token-level check for any forbidden keyword token
    try:
        tokens = duckdb.tokenize(clean_sql)
        for i, (offset, ttype) in enumerate(tokens):
            if ttype == duckdb.token_type.keyword:
                next_offset = (
                    tokens[i + 1][0] if i + 1 < len(tokens) else len(clean_sql)
                )
                word = clean_sql[offset:next_offset].strip().upper()
                if word in FORBIDDEN_KEYWORDS:
                    return (
                        False,
                        f"Forbidden keyword '{word}' is not permitted in read-only queries.",
                    )
    except Exception as exc:
        return False, f"SQL token validation failed: {exc}"

    return True, "Query is safe."


class SafeSQLDatabase(SQLDatabase):
    """SQLDatabase subclass enforcing read-only query safety before execution."""

    def run(
        self,
        command: Union[str, Executable],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        sql_str = (
            str(command.compile(compile_kwargs={"literal_binds": True}))
            if isinstance(command, Executable)
            else str(command)
        )
        is_safe, error_msg = validate_sql(sql_str)
        if not is_safe:
            raise ValueError(error_msg)
        return super().run(command, *args, **kwargs)

    def run_no_throw(
        self,
        command: Union[str, Executable],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        sql_str = (
            str(command.compile(compile_kwargs={"literal_binds": True}))
            if isinstance(command, Executable)
            else str(command)
        )
        is_safe, error_msg = validate_sql(sql_str)
        if not is_safe:
            return f"Error: {error_msg}"
        return super().run_no_throw(command, *args, **kwargs)
