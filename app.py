"""SqlSense: Streamlit + DuckDB + LangChain + Gemini."""
import os
import re
import sys
import tempfile

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import create_engine, text

from sql_safety import SafeSQLDatabase, validate_sql

load_dotenv(override=True)

# ---------------- Data ----------------

def _build_frames() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)

    countries = ["US", "UK", "DE", "FR", "IN", "BR", "JP", "CA"]
    customers = pd.DataFrame({
        "customer_id": range(1, 51),
        "name": [f"Customer {i}" for i in range(1, 51)],
        "country": rng.choice(countries, 50),
        "signup_date": pd.to_datetime("2023-01-01")
        + pd.to_timedelta(rng.integers(0, 540, 50), unit="D"),
    })

    categories = ["Electronics", "Apparel", "Home", "Beauty", "Sports"]
    products = pd.DataFrame({
        "product_id": range(1, 21),
        "name": [f"Product {i}" for i in range(1, 21)],
        "category": rng.choice(categories, 20),
        "price": np.round(rng.uniform(5, 500, 20), 2),
    })

    orders = pd.DataFrame({
        "order_id": range(1, 201),
        "customer_id": rng.integers(1, 51, 200),
        "order_date": pd.to_datetime("2024-01-01")
        + pd.to_timedelta(rng.integers(0, 540, 200), unit="D"),
        "status": rng.choice(["completed", "shipped", "cancelled"], 200, p=[0.7, 0.2, 0.1]),
    })

    order_items = pd.DataFrame({
        "order_id": rng.integers(1, 201, 400),
        "product_id": rng.integers(1, 21, 400),
        "quantity": rng.integers(1, 6, 400),
    })
    order_items["unit_price"] = order_items["product_id"].map(
        products.set_index("product_id")["price"]
    )

    return {
        "customers": customers,
        "products": products,
        "orders": orders,
        "order_items": order_items,
    }


@st.cache_resource
def get_resources(extra_tables: tuple[str, ...] = ()):
    """One DuckDB file + DB resources per Streamlit session.

    extra_tables: names of user-CSV tables already written to the duckdb file.
    Changing the tuple busts the cache so the agent sees the new tables.
    """
    # ponytail: file-backed duckdb, not :memory: -- duckdb-engine's :memory: path
    # trips a SQLAlchemy 2.0 post_exec bug on in-memory connections.
    db_path = _db_path()
    if not os.path.exists(db_path):
        seed = duckdb.connect(db_path)
        for name, df in _build_frames().items():
            seed.register(name, df)
            seed.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}")
        seed.close()  # release the single-writer lock before SQLAlchemy opens it

    engine = create_engine(f"duckdb:///{db_path}")
    tables = ["customers", "products", "orders", "order_items", *extra_tables]
    db = SafeSQLDatabase(engine=engine, include_tables=tables)

    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        temperature=0,
    )
    return engine, db, llm


DEMO_TABLES = ("customers", "products", "orders", "order_items")


def _db_path() -> str:
    return os.path.join(tempfile.gettempdir(), "sqlsense_demo.duckdb")


def _reset_to_demo_tables(engine) -> None:
    """Ensure the database only contains the clean demo tables, removing stale uploaded tables."""
    frames = _build_frames()
    with engine.begin() as c:
        existing = pd.read_sql(text("SHOW TABLES"), c)["name"].tolist()
        # Drop any non-demo uploaded tables
        for t in existing:
            if t not in DEMO_TABLES:
                c.execute(text(f'DROP TABLE IF EXISTS "{t}"'))
        # Ensure all demo tables are present
        for name, df in frames.items():
            if name not in existing:
                c.connection.dbapi_connection.register("_demo_stage", df)
                c.execute(text(f'CREATE TABLE "{name}" AS SELECT * FROM _demo_stage'))
                c.connection.dbapi_connection.unregister("_demo_stage")


def add_csv_table(engine, name: str, df: pd.DataFrame) -> None:
    # Write via the live engine -- duckdb allows one writer, and this is it.
    with engine.begin() as c:
        c.connection.dbapi_connection.register("_csv_stage", df)
        c.execute(text(f'DROP TABLE IF EXISTS "{name}"'))
        c.execute(text(f'CREATE TABLE "{name}" AS SELECT * FROM _csv_stage'))
        c.connection.dbapi_connection.unregister("_csv_stage")


# ---------------- SQL extraction & generation ----------------

_SQL_FENCE = re.compile(r"```sql\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_GENERIC_FENCE = re.compile(r"```\s*(.*?)```", re.DOTALL)


def _sanitize_for_history(text_val: str) -> str:
    """Redact any potential API keys or credentials from text before saving to session history."""
    if not isinstance(text_val, str):
        return text_val
    sanitized = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_API_KEY]", text_val)
    sanitized = re.sub(r"AQ\.[0-9A-Za-z-_]{40,}", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(
        r"(?i)(api[_-]?key\s*[:=]\s*)['\"][^'\"]+['\"]", r"\1'[REDACTED]'", sanitized
    )
    return sanitized


def _to_text(output) -> str:
    # langchain 1.x returns output as a list of content blocks, not a str.
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = []
        for block in output:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return str(output)


def extract_sql(result) -> str | None:
    if isinstance(result, dict):
        for action, _ in reversed(result.get("intermediate_steps", [])):
            if getattr(action, "tool", "") == "sql_db_query":
                inp = action.tool_input
                return inp.get("query") if isinstance(inp, dict) else str(inp)
        raw = _to_text(result.get("output", ""))
    else:
        raw = _to_text(result)

    fences = _SQL_FENCE.findall(raw)
    if fences:
        return fences[-1].strip()
    generic = _GENERIC_FENCE.findall(raw)
    if generic:
        val = generic[-1].strip()
        if val.lower().startswith("sql\n") or val.lower().startswith("sql "):
            val = val[4:].strip()
        return val
    cleaned = raw.strip()
    if cleaned.lower().startswith("sql\n") or cleaned.lower().startswith("sql "):
        cleaned = cleaned[4:].strip()
    return cleaned if cleaned else None


def generate_sql_query(llm, db, question: str, history: list = None) -> str:
    """Prompt the LLM to write a single read-only DuckDB SQL query without executing it."""
    schema = db.get_table_info()
    history_context = ""
    if history:
        turns = []
        for h in history[-6:]:  # last 3 turns
            if h.get("role") == "user":
                turns.append(f"User: {h.get('content', '')}")
            elif h.get("role") == "assistant" and h.get("sql"):
                turns.append(f"Assistant SQL: {h.get('sql', '')}")
        if turns:
            history_context = "Previous Conversation Context:\n" + "\n".join(turns) + "\n\n"

    prompt = (
        "You are an expert DuckDB SQL developer.\n"
        "Your task is to write a single, read-only DuckDB SQL query that answers the user's question.\n\n"
        "Rules:\n"
        "1. Write ONLY read-only SELECT queries (CTEs using WITH are allowed).\n"
        "2. Do NOT write any INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or TRUNCATE statements.\n"
        "3. Output ONLY ONE SQL statement.\n"
        "4. Enclose your SQL query in a ```sql ... ``` block.\n\n"
        f"Database Schema:\n{schema}\n\n"
        f"{history_context}"
        f"User Question: {question}\n\n"
        "SQL Query:"
    )
    response = llm.invoke(prompt)
    output_text = _to_text(response.content if hasattr(response, "content") else response)
    sql = extract_sql(output_text)
    return sql or ""


# ---------------- Chart heuristic ----------------

def make_chart(df: pd.DataFrame):
    if df.empty or len(df.columns) < 2:
        return None
    dts = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cats = [
        c for c in df.columns
        if c not in nums and c not in dts and df[c].nunique() <= 30
    ]
    if dts and nums:
        d = df.sort_values(dts[0])
        return px.line(d, x=dts[0], y=nums[0])
    if cats and nums:
        return px.bar(df.head(30), x=cats[0], y=nums[0])
    if len(nums) >= 2:
        return px.scatter(df, x=nums[0], y=nums[1])
    return None


# ---------------- UI ----------------

def _sidebar(engine) -> None:
    with st.sidebar:
        st.header("Actions")
        if st.button("Clear Chat", use_container_width=True, key="clear_chat_sidebar"):
            st.session_state.messages = []
            st.rerun()

        st.header("Schema")
        with engine.connect() as c:
            tables = pd.read_sql(text("SHOW TABLES"), c)["name"].tolist()
            for t in tables:
                cols = pd.read_sql(text(f'DESCRIBE "{t}"'), c)
                with st.expander(f"**{t}** ({len(cols)} cols)"):
                    st.dataframe(cols[["column_name", "column_type"]],
                                 hide_index=True, use_container_width=True)

        st.header("Upload CSV")
        up = st.file_uploader("Add a table", type="csv", label_visibility="collapsed")
        if up is not None:
            name = re.sub(r"\W+", "_", os.path.splitext(up.name)[0]).strip("_").lower()
            if name and name not in st.session_state.get("csv_tables", []):
                try:
                    add_csv_table(engine, name, pd.read_csv(up))
                except Exception as exc:
                    st.error(f"CSV load failed: {exc}")
                    return
                st.session_state.setdefault("csv_tables", []).append(name)
                st.success(f"Added table `{name}`")
                st.rerun()  # bust agent cache so it sees the new table


def main() -> None:
    st.set_page_config(page_title="SqlSense", layout="wide")
    st.title("SqlSense")
    st.caption("Natural Language to SQL Analytics • Ask business questions over the demo e-commerce dataset.")

    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY is not set. Copy .env.example to .env and fill it in.")
        st.stop()

    engine, db, llm = get_resources(tuple(st.session_state.get("csv_tables", [])))

    if "session_initialized" not in st.session_state:
        st.session_state.session_initialized = True
        st.session_state.csv_tables = []
        _reset_to_demo_tables(engine)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    _sidebar(engine)

    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            elif msg["role"] == "assistant":
                original_sql = msg.get("original_sql", msg.get("sql", ""))
                current_sql = msg.get("current_sql", original_sql)
                show_preview = msg.get("show_preview", False)
                is_valid = msg.get("is_valid", True)
                error_msg = msg.get("error_msg")
                df = msg.get("df")
                exec_error = msg.get("exec_error")

                # Preview toggle button (SQL hidden by default)
                preview_label = "Hide SQL" if show_preview else "Preview SQL"
                if st.button(preview_label, key=f"toggle_preview_{idx}"):
                    msg["show_preview"] = not show_preview
                    st.rerun()

                # Editable SQL preview
                if show_preview:
                    editor_key = f"sql_editor_{idx}"
                    if editor_key not in st.session_state:
                        st.session_state[editor_key] = current_sql

                    edited_sql = st.text_area(
                        "SQL Query Editor",
                        value=st.session_state[editor_key],
                        key=editor_key,
                        height=140,
                    )

                    # Detect user modifications
                    is_modified = (edited_sql.strip() != original_sql.strip())
                    msg["current_sql"] = edited_sql
                    msg["is_modified"] = is_modified

                    if is_modified:
                        st.info("Modified SQL detected. Click 'Execute Modified SQL' to run your changes.")
                        if st.button("Execute Modified SQL", key=f"exec_mod_{idx}", type="primary"):
                            with st.spinner("Validating and executing modified SQL..."):
                                valid, reason = validate_sql(edited_sql)
                                if not valid:
                                    msg["is_valid"] = False
                                    msg["error_msg"] = reason
                                    msg["exec_error"] = None
                                    st.rerun()
                                else:
                                    try:
                                        with engine.connect() as c:
                                            mod_df = pd.read_sql(text(edited_sql), c)
                                        msg["df"] = mod_df
                                        msg["is_valid"] = True
                                        msg["error_msg"] = None
                                        msg["exec_error"] = None
                                        msg["executed"] = True
                                        st.rerun()
                                    except Exception as exc:
                                        msg["exec_error"] = str(exc)
                                        st.rerun()
                    else:
                        st.caption("SQL matches original generated query.")

                # Provide clear message if generated or modified SQL is invalid
                if not is_valid:
                    st.error(f"Invalid SQL query: {error_msg}")
                elif exec_error:
                    st.error(f"Query execution failed: {exec_error}")
                else:
                    # Results display below SQL preview
                    if df is not None:
                        st.dataframe(df, use_container_width=True)
                        st.download_button(
                            label="Download as CSV",
                            data=df.to_csv(index=False).encode("utf-8"),
                            file_name=f"query_results_{idx + 1}.csv",
                            mime="text/csv",
                            key=f"csv_download_{idx}",
                        )
                        chart = make_chart(df)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)

    question = st.chat_input("e.g. Top 5 customers by total revenue")
    if not question:
        return

    sanitized_question = _sanitize_for_history(question)
    st.session_state.messages.append({"role": "user", "content": sanitized_question})

    with st.spinner("Generating and executing SQL with Gemini..."):
        try:
            sql = generate_sql_query(
                llm, db, sanitized_question, history=st.session_state.messages[:-1]
            )
            is_valid, error_msg = validate_sql(sql)
        except Exception as exc:
            sql = ""
            is_valid = False
            error_msg = f"Failed to generate SQL: {exc}"

        df = None
        exec_error = None
        executed = False

        # 1. Automatic execution of valid generated SQL immediately
        if is_valid and sql:
            try:
                with engine.connect() as c:
                    df = pd.read_sql(text(sql), c)
                executed = True
            except Exception as exc:
                exec_error = str(exc)
                executed = True

    st.session_state.messages.append({
        "role": "assistant",
        "original_sql": sql,
        "current_sql": sql,
        "is_modified": False,
        "show_preview": False,
        "executed": executed,
        "df": df,
        "is_valid": is_valid,
        "error_msg": error_msg,
        "exec_error": exec_error,
    })
    st.rerun()


def _smoke() -> None:
    """ponytail: `python app.py smoke` -- full testing is `streamlit run app.py`."""
    frames = _build_frames()
    c = duckdb.connect(":memory:")
    for n, d in frames.items():
        c.register(n, d)
    assert c.execute("SELECT COUNT(*) FROM orders").fetchone()[0] > 0
    assert validate_sql("SELECT * FROM customers")[0] is True
    assert validate_sql("DROP TABLE customers")[0] is False
    assert validate_sql("SELECT 1; DROP TABLE customers;")[0] is False
    assert validate_sql("TRUNCATE TABLE orders")[0] is False
    print("smoke ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        _smoke()
    else:
        main()
