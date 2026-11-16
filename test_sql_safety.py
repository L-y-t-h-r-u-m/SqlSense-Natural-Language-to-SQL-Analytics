"""Tests for SQL query safety checks."""
import unittest
import duckdb
from sqlalchemy import create_engine

from sql_safety import SafeSQLDatabase, validate_sql


class TestSQLSafety(unittest.TestCase):
    """Test suite for validate_sql and SafeSQLDatabase."""

    # ---------------- 1. Safe Queries ----------------

    def test_safe_simple_select(self):
        safe, msg = validate_sql("SELECT * FROM customers")
        self.assertTrue(safe, msg)

    def test_safe_select_with_where(self):
        safe, msg = validate_sql("SELECT customer_id, name FROM customers WHERE country = 'US'")
        self.assertTrue(safe, msg)

    def test_safe_select_with_join_and_group_by(self):
        sql = (
            "SELECT c.country, COUNT(o.order_id) AS total "
            "FROM customers c JOIN orders o ON c.customer_id = o.customer_id "
            "GROUP BY c.country ORDER BY total DESC LIMIT 5"
        )
        safe, msg = validate_sql(sql)
        self.assertTrue(safe, msg)

    def test_safe_select_with_cte(self):
        sql = (
            "WITH high_val AS ("
            "  SELECT customer_id, SUM(unit_price * quantity) as total "
            "  FROM orders JOIN order_items ON orders.order_id = order_items.order_id "
            "  GROUP BY customer_id"
            ") "
            "SELECT * FROM high_val WHERE total > 100 ORDER BY total DESC"
        )
        safe, msg = validate_sql(sql)
        self.assertTrue(safe, msg)

    def test_safe_select_with_semicolon_and_whitespace(self):
        safe, msg = validate_sql("SELECT 1;   \n")
        self.assertTrue(safe, msg)

    def test_safe_select_with_comments(self):
        sql = "-- Top customers query\nSELECT * FROM customers; -- end of line comment"
        safe, msg = validate_sql(sql)
        self.assertTrue(safe, msg)

        block_comment_sql = "/* Block comment\n multi-line */ SELECT 1;"
        safe, msg = validate_sql(block_comment_sql)
        self.assertTrue(safe, msg)

    def test_safe_select_containing_keywords_in_string_literals(self):
        # Destructive words inside string constants should NOT trigger false positives
        sql = "SELECT * FROM products WHERE name = 'Create Your Own Mug' AND description = 'DROP TABLE';"
        safe, msg = validate_sql(sql)
        self.assertTrue(safe, msg)

    def test_safe_select_with_string_functions(self):
        sql = "SELECT REPLACE(name, 'Customer', 'Client') FROM customers"
        safe, msg = validate_sql(sql)
        self.assertTrue(safe, msg)

    def test_safe_introspection(self):
        safe, msg = validate_sql("DESCRIBE customers")
        self.assertTrue(safe, msg)
        safe, msg = validate_sql("SHOW TABLES")
        self.assertTrue(safe, msg)

    # ---------------- 2. Block Destructive / Write Operations ----------------

    def test_block_insert(self):
        safe, msg = validate_sql("INSERT INTO customers VALUES (100, 'Hacker', 'US', '2024-01-01')")
        self.assertFalse(safe)
        self.assertIn("INSERT", msg)

    def test_block_update(self):
        safe, msg = validate_sql("UPDATE products SET price = 0 WHERE product_id = 1")
        self.assertFalse(safe)
        self.assertIn("UPDATE", msg)

    def test_block_delete(self):
        safe, msg = validate_sql("DELETE FROM orders WHERE order_id = 1")
        self.assertFalse(safe)
        self.assertIn("DELETE", msg)

    def test_block_drop(self):
        safe, msg = validate_sql("DROP TABLE customers")
        self.assertFalse(safe)
        self.assertIn("DROP", msg)

    def test_block_alter(self):
        safe, msg = validate_sql("ALTER TABLE customers ADD COLUMN balance INT")
        self.assertFalse(safe)
        self.assertIn("ALTER", msg)

    def test_block_create(self):
        safe, msg = validate_sql("CREATE TABLE evil (x INT)")
        self.assertFalse(safe)
        self.assertIn("CREATE", msg)

    def test_block_create_table_as_select(self):
        safe, msg = validate_sql("CREATE TABLE evil AS SELECT * FROM customers")
        self.assertFalse(safe)
        self.assertIn("CREATE", msg)

    def test_block_truncate(self):
        safe, msg = validate_sql("TRUNCATE TABLE orders")
        self.assertFalse(safe)
        self.assertTrue("TRUNCATE" in msg or "DELETE" in msg)

    def test_block_copy(self):
        safe, msg = validate_sql("COPY customers TO 'customers.csv'")
        self.assertFalse(safe)
        self.assertIn("COPY", msg)

    def test_block_attach(self):
        safe, msg = validate_sql("ATTACH 'evil.duckdb'")
        self.assertFalse(safe)
        self.assertIn("ATTACH", msg)

    # ---------------- 3. Block Multiple Statements ----------------

    def test_block_multi_statement_drop_injection(self):
        safe, msg = validate_sql("SELECT 1; DROP TABLE customers;")
        self.assertFalse(safe)
        self.assertIn("Multiple SQL statements", msg)

    def test_block_multi_statement_selects(self):
        safe, msg = validate_sql("SELECT 1; SELECT 2;")
        self.assertFalse(safe)
        self.assertIn("Multiple SQL statements", msg)

    def test_block_multi_statement_with_comments(self):
        safe, msg = validate_sql("SELECT * FROM customers; -- comment\n DROP TABLE customers;")
        self.assertFalse(safe)
        self.assertIn("Multiple SQL statements", msg)

    # ---------------- 4. Empty and Syntax Errors ----------------

    def test_block_empty_query(self):
        safe, msg = validate_sql("")
        self.assertFalse(safe)
        self.assertIn("empty", msg)

    def test_block_whitespace_only(self):
        safe, msg = validate_sql("   \n\t  ")
        self.assertFalse(safe)
        self.assertIn("empty", msg)

    def test_block_comment_only(self):
        safe, msg = validate_sql("-- only a comment")
        self.assertFalse(safe)
        self.assertIn("no executable", msg)

    def test_block_syntax_error(self):
        safe, msg = validate_sql("SELECT * FORM customers")
        self.assertFalse(safe)
        self.assertIn("syntax", msg.lower())

    # ---------------- 5. SafeSQLDatabase Integration ----------------

    def test_safesqldatabase_blocks_execution(self):
        # Create an in-memory test DuckDB database
        seed = duckdb.connect(":memory:")
        seed.execute("CREATE TABLE products (id INT, price INT); INSERT INTO products VALUES (1, 50)")
        seed.close()

        # Test with file-backed connection via engine
        import tempfile, os
        db_file = os.path.join(tempfile.gettempdir(), "test_safety_suite.duckdb")
        if os.path.exists(db_file):
            os.remove(db_file)
        c = duckdb.connect(db_file)
        c.execute("CREATE TABLE products (id INT, price INT); INSERT INTO products VALUES (1, 50)")
        c.close()

        engine = create_engine(f"duckdb:///{db_file}")
        safe_db = SafeSQLDatabase(engine=engine, include_tables=["products"])

        # Allowed read
        res = safe_db.run_no_throw("SELECT * FROM products")
        self.assertIn("50", str(res))

        # Blocked write via run_no_throw returns error string without touching db
        err = safe_db.run_no_throw("DROP TABLE products")
        self.assertIn("Error:", err)
        self.assertIn("DROP", err)

        # Blocked multiple statements
        multi_err = safe_db.run_no_throw("SELECT 1; DROP TABLE products;")
        self.assertIn("Error:", multi_err)
        self.assertIn("Multiple SQL statements", multi_err)

        # Verify products table is still intact!
        res_after = safe_db.run_no_throw("SELECT COUNT(*) FROM products")
        self.assertIn("1", str(res_after))

        # Blocked write via run() raises ValueError
        with self.assertRaises(ValueError):
            safe_db.run("DELETE FROM products")

        engine.dispose()
        if os.path.exists(db_file):
            os.remove(db_file)

    # ---------------- 6. History Credential Sanitization ----------------

    def test_sanitize_for_history(self):
        from app import _sanitize_for_history

        normal_text = "Show top 5 customers by revenue"
        self.assertEqual(_sanitize_for_history(normal_text), normal_text)

        with_aiza = "My key is AIzaSyD12345678901234567890123456789012 what is my revenue?"
        sanitized = _sanitize_for_history(with_aiza)
        self.assertNotIn("AIzaSyD", sanitized)
        self.assertIn("[REDACTED_API_KEY]", sanitized)

        with_aq = "Using key AQ.Ab8DummyMockKeyForTestingOnly1234567890abcdef1234567890 for queries"
        sanitized_aq = _sanitize_for_history(with_aq)
        self.assertNotIn("AQ.Ab8DummyMockKey", sanitized_aq)
        self.assertIn("[REDACTED_API_KEY]", sanitized_aq)

    # ---------------- 7. Generated vs Modified SQL Workflow ----------------

    def test_workflow_generated_and_modified_execution(self):
        import pandas as pd
        from sqlalchemy import text

        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE items (id INT, name VARCHAR, price INT); INSERT INTO items VALUES (1, 'Widget', 10), (2, 'Gadget', 25)")
        
        # Workflow A: Generated SQL (e.g. SELECT * FROM items WHERE price > 15)
        generated_sql = "SELECT * FROM items WHERE price > 15"
        safe, msg = validate_sql(generated_sql)
        self.assertTrue(safe)
        # Automatically executes
        df_gen = con.execute(generated_sql).fetchdf()
        self.assertEqual(len(df_gen), 1)
        self.assertEqual(df_gen.iloc[0]["name"], "Gadget")

        # Workflow B: User modifies SQL (e.g. changes price > 15 to price >= 10)
        modified_sql = "SELECT * FROM items WHERE price >= 10 ORDER BY price DESC"
        is_modified = (modified_sql.strip() != generated_sql.strip())
        self.assertTrue(is_modified)

        # Safety validation before manual execution
        safe_mod, msg_mod = validate_sql(modified_sql)
        self.assertTrue(safe_mod)
        # Executes modified SQL on demand
        df_mod = con.execute(modified_sql).fetchdf()
        self.assertEqual(len(df_mod), 2)

        # Workflow C: User tries to modify SQL with destructive command
        unsafe_modified_sql = "DROP TABLE items;"
        is_unsafe_mod = (unsafe_modified_sql.strip() != generated_sql.strip())
        self.assertTrue(is_unsafe_mod)
        safe_drop, drop_msg = validate_sql(unsafe_modified_sql)
        self.assertFalse(safe_drop)
        self.assertIn("DROP", drop_msg)

        con.close()


if __name__ == "__main__":
    unittest.main()
