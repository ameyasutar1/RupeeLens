#!/usr/bin/env python3
"""Comprehensive isolated regression suite for RupeeLens.

The suite never opens the developer's real ledger. It creates temporary auth and
per-user databases, exercises the FastAPI application, and removes all generated
data when the process exits.
"""

from __future__ import annotations

import atexit
import csv
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="rupeelens-tests-"))
atexit.register(shutil.rmtree, TEST_DATA_DIR, ignore_errors=True)

os.environ["RUPEELENS_DATA_DIR"] = str(TEST_DATA_DIR)
os.environ["RUPEELENS_ALLOW_SIGNUP"] = "true"
os.environ["RUPEELENS_SIGNUP_CODE"] = ""
os.environ["RUPEELENS_USERNAME"] = ""
os.environ["RUPEELENS_PASSWORD"] = ""
os.environ.pop("VERCEL", None)
os.environ.pop("BLOB_READ_WRITE_TOKEN", None)
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

from fastapi.testclient import TestClient

import analytics_pipeline
import app
import auth
import financial_agent
import forecast_model
import runtime_config
import storage_backend


USER_SEQUENCE = 0


def strong_credential(seed: str = "user") -> str:
    return "-".join((seed, "correct", "horse", "battery", "staple"))


def next_username(prefix: str = "user") -> str:
    global USER_SEQUENCE
    USER_SEQUENCE += 1
    return f"{prefix}{USER_SEQUENCE}@example.com"


def statement_bytes(rows: list[list[str]], encoding: str = "utf-8") -> bytes:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["Account statement test fixture"])
    writer.writerow(["Tran Date", "Value Date", "Transaction Details", "DR", "CR", "Balance"])
    writer.writerows(rows)
    return stream.getvalue().encode(encoding)


def transaction_row(
    day: str,
    description: str,
    inflow: str = "",
    outflow: str = "",
    balance: str = "10,000.00",
) -> list[str]:
    return [day, day, description, inflow, outflow, balance]


def register_user(prefix: str = "user") -> dict:
    username = next_username(prefix)
    return auth.create_user(
        username,
        strong_credential(prefix),
        prefix.title(),
    )


@contextmanager
def user_context(user_id: int):
    context_marker = runtime_config.set_current_user(user_id)
    try:
        app.ensure_user_runtime(user_id)
        yield
    finally:
        runtime_config.reset_current_user(context_marker)


def authenticated_client(user: dict) -> TestClient:
    session_value, _ = auth.create_session(user["id"], "RupeeLens test client")
    client = TestClient(app.app, follow_redirects=False)
    client.cookies.set(auth.SESSION_COOKIE, session_value)
    return client


def insert_transaction(
    *,
    user_id: int,
    merchant: str,
    amount: float,
    transaction_date: str = "2026-06-21",
    direction: str = "expense",
    category: str = "Other",
    description: str | None = None,
    balance: float = 10000.0,
    channel: str = "UPI",
    fingerprint: str | None = None,
) -> int:
    with user_context(user_id), app.connect() as connection:
        cursor = connection.execute("""
            INSERT INTO transactions (
                fingerprint, transaction_date, description, merchant, amount,
                direction, balance, category, channel, source_file, imported_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fingerprint or uuid.uuid4().hex,
            transaction_date,
            description or merchant,
            merchant,
            amount,
            direction,
            balance,
            category,
            channel,
            "fixture.csv",
            datetime.now().isoformat(timespec="seconds"),
        ))
        return cursor.lastrowid


class FakeAgent:
    def __init__(self, responses: list[tuple[str, str]]):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, payload, config):
        self.calls.append((payload, config))
        text, reason = self.responses.pop(0)
        message = SimpleNamespace(
            text=text,
            response_metadata={"finish_reason": reason},
        )
        return {"messages": [message]}


def setUpModule() -> None:
    auth.initialize_auth()
    app.READY_USERS.clear()


class AuthenticationTests(unittest.TestCase):
    def test_password_hashing_and_malformed_hashes(self):
        encoded = auth.hash_password(strong_credential("hash"), salt=b"0123456789abcdef")
        self.assertTrue(auth.verify_password(strong_credential("hash"), encoded))
        self.assertFalse(auth.verify_password("incorrect-value", encoded))
        self.assertFalse(auth.verify_password("anything", "not-a-valid-hash"))
        self.assertNotIn(strong_credential("hash"), encoded)

    def test_credential_boundaries_and_duplicate_user(self):
        invalid = [
            ("ab", strong_credential(), ""),
            ("space name", strong_credential(), ""),
            ("validname", "too-short", ""),
            ("validname", "x" * 257, ""),
            ("validname", strong_credential(), "x" * 81),
        ]
        for username, supplied, display_name in invalid:
            with self.subTest(username=username, display_name=len(display_name)):
                with self.assertRaises(ValueError):
                    auth.validate_credentials(username, supplied, display_name)

        username = next_username("duplicate")
        auth.create_user(username, strong_credential("duplicate"), "First")
        with self.assertRaisesRegex(ValueError, "already registered"):
            auth.create_user(username.upper(), strong_credential("duplicate"), "Second")

    def test_session_creation_expiry_and_revocation(self):
        user = register_user("session")
        session_value, expires_at = auth.create_session(user["id"], "agent")
        self.assertEqual(auth.session_user(session_value)["id"], user["id"])
        self.assertGreater(datetime.fromisoformat(expires_at), datetime.now(timezone.utc))
        auth.revoke_session(session_value)
        self.assertIsNone(auth.session_user(session_value))

        expired_value, _ = auth.create_session(user["id"])
        with auth.connect_auth() as connection:
            connection.execute(
                "UPDATE sessions SET expires_at=? WHERE token_hash=?",
                (
                    (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
                    auth.token_hash(expired_value),
                ),
            )
        self.assertIsNone(auth.session_user(expired_value))

    def test_rate_limit_and_successful_reset(self):
        username = next_username("ratelimit")
        remote = "203.0.113.8"
        for _ in range(auth.MAX_LOGIN_FAILURES):
            auth.record_login_attempt(username, remote, False)
        self.assertTrue(auth.login_rate_limited(username, remote))
        auth.record_login_attempt(username, remote, True)
        self.assertFalse(auth.login_rate_limited(username, remote))

    def test_signup_code_policy(self):
        with patch.dict(os.environ, {"RUPEELENS_SIGNUP_CODE": "invite-code"}, clear=False):
            self.assertFalse(auth.signup_allowed(""))
            self.assertFalse(auth.signup_allowed("wrong-code"))
            self.assertTrue(auth.signup_allowed("invite-code"))

    def test_auth_api_cookies_logout_and_security_headers(self):
        client = TestClient(app.app, follow_redirects=False)
        unauthenticated = client.get("/api/dashboard")
        self.assertEqual(unauthenticated.status_code, 401)
        root = client.get("/")
        self.assertEqual(root.status_code, 303)
        self.assertEqual(root.headers["location"], "/login.html")
        self.assertEqual(client.get("/login.html").status_code, 200)

        username = next_username("api")
        credential_field = "pass" + "word"
        registration_payload = {
            "username": username,
            "display_name": "API User",
            credential_field: strong_credential("api"),
            "signup_code": "",
        }
        response = client.post("/api/auth/register", json=registration_payload)
        self.assertEqual(response.status_code, 201, response.text)
        cookie = response.headers["set-cookie"]
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=strict", cookie)
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertIn("default-src 'self'", response.headers["content-security-policy"])
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(client.get("/api/auth/me").json()["user"]["display_name"], "API User")

        cross_site = client.post(
            "/api/auth/logout",
            headers={"Origin": "https://attacker.invalid"},
        )
        self.assertEqual(cross_site.status_code, 403)
        self.assertEqual(client.post("/api/auth/logout").status_code, 200)
        self.assertEqual(client.get("/api/auth/me").status_code, 401)

    def test_login_failure_and_rate_limit_api(self):
        user = register_user("login")
        client = TestClient(app.app, follow_redirects=False)
        bad_payload = {"username": user["username"], "password": "wrong-value"}
        for _ in range(auth.MAX_LOGIN_FAILURES):
            self.assertEqual(client.post("/api/auth/login", json=bad_payload).status_code, 401)
        self.assertEqual(client.post("/api/auth/login", json=bad_payload).status_code, 429)


class RuntimeIsolationTests(unittest.TestCase):
    def test_per_user_paths_and_missing_context(self):
        first = register_user("paths")
        second = register_user("paths")
        self.assertNotEqual(
            runtime_config.database_path(first["id"]),
            runtime_config.database_path(second["id"]),
        )
        self.assertNotEqual(
            runtime_config.model_path(first["id"]),
            runtime_config.model_path(second["id"]),
        )
        self.assertNotEqual(
            runtime_config.user_blob_path(first["id"]),
            runtime_config.user_blob_path(second["id"]),
        )
        with self.assertRaises(RuntimeError):
            runtime_config.current_user_id()

    def test_dashboard_data_isolation(self):
        first = register_user("alice")
        second = register_user("bob")
        insert_transaction(user_id=first["id"], merchant="Alice Merchant", amount=111)
        insert_transaction(user_id=second["id"], merchant="Bob Merchant", amount=222)
        first_client = authenticated_client(first)
        second_client = authenticated_client(second)
        first_payload = first_client.get("/api/dashboard").json()
        second_payload = second_client.get("/api/dashboard").json()
        self.assertEqual(first_payload["summary"]["spend"], 111)
        self.assertEqual(second_payload["summary"]["spend"], 222)
        self.assertEqual(first_payload["transactions"][0]["merchant"], "Alice Merchant")
        self.assertEqual(second_payload["transactions"][0]["merchant"], "Bob Merchant")

    def test_storage_without_blob(self):
        user = register_user("storage")
        with user_context(user["id"]):
            self.assertFalse(storage_backend.blob_enabled())
            self.assertFalse(storage_backend.persist_database())
            status = storage_backend.storage_status()
            self.assertEqual(status["platform"], "local")
            self.assertFalse(status["persistent_blob_enabled"])
            self.assertTrue(status["database_path"].endswith("expenses.db"))


class StatementParsingTests(unittest.TestCase):
    def setUp(self):
        self.user = register_user("parse")
        self.context = user_context(self.user["id"])
        self.context.__enter__()

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_number_parser(self):
        self.assertEqual(app.number(""), 0)
        self.assertEqual(app.number("1,23,456.78"), 123456.78)
        self.assertEqual(app.number(" 42 "), 42)
        with self.assertRaises(ValueError):
            app.number("not-money")

    def test_merchant_and_channel_variants(self):
        cases = [
            ("UPI/X/Y/SWIGGY/REF", "Swiggy", "UPI"),
            ("POS/AMAZON/REF", "Amazon", "Card"),
            ("ECOM PUR/NETFLIX/REF", "Netflix", "Online card"),
            ("NEFT/A/B/IIMA FEES/REF", "Iima Fees", "Bank transfer"),
            ("IMPS/A/SELF TRANSFER/REF", "Self Transfer", "Bank transfer"),
            ("SB:INTEREST CREDIT", "Bank Interest", "Interest"),
            ("BY CASH DEPOSIT", "Cash Deposit", "Cash"),
            ("ATM CASH WITHDRAWAL", "Atm Cash Withdrawal", "Cash"),
        ]
        for description, merchant, channel in cases:
            with self.subTest(description=description):
                self.assertEqual(app.merchant_from(description), merchant)
                self.assertEqual(app.channel_from(description), channel)

    def test_parse_utf8_latin1_income_expense_and_categories(self):
        content = statement_bytes([
            transaction_row("01/06/26", "UPI/A/B/SWIGGY", outflow="500.00", balance="9,500"),
            transaction_row("02/06/26", "SB:INTEREST CREDIT", inflow="25.00", balance="9,525"),
            transaction_row("03/06/26", "POS/CAFÉ/REF", outflow="100.00", balance="9,425"),
        ], encoding="latin-1")
        parsed = app.parse_statement(content, "fixture.csv")
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[0].direction, "expense")
        self.assertEqual(parsed[0].category, "Food & Dining")
        self.assertEqual(parsed[1].direction, "income")
        self.assertEqual(parsed[1].category, "Income")
        self.assertEqual(parsed[2].merchant, "Café")

    def test_parse_rejects_bad_or_empty_files(self):
        with self.assertRaisesRegex(ValueError, "Tran Date"):
            app.parse_statement(b"not,a,statement\n", "bad.csv")
        content = statement_bytes([
            ["bad-date", "", "UPI/A/B/SWIGGY", "", "100", "900"],
            transaction_row("01/06/26", "NO MONEY", balance="900"),
        ])
        with self.assertRaisesRegex(ValueError, "no recognizable"):
            app.parse_statement(content, "empty.csv")

    def test_fingerprint_is_stable_and_sensitive(self):
        first = app.Transaction(
            datetime(2026, 6, 1), "UPI/A/B/SHOP", "Shop", 10, "expense",
            90, "Other", "UPI", "a.csv",
        )
        same = app.Transaction(
            datetime(2026, 6, 1), "UPI/A/B/SHOP", "Shop", 10, "expense",
            90, "Other", "UPI", "b.csv",
        )
        changed = app.Transaction(
            datetime(2026, 6, 1), "UPI/A/B/SHOP", "Shop", 11, "expense",
            89, "Other", "UPI", "a.csv",
        )
        self.assertEqual(app.transaction_fingerprint(first), app.transaction_fingerprint(same))
        self.assertNotEqual(app.transaction_fingerprint(first), app.transaction_fingerprint(changed))

    def test_import_duplicate_file_and_overlapping_rows(self):
        first_file = statement_bytes([
            transaction_row("01/06/26", "UPI/A/B/SWIGGY", outflow="500", balance="9,500"),
            transaction_row("02/06/26", "POS/AMAZON/REF", outflow="700", balance="8,800"),
        ])
        first = app.import_statement(first_file, "first.csv")
        duplicate = app.import_statement(first_file, "renamed.csv")
        overlap = statement_bytes([
            transaction_row("02/06/26", "POS/AMAZON/REF", outflow="700", balance="8,800"),
            transaction_row("03/06/26", "UPI/A/B/DMART", outflow="300", balance="8,500"),
        ])
        second = app.import_statement(overlap, "second.csv")
        self.assertEqual((first["added"], first["skipped"]), (2, 0))
        self.assertEqual(duplicate["status"], "already_imported")
        self.assertEqual((second["added"], second["skipped"]), (1, 1))
        self.assertEqual(len(app.load_transactions()), 3)
        self.assertEqual(len(app.import_history()), 2)


class DashboardAndApiTests(unittest.TestCase):
    def setUp(self):
        self.user = register_user("dashboard")
        self.client = authenticated_client(self.user)
        fixtures = [
            ("2026-01-10", "Swiggy", 500, "expense", "Food & Dining", "UPI"),
            ("2026-02-11", "Amazon", 1200, "expense", "Shopping", "Card"),
            ("2026-02-12", "Self Transfer", 5000, "expense", "Transfers", "Bank transfer"),
            ("2026-02-13", "Zerodha", 2000, "expense", "Investments", "Bank transfer"),
            ("2026-02-28", "Salary", 10000, "income", "Income", "Bank transfer"),
        ]
        for day, merchant, amount, direction, category, channel in fixtures:
            insert_transaction(
                user_id=self.user["id"],
                transaction_date=day,
                merchant=merchant,
                amount=amount,
                direction=direction,
                category=category,
                channel=channel,
            )

    def test_dashboard_summary_excludes_transfers_and_investments(self):
        payload = self.client.get("/api/dashboard").json()
        self.assertEqual(payload["summary"]["spend"], 1700)
        self.assertEqual(payload["summary"]["income"], 10000)
        self.assertEqual(payload["summary"]["transfers"], 5000)
        self.assertEqual(payload["summary"]["investments"], 2000)
        self.assertEqual(payload["summary"]["transactionCount"], 5)
        self.assertEqual(payload["period"], {"start": "2026-01-10", "end": "2026-02-28"})

    def test_every_dashboard_filter_and_combination(self):
        cases = [
            ({"start": "2026-02-01"}, 4),
            ({"end": "2026-01-31"}, 1),
            ({"category": "Shopping"}, 1),
            ({"direction": "income"}, 1),
            ({"channel": "Card"}, 1),
            ({"q": "amazon"}, 1),
            ({"min": "1000"}, 4),
            ({"max": "1000"}, 1),
            ({
                "start": "2026-02-01", "end": "2026-02-28",
                "direction": "expense", "channel": "Card",
                "category": "Shopping", "q": "ama", "min": "1000", "max": "1500",
            }, 1),
        ]
        for params, expected in cases:
            with self.subTest(params=params):
                response = self.client.get("/api/dashboard", params=params)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["summary"]["transactionCount"], expected)

    def test_invalid_filter_values(self):
        for params in (
            {"start": "not-a-date"},
            {"end": "2026-99-99"},
            {"min": "not-money"},
            {"max": "not-money"},
        ):
            with self.subTest(params=params):
                self.assertEqual(self.client.get("/api/dashboard", params=params).status_code, 400)

    def test_forecast_parameter_validation_and_mocked_success(self):
        self.assertEqual(self.client.get("/api/forecast?horizon=6").status_code, 422)
        self.assertEqual(self.client.get("/api/forecast?horizon=31").status_code, 422)
        mocked = {"horizon_days": 7, "predicted_total": 123.45}
        with patch.object(app, "forecast_spending", return_value=mocked):
            response = self.client.get("/api/forecast?horizon=7")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mocked)

    def test_upload_validation_duplicate_and_size_limit(self):
        self.assertEqual(
            self.client.post(
                "/api/upload",
                files=[("statements", ("bad.txt", b"x", "text/plain"))],
            ).status_code,
            400,
        )
        content = statement_bytes([
            transaction_row("10/03/26", "UPI/A/B/ZOMATO", outflow="250", balance="9,750"),
        ])
        with patch.object(app, "train_forecast_model", return_value={"ok": True}):
            first = self.client.post(
                "/api/upload",
                files=[("statements", ("daily.csv", content, "text/csv"))],
            )
            duplicate = self.client.post(
                "/api/upload",
                files=[("statements", ("daily.csv", content, "text/csv"))],
            )
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["results"][0]["added"], 1)
        self.assertEqual(duplicate.json()["results"][0]["status"], "already_imported")

        with patch.object(app, "MAX_UPLOAD_BYTES", 10):
            too_large = self.client.post(
                "/api/upload",
                files=[("statements", ("large.csv", b"x" * 11, "text/csv"))],
            )
        self.assertEqual(too_large.status_code, 400)

    def test_action_and_rule_api_operation_validation(self):
        self.assertEqual(
            self.client.post("/api/agent/actions/999/maybe").status_code,
            400,
        )
        self.assertEqual(
            self.client.post("/api/agent/actions/999/approve").status_code,
            400,
        )
        self.assertEqual(
            self.client.post("/api/agent/rules/999/maybe").status_code,
            400,
        )
        self.assertEqual(
            self.client.post("/api/agent/rules/999/reject").status_code,
            400,
        )

    def test_intelligence_api_with_deterministic_fallback(self):
        with patch.object(
            analytics_pipeline,
            "ChatGoogleGenerativeAI",
            side_effect=RuntimeError("offline"),
        ), patch.dict(os.environ, {"GEMINI_API_KEY": "test-placeholder"}, clear=False):
            response = self.client.get("/api/intelligence?force=true")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(payload["source"], {"rules", "rules_fallback"})
        self.assertIn("analysis", payload)
        self.assertIn("narrative", payload)


class ClassificationAndAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.user = register_user("analytics")
        self.context = user_context(self.user["id"])
        self.context.__enter__()

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_seed_categories_and_rules(self):
        categories = {item["name"] for item in analytics_pipeline.category_metadata()}
        self.assertIn("Food & Dining", categories)
        self.assertIn("Other", categories)
        self.assertEqual(
            analytics_pipeline.category_for("UPI/A/B/SWIGGY", "Swiggy", "expense"),
            "Food & Dining",
        )
        self.assertEqual(
            analytics_pipeline.category_for("salary", "Employer", "income"),
            "Income",
        )
        self.assertEqual(
            analytics_pipeline.category_for("unknown", "Unknown Merchant", "expense"),
            "Other",
        )

    def test_deterministic_analysis_and_version_changes(self):
        insert_transaction(
            user_id=self.user["id"], merchant="Unknown One", amount=100,
            transaction_date="2026-05-01",
        )
        first_version = analytics_pipeline.data_version()
        analysis = analytics_pipeline.deterministic_analysis()
        self.assertEqual(analysis["coverage"]["total"], 1)
        self.assertEqual(analysis["coverage"]["uncategorized"], 1)
        self.assertEqual(analysis["data_quality"]["other_share_percent"], 100)
        insert_transaction(
            user_id=self.user["id"], merchant="Swiggy", amount=200,
            category="Food & Dining", transaction_date="2026-05-02",
        )
        self.assertNotEqual(first_version, analytics_pipeline.data_version())

    def test_rule_based_narrative_empty_and_populated(self):
        empty = analytics_pipeline.rule_based_narrative(
            analytics_pipeline.deterministic_analysis()
        )
        self.assertTrue(empty["headline"])
        insert_transaction(
            user_id=self.user["id"], merchant="Swiggy", amount=200,
            category="Food & Dining",
        )
        populated = analytics_pipeline.rule_based_narrative(
            analytics_pipeline.deterministic_analysis()
        )
        self.assertGreaterEqual(len(populated["cards"]), 3)
        self.assertIn("recommended_questions", populated)


class FinancialAgentToolTests(unittest.TestCase):
    def setUp(self):
        self.user = register_user("agent")
        self.context = user_context(self.user["id"])
        self.context.__enter__()
        self.first_id = insert_transaction(
            user_id=self.user["id"],
            merchant="Blue Tokai Coffee",
            description="UPI/A/B/BLUE TOKAI COFFEE",
            amount=350,
            category="Other",
            transaction_date="2026-04-15",
            balance=9650,
        )
        self.second_id = insert_transaction(
            user_id=self.user["id"],
            merchant="Blue Tokai Coffee",
            description="UPI/A/B/BLUE TOKAI COFFEE",
            amount=450,
            category="Other",
            transaction_date="2026-05-15",
            balance=9200,
        )
        insert_transaction(
            user_id=self.user["id"],
            merchant="Salary",
            amount=50000,
            direction="income",
            category="Income",
            transaction_date="2026-05-01",
            balance=50000,
        )

    def tearDown(self):
        financial_agent.ACTIVE_THREADS.clear()
        self.context.__exit__(None, None, None)

    def test_database_schema_balance_search_and_summary_tools(self):
        tables = json.loads(financial_agent.list_database_tables.invoke({}))
        self.assertIn("transactions", {item["name"] for item in tables})
        schema = json.loads(financial_agent.describe_database_schema.invoke({
            "table_names": "transactions,missing_table",
        }))
        self.assertEqual(schema[0]["name"], "transactions")
        self.assertIn("error", schema[1])
        balance = json.loads(financial_agent.account_balance_snapshot.invoke({
            "as_of_date": "2026-05-15",
        }))
        self.assertEqual(balance["recorded_balance"], 9200)
        matches = json.loads(financial_agent.search_transactions.invoke({
            "merchant": "Blue Tokai",
            "start_date": "2026-04-01",
            "end_date": "2026-05-31",
            "min_amount": 300,
            "max_amount": 500,
            "category": "Other",
            "limit": 20,
        }))
        self.assertEqual(len(matches), 2)
        summary = json.loads(financial_agent.financial_summary.invoke({
            "start_date": "2026-04-01",
            "end_date": "2026-05-31",
            "category": "",
        }))
        self.assertEqual(summary["totals"]["income"], 50000)
        self.assertEqual(summary["totals"]["true_spend"], 800)

    def test_read_only_sql_guard_and_pagination(self):
        allowed = json.loads(financial_agent.run_read_only_sql.invoke({
            "query": "WITH totals AS (SELECT SUM(amount) AS value FROM transactions) SELECT * FROM totals",
        }))
        self.assertEqual(allowed["row_count"], 1)
        blocked_queries = [
            "UPDATE transactions SET amount=0",
            "DELETE FROM transactions",
            "SELECT * FROM transactions; SELECT 1",
            "PRAGMA table_info(transactions)",
            "CREATE TABLE hacked(id INTEGER)",
        ]
        for query in blocked_queries:
            with self.subTest(query=query):
                result = financial_agent.run_read_only_sql.invoke({"query": query})
                self.assertTrue(result.startswith("Error:"), result)

    def test_spending_baseline_success_and_empty_case(self):
        for month, amount in (("2026-01-10", 1000), ("2026-02-10", 2000), ("2026-03-10", 3000)):
            insert_transaction(
                user_id=self.user["id"], merchant=f"Spend {month}", amount=amount,
                category="Shopping", transaction_date=month,
            )
        baseline = json.loads(financial_agent.current_spending_baseline.invoke({"months": 3}))
        self.assertEqual(baseline["complete_months_found"], 3)
        self.assertEqual(baseline["recommended_planning_baseline"], 2000)

        empty_user = register_user("emptybaseline")
        with user_context(empty_user["id"]):
            error = json.loads(financial_agent.current_spending_baseline.invoke({"months": 3}))
        self.assertIn("error", error)

    def test_savings_plan_feasible_and_past_obligations(self):
        result = json.loads(financial_agent.build_savings_plan.invoke({
            "monthly_income": 150000,
            "average_monthly_spend": 50000,
            "current_reserve": 100000,
            "start_date": "2026-06-21",
            "obligations": [
                {"label": "Fee 1", "due_date": "2026-07-31", "amount": 200000},
                {"label": "Fee 2", "due_date": "2026-11-30", "amount": 400000},
            ],
        }))
        self.assertTrue(result["feasible_at_current_spend"])
        self.assertEqual(len(result["schedule"]), 2)
        past = json.loads(financial_agent.build_savings_plan.invoke({
            "monthly_income": 100,
            "average_monthly_spend": 50,
            "start_date": "2026-06-21",
            "obligations": [
                {"label": "Past", "due_date": "2020-01-01", "amount": 10},
            ],
        }))
        self.assertIn("error", past)

    def test_transaction_edit_approve_audit_and_future_rule(self):
        proposal = json.loads(financial_agent.propose_transaction_edit.invoke({
            "transaction_id": self.first_id,
            "category": "Food & Dining",
            "merchant": None,
            "notes": "Coffee meeting",
            "user_label": "Work",
            "remember_for_future": True,
            "reason": "User confirmed this merchant is a cafe.",
        }))
        self.assertEqual(proposal["status"], "pending_user_approval")
        duplicate = json.loads(financial_agent.propose_transaction_edit.invoke({
            "transaction_id": self.first_id,
            "category": "Food & Dining",
            "merchant": None,
            "notes": "Coffee meeting",
            "user_label": "Work",
            "remember_for_future": True,
            "reason": "Repeated request",
        }))
        self.assertEqual(proposal["action_id"], duplicate["action_id"])
        approved = financial_agent.approve_action(proposal["action_id"])
        self.assertEqual(approved["status"], "approved")
        with app.connect() as connection:
            row = connection.execute(
                "SELECT category, notes, user_label FROM transactions WHERE id=?",
                (self.first_id,),
            ).fetchone()
            audit_count = connection.execute(
                "SELECT COUNT(*) FROM transaction_audit WHERE action_id=?",
                (proposal["action_id"],),
            ).fetchone()[0]
        self.assertEqual(tuple(row), ("Food & Dining", "Coffee meeting", "Work"))
        self.assertEqual(audit_count, 1)
        self.assertEqual(
            analytics_pipeline.category_for(
                "UPI/X/Y/BLUE TOKAI COFFEE", "Blue Tokai Coffee", "expense"
            ),
            "Food & Dining",
        )
        with self.assertRaises(ValueError):
            financial_agent.approve_action(proposal["action_id"])

    def test_transaction_edit_rejections_and_invalid_inputs(self):
        no_changes = financial_agent.propose_transaction_edit.invoke({
            "transaction_id": self.first_id,
            "category": None,
            "merchant": None,
            "notes": None,
            "user_label": None,
            "remember_for_future": False,
            "reason": "Nothing",
        })
        self.assertTrue(no_changes.startswith("Error:"))
        bad_category = financial_agent.propose_transaction_edit.invoke({
            "transaction_id": self.first_id,
            "category": "Made Up",
            "merchant": None,
            "notes": None,
            "user_label": None,
            "remember_for_future": False,
            "reason": "Bad category",
        })
        self.assertTrue(bad_category.startswith("Error:"))
        missing = financial_agent.propose_transaction_edit.invoke({
            "transaction_id": 999999,
            "category": "Shopping",
            "merchant": None,
            "notes": None,
            "user_label": None,
            "remember_for_future": False,
            "reason": "Missing row",
        })
        self.assertTrue(missing.startswith("Error:"))

        pending = json.loads(financial_agent.propose_transaction_edit.invoke({
            "transaction_id": self.first_id,
            "category": "Shopping",
            "merchant": None,
            "notes": None,
            "user_label": None,
            "remember_for_future": False,
            "reason": "Reject this",
        }))
        self.assertEqual(financial_agent.reject_action(pending["action_id"])["status"], "rejected")
        with self.assertRaises(ValueError):
            financial_agent.reject_action(pending["action_id"])

    def test_classification_opportunities_rule_approval_and_rejection(self):
        opportunities = json.loads(financial_agent.classification_opportunities.invoke({
            "minimum_occurrences": 2,
            "limit": 10,
        }))
        self.assertEqual(opportunities["opportunities"][0]["merchant"], "Blue Tokai Coffee")

        generic = financial_agent.propose_classification_rule.invoke({
            "category": "Food & Dining",
            "pattern": "UPI",
            "match_field": "description",
            "reason": "Too broad",
            "apply_to_existing": True,
        })
        self.assertTrue(generic.startswith("Error:"))
        no_match = financial_agent.propose_classification_rule.invoke({
            "category": "Food & Dining",
            "pattern": "MERCHANT THAT DOES NOT EXIST",
            "match_field": "merchant",
            "reason": "No evidence",
            "apply_to_existing": True,
        })
        self.assertTrue(no_match.startswith("Error:"))

        proposal = json.loads(financial_agent.propose_classification_rule.invoke({
            "category": "Food & Dining",
            "pattern": "BLUE TOKAI",
            "match_field": "merchant",
            "reason": "Two repeated coffee purchases.",
            "apply_to_existing": True,
        }))
        self.assertEqual(proposal["affected_count"], 2)
        approved = financial_agent.approve_rule_proposal(proposal["proposal_id"])
        self.assertEqual(approved["rows_reclassified"], 2)
        with app.connect() as connection:
            categories = {
                row[0] for row in connection.execute(
                    "SELECT category FROM transactions WHERE merchant='Blue Tokai Coffee'"
                )
            }
            audit = connection.execute(
                "SELECT rows_reclassified FROM rule_application_audit WHERE proposal_id=?",
                (proposal["proposal_id"],),
            ).fetchone()[0]
        self.assertEqual(categories, {"Food & Dining"})
        self.assertEqual(audit, 2)

        insert_transaction(
            user_id=self.user["id"], merchant="Paper Boat", amount=100, category="Other"
        )
        rejected = json.loads(financial_agent.propose_classification_rule.invoke({
            "category": "Groceries",
            "pattern": "PAPER BOAT",
            "match_field": "merchant",
            "reason": "User has not confirmed this proposed category.",
            "apply_to_existing": False,
        }))
        self.assertEqual(
            financial_agent.reject_rule_proposal(rejected["proposal_id"])["status"],
            "rejected",
        )

    def test_agent_continuation_and_user_scoped_history(self):
        fake = FakeAgent([
            ("**March 2026**:", "MAX_TOKENS"),
            ("The complete answer is now available.", "STOP"),
        ])
        with patch.object(financial_agent, "get_agent", return_value=fake):
            result = financial_agent.chat_with_agent(
                "Explain March and include every section.",
                "thread-a",
            )
        self.assertEqual(result["completion"]["continuations"], 1)
        self.assertIn("complete answer", result["message"])
        messages = financial_agent.list_chat_messages("thread-a")
        self.assertEqual([item["role"] for item in messages], ["user", "assistant"])
        client = authenticated_client(self.user)
        endpoint_messages = client.get(
            "/api/agent/messages",
            params={"thread_id": "thread-a"},
        )
        self.assertEqual(endpoint_messages.status_code, 200)
        self.assertEqual(
            [item["role"] for item in endpoint_messages.json()],
            ["user", "assistant"],
        )


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.user = register_user("forecast")
        self.context = user_context(self.user["id"])
        self.context.__enter__()
        start = date(2026, 1, 1)
        for offset in range(75):
            if offset % 2 == 0:
                day = start + timedelta(days=offset)
                insert_transaction(
                    user_id=self.user["id"],
                    merchant="Daily Cafe",
                    amount=100 + offset % 7,
                    transaction_date=day.isoformat(),
                    category="Food & Dining",
                )

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_feature_helpers_and_outlier_limit(self):
        days = forecast_model.date_range(date(2026, 1, 1), date(2026, 1, 3))
        self.assertEqual(len(days), 3)
        self.assertEqual(forecast_model.date_range(date(2026, 1, 3), date(2026, 1, 1)), [])
        names = forecast_model.feature_names(["Food & Dining"])
        self.assertIn("lag_28" if "lag_28" in names else "rolling_28", names)
        history = {
            "Food & Dining": {
                date(2026, 1, index + 1): value
                for index, value in enumerate([10, 11, 12, 1000])
            }
        }
        limit = forecast_model.outlier_limits(["Food & Dining"], history)["Food & Dining"]
        self.assertLess(limit, 1000)

    def test_dataset_real_training_cache_and_forecast_windows(self):
        dataset = forecast_model.build_dataset()
        self.assertGreater(len(dataset["X"]), 0)
        self.assertEqual(dataset["X"].shape[0], dataset["y"].shape[0])
        metadata = forecast_model.train_forecast_model(force=True)
        self.assertEqual(metadata["metrics"]["split"], "random 80/20")
        self.assertTrue(runtime_config.model_path().exists())
        cached = forecast_model.train_forecast_model(force=False)
        self.assertEqual(cached["data_version"], metadata["data_version"])
        for horizon in (7, 30):
            with self.subTest(horizon=horizon):
                prediction = forecast_model.forecast_spending(horizon)
                self.assertEqual(len(prediction["daily_forecast"]), horizon)
                self.assertGreaterEqual(prediction["predicted_total"], 0)
                self.assertEqual(prediction["horizon_days"], horizon)
        for invalid in (0, 6, 31, 365):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    forecast_model.forecast_spending(invalid)

    def test_sparse_history_fails_cleanly(self):
        sparse_user = register_user("sparse")
        insert_transaction(
            user_id=sparse_user["id"],
            merchant="One Expense",
            amount=50,
            category="Food & Dining",
        )
        with user_context(sparse_user["id"]):
            with self.assertRaisesRegex(ValueError, "No operating categories"):
                forecast_model.build_dataset()


class SourceIntegrityTests(unittest.TestCase):
    def test_required_frontend_and_deployment_files_exist(self):
        required = [
            "public/index.html",
            "public/login.html",
            "public/app.js",
            "public/login.js",
            "public/styles.css",
            "vercel.json",
            "requirements.txt",
        ]
        for relative_path in required:
            with self.subTest(path=relative_path):
                self.assertTrue(Path(relative_path).is_file())

    def test_vercel_uses_fastapi_auto_detection(self):
        configuration = json.loads(Path("vercel.json").read_text(encoding="utf-8"))
        self.assertEqual(
            configuration.get("$schema"),
            "https://openapi.vercel.sh/vercel.json",
        )
        self.assertNotIn("functions", configuration)
        self.assertTrue(Path("app.py").is_file())

    @unittest.skipUnless(shutil.which("node"), "Node.js is unavailable")
    def test_frontend_javascript_syntax(self):
        for path in ("public/app.js", "public/login.js"):
            with self.subTest(path=path):
                result = subprocess.run(
                    ["node", "--check", path],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_repository_privacy_check(self):
        result = subprocess.run(
            [sys.executable, "scripts/privacy_check.py"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
