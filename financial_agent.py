"""LangChain Gemini financial agent with guarded SQLite tools."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import statistics
from calendar import monthrange
from datetime import date, datetime
from pathlib import Path
from threading import Lock
from typing import Literal

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from analytics_pipeline import categories_by_type, category_metadata, operating_categories
from forecast_model import forecast_spending

ROOT = Path(__file__).resolve().parent
DATABASE = ROOT / "expenses.db"
ENV_FILE = ROOT / ".env"
MODEL_NAME = "gemini-2.5-flash"
EDITABLE_FIELDS = {"category", "merchant", "notes", "user_label"}
AGENT_LOCK = Lock()
AGENT = None
ACTIVE_THREADS: set[str] = set()


def load_local_environment() -> None:
    if not ENV_FILE.exists():
        return
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def connect(read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        connection = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True, timeout=30)
    else:
        connection = sqlite3.connect(DATABASE, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def rows_as_json(rows: list[sqlite3.Row]) -> str:
    return json.dumps([dict(row) for row in rows], ensure_ascii=False, default=str)


class TransactionSearch(BaseModel):
    merchant: str = Field(default="", description="Merchant or description text to search for.")
    start_date: str = Field(default="", description="Optional inclusive date in YYYY-MM-DD format.")
    end_date: str = Field(default="", description="Optional inclusive date in YYYY-MM-DD format.")
    min_amount: float = Field(default=0, ge=0)
    max_amount: float = Field(default=0, ge=0)
    category: str = Field(default="", description="Optional exact category.")
    limit: int = Field(default=20, ge=1, le=50)


class FinancialSummary(BaseModel):
    start_date: str = Field(default="", description="Optional inclusive date in YYYY-MM-DD format.")
    end_date: str = Field(default="", description="Optional inclusive date in YYYY-MM-DD format.")
    category: str = Field(default="", description="Optional exact category.")


class ProposedEdit(BaseModel):
    transaction_id: int = Field(description="Exact transaction ID returned by search_transactions.")
    category: str | None = Field(default=None, description="Replacement category, if needed.")
    merchant: str | None = Field(default=None, description="Corrected merchant name, if needed.")
    notes: str | None = Field(default=None, description="A factual personal note explaining the expense.")
    user_label: str | None = Field(default=None, description="A short custom label such as Work, Personal, Reimbursable.")
    remember_for_future: bool = Field(
        default=False,
        description="Only true when the user explicitly wants this merchant categorized this way in future imports.",
    )
    reason: str = Field(description="Clear explanation of why this change matches the user's instruction.")


class ScheduledObligation(BaseModel):
    label: str = Field(description="Name of the future payment.")
    due_date: str = Field(description="Payment deadline in YYYY-MM-DD format.")
    amount: float = Field(gt=0, description="Amount due in INR.")


class SavingsPlanInput(BaseModel):
    monthly_income: float = Field(gt=0, description="Current monthly take-home income in INR.")
    average_monthly_spend: float = Field(ge=0, description="Expected monthly operating spend in INR.")
    obligations: list[ScheduledObligation] = Field(description="Future unpaid obligations only.")
    current_reserve: float = Field(default=0, ge=0, description="Money already reserved for these obligations.")
    start_date: str = Field(default="", description="Planning date in YYYY-MM-DD. Leave empty for today.")


class SpendingBaselineInput(BaseModel):
    months: int = Field(
        default=3, ge=2, le=12,
        description="Number of recent complete months to use for the spending baseline.",
    )


class AccountPositionInput(BaseModel):
    as_of_date: str = Field(
        default="",
        description="Optional YYYY-MM-DD date. Returns the latest recorded balance on or before this date.",
    )


class ForecastInput(BaseModel):
    horizon_days: int = Field(
        default=7, ge=7, le=30,
        description="Prediction window from 7 to 30 days.",
    )


@tool
def database_guide() -> str:
    """Return the financial database meaning, available categories, and safe analysis conventions."""
    return json.dumps({
        "table": "transactions",
        "meaning": {
            "direction": "expense means money left the account; income means money entered",
            "amount": "positive INR amount",
            "balance": "account balance recorded by the bank immediately after that transaction",
            "transaction_date": "ISO date YYYY-MM-DD",
            "category": "analytical category",
            "notes": "user-confirmed context about what the expense was",
            "user_label": "user-defined tag",
        },
        "categories": [item["name"] for item in category_metadata()],
        "important_rule": "Transfers and Investments are excluded from true operating/lifestyle spending.",
    })


@tool
def list_database_tables() -> str:
    """List every readable table and view in the SQLite database."""
    with connect(read_only=True) as connection:
        rows = connection.execute("""
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name
        """).fetchall()
    return rows_as_json(rows)


@tool
def describe_database_schema(table_names: str = "") -> str:
    """Return CREATE statements, columns, indexes, and sample rows for requested tables or all tables."""
    requested = [name.strip() for name in table_names.split(",") if name.strip()]
    with connect(read_only=True) as connection:
        available_rows = connection.execute("""
            SELECT name, type, sql
            FROM sqlite_master
            WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """).fetchall()
        available = {row["name"]: row for row in available_rows}
        selected = requested or list(available)
        result = []
        for name in selected:
            if name not in available:
                result.append({"name": name, "error": "Table or view not found."})
                continue
            quoted = '"' + name.replace('"', '""') + '"'
            columns = [
                dict(row) for row in connection.execute(f"PRAGMA table_info({quoted})")
            ]
            indexes = [
                dict(row) for row in connection.execute(f"PRAGMA index_list({quoted})")
            ]
            samples = [
                dict(row) for row in connection.execute(f"SELECT * FROM {quoted} LIMIT 3")
            ]
            result.append({
                "name": name,
                "type": available[name]["type"],
                "create_sql": available[name]["sql"],
                "columns": columns,
                "indexes": indexes,
                "sample_rows": samples,
            })
    return json.dumps(result, ensure_ascii=False, default=str)


@tool(args_schema=AccountPositionInput)
def account_balance_snapshot(as_of_date: str = "") -> str:
    """Return the latest recorded bank balance and its statement date from the ledger."""
    conditions = []
    parameters: list[object] = []
    if as_of_date:
        conditions.append("transaction_date <= ?")
        parameters.append(as_of_date)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with connect(read_only=True) as connection:
        latest = connection.execute(f"""
            SELECT id, transaction_date, balance, merchant, amount, direction,
                   source_file, imported_at
            FROM transactions
            {where}
            ORDER BY transaction_date DESC, id DESC
            LIMIT 1
        """, parameters).fetchone()
        if not latest:
            return json.dumps({"error": "No balance record was found for that date."})
        recent = connection.execute(f"""
            SELECT id, transaction_date, balance, merchant, amount, direction
            FROM transactions
            {where}
            ORDER BY transaction_date DESC, id DESC
            LIMIT 5
        """, parameters).fetchall()
    return json.dumps({
        "recorded_balance": round(latest["balance"], 2),
        "as_of_date": latest["transaction_date"],
        "last_transaction_id": latest["id"],
        "last_transaction": {
            "merchant": latest["merchant"],
            "amount": round(latest["amount"], 2),
            "direction": latest["direction"],
        },
        "source_file": latest["source_file"],
        "recent_balance_records": [dict(row) for row in recent],
        "important_note": (
            "This is the latest closing balance present in the imported statement, "
            "not a live bank balance."
        ),
    }, ensure_ascii=False)


@tool(args_schema=ForecastInput)
def predict_future_spending(horizon_days: int = 7) -> str:
    """Run the trained XGBoost spending forecast for the next 7 to 30 days."""
    return json.dumps(forecast_spending(horizon_days), ensure_ascii=False)


@tool(args_schema=TransactionSearch)
def search_transactions(
    merchant: str = "",
    start_date: str = "",
    end_date: str = "",
    min_amount: float = 0,
    max_amount: float = 0,
    category: str = "",
    limit: int = 20,
) -> str:
    """Find individual transactions before explaining or proposing an edit."""
    conditions = ["1 = 1"]
    parameters: list[object] = []
    if merchant:
        conditions.append("(merchant LIKE ? OR description LIKE ? OR notes LIKE ?)")
        term = f"%{merchant}%"
        parameters.extend((term, term, term))
    if start_date:
        conditions.append("transaction_date >= ?")
        parameters.append(start_date)
    if end_date:
        conditions.append("transaction_date <= ?")
        parameters.append(end_date)
    if min_amount:
        conditions.append("amount >= ?")
        parameters.append(min_amount)
    if max_amount:
        conditions.append("amount <= ?")
        parameters.append(max_amount)
    if category:
        conditions.append("category = ?")
        parameters.append(category)
    parameters.append(limit)
    query = f"""
        SELECT id, transaction_date, merchant, amount, direction, category,
               channel, balance, notes, user_label, description
        FROM transactions
        WHERE {' AND '.join(conditions)}
        ORDER BY transaction_date DESC, id DESC
        LIMIT ?
    """
    with connect(read_only=True) as connection:
        return rows_as_json(connection.execute(query, parameters).fetchall())


@tool(args_schema=FinancialSummary)
def financial_summary(start_date: str = "", end_date: str = "", category: str = "") -> str:
    """Calculate reliable income, spending, transfer, category, and merchant summaries."""
    conditions = ["1 = 1"]
    parameters: list[object] = []
    if start_date:
        conditions.append("transaction_date >= ?")
        parameters.append(start_date)
    if end_date:
        conditions.append("transaction_date <= ?")
        parameters.append(end_date)
    if category:
        conditions.append("category = ?")
        parameters.append(category)
    where = " AND ".join(conditions)
    operating = operating_categories()
    transfers = categories_by_type("transfer")
    investments = categories_by_type("investment")
    operating_marks = ",".join("?" for _ in operating)
    transfer_marks = ",".join("?" for _ in transfers)
    investment_marks = ",".join("?" for _ in investments)
    with connect(read_only=True) as connection:
        totals = connection.execute(f"""
            SELECT
                ROUND(SUM(CASE WHEN direction='income' THEN amount ELSE 0 END), 2) AS income,
                ROUND(SUM(CASE WHEN direction='expense' AND category IN ({operating_marks})
                    THEN amount ELSE 0 END), 2) AS true_spend,
                ROUND(SUM(CASE WHEN direction='expense' AND category IN ({transfer_marks}) THEN amount ELSE 0 END), 2) AS transfers,
                ROUND(SUM(CASE WHEN direction='expense' AND category IN ({investment_marks}) THEN amount ELSE 0 END), 2) AS investments,
                COUNT(*) AS transaction_count
            FROM transactions WHERE {where}
        """, (*operating, *transfers, *investments, *parameters)).fetchone()
        categories = connection.execute(f"""
            SELECT category, ROUND(SUM(amount), 2) AS amount, COUNT(*) AS transactions
            FROM transactions
            WHERE {where} AND direction='expense'
            GROUP BY category ORDER BY amount DESC LIMIT 12
        """, parameters).fetchall()
        merchants = connection.execute(f"""
            SELECT merchant, ROUND(SUM(amount), 2) AS amount, COUNT(*) AS transactions
            FROM transactions
            WHERE {where} AND direction='expense' AND category IN ({operating_marks})
            GROUP BY merchant ORDER BY amount DESC LIMIT 10
        """, (*parameters, *operating)).fetchall()
    return json.dumps({
        "totals": dict(totals),
        "categories": [dict(row) for row in categories],
        "top_merchants": [dict(row) for row in merchants],
    }, ensure_ascii=False)


@tool(args_schema=SpendingBaselineInput)
def current_spending_baseline(months: int = 3) -> str:
    """Return a robust monthly operating-spend baseline from recent complete ledger months."""
    operating = operating_categories()
    marks = ",".join("?" for _ in operating)
    with connect(read_only=True) as connection:
        latest_value = connection.execute(
            "SELECT MAX(transaction_date) AS latest FROM transactions"
        ).fetchone()["latest"]
        if not latest_value:
            return json.dumps({"error": "The ledger contains no transactions."})
        latest = datetime.strptime(latest_value, "%Y-%m-%d").date()
        current_month_complete = latest.day == monthrange(latest.year, latest.month)[1]
        end_year, end_month = latest.year, latest.month
        if not current_month_complete:
            end_month -= 1
            if end_month == 0:
                end_month = 12
                end_year -= 1
        end_date = date(end_year, end_month, monthrange(end_year, end_month)[1])
        start_month_index = end_year * 12 + end_month - (months - 1)
        start_year, start_month_zero = divmod(start_month_index - 1, 12)
        start_date = date(start_year, start_month_zero + 1, 1)
        rows = connection.execute(f"""
            SELECT SUBSTR(transaction_date, 1, 7) AS month,
                   ROUND(SUM(amount), 2) AS spend
            FROM transactions
            WHERE direction='expense' AND category IN ({marks})
              AND transaction_date BETWEEN ? AND ?
            GROUP BY month ORDER BY month
        """, (*operating, start_date.isoformat(), end_date.isoformat())).fetchall()
    monthly = [dict(row) for row in rows]
    values = [row["spend"] for row in monthly]
    if not values:
        return json.dumps({"error": "No complete-month operating spending was found."})
    average = round(statistics.mean(values), 2)
    median = round(statistics.median(values), 2)
    return json.dumps({
        "period_start": start_date.isoformat(),
        "period_end": end_date.isoformat(),
        "complete_months_requested": months,
        "complete_months_found": len(values),
        "monthly_operating_spend": monthly,
        "average_monthly_spend": average,
        "median_monthly_spend": median,
        "recommended_planning_baseline": median,
        "method": (
            "Median recent complete-month operating spend is recommended because it is "
            "less distorted by large one-off expenses."
        ),
    }, ensure_ascii=False)


@tool
def run_read_only_sql(query: str) -> str:
    """Run any single read-only SQLite SELECT or WITH query across the complete database."""
    normalized = re.sub(r"\s+", " ", query.strip()).lower()
    if not normalized.startswith(("select ", "with ")):
        return "Error: only SELECT or WITH queries are permitted."
    if ";" in query.rstrip(";"):
        return "Error: only one SQL statement is permitted."
    blocked = (
        "insert", "update", "delete", "drop", "alter", "attach", "detach",
        "pragma", "vacuum", "replace", "create", "reindex",
    )
    if any(re.search(rf"\b{re.escape(word.strip())}\b", normalized) for word in blocked):
        return "Error: the query contains a prohibited operation."
    if " limit " not in normalized:
        query = f"{query.rstrip(';')} LIMIT 500"
    try:
        with connect(read_only=True) as connection:
            cursor = connection.execute(query)
            rows = cursor.fetchall()
            return json.dumps({
                "columns": [item[0] for item in cursor.description or []],
                "row_count": len(rows),
                "rows": [dict(row) for row in rows],
                "truncated_at": 500 if len(rows) == 500 else None,
            }, ensure_ascii=False, default=str)
    except sqlite3.Error as error:
        return f"SQL error: {error}"


@tool(args_schema=ProposedEdit)
def propose_transaction_edit(
    transaction_id: int,
    reason: str,
    category: str | None = None,
    merchant: str | None = None,
    notes: str | None = None,
    user_label: str | None = None,
    remember_for_future: bool = False,
) -> str:
    """Create a pending transaction correction for user approval; this does not modify the transaction."""
    changes = {
        key: value.strip()
        for key, value in {
            "category": category,
            "merchant": merchant,
            "notes": notes,
            "user_label": user_label,
        }.items()
        if value is not None and value.strip()
    }
    if not changes:
        return "Error: at least one editable field must be supplied."
    allowed_categories = [item["name"] for item in category_metadata()]
    if "category" in changes and changes["category"] not in allowed_categories:
        return f"Error: category must be one of {', '.join(allowed_categories)}."
    if remember_for_future:
        if "category" not in changes:
            return "Error: remember_for_future requires a category change."
        changes["_remember_for_future"] = True
    with connect() as connection:
        transaction = connection.execute(
            "SELECT id, transaction_date, merchant, amount, category, notes, user_label "
            "FROM transactions WHERE id = ?", (transaction_id,)
        ).fetchone()
        if not transaction:
            return "Error: transaction not found. Search again for the exact transaction ID."
        duplicate = connection.execute("""
            SELECT id FROM agent_actions
            WHERE transaction_id = ? AND proposed_changes = ? AND status = 'pending'
        """, (transaction_id, json.dumps(changes, sort_keys=True))).fetchone()
        if duplicate:
            action_id = duplicate["id"]
        else:
            cursor = connection.execute("""
                INSERT INTO agent_actions (
                    transaction_id, proposed_changes, reason, status, created_at
                ) VALUES (?, ?, ?, 'pending', ?)
            """, (
                transaction_id, json.dumps(changes, sort_keys=True),
                reason.strip(), datetime.now().isoformat(timespec="seconds"),
            ))
            action_id = cursor.lastrowid
    return json.dumps({
        "status": "pending_user_approval",
        "action_id": action_id,
        "transaction": dict(transaction),
        "proposed_changes": changes,
        "message": "The database has not been changed. Ask the user to approve the pending action in the dashboard.",
    }, ensure_ascii=False)


@tool(args_schema=SavingsPlanInput)
def build_savings_plan(
    monthly_income: float,
    average_monthly_spend: float,
    obligations: list[ScheduledObligation],
    current_reserve: float = 0,
    start_date: str = "",
) -> str:
    """Calculate a deadline-aware monthly reserve plan for future fees or other obligations."""
    planning_date = (
        datetime.strptime(start_date, "%Y-%m-%d").date()
        if start_date else date.today()
    )
    future = sorted(
        (
            item for item in obligations
            if datetime.strptime(item.due_date, "%Y-%m-%d").date() >= planning_date
        ),
        key=lambda item: item.due_date,
    )
    if not future:
        return json.dumps({"error": "No future obligations were supplied."})

    cumulative_due = 0.0
    binding_reserve = 0.0
    schedule = []
    for item in future:
        due = datetime.strptime(item.due_date, "%Y-%m-%d").date()
        months = max(
            1,
            (due.year - planning_date.year) * 12
            + due.month - planning_date.month
            + (1 if due.day >= planning_date.day else 0),
        )
        cumulative_due += item.amount
        required_monthly = max(0, (cumulative_due - current_reserve) / months)
        binding_reserve = max(binding_reserve, required_monthly)
        schedule.append({
            "label": item.label,
            "due_date": item.due_date,
            "amount": round(item.amount, 2),
            "months_available": months,
            "cumulative_due": round(cumulative_due, 2),
            "minimum_monthly_reserve_by_this_deadline": round(required_monthly, 2),
        })

    available_after_spend = monthly_income - average_monthly_spend
    remaining_buffer = available_after_spend - binding_reserve
    return json.dumps({
        "planning_date": planning_date.isoformat(),
        "monthly_income": round(monthly_income, 2),
        "average_monthly_spend": round(average_monthly_spend, 2),
        "available_after_spend": round(available_after_spend, 2),
        "required_monthly_reserve": round(binding_reserve, 2),
        "remaining_monthly_buffer": round(remaining_buffer, 2),
        "feasible_at_current_spend": remaining_buffer >= 0,
        "current_reserve": round(current_reserve, 2),
        "schedule": schedule,
        "method": "The monthly reserve is the highest cumulative funding requirement across all deadlines.",
    }, ensure_ascii=False)


TOOLS = [
    database_guide,
    list_database_tables,
    describe_database_schema,
    account_balance_snapshot,
    predict_future_spending,
    search_transactions,
    financial_summary,
    current_spending_baseline,
    run_read_only_sql,
    build_savings_plan,
    propose_transaction_edit,
]

SYSTEM_PROMPT = """
You are Artha, the user's personal financial analyst and ledger assistant.
You analyze a private SQLite transaction database denominated in INR.

Your style:
- Be concise, numerate, calm, and specific.
- Cite exact dates, merchants, amounts, and transaction IDs when discussing individual rows.
- Distinguish income, true spending, transfers, and investments.
- Explain assumptions and uncertainty. Never invent a transaction.
- Reproduce dates, amounts, periods, and calculation bases from tool results exactly.
- Never invent or reconstruct a salary, balance, installment amount, deadline, reserve,
  or financial commitment that is absent from the current conversation or tool results.
- If the user refers to a plan "we discussed" but the required figures are unavailable,
  ask them to restate the missing figures instead of guessing.
- Fully answer every distinct part of a multi-line or multi-intent question.
- For complex requests, silently create a checklist and do not finish until every item is addressed.
- Prefer Markdown tables or clearly numbered sections for schedules and month-by-month analysis.
- End complex answers with a brief direct recommendation or conclusion.

Tool discipline:
- You have complete read access to the SQLite database.
- For open-ended analysis, use list_database_tables, inspect the relevant tables with
  describe_database_schema, write the SQL yourself, and execute it with run_read_only_sql.
- You may join, aggregate, filter, use CTEs, window functions, subqueries, and inspect
  sqlite_master through read-only SQL.
- SQL results are capped at 500 rows per call; use LIMIT and OFFSET to paginate when needed.
- Use database_guide for the financial meaning of important fields and category conventions.
- Use account_balance_snapshot whenever the user asks how much money is currently in the
  account, asks for the latest/closing balance, or wants available savings included in a plan.
- Describe it as the latest imported statement balance and always state its exact as-of date.
- Use financial_summary for broad analysis and search_transactions for individual expenses.
- Use predict_future_spending when the user asks what they may spend over the next week
  or month, and clearly state the model confidence and uncertainty range.
- Before any savings, affordability, salary, installment, or future-obligation plan,
  you MUST call current_spending_baseline and use its recommended_planning_baseline.
- When the user asks to include existing bank money in a savings plan, call
  account_balance_snapshot and pass the available amount as current_reserve to build_savings_plan.
- Do not assume the entire account balance is reserved for fees unless the user explicitly says so.
  Show both the balance and the effect if it were fully allocated, then ask or state the assumption.
- Use run_read_only_sql freely whenever direct SQL is the clearest way to answer the question.
- Use build_savings_plan for deadline-based fees, installments, sinking funds, or future obligations.
- Never ask for, expose, or repeat API keys or sensitive bank identifiers.
- Never claim a database edit has happened merely because the user requested it.
- To correct an expense, first identify the exact transaction, then call propose_transaction_edit.
- Set remember_for_future=true only when the user explicitly says this merchant should always use that category.
- A proposed edit requires explicit user approval in the dashboard before it is committed.
- The SQL connection is physically read-only. Never attempt write SQL; transaction corrections
  must still use propose_transaction_edit and require user approval.

When the user's reference is ambiguous (for example, multiple matching Swiggy payments),
show the likely matches and ask which transaction they mean instead of guessing.
"""


def get_agent():
    global AGENT
    if AGENT is None:
        load_local_environment()
        if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is missing from .env.")
        model = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=0.2,
            max_tokens=8192,
            thinking_budget=0,
            retries=2,
        )
        AGENT = create_agent(
            model=model,
            tools=TOOLS,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
            name="artha_financial_agent",
        )
    return AGENT


def save_message(thread_id: str, role: Literal["user", "assistant"], content: str) -> None:
    with connect() as connection:
        connection.execute("""
            INSERT INTO agent_messages (thread_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
        """, (thread_id, role, content, datetime.now().isoformat(timespec="seconds")))


def list_chat_messages(thread_id: str, limit: int = 50) -> list[dict]:
    with connect(read_only=True) as connection:
        rows = connection.execute("""
            SELECT role, content, created_at FROM (
                SELECT id, role, content, created_at
                FROM agent_messages WHERE thread_id = ?
                ORDER BY id DESC LIMIT ?
            ) ORDER BY id
        """, (thread_id, limit)).fetchall()
    return [dict(row) for row in rows]


def finish_reason(message) -> str:
    metadata = getattr(message, "response_metadata", {}) or {}
    reason = metadata.get("finish_reason") or metadata.get("finishReason")
    if not reason:
        candidate = metadata.get("candidate", {})
        reason = candidate.get("finish_reason") or candidate.get("finishReason")
    return str(reason or "").upper()


def response_looks_incomplete(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return True
    if stripped.endswith((",", ":", ";", "-", "(", "[", "**", "*")):
        return True
    if stripped.count("**") % 2:
        return True
    return False


def chat_with_agent(message: str, thread_id: str) -> dict:
    save_message(thread_id, "user", message)
    if thread_id in ACTIVE_THREADS:
        input_messages = [{"role": "user", "content": message}]
    else:
        input_messages = [
            {"role": item["role"], "content": item["content"]}
            for item in list_chat_messages(thread_id, limit=50)
        ]
    with AGENT_LOCK:
        result = get_agent().invoke(
            {"messages": input_messages},
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 50,
            },
        )
        ACTIVE_THREADS.add(thread_id)
        final_message = result["messages"][-1]
        response_parts = [str(final_message.text)]
        reasons = [finish_reason(final_message)]
        continuation_count = 0
        while (
            "MAX_TOKEN" in reasons[-1]
            or response_looks_incomplete(response_parts[-1])
        ) and continuation_count < 2:
            continuation_count += 1
            continuation = get_agent().invoke(
                {"messages": [{
                    "role": "user",
                    "content": (
                        "Your previous answer was cut off. Continue exactly where it stopped. "
                        "Complete every remaining part of my original question. Do not repeat "
                        "sections already provided."
                    ),
                }]},
                config={
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": 50,
                },
            )
            final_message = continuation["messages"][-1]
            response_parts.append(str(final_message.text))
            reasons.append(finish_reason(final_message))
    response = "\n\n".join(part.strip() for part in response_parts if part.strip())
    print(
        f"[Artha] thread={thread_id} chars={len(response)} "
        f"finish={reasons} continuations={continuation_count}"
    )
    save_message(thread_id, "assistant", response)
    return {
        "message": response,
        "model": MODEL_NAME,
        "completion": {
            "finish_reasons": reasons,
            "continuations": continuation_count,
        },
        "actions": list_pending_actions(),
    }


def list_pending_actions() -> list[dict]:
    with connect(read_only=True) as connection:
        rows = connection.execute("""
            SELECT a.id, a.transaction_id, a.proposed_changes, a.reason, a.status,
                   a.created_at, t.transaction_date, t.merchant, t.amount,
                   t.category, t.notes, t.user_label
            FROM agent_actions a
            JOIN transactions t ON t.id = a.transaction_id
            WHERE a.status = 'pending'
            ORDER BY a.id DESC
        """).fetchall()
    actions = []
    for row in rows:
        item = dict(row)
        item["proposed_changes"] = json.loads(item["proposed_changes"])
        actions.append(item)
    return actions


def approve_action(action_id: int) -> dict:
    with connect() as connection:
        action = connection.execute(
            "SELECT * FROM agent_actions WHERE id = ?", (action_id,)
        ).fetchone()
        if not action or action["status"] != "pending":
            raise ValueError("Pending action not found.")
        changes = json.loads(action["proposed_changes"])
        remember_for_future = bool(changes.pop("_remember_for_future", False))
        if not changes or not set(changes).issubset(EDITABLE_FIELDS):
            raise ValueError("Action contains invalid fields.")
        transaction = connection.execute(
            "SELECT * FROM transactions WHERE id = ?", (action["transaction_id"],)
        ).fetchone()
        before = {field: transaction[field] for field in changes}
        assignments = ", ".join(f'"{field}" = ?' for field in changes)
        now = datetime.now().isoformat(timespec="seconds")
        parameters = [changes[field] for field in changes] + [now, action["transaction_id"]]
        connection.execute(
            f'UPDATE transactions SET {assignments}, updated_at = ? WHERE id = ?',
            parameters,
        )
        if remember_for_future and "category" in changes:
            connection.execute("""
                INSERT OR IGNORE INTO classification_rules (
                    category, pattern, match_field, priority, active, source, created_at
                ) VALUES (?, ?, 'merchant', 5, 1, 'user_approved', ?)
            """, (changes["category"], transaction["merchant"].upper(), now))
        connection.execute("""
            INSERT INTO transaction_audit (
                transaction_id, action_id, before_values, after_values, reason, changed_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            action["transaction_id"], action_id, json.dumps(before),
            json.dumps(changes), action["reason"], now,
        ))
        connection.execute(
            "UPDATE agent_actions SET status='approved', resolved_at=? WHERE id=?",
            (now, action_id),
        )
    return {"status": "approved", "action_id": action_id, "transaction_id": action["transaction_id"]}


def reject_action(action_id: int) -> dict:
    with connect() as connection:
        cursor = connection.execute("""
            UPDATE agent_actions SET status='rejected', resolved_at=?
            WHERE id=? AND status='pending'
        """, (datetime.now().isoformat(timespec="seconds"), action_id))
        if not cursor.rowcount:
            raise ValueError("Pending action not found.")
    return {"status": "rejected", "action_id": action_id}
