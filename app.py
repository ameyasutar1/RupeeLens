#!/usr/bin/env python3
"""Persistent local expense intelligence dashboard."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import sqlite3
from threading import Lock
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import uvicorn
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from analytics_pipeline import (
    categories_by_type,
    category_for,
    category_metadata,
    generate_intelligence,
    initialize_pipeline,
    operating_categories,
)
from auth import (
    SESSION_COOKIE,
    authenticate,
    bootstrap_environment_user,
    create_session,
    create_user,
    initialize_auth,
    login_rate_limited,
    record_login_attempt,
    revoke_session,
    session_user,
    signup_allowed,
)
from financial_agent import (
    approve_action,
    approve_rule_proposal,
    chat_with_agent,
    list_chat_messages,
    list_pending_actions,
    list_pending_rule_proposals,
    reject_action,
    reject_rule_proposal,
)
from forecast_model import (
    forecast_spending,
    initialize_forecasting,
    train_forecast_model,
)
from runtime_config import (
    IS_VERCEL,
    STATIC_DIR,
    database_path,
    reset_current_user,
    set_current_user,
)
from storage_backend import (
    persist_auth_database,
    persist_database,
    restore_auth_database,
    restore_database,
    storage_status,
)


ROOT = Path(__file__).resolve().parent
STATEMENT_PATTERN = "AcctStatement_*.csv"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
READY_USERS: set[int] = set()
RUNTIME_LOCK = Lock()


@dataclass(frozen=True)
class Transaction:
    date: datetime
    description: str
    merchant: str
    amount: float
    direction: str
    balance: float
    category: str
    channel: str
    source: str

    def as_dict(self) -> dict:
        return {
            "date": self.date.strftime("%Y-%m-%d"),
            "description": self.description,
            "merchant": self.merchant,
            "amount": round(self.amount, 2),
            "direction": self.direction,
            "balance": round(self.balance, 2),
            "category": self.category,
            "channel": self.channel,
            "source": self.source,
        }


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(database_path(), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def initialize_database() -> None:
    with connect() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS imports (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                file_hash TEXT NOT NULL UNIQUE,
                imported_at TEXT NOT NULL,
                rows_found INTEGER NOT NULL,
                rows_added INTEGER NOT NULL,
                rows_skipped INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY,
                fingerprint TEXT NOT NULL UNIQUE,
                transaction_date TEXT NOT NULL,
                description TEXT NOT NULL,
                merchant TEXT NOT NULL,
                amount REAL NOT NULL,
                direction TEXT NOT NULL,
                balance REAL NOT NULL,
                category TEXT NOT NULL,
                channel TEXT NOT NULL,
                source_file TEXT NOT NULL,
                imported_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date);
            CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category);
            CREATE INDEX IF NOT EXISTS idx_transactions_merchant ON transactions(merchant);
            CREATE TABLE IF NOT EXISTS agent_actions (
                id INTEGER PRIMARY KEY,
                transaction_id INTEGER NOT NULL,
                proposed_changes TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                FOREIGN KEY(transaction_id) REFERENCES transactions(id)
            );
            CREATE TABLE IF NOT EXISTS transaction_audit (
                id INTEGER PRIMARY KEY,
                transaction_id INTEGER NOT NULL,
                action_id INTEGER,
                before_values TEXT NOT NULL,
                after_values TEXT NOT NULL,
                reason TEXT NOT NULL,
                changed_at TEXT NOT NULL,
                FOREIGN KEY(transaction_id) REFERENCES transactions(id),
                FOREIGN KEY(action_id) REFERENCES agent_actions(id)
            );
            CREATE TABLE IF NOT EXISTS agent_messages (
                id INTEGER PRIMARY KEY,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_agent_messages_thread
            ON agent_messages(thread_id, id);
        """)
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(transactions)")
        }
        for column, declaration in (
            ("notes", "TEXT NOT NULL DEFAULT ''"),
            ("user_label", "TEXT NOT NULL DEFAULT ''"),
            ("updated_at", "TEXT"),
        ):
            if column not in columns:
                connection.execute(
                    f'ALTER TABLE transactions ADD COLUMN "{column}" {declaration}'
                )


def number(value: str) -> float:
    cleaned = value.strip().replace(",", "")
    return float(cleaned) if cleaned else 0.0


def merchant_from(description: str) -> str:
    parts = [part.strip() for part in description.split("/")]
    upper = description.upper()
    if upper.startswith("UPI/") and len(parts) > 3:
        merchant = parts[3]
    elif upper.startswith(("POS/", "ECOM PUR/")) and len(parts) > 1:
        merchant = parts[1]
    elif upper.startswith(("NEFT/", "RTGS/")) and len(parts) > 3:
        merchant = parts[3]
    elif upper.startswith(("MOB/", "IFT/", "IMPS/")) and len(parts) > 2:
        merchant = parts[2]
    elif upper.startswith("SB:"):
        merchant = "Bank interest"
    elif upper.startswith("BY CASH DEPOSIT"):
        merchant = "Cash deposit"
    else:
        merchant = parts[0]
    merchant = re.sub(r"\s+", " ", merchant).strip(" -")
    return merchant.title() if merchant else "Unknown"


def channel_from(description: str) -> str:
    upper = description.upper()
    if upper.startswith("UPI/"):
        return "UPI"
    if upper.startswith("POS/"):
        return "Card"
    if upper.startswith("ECOM"):
        return "Online card"
    if upper.startswith(("NEFT/", "RTGS/", "IMPS/", "IFT/", "MOB/")):
        return "Bank transfer"
    if "CASH" in upper or "ATM" in upper:
        return "Cash"
    if upper.startswith("SB:"):
        return "Interest"
    return "Other"


def parse_statement(content: bytes, filename: str) -> list[Transaction]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    rows = list(csv.reader(io.StringIO(text)))
    try:
        header_index = next(
            index for index, row in enumerate(rows)
            if row and row[0].strip() == "Tran Date"
        )
    except StopIteration as error:
        raise ValueError("Could not find the 'Tran Date' statement header.") from error

    transactions = []
    for row in rows[header_index + 1:]:
        if not row or len(row) < 6 or not re.fullmatch(r"\d{2}/\d{2}/\d{2}", row[0].strip()):
            continue
        inflow = number(row[3])
        outflow = number(row[4])
        if not inflow and not outflow:
            continue
        direction = "income" if inflow else "expense"
        description = row[2].strip()
        merchant = merchant_from(description)
        transactions.append(Transaction(
            date=datetime.strptime(row[0].strip(), "%d/%m/%y"),
            description=description,
            merchant=merchant,
            amount=inflow or outflow,
            direction=direction,
            balance=number(row[5]),
            category=category_for(description, merchant, direction),
            channel=channel_from(description),
            source=filename,
        ))
    if not transactions:
        raise ValueError("The file contains no recognizable transaction rows.")
    return transactions


def transaction_fingerprint(item: Transaction) -> str:
    raw = "|".join((
        item.date.strftime("%Y-%m-%d"),
        item.description,
        f"{item.amount:.2f}",
        item.direction,
        f"{item.balance:.2f}",
    ))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def import_statement(content: bytes, filename: str) -> dict:
    file_hash = hashlib.sha256(content).hexdigest()
    with connect() as connection:
        existing = connection.execute(
            "SELECT * FROM imports WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        if existing:
            return {
                "filename": filename,
                "status": "already_imported",
                "found": existing["rows_found"],
                "added": 0,
                "skipped": existing["rows_found"],
            }

        transactions = parse_statement(content, filename)
        imported_at = datetime.now().isoformat(timespec="seconds")
        added = 0
        for item in transactions:
            cursor = connection.execute("""
                INSERT OR IGNORE INTO transactions (
                    fingerprint, transaction_date, description, merchant, amount,
                    direction, balance, category, channel, source_file, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                transaction_fingerprint(item), item.date.strftime("%Y-%m-%d"),
                item.description, item.merchant, item.amount, item.direction,
                item.balance, item.category, item.channel, filename, imported_at,
            ))
            added += cursor.rowcount
        skipped = len(transactions) - added
        connection.execute("""
            INSERT INTO imports (
                filename, file_hash, imported_at, rows_found, rows_added, rows_skipped
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (filename, file_hash, imported_at, len(transactions), added, skipped))
    return {
        "filename": filename,
        "status": "imported",
        "found": len(transactions),
        "added": added,
        "skipped": skipped,
    }


def bootstrap_existing_statements() -> None:
    if IS_VERCEL:
        return
    for path in sorted(ROOT.glob(STATEMENT_PATTERN)):
        import_statement(path.read_bytes(), path.name)


def load_transactions() -> list[Transaction]:
    with connect() as connection:
        rows = connection.execute("""
            SELECT transaction_date, description, merchant, amount, direction,
                   balance, category, channel, source_file
            FROM transactions
            ORDER BY transaction_date, id
        """).fetchall()
    return [
        Transaction(
            date=datetime.strptime(row["transaction_date"], "%Y-%m-%d"),
            description=row["description"],
            merchant=row["merchant"],
            amount=row["amount"],
            direction=row["direction"],
            balance=row["balance"],
            category=row["category"],
            channel=row["channel"],
            source=row["source_file"],
        )
        for row in rows
    ]


def filtered_transactions(params: dict[str, list[str]]) -> list[Transaction]:
    start = params.get("start", [""])[0]
    end = params.get("end", [""])[0]
    category = params.get("category", ["all"])[0]
    direction = params.get("direction", ["all"])[0]
    channel = params.get("channel", ["all"])[0]
    query = params.get("q", [""])[0].strip().lower()
    minimum = number(params.get("min", [""])[0])
    maximum = number(params.get("max", [""])[0])

    items = load_transactions()
    if start:
        start_date = datetime.strptime(start, "%Y-%m-%d")
        items = [item for item in items if item.date >= start_date]
    if end:
        end_date = datetime.strptime(end, "%Y-%m-%d")
        items = [item for item in items if item.date <= end_date]
    if category != "all":
        items = [item for item in items if item.category == category]
    if direction != "all":
        items = [item for item in items if item.direction == direction]
    if channel != "all":
        items = [item for item in items if item.channel == channel]
    if minimum:
        items = [item for item in items if item.amount >= minimum]
    if maximum:
        items = [item for item in items if item.amount <= maximum]
    if query:
        items = [
            item for item in items
            if query in item.merchant.lower() or query in item.description.lower()
        ]
    return items


def sum_amount(items: Iterable[Transaction]) -> float:
    return round(sum(item.amount for item in items), 2)


def dashboard_payload(items: list[Transaction]) -> dict:
    category_config = category_metadata()
    category_colors = {item["name"]: item["color"] for item in category_config}
    operating = operating_categories()
    transfer_categories = categories_by_type("transfer")
    investment_categories = categories_by_type("investment")
    expenses = [item for item in items if item.direction == "expense"]
    income = [item for item in items if item.direction == "income"]
    transfers = [item for item in expenses if item.category in transfer_categories]
    investments = [item for item in expenses if item.category in investment_categories]
    true_spend = [
        item for item in expenses
        if item.category in operating
    ]

    category_totals: defaultdict[str, float] = defaultdict(float)
    category_counts: Counter[str] = Counter()
    merchant_totals: defaultdict[str, float] = defaultdict(float)
    merchant_counts: Counter[str] = Counter()
    channel_totals: defaultdict[str, float] = defaultdict(float)
    monthly: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {"spend": 0.0, "income": 0.0, "transfers": 0.0}
    )
    daily: defaultdict[str, float] = defaultdict(float)

    for item in true_spend:
        category_totals[item.category] += item.amount
        category_counts[item.category] += 1
        merchant_totals[item.merchant] += item.amount
        merchant_counts[item.merchant] += 1
        channel_totals[item.channel] += item.amount
        daily[item.date.strftime("%Y-%m-%d")] += item.amount

    for item in items:
        month = item.date.strftime("%Y-%m")
        if item.direction == "income":
            monthly[month]["income"] += item.amount
        elif item.category in transfer_categories:
            monthly[month]["transfers"] += item.amount
        elif item.category in operating:
            monthly[month]["spend"] += item.amount

    total_spend = sum_amount(true_spend)
    total_income = sum_amount(income)
    category_data = [
        {
            "name": name,
            "amount": round(amount, 2),
            "count": category_counts[name],
            "color": category_colors.get(name, "#94a0ad"),
            "share": round(amount / total_spend * 100, 1) if total_spend else 0,
        }
        for name, amount in sorted(category_totals.items(), key=lambda pair: pair[1], reverse=True)
    ]
    merchant_data = [
        {"name": name, "amount": round(amount, 2), "count": merchant_counts[name]}
        for name, amount in sorted(merchant_totals.items(), key=lambda pair: pair[1], reverse=True)[:12]
    ]
    monthly_data = [
        {"month": month, **{key: round(value, 2) for key, value in totals.items()}}
        for month, totals in sorted(monthly.items())
    ]

    recurring = []
    recurring_groups: defaultdict[str, list[Transaction]] = defaultdict(list)
    for item in true_spend:
        recurring_groups[item.merchant].append(item)
    for merchant, merchant_items in recurring_groups.items():
        active_months = {item.date.strftime("%Y-%m") for item in merchant_items}
        if len(merchant_items) >= 3 and len(active_months) >= 2:
            recurring.append({
                "merchant": merchant,
                "count": len(merchant_items),
                "months": len(active_months),
                "total": sum_amount(merchant_items),
                "average": round(sum_amount(merchant_items) / len(merchant_items), 2),
            })
    recurring.sort(key=lambda item: item["total"], reverse=True)

    largest = sorted(true_spend, key=lambda item: item.amount, reverse=True)[:8]
    active_months = max(len({item.date.strftime("%Y-%m") for item in items}), 1)
    avg_monthly = round(total_spend / active_months, 2)
    top_five_spend = sum(item["amount"] for item in merchant_data[:5])
    operating_surplus = round(total_income - total_spend, 2)
    savings_rate = round(operating_surplus / total_income * 100, 1) if total_income else 0
    concentration = round(top_five_spend / total_spend * 100, 1) if total_spend else 0
    average_expense = round(total_spend / len(true_spend), 2) if true_spend else 0

    insights = []
    if category_data:
        top = category_data[0]
        insights.append({
            "title": f"{top['name']} leads spending",
            "text": f"₹{top['amount']:,.0f} across {top['count']} payments, {top['share']}% of operating spend.",
        })
    if merchant_data:
        top = merchant_data[0]
        insights.append({
            "title": f"{top['name']} is the top merchant",
            "text": f"₹{top['amount']:,.0f} over {top['count']} transactions.",
        })
    if largest:
        insights.append({
            "title": "Largest direct expense",
            "text": f"₹{largest[0].amount:,.0f} paid to {largest[0].merchant} on {largest[0].date.strftime('%d %b %Y')}.",
        })
    insights.append({
        "title": "Analyst view",
        "text": f"Operating surplus is ₹{operating_surplus:,.0f}; the top five merchants represent {concentration}% of spend.",
    })

    return {
        "period": {
            "start": min((item.date for item in items), default=None).strftime("%Y-%m-%d") if items else None,
            "end": max((item.date for item in items), default=None).strftime("%Y-%m-%d") if items else None,
        },
        "summary": {
            "spend": total_spend,
            "income": total_income,
            "transfers": sum_amount(transfers),
            "investments": sum_amount(investments),
            "net": round(total_income - sum_amount(expenses), 2),
            "operatingSurplus": operating_surplus,
            "savingsRate": savings_rate,
            "concentration": concentration,
            "averageExpense": average_expense,
            "averageMonthly": avg_monthly,
            "transactionCount": len(items),
            "expenseCount": len(expenses),
        },
        "configuration": {
            "categories": category_config,
            "channels": sorted({item.channel for item in load_transactions()}),
            "operatingCategories": sorted(operating),
        },
        "categories": category_data,
        "merchants": merchant_data,
        "channels": [
            {"name": name, "amount": round(amount, 2)}
            for name, amount in sorted(channel_totals.items(), key=lambda pair: pair[1], reverse=True)
        ],
        "monthly": monthly_data,
        "daily": [{"date": date, "amount": round(amount, 2)} for date, amount in sorted(daily.items())],
        "recurring": recurring[:8],
        "largest": [item.as_dict() for item in largest],
        "insights": insights,
        "transactions": [item.as_dict() for item in sorted(items, key=lambda item: item.date, reverse=True)],
    }


def import_history() -> list[dict]:
    with connect() as connection:
        rows = connection.execute("""
            SELECT filename, imported_at, rows_found, rows_added, rows_skipped
            FROM imports ORDER BY id DESC LIMIT 20
        """).fetchall()
    return [dict(row) for row in rows]


class AgentChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class AuthCredentials(BaseModel):
    username: str
    password: str


class RegistrationRequest(AuthCredentials):
    display_name: str = ""
    signup_code: str = ""


def ensure_user_runtime(user_id: int) -> None:
    if user_id in READY_USERS:
        return
    with RUNTIME_LOCK:
        if user_id in READY_USERS:
            return
        restore_database()
        initialize_database()
        initialize_pipeline()
        initialize_forecasting()
        READY_USERS.add(user_id)


@asynccontextmanager
async def lifespan(_: FastAPI):
    restore_auth_database()
    initialize_auth()
    try:
        bootstrap_environment_user()
    except ValueError as error:
        print(f"Authentication bootstrap skipped: {error}")
    persist_auth_database()
    print("RupeeLens authentication and per-user storage initialized.")
    yield


app = FastAPI(
    title="Expense Intelligence API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    if IS_VERCEL:
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
    if request.url.path.startswith("/api/auth/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.middleware("http")
async def authenticate_dashboard(request: Request, call_next):
    public_paths = {
        "/login.html", "/login.js", "/styles.css",
        "/api/auth/login", "/api/auth/register", "/api/auth/status",
    }
    path = request.url.path
    if path in public_paths or path.startswith("/favicon"):
        return await call_next(request)

    user = session_user(request.cookies.get(SESSION_COOKIE, ""))
    if not user:
        if path.startswith("/api/"):
            return JSONResponse({"error": "Authentication required."}, status_code=401)
        return RedirectResponse("/login.html", status_code=303)

    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("origin", "")
        expected_origin = f"{request.url.scheme}://{request.url.netloc}"
        if origin and origin != expected_origin:
            return JSONResponse({"error": "Cross-site request rejected."}, status_code=403)

    request.state.user = user
    token = set_current_user(user["id"])
    try:
        ensure_user_runtime(user["id"])
        return await call_next(request)
    finally:
        reset_current_user(token)


def error_response(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=30 * 24 * 60 * 60,
        httponly=True,
        secure=IS_VERCEL,
        samesite="strict",
        path="/",
    )


@app.get("/api/auth/status")
def get_auth_status():
    return {
        "signup_available": signup_allowed("") or bool(os.getenv("RUPEELENS_SIGNUP_CODE")),
        "signup_code_required": bool(os.getenv("RUPEELENS_SIGNUP_CODE")),
    }


@app.post("/api/auth/register")
def register_account(payload: RegistrationRequest, request: Request):
    try:
        user = create_user(
            payload.username,
            payload.password,
            payload.display_name,
            payload.signup_code,
        )
        token, _ = create_session(user["id"], request.headers.get("user-agent", ""))
        persist_auth_database()
        response = JSONResponse({"user": user}, status_code=201)
        set_session_cookie(response, token)
        return response
    except ValueError as error:
        return error_response(str(error), 400)


@app.post("/api/auth/login")
def login_account(payload: AuthCredentials, request: Request):
    remote_address = request.client.host if request.client else "unknown"
    if login_rate_limited(payload.username, remote_address):
        return error_response(
            "Too many failed attempts. Try again in 15 minutes.",
            429,
        )
    user = authenticate(payload.username, payload.password)
    if not user:
        record_login_attempt(payload.username, remote_address, False)
        persist_auth_database()
        return error_response("Invalid username or password.", 401)
    record_login_attempt(payload.username, remote_address, True)
    token, _ = create_session(user["id"], request.headers.get("user-agent", ""))
    persist_auth_database()
    response = JSONResponse({"user": user})
    set_session_cookie(response, token)
    return response


@app.get("/api/auth/me")
def get_current_account(request: Request):
    return {"user": request.state.user}


@app.post("/api/auth/logout")
def logout_account(request: Request):
    revoke_session(request.cookies.get(SESSION_COOKIE, ""))
    persist_auth_database()
    response = JSONResponse({"status": "logged_out"})
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@app.get("/api/dashboard")
def get_dashboard(
    start: str = "",
    end: str = "",
    category: str = "all",
    direction: str = "all",
    channel: str = "all",
    q: str = "",
    min_amount: str = Query(default="", alias="min"),
    max_amount: str = Query(default="", alias="max"),
):
    params = {
        "start": [start],
        "end": [end],
        "category": [category],
        "direction": [direction],
        "channel": [channel],
        "q": [q],
        "min": [min_amount],
        "max": [max_amount],
    }
    try:
        return dashboard_payload(filtered_transactions(params))
    except ValueError as error:
        return error_response(str(error), 400)


@app.get("/api/imports")
def get_imports():
    return import_history()


@app.get("/api/runtime")
def get_runtime():
    return storage_status()


@app.get("/api/intelligence")
def get_intelligence(force: bool = False):
    result = generate_intelligence(force=force)
    persist_database()
    return result


@app.get("/api/forecast")
def get_forecast(horizon: int = Query(default=7, ge=7, le=30)):
    try:
        result = forecast_spending(horizon)
        persist_database()
        return result
    except ValueError as error:
        return error_response(str(error), 400)
    except Exception as error:
        return error_response(f"Forecast failed: {error}", 500)


@app.get("/api/agent/actions")
def get_agent_actions():
    return list_pending_actions()


@app.get("/api/agent/messages")
def get_agent_messages(thread_id: str = "default"):
    return list_chat_messages(thread_id)


@app.get("/api/agent/rules")
def get_agent_rule_proposals():
    return list_pending_rule_proposals()


@app.post("/api/agent/chat")
def post_agent_chat(payload: AgentChatRequest):
    message = payload.message.strip()
    thread_id = payload.thread_id.strip() or "default"
    if not message:
        return error_response("Message cannot be empty.", 400)
    try:
        result = chat_with_agent(message, thread_id)
        persist_database()
        return result
    except ValueError as error:
        return error_response(str(error), 400)
    except Exception as error:
        return error_response(f"Agent request failed: {error}", 500)


@app.post("/api/agent/actions/{action_id}/{operation}")
def resolve_agent_action(
    action_id: int,
    operation: str,
):
    if operation not in {"approve", "reject"}:
        return error_response("Operation must be approve or reject.", 400)
    try:
        result = approve_action(action_id) if operation == "approve" else reject_action(action_id)
        persist_database()
        return result
    except ValueError as error:
        return error_response(str(error), 400)


@app.post("/api/agent/rules/{proposal_id}/{operation}")
def resolve_agent_rule_proposal(
    proposal_id: int,
    operation: str,
):
    if operation not in {"approve", "reject"}:
        return error_response("Operation must be approve or reject.", 400)
    try:
        result = (
            approve_rule_proposal(proposal_id)
            if operation == "approve"
            else reject_rule_proposal(proposal_id)
        )
        persist_database()
        return result
    except ValueError as error:
        return error_response(str(error), 400)


@app.post("/api/upload")
async def upload_statements(
    statements: list[UploadFile] = File(...),
):
    try:
        results = []
        total_size = 0
        for upload in statements:
            filename = Path(upload.filename or "statement.csv").name
            if not filename.lower().endswith(".csv"):
                raise ValueError(f"{filename}: only CSV statements are supported.")
            content = await upload.read()
            total_size += len(content)
            if total_size > MAX_UPLOAD_BYTES:
                raise ValueError("Combined upload size must not exceed 20 MB.")
            results.append(import_statement(content, filename))
        if not results:
            raise ValueError("No CSV file was attached.")

        added = sum(result["added"] for result in results)
        forecast_retrained = False
        forecast_warning = ""
        if added:
            try:
                train_forecast_model(force=True)
                forecast_retrained = True
            except Exception as error:
                forecast_warning = str(error)
        persist_database()
        return {
            "results": results,
            "history": import_history(),
            "intelligence_stale": added > 0,
            "forecast_retrained": forecast_retrained,
            "forecast_warning": forecast_warning,
        }
    except (ValueError, csv.Error) as error:
        return error_response(str(error), 400)
    except Exception as error:
        return error_response(f"Import failed: {error}", 500)


@app.get("/")
def get_index():
    return RedirectResponse("/index.html")


if not IS_VERCEL:
    app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
