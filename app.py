#!/usr/bin/env python3
"""Persistent local expense intelligence dashboard."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from email.parser import BytesParser
from email.policy import default
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse

from analytics_pipeline import (
    categories_by_type,
    category_for,
    category_metadata,
    generate_intelligence,
    initialize_pipeline,
    operating_categories,
)
from financial_agent import (
    approve_action,
    chat_with_agent,
    list_chat_messages,
    list_pending_actions,
    reject_action,
)
from forecast_model import (
    forecast_spending,
    initialize_forecasting,
    train_forecast_model,
)


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
DATABASE = ROOT / "expenses.db"
STATEMENT_PATTERN = "AcctStatement_*.csv"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


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
    connection = sqlite3.connect(DATABASE, timeout=30)
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


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def send_json(self, payload: dict | list, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/dashboard":
            self.send_json(dashboard_payload(filtered_transactions(parse_qs(parsed.query))))
            return
        if parsed.path == "/api/imports":
            self.send_json(import_history())
            return
        if parsed.path == "/api/intelligence":
            force = parse_qs(parsed.query).get("force", ["false"])[0].lower() == "true"
            self.send_json(generate_intelligence(force=force))
            return
        if parsed.path == "/api/forecast":
            try:
                horizon = int(parse_qs(parsed.query).get("horizon", ["7"])[0])
                self.send_json(forecast_spending(horizon))
            except ValueError as error:
                self.send_json({"error": str(error)}, 400)
            except Exception as error:
                self.send_json({"error": f"Forecast failed: {error}"}, 500)
            return
        if parsed.path == "/api/agent/actions":
            self.send_json(list_pending_actions())
            return
        if parsed.path == "/api/agent/messages":
            thread_id = parse_qs(parsed.query).get("thread_id", ["default"])[0]
            self.send_json(list_chat_messages(thread_id))
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/agent/chat":
            try:
                payload = self.read_json_body()
                message = str(payload.get("message", "")).strip()
                thread_id = str(payload.get("thread_id", "default")).strip() or "default"
                if not message:
                    raise ValueError("Message cannot be empty.")
                self.send_json(chat_with_agent(message, thread_id))
            except ValueError as error:
                self.send_json({"error": str(error)}, 400)
            except Exception as error:
                self.send_json({"error": f"Agent request failed: {error}"}, 500)
            return
        action_match = re.fullmatch(r"/api/agent/actions/(\d+)/(approve|reject)", parsed.path)
        if action_match:
            try:
                action_id = int(action_match.group(1))
                operation = action_match.group(2)
                result = approve_action(action_id) if operation == "approve" else reject_action(action_id)
                self.send_json(result)
            except ValueError as error:
                self.send_json({"error": str(error)}, 400)
            return
        if parsed.path != "/api/upload":
            self.send_json({"error": "Not found"}, 404)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if not content_length or content_length > MAX_UPLOAD_BYTES:
                raise ValueError("Upload must be between 1 byte and 20 MB.")
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                raise ValueError("Expected a multipart file upload.")
            body = self.rfile.read(content_length)
            message = BytesParser(policy=default).parsebytes(
                f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
            )
            results = []
            for part in message.iter_attachments():
                filename = Path(part.get_filename() or "statement.csv").name
                if not filename.lower().endswith(".csv"):
                    raise ValueError(f"{filename}: only CSV statements are supported.")
                results.append(import_statement(part.get_payload(decode=True), filename))
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
            self.send_json({
                "results": results,
                "history": import_history(),
                "intelligence_stale": added > 0,
                "forecast_retrained": forecast_retrained,
                "forecast_warning": forecast_warning,
            })
        except (ValueError, csv.Error) as error:
            self.send_json({"error": str(error)}, 400)
        except Exception as error:
            self.send_json({"error": f"Import failed: {error}"}, 500)

    def read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        if not content_length or content_length > 1_000_000:
            raise ValueError("Invalid request size.")
        try:
            return json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError as error:
            raise ValueError("Invalid JSON request.") from error


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    count = len(load_transactions())
    print(f"Expense dashboard running at http://{host}:{port}")
    print(f"SQLite database: {DATABASE.name} · {count:,} transactions")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


initialize_database()
initialize_pipeline()
initialize_forecasting()
bootstrap_existing_statements()

if __name__ == "__main__":
    run()
