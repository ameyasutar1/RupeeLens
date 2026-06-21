"""Database-driven classification and adaptive financial intelligence pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import statistics
from calendar import monthrange
from collections import defaultdict
from datetime import datetime
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from runtime_config import ENV_FILE, database_path


MODEL_NAME = "gemini-2.5-flash"

DEFAULT_CATEGORIES = [
    ("Income", "#23875b", "income", 0, 10),
    ("Transfers", "#7c6cf2", "transfer", 0, 20),
    ("Investments", "#137d68", "investment", 0, 30),
    ("Education", "#2e7dff", "operating", 1, 40),
    ("Insurance", "#d95d90", "operating", 1, 50),
    ("Food & Dining", "#ff6b45", "operating", 1, 60),
    ("Groceries", "#34b27b", "operating", 1, 70),
    ("Shopping", "#f2b84b", "operating", 1, 80),
    ("Transport", "#22a7b8", "operating", 1, 90),
    ("Health", "#ef5f67", "operating", 1, 100),
    ("Subscriptions", "#8d69b5", "operating", 1, 110),
    ("Bills & Utilities", "#6b8afd", "operating", 1, 120),
    ("Other", "#94a0ad", "operating", 1, 999),
]

DEFAULT_RULES = {
    "Education": (
        "UNIVERSITY", "COLLEGE", "INSTITUTE", "SCHOOL", "TUITION",
        "COURSE", "EDUCATION", "UDEMY", "COURSERA",
    ),
    "Insurance": ("LIFE INSUR", "GENERAL INSU", "INSURANCE", "PREMIUM"),
    "Investments": ("ICCL GROW", "MUTUAL", "ZERODHA", "GROWW", "UPSTOX"),
    "Transfers": ("P2A", "TPFT", "RTGS/", "IMPS/", "AC XFR", "SELF TRANSFER"),
    "Food & Dining": (
        "SWIGGY", "ZOMATO", "RESTAURANT", "CAFE", "COFFEE", "CATERER",
        "CANTEEN", "FOOD", "DINING", "MCDONALD", "DOMINO",
    ),
    "Groceries": (
        "DMART", "SUPER MARKET", "SUPERMARKET", "SUPER SHOPPE", "INSTAMART",
        "BLINKIT", "ZEPTO", "BIGBASKET", "GROCERY", "FRESHMART",
    ),
    "Shopping": (
        "AMAZON", "MYNTRA", "H AND M", "H&M", "DECATHLON", "LENSKART", "FLIPKART",
        "AJIO", "RELIANCE RETAIL", "ZUDIO", "IKEA", "NYKAA",
    ),
    "Transport": (
        "METRO", "UBER", "OLA", "RAPIDO", "INDIGO", "CLEARTRIP", "IRCTC",
        "PETROLEUM", "PETROL", "FUEL", "MACHNIFY", "BIKE SERVICE", "MAKEMYTRIP",
    ),
    "Health": ("APPLE MED", "MEDICO", "HOSP", "PHARMA", "MEDICAL", "CLINIC", "HEALTH"),
    "Subscriptions": (
        "GOOGLE", "PLAYSTATION", "NETFLIX", "SPOTIFY", "YOUTUBE", "APPLE.COM",
        "MICROSOFT", "ADOBE", "CLOUD", "HOTSTAR", "JIOHOTSTAR",
    ),
    "Bills & Utilities": (
        "ELECTRIC", "MAHAVITARAN", "AIRTEL", "JIO ", "VODAFONE", "BROADBAND",
        "MSEDCL", "BILLPAY", "GAS", "RECHARGE",
    ),
}


def connect(read_only: bool = False) -> sqlite3.Connection:
    database = database_path()
    if read_only:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30)
    else:
        connection = sqlite3.connect(database, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def initialize_pipeline() -> None:
    with connect() as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS category_config (
                name TEXT PRIMARY KEY,
                color TEXT NOT NULL,
                category_type TEXT NOT NULL DEFAULT 'operating',
                is_operating INTEGER NOT NULL DEFAULT 1,
                display_order INTEGER NOT NULL DEFAULT 999,
                active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS classification_rules (
                id INTEGER PRIMARY KEY,
                category TEXT NOT NULL,
                pattern TEXT NOT NULL,
                match_field TEXT NOT NULL DEFAULT 'description',
                priority INTEGER NOT NULL DEFAULT 100,
                active INTEGER NOT NULL DEFAULT 1,
                source TEXT NOT NULL DEFAULT 'system',
                created_at TEXT NOT NULL,
                UNIQUE(category, pattern, match_field)
            );
            CREATE TABLE IF NOT EXISTS classification_rule_proposals (
                id INTEGER PRIMARY KEY,
                category TEXT NOT NULL,
                pattern TEXT NOT NULL,
                match_field TEXT NOT NULL DEFAULT 'merchant',
                priority INTEGER NOT NULL DEFAULT 5,
                reason TEXT NOT NULL,
                apply_to_existing INTEGER NOT NULL DEFAULT 1,
                affected_count INTEGER NOT NULL DEFAULT 0,
                sample_transactions TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                resolved_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_rule_proposals_status
            ON classification_rule_proposals(status, id);
            CREATE TABLE IF NOT EXISTS rule_application_audit (
                id INTEGER PRIMARY KEY,
                proposal_id INTEGER NOT NULL,
                rule_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                pattern TEXT NOT NULL,
                match_field TEXT NOT NULL,
                rows_reclassified INTEGER NOT NULL DEFAULT 0,
                applied_at TEXT NOT NULL,
                FOREIGN KEY(proposal_id) REFERENCES classification_rule_proposals(id),
                FOREIGN KEY(rule_id) REFERENCES classification_rules(id)
            );
            CREATE TABLE IF NOT EXISTS intelligence_snapshots (
                id INTEGER PRIMARY KEY,
                data_version TEXT NOT NULL UNIQUE,
                deterministic_data TEXT NOT NULL,
                ai_narrative TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
        category_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(category_config)")
        }
        if "category_type" not in category_columns:
            connection.execute(
                "ALTER TABLE category_config ADD COLUMN category_type TEXT NOT NULL DEFAULT 'operating'"
            )
        connection.executemany("""
            INSERT OR IGNORE INTO category_config
                (name, color, category_type, is_operating, display_order)
            VALUES (?, ?, ?, ?, ?)
        """, DEFAULT_CATEGORIES)
        connection.executemany(
            "UPDATE category_config SET category_type=? WHERE name=?",
            [(item[2], item[0]) for item in DEFAULT_CATEGORIES],
        )
        now = datetime.now().isoformat(timespec="seconds")
        priority = 10
        for category, patterns in DEFAULT_RULES.items():
            for pattern in patterns:
                connection.execute("""
                    INSERT OR IGNORE INTO classification_rules
                        (category, pattern, match_field, priority, source, created_at)
                    VALUES (?, ?, 'description', ?, 'seed', ?)
                """, (category, pattern, priority, now))
            priority += 10


def category_for(description: str, merchant: str, direction: str) -> str:
    if direction == "income":
        return "Income"
    with connect(read_only=True) as connection:
        rules = connection.execute("""
            SELECT category, pattern, match_field
            FROM classification_rules
            WHERE active = 1
            ORDER BY priority, id
        """).fetchall()
    values = {"description": description.upper(), "merchant": merchant.upper()}
    for rule in rules:
        if rule["pattern"].upper() in values.get(rule["match_field"], values["description"]):
            return rule["category"]
    return "Other"


def category_metadata() -> list[dict]:
    with connect(read_only=True) as connection:
        rows = connection.execute("""
            SELECT name, color, category_type, is_operating, display_order
            FROM category_config WHERE active = 1
            ORDER BY display_order, name
        """).fetchall()
    return [dict(row) for row in rows]


def operating_categories() -> set[str]:
    return {
        item["name"] for item in category_metadata()
        if item["is_operating"]
    }


def categories_by_type(category_type: str) -> set[str]:
    return {
        item["name"] for item in category_metadata()
        if item["category_type"] == category_type
    }


def data_version() -> str:
    with connect(read_only=True) as connection:
        row = connection.execute("""
            SELECT COUNT(*) AS count, COALESCE(MAX(imported_at), '') AS imported,
                   COALESCE(MAX(updated_at), '') AS updated,
                   ROUND(COALESCE(SUM(amount), 0), 2) AS amount
            FROM transactions
        """).fetchone()
        rules = connection.execute("""
            SELECT category, pattern, match_field, priority, active
            FROM classification_rules ORDER BY id
        """).fetchall()
        categories = connection.execute("""
            SELECT name, color, category_type, is_operating, display_order, active
            FROM category_config ORDER BY name
        """).fetchall()
    raw = json.dumps({
        "data": dict(row),
        "rules": [dict(item) for item in rules],
        "categories": [dict(item) for item in categories],
    }, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def deterministic_analysis() -> dict:
    operating = operating_categories()
    placeholders = ",".join("?" for _ in operating)
    with connect(read_only=True) as connection:
        monthly_rows = connection.execute(f"""
            SELECT SUBSTR(transaction_date, 1, 7) AS month,
                   ROUND(SUM(CASE WHEN direction='income' THEN amount ELSE 0 END), 2) AS income,
                   ROUND(SUM(CASE WHEN direction='expense' AND category IN ({placeholders})
                       THEN amount ELSE 0 END), 2) AS spend
            FROM transactions GROUP BY month ORDER BY month
        """, tuple(operating)).fetchall()
        category_rows = connection.execute(f"""
            SELECT category, ROUND(SUM(amount), 2) AS amount, COUNT(*) AS count
            FROM transactions
            WHERE direction='expense' AND category IN ({placeholders})
            GROUP BY category ORDER BY amount DESC
        """, tuple(operating)).fetchall()
        merchant_rows = connection.execute(f"""
            SELECT merchant, ROUND(SUM(amount), 2) AS amount, COUNT(*) AS count
            FROM transactions
            WHERE direction='expense' AND category IN ({placeholders})
            GROUP BY merchant ORDER BY amount DESC LIMIT 15
        """, tuple(operating)).fetchall()
        expense_rows = connection.execute(f"""
            SELECT id, transaction_date, merchant, amount, category
            FROM transactions
            WHERE direction='expense' AND category IN ({placeholders})
            ORDER BY amount DESC LIMIT 20
        """, tuple(operating)).fetchall()
        quality = connection.execute("""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN category='Other' THEN 1 ELSE 0 END) AS uncategorized,
                   COUNT(DISTINCT merchant) AS merchants,
                   MIN(transaction_date) AS first_date,
                   MAX(transaction_date) AS last_date
            FROM transactions
        """).fetchone()
        uncategorized = connection.execute("""
            SELECT merchant, COUNT(*) AS count, ROUND(SUM(amount), 2) AS amount
            FROM transactions
            WHERE direction='expense' AND category='Other'
            GROUP BY merchant ORDER BY amount DESC LIMIT 10
        """).fetchall()

    monthly = [dict(row) for row in monthly_rows]
    categories = [dict(row) for row in category_rows]
    merchants = [dict(row) for row in merchant_rows]
    largest = [dict(row) for row in expense_rows]
    latest = monthly[-1] if monthly else {"month": "", "spend": 0, "income": 0}
    previous = monthly[-2] if len(monthly) > 1 else {"month": "", "spend": 0, "income": 0}
    comparison_basis = "full_month"
    if quality["last_date"] and latest["month"]:
        last_date = datetime.strptime(quality["last_date"], "%Y-%m-%d")
        if last_date.day < monthrange(last_date.year, last_date.month)[1]:
            previous_year = last_date.year if last_date.month > 1 else last_date.year - 1
            previous_month = last_date.month - 1 if last_date.month > 1 else 12
            previous_day = min(last_date.day, monthrange(previous_year, previous_month)[1])
            previous_start = f"{previous_year:04d}-{previous_month:02d}-01"
            previous_end = f"{previous_year:04d}-{previous_month:02d}-{previous_day:02d}"
            with connect(read_only=True) as connection:
                comparable = connection.execute(f"""
                    SELECT ROUND(COALESCE(SUM(amount), 0), 2) AS spend
                    FROM transactions
                    WHERE direction='expense' AND category IN ({placeholders})
                      AND transaction_date BETWEEN ? AND ?
                """, (*operating, previous_start, previous_end)).fetchone()
            previous = {
                "month": f"{previous_year:04d}-{previous_month:02d}",
                "spend": comparable["spend"],
                "income": previous.get("income", 0),
                "through_day": previous_day,
            }
            latest = {**latest, "through_day": last_date.day}
            comparison_basis = "month_to_date"
    spend_change = (
        round((latest["spend"] - previous["spend"]) / previous["spend"] * 100, 1)
        if previous["spend"] else 0
    )
    category_total = sum(row["amount"] for row in categories)
    concentration = (
        round(sum(row["amount"] for row in merchants[:5]) / category_total * 100, 1)
        if category_total else 0
    )
    amounts = [row["amount"] for row in largest]
    anomaly_threshold = (
        round(statistics.median(amounts) * 2.5, 2) if amounts else 0
    )
    anomalies = [row for row in largest if row["amount"] >= anomaly_threshold][:6]
    other_share = round(quality["uncategorized"] / quality["total"] * 100, 1) if quality["total"] else 0

    findings = []
    if latest["month"]:
        direction = "higher" if spend_change > 0 else "lower"
        comparison_note = (
            f" through day {latest['through_day']}"
            if comparison_basis == "month_to_date" else ""
        )
        findings.append({
            "type": "trend",
            "title": f"Latest month is {abs(spend_change):.1f}% {direction}",
            "detail": f"{latest['month']}{comparison_note} operating spend was ₹{latest['spend']:,.0f}, compared with ₹{previous['spend']:,.0f} on the same basis.",
            "severity": "watch" if spend_change > 20 else "positive" if spend_change < -10 else "neutral",
        })
    if categories:
        findings.append({
            "type": "mix",
            "title": f"{categories[0]['category']} dominates the expense mix",
            "detail": f"₹{categories[0]['amount']:,.0f} across {categories[0]['count']} transactions.",
            "severity": "neutral",
        })
    if other_share > 20:
        findings.append({
            "type": "data_quality",
            "title": "Classification quality needs attention",
            "detail": f"{other_share}% of transactions remain in Other. Teaching Artha about these expenses will improve future analysis.",
            "severity": "watch",
        })
    return {
        "version": data_version(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "coverage": dict(quality),
        "latest_month": latest,
        "previous_month": previous,
        "spend_change_percent": spend_change,
        "comparison_basis": comparison_basis,
        "merchant_concentration_percent": concentration,
        "category_mix": categories,
        "top_merchants": merchants,
        "anomalies": anomalies,
        "findings": findings,
        "data_quality": {"other_share_percent": other_share},
        "classification_opportunities": [dict(row) for row in uncategorized],
    }


class NarrativeCard(BaseModel):
    label: str
    value: str
    context: str
    tone: Literal["positive", "neutral", "watch"]


class IntelligenceNarrative(BaseModel):
    headline: str
    executive_summary: str
    cards: list[NarrativeCard]
    findings: list[str]
    watchlist: list[str] = Field(default_factory=list)
    recommended_questions: list[str] = Field(default_factory=list)
    layout_priority: list[
        Literal["trend", "categories", "merchants", "anomalies", "data_quality"]
    ] = Field(default_factory=lambda: [
        "trend", "categories", "data_quality", "merchants", "anomalies"
    ])


def load_environment() -> None:
    if not ENV_FILE.exists():
        return
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def rule_based_narrative(analysis: dict) -> dict:
    latest = analysis["latest_month"]
    top = analysis["category_mix"][0] if analysis["category_mix"] else {"category": "No category", "amount": 0}
    return {
        "headline": "Your financial picture has been recalculated from the latest ledger.",
        "executive_summary": (
            f"Operating spend in {latest['month'] or 'the latest period'} is ₹{latest['spend']:,.0f}. "
            f"{top['category']} is currently the largest category, while "
            f"{analysis['data_quality']['other_share_percent']}% of transactions remain uncategorized."
        ),
        "cards": [
            {"label": "Latest operating spend", "value": f"₹{latest['spend']:,.0f}", "context": latest["month"], "tone": "neutral"},
            {"label": "Month-on-month movement", "value": f"{analysis['spend_change_percent']:+.1f}%", "context": "versus previous month", "tone": "watch" if analysis["spend_change_percent"] > 20 else "neutral"},
            {"label": "Merchant concentration", "value": f"{analysis['merchant_concentration_percent']}%", "context": "top five merchant share", "tone": "neutral"},
        ],
        "findings": [item["detail"] for item in analysis["findings"]] or ["More data is needed for reliable trend analysis."],
        "watchlist": [f"Review {item['merchant']} at ₹{item['amount']:,.0f}" for item in analysis["anomalies"]],
        "recommended_questions": [
            "Why did spending change in the latest month?",
            "Which Other transactions should I classify?",
            "Which merchants are becoming more expensive?",
        ],
        "layout_priority": ["trend", "categories", "data_quality", "merchants", "anomalies"],
    }


def generate_intelligence(force: bool = False) -> dict:
    analysis = deterministic_analysis()
    version = analysis["version"]
    with connect(read_only=True) as connection:
        cached = connection.execute(
            "SELECT * FROM intelligence_snapshots WHERE data_version = ?", (version,)
        ).fetchone()
    if cached and not force:
        return {
            "version": version,
            "created_at": cached["created_at"],
            "analysis": json.loads(cached["deterministic_data"]),
            "narrative": json.loads(cached["ai_narrative"]),
            "source": "cache",
        }

    narrative = rule_based_narrative(analysis)
    source = "rules"
    try:
        load_environment()
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            model = ChatGoogleGenerativeAI(
                model=MODEL_NAME, temperature=0.2, max_tokens=2400,
                thinking_budget=0, retries=2
            ).with_structured_output(IntelligenceNarrative, method="json_schema")
            prompt = f"""
You are the narrative layer of a personal finance dashboard.
Use only the supplied deterministic analysis. Do not invent facts.
The dashboard should evolve with the data: prioritize the most decision-useful sections,
write a concise executive summary, and create KPI cards whose labels and values reflect
what is currently important. INR currency. Avoid financial advice or moral judgment.

ANALYSIS:
{json.dumps(analysis, ensure_ascii=False)}
"""
            response = model.invoke(prompt)
            narrative = response.model_dump()
            fallback = rule_based_narrative(analysis)
            for key, minimum in (("cards", 3), ("findings", 3), ("recommended_questions", 3)):
                if len(narrative.get(key, [])) < minimum:
                    existing = narrative.get(key, [])
                    additions = [item for item in fallback[key] if item not in existing]
                    narrative[key] = (existing + additions)[:max(minimum, len(existing))]
            if not narrative.get("layout_priority"):
                narrative["layout_priority"] = fallback["layout_priority"]
            source = "gemini"
    except Exception:
        source = "rules_fallback"

    created_at = datetime.now().isoformat(timespec="seconds")
    with connect() as connection:
        connection.execute("""
            INSERT OR REPLACE INTO intelligence_snapshots
                (data_version, deterministic_data, ai_narrative, model, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            version, json.dumps(analysis, ensure_ascii=False),
            json.dumps(narrative, ensure_ascii=False), MODEL_NAME, created_at,
        ))
    return {
        "version": version,
        "created_at": created_at,
        "analysis": analysis,
        "narrative": narrative,
        "source": source,
    }
