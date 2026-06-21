# Expense Intelligence Dashboard

A persistent local Python, HTML, CSS, JavaScript, and SQLite dashboard for supported bank-statement CSV exports.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your own Gemini API key to `.env`, then run:

```bash
python app.py
```

Open `http://127.0.0.1:8000`.

On first run, supported statement CSV files already in the folder are imported into the local `expenses.db`.

Use **Import statement** or the dashboard drop zone whenever you download a new CSV. The import process:

- appends new transactions to SQLite;
- skips an identical file if it was already imported;
- skips overlapping transaction rows in cumulative statements;
- records an import audit trail;
- refreshes the analysis immediately.

Statement data never leaves the computer.

## Privacy before publishing

The repository includes a strict `.gitignore` for:

- `.env` and local API keys;
- bank statement CSV files;
- SQLite databases and WAL files;
- trained model artifacts;
- Python environments, caches, logs, and OS metadata.

Run this before every push:

```bash
python scripts/privacy_check.py
```

Never force-add an ignored financial or secret file.

## Artha AI agent

The dashboard includes a LangChain agent powered by `gemini-2.5-flash`. It loads `GEMINI_API_KEY` from `.env`.

Artha can:

- answer natural-language questions about the ledger;
- inspect every SQLite table and schema, then write and execute her own read-only SQL;
- calculate financial summaries and trends;
- search exact transactions by merchant, date, amount, or category;
- run guarded read-only SQL analysis;
- propose category, merchant, note, or custom-label corrections.

Artha has complete read access across the SQLite database, including transactions, imports, configuration, audit history, and intelligence snapshots. She can use joins, CTEs, window functions, aggregations, subqueries, and schema metadata. SQL result sets are capped at 500 rows per call and can be paginated.

The SQL connection is opened in SQLite read-only mode and the SQL tool accepts only a single `SELECT` or `WITH` statement. The model cannot execute raw write SQL. Every correction appears as a pending action in the dashboard and requires explicit approval. Approved changes are stored in `transaction_audit`.

## Adaptive analytics pipeline

The dashboard is intentionally split into five backend stages:

1. **Ingestion** appends duplicate-safe bank transactions to SQLite.
2. **Classification rules** in `classification_rules` categorize new transactions.
3. **Deterministic analytics** calculates comparable trends, category mix, anomalies, merchant concentration, and data-quality gaps.
4. **Gemini narrative** converts those verified numbers into an executive briefing, adaptive KPI cards, watchlists, and suggested questions.
5. **Versioned snapshots** cache the narrative in `intelligence_snapshots`. New transactions, approved edits, category configuration, or rule changes create a new data version and therefore a new dashboard story.

Categories, colors, display order, and financial roles are stored in `category_config`; the frontend reads them from the API rather than maintaining a fixed list.

If you explicitly ask Artha to remember a category for a merchant, the proposed change still requires approval. Approval updates the transaction and creates a reusable `user_approved` classification rule for future imports.

## XGBoost spending forecast

The dashboard trains a two-stage XGBoost model that predicts daily spending probability and conditional spend amount by category.

- Uses a reproducible random 80/20 split (`random_state=42`).
- Removes per-category outliers with the IQR upper fence before splitting.
- Uses calendar, category, lag-1/7/14, rolling-7/28, and recent-frequency features.
- Excludes categories with fewer than 12 active spending days from behavioral forecasts.
- Produces recursive forecasts from 7 to 30 days after the newest imported transaction.
- Stores the model at `artifacts/spending_forecast.joblib`.
- Records each training run and holdout metrics in `forecast_model_runs`.
- Retrains automatically on all accumulated history whenever an uploaded CSV adds transactions.

Personal spending is noisy and the current holdout score is low, so forecasts are presented with validation metrics and uncertainty rather than as guaranteed future expenses.

## How totals are interpreted

In the currently supported export format, money entering the account appears in the `DR` field and money leaving appears in `CR`. The dashboard follows the balance movement rather than relying on the conventional meaning of those labels.

Lifestyle spending excludes account/person transfers and investments. Those amounts remain visible in separate summary cards and in the full ledger. Analyst KPIs show operating surplus, savings capacity, merchant concentration, recurring merchants, and spending composition.
