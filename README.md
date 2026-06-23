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

The backend is served by FastAPI and Uvicorn. Interactive API documentation is
available locally at `http://127.0.0.1:8000/docs`.

On first run, supported statement CSV files already in the folder are imported into the local `expenses.db`.

## Test suite

Run the isolated comprehensive regression suite with:

```bash
python test.py
```

`test.py` creates temporary per-user databases and never opens the real ledger. It covers
authentication, sessions, rate limiting, cross-user isolation, statement parsing, duplicate
imports, every dashboard filter, upload validation, deterministic analytics, SQL safety,
transaction approvals, adaptive classification rules, chat continuation/history, savings
calculations, a real XGBoost training and 7/30-day forecast cycle, frontend JavaScript syntax,
deployment files, and the repository privacy scan.

Gemini and Vercel Blob network calls are intentionally replaced by deterministic local
boundaries; those external services require separate deployment smoke tests.

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

## Deploy to Vercel

Vercel automatically detects the top-level FastAPI `app` in `app.py`. Do not add
`app.py` under the `functions` block in `vercel.json`; current Vercel function globs
there apply to files inside `api/`, while FastAPI framework detection needs no route
configuration. The repository includes a minimal `vercel.json` and `.python-version`.

1. Import the GitHub repository in Vercel and keep the **FastAPI** preset.
2. Keep the root directory as `./`.
3. Create a **private Vercel Blob** store for persistent database snapshots.
4. Add these Environment Variables in Vercel:

   - `GEMINI_API_KEY`
   - `BLOB_READ_WRITE_TOKEN` — automatically available when the Blob store is connected
   - `RUPEELENS_BLOB_PREFIX=rupeelens`
   - `RUPEELENS_USERNAME=rupeelens` — optional first-account bootstrap
   - `RUPEELENS_PASSWORD` — optional bootstrap password, at least 12 characters
   - `RUPEELENS_SIGNUP_CODE` — required to create accounts through the deployed login page

5. Deploy.

On Vercel, the application restores the private SQLite snapshot from Blob into its
writable `/tmp` directory on cold start. Mutating operations checkpoint and upload
the database before returning. Without `BLOB_READ_WRITE_TOKEN`, the application still
starts, but uploaded data is temporary and may disappear when the function instance
is recycled.

Check deployment storage status at `/api/runtime`.

The application uses scrypt password hashing and revocable random session tokens stored in
`HttpOnly`, `SameSite=Strict`, secure cookies on Vercel. Failed logins are rate-limited.
Every account receives a separate SQLite ledger, forecast model, chat history, adaptive
rules, and private Blob snapshot, so one user cannot query or render another user's data.

For an existing single-user installation, the first account automatically adopts the legacy
`expenses.db` and model artifacts. Additional accounts begin with an empty private workspace.

This snapshot design is intended for a private, single-user personal dashboard. It is
not a concurrent multi-tenant database architecture.

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

Artha can also inspect repeated merchants that remain in **Other** and propose reusable
classification rules. The model cannot activate a rule directly: each proposal shows the
target category, match pattern, sample transactions, and affected row count in the dashboard.
After approval, the rule is stored as `llm_approved`, matching historical **Other** expenses
can be backfilled, and every application is recorded in `rule_application_audit`. New CSV
imports then use the learned rule automatically.

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
