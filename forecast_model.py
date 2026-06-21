"""XGBoost spending forecast pipeline for daily category-level predictions."""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from analytics_pipeline import category_metadata, data_version, operating_categories


ROOT = Path(__file__).resolve().parent
DATABASE = ROOT / "expenses.db"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "spending_forecast.joblib"
RANDOM_STATE = 42
MIN_HISTORY_DAYS = 28
MIN_CATEGORY_ACTIVE_DAYS = 12


def connect(read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        connection = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True, timeout=30)
    else:
        connection = sqlite3.connect(DATABASE, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def initialize_forecasting() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    with connect() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS forecast_model_runs (
                id INTEGER PRIMARY KEY,
                data_version TEXT NOT NULL,
                trained_at TEXT NOT NULL,
                train_rows INTEGER NOT NULL,
                test_rows INTEGER NOT NULL,
                outliers_removed INTEGER NOT NULL,
                metrics TEXT NOT NULL,
                feature_importance TEXT NOT NULL,
                model_path TEXT NOT NULL
            )
        """)


def date_range(start: date, end: date) -> list[date]:
    return [
        start + timedelta(days=offset)
        for offset in range((end - start).days + 1)
    ]


def load_daily_history() -> tuple[list[str], list[date], dict[str, dict[date, float]]]:
    configured = sorted(operating_categories())
    marks = ",".join("?" for _ in configured)
    with connect(read_only=True) as connection:
        eligible_rows = connection.execute(f"""
            SELECT category, COUNT(DISTINCT transaction_date) AS active_days
            FROM transactions
            WHERE direction='expense' AND category IN ({marks})
            GROUP BY category
            HAVING active_days >= ?
            ORDER BY category
        """, (*configured, MIN_CATEGORY_ACTIVE_DAYS)).fetchall()
    categories = [row["category"] for row in eligible_rows]
    if not categories:
        raise ValueError("No operating categories are configured.")
    marks = ",".join("?" for _ in categories)
    with connect(read_only=True) as connection:
        rows = connection.execute(f"""
            SELECT transaction_date, category, ROUND(SUM(amount), 2) AS amount
            FROM transactions
            WHERE direction='expense' AND category IN ({marks})
            GROUP BY transaction_date, category
            ORDER BY transaction_date
        """, tuple(categories)).fetchall()
    if not rows:
        raise ValueError("No operating expense history is available.")
    first = datetime.strptime(rows[0]["transaction_date"], "%Y-%m-%d").date()
    last = datetime.strptime(rows[-1]["transaction_date"], "%Y-%m-%d").date()
    dates = date_range(first, last)
    history = {category: {day: 0.0 for day in dates} for category in categories}
    for row in rows:
        day = datetime.strptime(row["transaction_date"], "%Y-%m-%d").date()
        history[row["category"]][day] = float(row["amount"])
    return categories, dates, history


def feature_names(categories: list[str]) -> list[str]:
    return [
        *(f"category={category}" for category in categories),
        "day_of_week_sin", "day_of_week_cos", "day_of_month_sin",
        "day_of_month_cos", "month_sin", "month_cos", "is_weekend",
        "lag_1", "lag_7", "lag_14", "rolling_7", "rolling_28",
        "spend_days_28",
    ]


def feature_row(
    day: date,
    category: str,
    categories: list[str],
    category_history: dict[date, float],
) -> list[float]:
    values = [1.0 if item == category else 0.0 for item in categories]
    values.extend([
        math.sin(2 * math.pi * day.weekday() / 7),
        math.cos(2 * math.pi * day.weekday() / 7),
        math.sin(2 * math.pi * day.day / 31),
        math.cos(2 * math.pi * day.day / 31),
        math.sin(2 * math.pi * day.month / 12),
        math.cos(2 * math.pi * day.month / 12),
        1.0 if day.weekday() >= 5 else 0.0,
    ])
    prior = [
        category_history.get(day - timedelta(days=offset), 0.0)
        for offset in range(1, 29)
    ]
    values.extend([
        prior[0],
        prior[6],
        prior[13],
        float(np.mean(prior[:7])),
        float(np.mean(prior)),
        float(sum(value > 0 for value in prior)),
    ])
    return values


def outlier_limits(
    categories: list[str],
    history: dict[str, dict[date, float]],
) -> dict[str, float]:
    limits = {}
    for category in categories:
        positive = np.array(
            [amount for amount in history[category].values() if amount > 0],
            dtype=float,
        )
        if len(positive) < 4:
            limits[category] = float(positive.max()) if len(positive) else 0.0
            continue
        q1, q3 = np.percentile(positive, [25, 75])
        limits[category] = float(q3 + 1.5 * (q3 - q1))
    return limits


def build_dataset() -> dict:
    categories, dates, history = load_daily_history()
    limits = outlier_limits(categories, history)
    features = []
    targets = []
    removed = 0
    for day in dates[MIN_HISTORY_DAYS:]:
        for category in categories:
            target = history[category][day]
            if target > 0 and limits[category] and target > limits[category]:
                removed += 1
                continue
            features.append(feature_row(day, category, categories, history[category]))
            targets.append(target)
    return {
        "X": np.asarray(features, dtype=np.float32),
        "y": np.asarray(targets, dtype=np.float32),
        "categories": categories,
        "dates": dates,
        "outlier_limits": limits,
        "outliers_removed": removed,
        "feature_names": feature_names(categories),
    }


def train_forecast_model(force: bool = False) -> dict:
    initialize_forecasting()
    version = data_version()
    if MODEL_PATH.exists() and not force:
        bundle = joblib.load(MODEL_PATH)
        if bundle.get("data_version") == version:
            return bundle["metadata"]

    dataset = build_dataset()
    excluded_categories = sorted(set(operating_categories()) - set(dataset["categories"]))
    X_train, X_test, y_train, y_test = train_test_split(
        dataset["X"], dataset["y"], test_size=0.20,
        random_state=RANDOM_STATE, shuffle=True,
    )
    event_model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=350,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.2,
        random_state=RANDOM_STATE,
        n_jobs=4,
        eval_metric="logloss",
    )
    event_model.fit(X_train, (y_train > 0).astype(np.int8))

    amount_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.15,
        reg_lambda=1.4,
        random_state=RANDOM_STATE,
        n_jobs=4,
    )
    positive_train = y_train > 0
    amount_model.fit(X_train[positive_train], np.log1p(y_train[positive_train]))
    event_probability = event_model.predict_proba(X_test)[:, 1]
    conditional_amount = np.maximum(0, np.expm1(amount_model.predict(X_test)))
    predictions = event_probability * conditional_amount
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
        "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 2),
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "mean_actual": round(float(np.mean(y_test)), 2),
        "positive_rate": round(float(np.mean(y_test > 0)), 4),
        "split": "random 80/20",
        "random_state": RANDOM_STATE,
    }
    importance_pairs = sorted(
        zip(
            dataset["feature_names"],
            (event_model.feature_importances_ + amount_model.feature_importances_) / 2,
        ),
        key=lambda item: item[1], reverse=True,
    )[:12]
    importance = [
        {"feature": name, "importance": round(float(value), 5)}
        for name, value in importance_pairs
    ]
    metadata = {
        "data_version": version,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "outliers_removed": dataset["outliers_removed"],
        "history_start": dataset["dates"][0].isoformat(),
        "history_end": dataset["dates"][-1].isoformat(),
        "categories": dataset["categories"],
        "excluded_categories": excluded_categories,
        "metrics": metrics,
        "feature_importance": importance,
    }
    bundle = {
        "event_model": event_model,
        "amount_model": amount_model,
        "data_version": version,
        "categories": dataset["categories"],
        "feature_names": dataset["feature_names"],
        "outlier_limits": dataset["outlier_limits"],
        "metadata": metadata,
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    with connect() as connection:
        connection.execute("""
            INSERT INTO forecast_model_runs (
                data_version, trained_at, train_rows, test_rows, outliers_removed,
                metrics, feature_importance, model_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            version, metadata["trained_at"], metadata["train_rows"],
            metadata["test_rows"], metadata["outliers_removed"],
            json.dumps(metrics), json.dumps(importance), str(MODEL_PATH),
        ))
    return metadata


def forecast_spending(horizon: int = 7) -> dict:
    if not 7 <= horizon <= 30:
        raise ValueError("Forecast horizon must be between 7 and 30 days.")
    train_forecast_model()
    bundle = joblib.load(MODEL_PATH)
    categories, dates, history = load_daily_history()
    if categories != bundle["categories"]:
        train_forecast_model(force=True)
        bundle = joblib.load(MODEL_PATH)
    event_model = bundle["event_model"]
    amount_model = bundle["amount_model"]
    start = dates[-1] + timedelta(days=1)
    future_days = [start + timedelta(days=offset) for offset in range(horizon)]
    daily_predictions = []
    category_totals: defaultdict[str, float] = defaultdict(float)
    for day in future_days:
        category_predictions = []
        for category in categories:
            row = np.asarray(
                [feature_row(day, category, categories, history[category])],
                dtype=np.float32,
            )
            event_probability = float(event_model.predict_proba(row)[0, 1])
            conditional_amount = max(
                0.0, float(np.expm1(amount_model.predict(row)[0]))
            )
            prediction = event_probability * conditional_amount
            cap = bundle["outlier_limits"].get(category, 0)
            if cap:
                prediction = min(prediction, cap)
            prediction = round(prediction, 2)
            history[category][day] = prediction
            category_totals[category] += prediction
            category_predictions.append({"category": category, "amount": prediction})
        category_predictions.sort(key=lambda item: item["amount"], reverse=True)
        daily_predictions.append({
            "date": day.isoformat(),
            "total": round(sum(item["amount"] for item in category_predictions), 2),
            "categories": category_predictions,
        })

    category_colors = {item["name"]: item["color"] for item in category_metadata()}
    category_forecast = [
        {
            "category": category,
            "amount": round(amount, 2),
            "share": 0,
            "color": category_colors.get(category, "#94a0ad"),
        }
        for category, amount in sorted(
            category_totals.items(), key=lambda item: item[1], reverse=True
        )
    ]
    total = round(sum(item["amount"] for item in category_forecast), 2)
    for item in category_forecast:
        item["share"] = round(item["amount"] / total * 100, 1) if total else 0
    mae = bundle["metadata"]["metrics"]["mae"]
    return {
        "horizon_days": horizon,
        "forecast_start": future_days[0].isoformat(),
        "forecast_end": future_days[-1].isoformat(),
        "predicted_total": total,
        "predicted_daily_average": round(total / horizon, 2),
        "top_category": category_forecast[0] if category_forecast else None,
        "category_forecast": category_forecast,
        "daily_forecast": daily_predictions,
        "uncertainty": {
            "lower_total": round(max(0, total - mae * math.sqrt(horizon)), 2),
            "upper_total": round(total + mae * math.sqrt(horizon), 2),
            "method": "Validation MAE scaled by the square root of the forecast horizon.",
        },
        "model": bundle["metadata"],
        "methodology": {
            "algorithm": "Two-stage XGBoost classifier + regressor",
            "target": (
                "Two-stage XGBoost hurdle model: spend-event probability multiplied "
                "by log1p conditional spend amount for each category-day."
            ),
            "split": "random 80/20",
            "outliers": "Removed per category using the IQR upper fence before splitting.",
            "sparse_categories": (
                f"Categories with fewer than {MIN_CATEGORY_ACTIVE_DAYS} active spending "
                "days are excluded from behavioral forecasts."
            ),
            "excluded_categories": bundle["metadata"]["excluded_categories"],
            "retraining": "Automatically retrained on all accumulated data after new CSV rows are imported.",
        },
    }
