"""Reusable, deterministic financial demo logic. No employer or customer data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RANDOM_SEED = 42
COLUMNS = ["revenue_m", "expenses_m", "assets_m", "debt_m", "rate_pct"]
FEATURES = ["revenue_m", "expenses_m", "assets_m", "debt_m", "rate_pct", "operating_margin_pct"]


def get_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make a synthetic reference ledger and a corrupted incoming extract."""
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range("2020-01-01", periods=72, freq="MS")
    idx = np.arange(len(dates))
    revenue = 84 + idx * .48 + 5 * np.sin(idx / 5) + rng.normal(0, 1.9, len(idx))
    expenses = 63 + idx * .34 + 3.5 * np.sin((idx - 1) / 5) + rng.normal(0, 1.4, len(idx))
    assets = 420 + idx * 2.9 + rng.normal(0, 6, len(idx))
    debt = 140 + idx * .66 + rng.normal(0, 2, len(idx))
    rate_pct = 2.5 + 1.3 * np.sin(idx / 11)
    trusted = pd.DataFrame({
        "month": dates,
        "revenue_m": np.round(revenue, 2),
        "expenses_m": np.round(expenses, 2),
        "assets_m": np.round(assets, 2),
        "debt_m": np.round(debt, 2),
        "rate_pct": np.round(rate_pct, 2),
    })
    raw = trusted.copy()
    raw.loc[12, "revenue_m"] = np.nan
    raw.loc[40, "expenses_m"] = round(float(raw.loc[40, "revenue_m"]) * 1.8, 2)
    raw.loc[57, "debt_m"] = round(float(raw.loc[57, "assets_m"]) * 1.35, 2)
    raw = pd.concat([raw, raw.iloc[[24]]], ignore_index=True).sort_values("month").reset_index(drop=True)
    return raw, trusted


def quality_issues(df: pd.DataFrame) -> pd.DataFrame:
    issues: list[dict[str, str]] = []
    for _, row in df.iterrows():
        month = pd.Timestamp(row["month"]).strftime("%Y-%m")
        for c in COLUMNS:
            if pd.isna(row[c]):
                issues.append({"Month": month, "Rule": "Missing value", "Details": f"{c} is null"})
        if pd.notna(row["expenses_m"]) and pd.notna(row["revenue_m"]) and row["expenses_m"] > row["revenue_m"] * 1.1:
            issues.append({"Month": month, "Rule": "Expense outlier", "Details": "Expenses > 110% of revenue"})
        if pd.notna(row["debt_m"]) and pd.notna(row["assets_m"]) and row["debt_m"] > row["assets_m"]:
            issues.append({"Month": month, "Rule": "Balance sheet check", "Details": "Debt exceeds total assets"})
    duplicate_months = df.loc[df.duplicated("month", keep="first"), "month"]
    for month in duplicate_months:
        issues.append({"Month": pd.Timestamp(month).strftime("%Y-%m"), "Rule": "Duplicate period", "Details": "A second record exists for the same month"})
    return pd.DataFrame(issues, columns=["Month", "Rule", "Details"])


def reconcile(raw: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile against a trusted, independent *synthetic* reference ledger.

    Never automatically overwrite production data from unverified observations.
    """
    unique_raw = raw.drop_duplicates("month", keep="first").copy().sort_values("month")
    ref = reference.set_index("month").sort_index()
    got = unique_raw.set_index("month").sort_index()
    if not got.index.equals(ref.index):
        raise ValueError("Reference ledger and incoming periods differ; manual review needed")
    changes: list[dict[str, str | float]] = []
    for col in COLUMNS:
        unmatched = ~np.isclose(got[col].to_numpy(dtype=float), ref[col].to_numpy(dtype=float), equal_nan=True)
        for ts in got.index[unmatched]:
            changes.append({"Month": ts.strftime("%Y-%m"), "Field": col,
                            "Incoming": got.at[ts, col], "Verified reference": ref.at[ts, col]})
    if len(raw) != len(unique_raw):
        for ts in raw.loc[raw.duplicated("month", keep="first"), "month"]:
            changes.append({"Month": ts.strftime("%Y-%m"), "Field": "duplicate", "Incoming": "duplicate row", "Verified reference": "removed"})
    return ref.reset_index(), pd.DataFrame(changes)


def prepare_model_df(verified: pd.DataFrame) -> pd.DataFrame:
    frame = verified.sort_values("month").copy()
    frame["operating_margin_pct"] = 100 * (frame["revenue_m"] - frame["expenses_m"]) / frame["revenue_m"]
    frame["next_month_expenses_m"] = frame["expenses_m"].shift(-1)
    return frame


def experiment(verified: pd.DataFrame) -> dict:
    """Time-ordered holdout; persistence baseline; one-month-ahead expense forecast."""
    features = prepare_model_df(verified)
    labeled = features.dropna(subset=["next_month_expenses_m"]).reset_index(drop=True)
    cut = int(len(labeled) * .8)
    train, test = labeled.iloc[:cut], labeled.iloc[cut:]
    reg = make_pipeline(StandardScaler(), Ridge(alpha=5))
    reg.fit(train[FEATURES], train["next_month_expenses_m"])
    yhat = reg.predict(test[FEATURES])
    persistence = test["expenses_m"].to_numpy()
    last = features.iloc[[-1]]
    return {
        "baseline_mae": float(mean_absolute_error(test["next_month_expenses_m"], persistence)),
        "model_mae": float(mean_absolute_error(test["next_month_expenses_m"], yhat)),
        "test": pd.DataFrame({"month": test["month"], "actual": test["next_month_expenses_m"],
                              "baseline": persistence, "model": yhat}),
        "next_forecast_m": float(reg.predict(last[FEATURES])[0]),
        "train_size": len(train), "test_size": len(test),
        "last_month": pd.Timestamp(last.iloc[0]["month"]),
        "first_test_month": pd.Timestamp(test.iloc[0]["month"]),
    }


def evidence_documents(verified: pd.DataFrame) -> list[dict[str, str]]:
    """All documents are generated from the trusted synthetic ledger."""
    df = verified.sort_values("month")
    last = df.iloc[-1]
    prior = df.iloc[-2]
    def month(r):
        return pd.Timestamp(r["month"]).strftime("%B %Y")
    d = [
        {"id": "S1", "title": "Verified latest-month financial summary", "text":
         f"{month(last)} revenue ${last.revenue_m:.2f}m, expenses ${last.expenses_m:.2f}m, "
         f"assets ${last.assets_m:.2f}m and debt ${last.debt_m:.2f}m. All values are synthetic and reconciled."},
        {"id": "S2", "title": "Verified previous-month comparison", "text":
         f"{month(prior)} revenue ${prior.revenue_m:.2f}m and expenses ${prior.expenses_m:.2f}m. "
         f"{month(last)} changes: revenue ${(last.revenue_m - prior.revenue_m):+.2f}m and "
         f"expenses ${(last.expenses_m - prior.expenses_m):+.2f}m. This does not establish causes."},
        {"id": "S3", "title": "Verified financial ratios", "text":
         f"{month(last)} debt/assets {last.debt_m / last.assets_m:.1%}; "
         f"operating margin {(last.revenue_m - last.expenses_m) / last.revenue_m:.1%}. "
         f"Rates indicator {last.rate_pct:.2f}%. Ratios are illustrative."},
        {"id": "S4", "title": "Data quality and governance", "text":
         "Demo source is a synthetic reference ledger. Incoming file intentionally has duplicate, missing, "
         "implausible expense, and debt greater than assets errors. Reconciliation replaces erroneous values "
         "only with known verified reference records. No historical causes, company filings, or actual investment "
         "recommendations are available in this demo."},
        {"id": "S5", "title": "Model card", "text":
         "Ridge regression estimates next-month operating expenses using month-end revenue, expenses, assets, "
         "debt, rates, and operating margin. Time-ordered train/test split. Compare with a persistence baseline. "
         "Synthetic data only; forecasts are not suitable for financial decision making."},
    ]
    return d


def retrieve(question: str, documents: list[dict[str, str]], k: int = 3) -> list[dict[str, str]]:
    if not question.strip():
        return []
    corpus = [f"{doc['title']} {doc['text']}" for doc in documents]
    vectors = TfidfVectorizer(stop_words="english").fit_transform(corpus + [question])
    sims = cosine_similarity(vectors[-1], vectors[:-1]).ravel()
    selected = np.argsort(-sims)[:k]
    return [documents[int(i)] for i in selected if sims[i] > 0]


def offline_summary(question: str, evidence: list[dict[str, str]]) -> str:
    if not evidence:
        return "No relevant verified evidence found. An analyst should review more sources before drawing a conclusion."
    return ("**Evidence lookup (not an LLM response):** These excerpts may help answer your question, "
            "but they do not establish causation. Open and verify the referenced synthetic records below.")
