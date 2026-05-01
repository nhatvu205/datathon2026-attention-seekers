"""Merge notebook 30 parquet features into sales-based modeling frames (by Date).

Only columns safe for Prophet-residual GBMs: exclude same-day Revenue/COGS leaks.
Calendar duplicates (vs v2 STATIC_FEATURES) are omitted where redundant.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd

FE_TRAIN_CANDIDATES = [
    "../data/features/train_features.parquet",
    "data/features/train_features.parquet",
]
FE_TEST_CANDIDATES = [
    "../data/features/test_features.parquet",
    "data/features/test_features.parquet",
]

# Subset kept when fe_stack_mode="focused" (from residual SHAP + gain plots; trim noise vs full parquet).
FOCUSED_STACK_FE_COLUMNS: frozenset[str] = frozenset(
    {
        "n_orders_lag1",
        "n_orders_roll7",
        "n_items_lag1",
        "n_items_roll7",
        "cr_proxy_orders_per_session_lag1",
        "cr_proxy_orders_per_session_roll7",
        "promo_order_pct_lag1",
        "promo_order_pct_roll7",
        "promo_x_aov_interaction_lag1",
        "promo_x_aov_interaction_roll7",
        "premium_margin_gap_daily_lag1",
        "premium_margin_gap_daily_roll7",
        "n_active_promos",
        "n_stackable_promos",
        "promo_category_coverage",
        "has_percentage_promo",
        "has_fixed_promo",
        "days_since_last_promo",
        "max_discount_pct",
        "promo_n_active_x_post2019",
        "promo_max_disc_x_post2019",
        "promo_stackable_x_post2019",
        "page_views_lag1",
        "page_views_roll7",
        "aov_line_approx_lag1",
        "aov_line_approx_roll7",
        "inv_overstock_pct",
        "inv_stock_on_hand",
        "sw_stockout_pct",
        "sw_fill_rate",
        "sessions_total_lag1",
        "sessions_total_roll7",
    }
)

# Bỏ web traffic + inventory (giảm signal fill YoY/ffill trên test); giữ promo + orders + margin.
PROMO_CORE_FE_COLUMNS: frozenset[str] = frozenset(
    c
    for c in FOCUSED_STACK_FE_COLUMNS
    if c
    not in {
        "page_views_lag1",
        "page_views_roll7",
        "sessions_total_lag1",
        "sessions_total_roll7",
        "inv_overstock_pct",
        "inv_stock_on_hand",
        "sw_stockout_pct",
        "sw_fill_rate",
    }
)


def resolve_first(paths: Iterable[str], label: str) -> str | None:
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def select_fe_columns_for_stack(
    all_columns: Iterable[str],
    static_features: list[str],
    *,
    fe_stack_mode: str = "full",
) -> list[str]:
    """Wide FE (no rev_/cogs_ lags), then optional ``focused`` subset for stacking GBMs.

    ``fe_stack_mode``: ``"full"`` = all safe wide columns; ``"focused"`` = intersection with
    :data:`FOCUSED_STACK_FE_COLUMNS`; ``"promo_core"`` = :data:`PROMO_CORE_FE_COLUMNS` (bỏ traffic/inventory).
    """
    skip = set(static_features) | {
        "Date",
        "Revenue",
        "COGS",
        "is_train",
        "Revenue_log",
        "COGS_Ratio",
    }
    cal_dup = {
        "sin_dow_1",
        "sin_dow_2",
        "cos_dow_1",
        "cos_dow_2",
        "dayofweek",
        "dayofmonth",
        "dayofyear",
        "month",
        "quarter",
        "year",
        "weekofyear",
        "is_weekend",
        "is_month_end",
        "is_month_start",
        "is_quarter_end",
        "sin_month_1",
        "sin_month_2",
        "sin_month_3",
        "cos_month_1",
        "cos_month_2",
        "cos_month_3",
        "tet_proximity_days",
        "is_tet_window",
        "is_pre_tet",
        "is_post_tet",
        "is_public_holiday",
        "is_year_end_season",
        "is_back_to_school",
        "post_2019_regime",
        "is_peak_may",
        "is_recovery_year_2022",
        "days_since_start",
        "year_index",
    }
    out: list[str] = []
    for c in all_columns:
        if c in skip or c in cal_dup:
            continue
        if c.startswith("rev_") or c.startswith("cogs_"):
            continue
        out.append(c)
    wide = sorted(set(out))
    if fe_stack_mode == "full":
        return wide
    if fe_stack_mode == "focused":
        return sorted(c for c in wide if c in FOCUSED_STACK_FE_COLUMNS)
    if fe_stack_mode == "promo_core":
        return sorted(c for c in wide if c in PROMO_CORE_FE_COLUMNS)
    raise ValueError(
        f"fe_stack_mode must be 'full', 'focused', or 'promo_core', got {fe_stack_mode!r}"
    )


def merge_parquet_fe(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    static_features: list[str],
    train_parquet: str | None = None,
    test_parquet: str | None = None,
    *,
    fe_stack_mode: str = "full",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], bool]:
    """Left-merge FE columns into train/test by Date. Returns (train, test, fe_cols, ok)."""
    tr_path = train_parquet or resolve_first(FE_TRAIN_CANDIDATES, "train_features.parquet")
    te_path = test_parquet or resolve_first(FE_TEST_CANDIDATES, "test_features.parquet")
    if not tr_path or not te_path:
        return df_train, df_test, [], False

    fe_tr = pd.read_parquet(tr_path)
    fe_te = pd.read_parquet(te_path)
    fe_cols = select_fe_columns_for_stack(fe_tr.columns, static_features, fe_stack_mode=fe_stack_mode)
    use_tr = fe_tr[["Date"] + [c for c in fe_cols if c in fe_tr.columns]].copy()
    use_te = fe_te[["Date"] + [c for c in fe_cols if c in fe_te.columns]].copy()
    use_tr = use_tr.drop_duplicates(subset=["Date"], keep="last")
    use_te = use_te.drop_duplicates(subset=["Date"], keep="last")
    fe_cols = [c for c in fe_cols if c in use_tr.columns and c != "Date"]

    df_train = df_train.merge(use_tr, on="Date", how="left")
    df_test = df_test.merge(use_te, on="Date", how="left")

    for c in fe_cols:
        if df_train[c].dtype in ("float64", "float32", "Int64", "int64") or np.issubdtype(
            df_train[c].dtype, np.floating
        ):
            med = float(np.nanmedian(df_train[c].to_numpy(dtype=float)))
            df_train[c] = df_train[c].fillna(med)
            df_test[c] = df_test[c].fillna(med)
        else:
            df_train[c] = pd.to_numeric(df_train[c], errors="coerce").fillna(0.0)
            df_test[c] = pd.to_numeric(df_test[c], errors="coerce").fillna(0.0)

    return df_train, df_test, fe_cols, True


def build_prophet_holidays_vn(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Vietnam-relevant holidays for Prophet (Tet + fixed solar dates)."""
    tet = pd.to_datetime(
        [
            "2013-02-10",
            "2014-01-31",
            "2015-02-19",
            "2016-02-08",
            "2017-01-28",
            "2018-02-16",
            "2019-02-05",
            "2020-01-25",
            "2021-02-12",
            "2022-02-01",
            "2023-01-22",
            "2024-02-10",
            "2025-01-29",
        ]
    )
    rows: list[dict] = []
    for ds in tet:
        if start - pd.Timedelta(days=20) <= ds <= end + pd.Timedelta(days=20):
            rows.append(
                {
                    "holiday": "tet",
                    "ds": ds,
                    "lower_window": -6,
                    "upper_window": 6,
                }
            )
    fixed = [(1, 1), (4, 30), (5, 1), (9, 2)]
    for y in range(start.year, end.year + 1):
        for m, d in fixed:
            try:
                ds = pd.Timestamp(year=y, month=m, day=d)
            except ValueError:
                continue
            if start <= ds <= end:
                rows.append(
                    {
                        "holiday": f"fixed_{m}_{d}",
                        "ds": ds,
                        "lower_window": 0,
                        "upper_window": 1,
                    }
                )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["holiday", "ds", "lower_window", "upper_window"])
