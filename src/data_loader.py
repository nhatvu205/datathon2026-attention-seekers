"""
Pipeline tải và merge 15 CSV thành các frame phân tích sẵn dùng.

Hàm công khai chính:
    get_master_frame()  → line-item đầy đủ join tất cả bảng
    build_daily_sales() → doanh thu / COGS / discount tổng hợp theo ngày
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "data_cleaned"


# ── raw loaders ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_products() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "products.csv")


@lru_cache(maxsize=1)
def load_customers() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "customers.csv", parse_dates=["signup_date"])


@lru_cache(maxsize=1)
def load_promotions() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "promotions.csv", parse_dates=["start_date", "end_date"])


@lru_cache(maxsize=1)
def load_geography() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "geography.csv")


@lru_cache(maxsize=1)
def load_orders() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date"])


@lru_cache(maxsize=1)
def load_order_items() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "order_items.csv")


@lru_cache(maxsize=1)
def load_payments() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "payments.csv")


@lru_cache(maxsize=1)
def load_shipments() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "shipments.csv", parse_dates=["ship_date", "delivery_date"])


@lru_cache(maxsize=1)
def load_returns() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "returns.csv", parse_dates=["return_date"])


@lru_cache(maxsize=1)
def load_reviews() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "reviews.csv", parse_dates=["review_date"])


@lru_cache(maxsize=1)
def load_sales() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"])


@lru_cache(maxsize=1)
def load_inventory() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "inventory.csv", parse_dates=["snapshot_date"])


@lru_cache(maxsize=1)
def load_web_traffic() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "web_traffic.csv", parse_dates=["date"])


# ── composite frames ───────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def build_line_items() -> pd.DataFrame:
    """
    Một dòng = một sản phẩm trong một đơn hàng, đầy đủ thông tin từ tất cả bảng.

    Các cột bổ sung:
        line_revenue   = quantity * unit_price - discount_amount
        line_cogs      = quantity * cogs  (từ products)
        delivery_days  = delivery_date - ship_date (nullable)
    """
    oi = load_order_items().copy()
    oi["line_revenue"] = oi["quantity"] * oi["unit_price"] - oi["discount_amount"]

    # join products → thêm category, segment, size, color, price, cogs
    oi = oi.merge(load_products(), on="product_id", how="left", suffixes=("", "_prod"))
    oi["line_cogs"] = oi["quantity"] * oi["cogs"]

    # join orders → thêm order_date, customer_id, zip, order_status, payment_method, ...
    oi = oi.merge(load_orders(), on="order_id", how="left")

    # join geography → thêm city, region, district
    oi = oi.merge(
        load_geography().rename(columns={"city": "geo_city"}),
        on="zip", how="left",
    )

    # join customers → thêm signup_date, gender, age_group, acquisition_channel
    oi = oi.merge(
        load_customers()[["customer_id", "signup_date", "gender", "age_group", "acquisition_channel"]],
        on="customer_id", how="left",
    )

    # join shipments → thêm ship_date, delivery_date, shipping_fee
    oi = oi.merge(
        load_shipments()[["order_id", "ship_date", "delivery_date", "shipping_fee"]],
        on="order_id", how="left",
    )
    oi["delivery_days"] = (oi["delivery_date"] - oi["ship_date"]).dt.days

    # join promo name cho promo_id chính (nullable)
    promo_map = load_promotions()[["promo_id", "promo_name", "promo_type", "discount_value"]].copy()
    oi = oi.merge(
        promo_map.rename(columns={"promo_name": "promo1_name", "promo_type": "promo1_type",
                                   "discount_value": "promo1_discount"}),
        on="promo_id", how="left",
    )

    return oi


@lru_cache(maxsize=1)
def build_daily_sales() -> pd.DataFrame:
    """
    Tổng hợp theo ngày từ line-items (khớp với sales.csv nhưng có thêm discount, COGS).

    Cột trả về:
        order_date, revenue, cogs, discount, gross_profit, gross_margin
    """
    li = build_line_items()
    daily = (
        li.groupby("order_date")
        .agg(
            revenue=("line_revenue", "sum"),
            cogs=("line_cogs", "sum"),
            discount=("discount_amount", "sum"),
            orders=("order_id", "nunique"),
            items=("order_id", "count"),
        )
        .reset_index()
        .rename(columns={"order_date": "date"})
    )
    daily["gross_profit"] = daily["revenue"] - daily["cogs"]
    daily["gross_margin"] = daily["gross_profit"] / daily["revenue"]
    return daily


@lru_cache(maxsize=1)
def get_master_frame() -> pd.DataFrame:
    """
    Alias rõ nghĩa cho build_line_items() — frame đầy đủ sẵn dùng cho EDA/modeling.
    """
    return build_line_items()


# ── quick sanity check ─────────────────────────────────────────────────────────

def _smoke_test() -> None:
    li = build_line_items()
    daily = build_daily_sales()

    assert li["order_id"].notna().all(), "order_id không được null"
    assert li["product_id"].notna().all(), "product_id không được null"
    assert (li["line_revenue"] >= 0).all(), "line_revenue âm"

    sales = load_sales()
    overlap = daily[daily["date"].isin(sales["Date"])]
    diff_pct = abs(overlap["revenue"].sum() - sales.loc[sales["Date"].isin(overlap["date"]), "Revenue"].sum())
    diff_pct /= sales["Revenue"].sum()
    assert diff_pct < 0.05, f"Daily revenue lệch sales.csv quá 5% ({diff_pct:.2%})"

    print(f"build_line_items : {len(li):,} rows, {li.shape[1]} cols")
    print(f"build_daily_sales: {len(daily):,} days | revenue total = {daily['revenue'].sum():,.0f}")
    print(f"Revenue vs sales.csv: diff = {diff_pct:.4%}  ✓")


if __name__ == "__main__":
    _smoke_test()
