"""Kiểm tra 15 CSV của cuộc thi: tồn tại, MD5, row count, date range cho sales."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EXPECTED_FILES = [
    ("products.csv",          ["product_id", "product_name", "category", "segment", "size", "color", "price", "cogs"]),
    ("customers.csv",         ["customer_id", "zip", "city", "signup_date", "gender", "age_group", "acquisition_channel"]),
    ("promotions.csv",        ["promo_id", "promo_name", "promo_type", "discount_value", "start_date", "end_date", "applicable_category", "promo_channel", "stackable_flag", "min_order_value"]),
    ("geography.csv",         ["zip", "city", "region", "district"]),
    ("orders.csv",            ["order_id", "order_date", "customer_id", "zip", "order_status", "payment_method", "device_type", "order_source"]),
    ("order_items.csv",       ["order_id", "product_id", "quantity", "unit_price", "discount_amount", "promo_id", "promo_id_2"]),
    ("payments.csv",          ["order_id", "payment_method", "payment_value", "installments"]),
    ("shipments.csv",         ["order_id", "ship_date", "delivery_date", "shipping_fee"]),
    ("returns.csv",           ["return_id", "order_id", "product_id", "return_date", "return_reason", "return_quantity", "refund_amount"]),
    ("reviews.csv",           ["review_id", "order_id", "product_id", "customer_id", "review_date", "rating", "review_title"]),
    ("sales.csv",             ["Date", "Revenue", "COGS"]),
    ("sample_submission.csv", ["Date", "Revenue", "COGS"]),
    ("inventory.csv",         ["snapshot_date", "product_id", "stock_on_hand"]),
    ("web_traffic.csv",       ["date", "sessions", "unique_visitors", "page_views", "bounce_rate", "avg_session_duration_sec", "traffic_source"]),
]


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def check() -> int:
    missing: list[str] = []
    rows: list[dict] = []

    for name, expected_cols in EXPECTED_FILES:
        path = DATA_DIR / name
        if not path.exists():
            missing.append(name)
            continue
        df = pd.read_csv(path, nrows=5)
        cols_ok = all(c in df.columns for c in expected_cols)
        full_rows = sum(1 for _ in path.open("rb")) - 1
        rows.append({
            "file": name,
            "rows": full_rows,
            "cols": len(pd.read_csv(path, nrows=0).columns),
            "schema_ok": "yes" if cols_ok else "NO",
            "md5": md5sum(path)[:12],
        })

    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    if missing:
        print("\nTHIẾU FILE:")
        for m in missing:
            print(f"  - {m}")

    sales = DATA_DIR / "sales.csv"
    if sales.exists():
        s = pd.read_csv(sales, parse_dates=["Date"])
        print(f"\nsales.csv range: {s['Date'].min().date()} → {s['Date'].max().date()}  ({len(s):,} rows)")

    sub = DATA_DIR / "sample_submission.csv"
    if sub.exists():
        s = pd.read_csv(sub, parse_dates=["Date"])
        print(f"sample_submission.csv range: {s['Date'].min().date()} → {s['Date'].max().date()}  ({len(s):,} rows)")

    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(check())
