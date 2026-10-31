
from pathlib import Path
import sqlite3
import pandas as pd

CLEAN = Path("data/clean")
DB = Path("data/instamart.db")

customers = pd.read_csv(CLEAN / "customers.csv")
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

checks = {
    "Customer count": len(customers) == 1200,
    "Product count": len(products) == 40,
    "Order count": len(orders) == 12309,
    "Unique order IDs": orders["order_id"].is_unique,
    "No pending orders": not orders["status"].eq("pending").any(),
    "Completed orders": (orders["status"] == "completed").sum() == 11110,
    "Cancelled orders": (orders["status"] == "cancelled").sum() == 1199,
    "No missing order amounts": orders["order_amount"].notna().all(),
    "Completed delivery times present": orders.loc[
        orders["status"] == "completed", "delivery_minutes"
    ].notna().all(),
    "Cancelled delivery times missing": orders.loc[
        orders["status"] == "cancelled", "delivery_minutes"
    ].isna().all(),
    "Six customer cities": customers["city"].nunique() == 6,
    "No missing customer ages": customers["age"].notna().all(),
    "Valid customer ages": customers["age"].between(10, 100).all(),
    "Correct completed revenue": round(
        orders.loc[
            orders["status"] == "completed", "order_amount"
        ].sum(), 2
    ) == 2998668.80,
}

with sqlite3.connect(DB) as connection:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }

checks["SQLite tables exist"] = {
    "customers", "products", "orders"
}.issubset(tables)

failed = []

for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}")
    if not passed:
        failed.append(name)

if failed:
    raise SystemExit(
        f"Validation failed: {', '.join(failed)}"
    )

print("ALL CHECKPOINTS PASSED")
