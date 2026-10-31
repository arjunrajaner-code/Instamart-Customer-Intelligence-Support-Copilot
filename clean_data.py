
from pathlib import Path
import pandas as pd
import sqlite3

RAW = Path("data/raw")
CLEAN = Path("data/clean")
CLEAN.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Load raw datasets
# --------------------------------------------------
customers = pd.read_csv(
    RAW / "customers.csv",
    keep_default_na=False
)

products = pd.read_csv(
    RAW / "products.csv",
    keep_default_na=False
)

orders = pd.read_csv(
    RAW / "orders.csv",
    keep_default_na=False
)

print("Raw data loaded successfully.")

# --------------------------------------------------
# 2. Clean customers
# --------------------------------------------------

customers["city"] = (
    customers["city"].astype(str).str.strip().str.title()
)

# Standardise gender values
gender_map = {
    "male": "male",
    "m": "male",
    "female": "female",
    "f": "female"
}

customers["gender"] = (
    customers["gender"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(gender_map)
    .fillna("unknown")
)

# Convert invalid ages to missing values
customers["age"] = pd.to_numeric(
    customers["age"], errors="coerce"
)

customers.loc[
    ~customers["age"].between(10, 100), "age"
] = float("nan")

# Fill missing ages using city median
city_medians = (
    customers.groupby("city")["age"].transform("median")
)

customers["age"] = customers["age"].fillna(city_medians)

# Convert signup dates
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="raise"
)

assert customers["age"].notna().all()
assert customers["city"].nunique() == 6

print("\nCustomer cleaning complete.")
print(customers["city"].value_counts())

print("\nGender counts:")
print(customers["gender"].value_counts())

# --------------------------------------------------
# 3. Clean orders
# --------------------------------------------------

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="raise"
)

# Remove exact duplicate rows
orders = orders.drop_duplicates().copy()

# Keep latest record for each order ID
orders = (
    orders.sort_values("order_ts")
    .drop_duplicates(subset="order_id", keep="last")
    .copy()
)

# Convert numerical columns
for column in [
    "quantity",
    "discount_pct",
    "order_amount",
    "delivery_minutes"
]:
    orders[column] = pd.to_numeric(
        orders[column], errors="coerce"
    )

# --------------------------------------------------
# 3.1 Handle negative delivery times
# --------------------------------------------------

negative_delivery = orders["delivery_minutes"] < 0

print(
    "\nNegative delivery times found:",
    negative_delivery.sum()
)

# Replace invalid negative values with NaN
orders.loc[
    negative_delivery,
    "delivery_minutes"
] = float("nan")

print(
    "Invalid delivery times converted to missing:",
    negative_delivery.sum()
)

# --------------------------------------------------
# 3.2 Calculate missing order amounts
# --------------------------------------------------

orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

orders["order_amount"] = orders["order_amount"].fillna(
    (
        orders["price_inr"]
        * orders["quantity"]
        * (1 - orders["discount_pct"] / 100)
    ).round(2)
)

# --------------------------------------------------
# 3.3 Handle cancelled and completed orders
# --------------------------------------------------

cancelled = orders["status"].str.strip().str.lower().eq("cancelled")
completed = orders["status"].str.strip().str.lower().eq("completed")

# Cancelled orders were never delivered
orders.loc[
    cancelled,
    "delivery_minutes"
] = float("nan")

# Calculate median delivery time by product category
all_category_medians = (
    orders.loc[completed]
    .groupby("category")["delivery_minutes"]
    .median()
)

# Identify missing completed-order delivery times
missing_completed = (
    completed & orders["delivery_minutes"].isna()
)

# Fill missing delivery times using category median
orders.loc[
    missing_completed,
    "delivery_minutes"
] = (
    orders.loc[missing_completed, "category"]
    .map(all_category_medians)
)

print(
    "\nMissing completed delivery times after cleaning:",
    orders.loc[completed, "delivery_minutes"].isna().sum()
)

# --------------------------------------------------
# 3.4 Final validation checks
# --------------------------------------------------

assert orders["order_id"].is_unique
assert not orders["status"].str.lower().eq("pending").any()
assert orders["order_amount"].notna().all()

assert orders.loc[
    completed, "delivery_minutes"
].notna().all()

assert orders.loc[
    cancelled, "delivery_minutes"
].isna().all()

assert not orders.loc[
    completed, "delivery_minutes"
].lt(0).any()

# Remove temporary product lookup columns
orders = orders.drop(
    columns=["price_inr", "category"]
)

# --------------------------------------------------
# 3.5 Format timestamps
# --------------------------------------------------

orders["order_ts"] = orders["order_ts"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

customers["signup_date"] = customers["signup_date"].dt.strftime(
    "%Y-%m-%d"
)

# --------------------------------------------------
# 4. Save clean CSV files
# --------------------------------------------------

customers.to_csv(
    CLEAN / "customers.csv",
    index=False
)

products.to_csv(
    CLEAN / "products.csv",
    index=False
)

orders.to_csv(
    CLEAN / "orders.csv",
    index=False
)

print("\nClean data saved successfully.")

print("Customers:", len(customers))
print("Products:", len(products))
print("Orders:", len(orders))

print(
    "Completed orders:",
    (orders["status"] == "completed").sum()
)

print(
    "Cancelled orders:",
    (orders["status"] == "cancelled").sum()
)

print(
    "Completed revenue INR:",
    round(
        orders.loc[
            orders["status"] == "completed",
            "order_amount"
        ].sum(),
        2
    )
)

# --------------------------------------------------
# 5. Create SQLite database
# --------------------------------------------------

db_path = "data/instamart.db"

with sqlite3.connect(db_path) as connection:

    customers.to_sql(
        "customers",
        connection,
        if_exists="replace",
        index=False
    )

    products.to_sql(
        "products",
        connection,
        if_exists="replace",
        index=False
    )

    orders.to_sql(
        "orders",
        connection,
        if_exists="replace",
        index=False
    )

print("\nSQLite database created:", db_path)

print("PART A CLEANING COMPLETED")
