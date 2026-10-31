
import sqlite3
import pandas as pd

DB = "data/instamart.db"

with sqlite3.connect(DB) as con:
    customers = pd.read_sql_query(
        "SELECT * FROM customers", con
    )
    products = pd.read_sql_query(
        "SELECT * FROM products", con
    )
    orders = pd.read_sql_query(
        "SELECT * FROM orders WHERE status = 'completed'", con
    )

orders["order_amount"] = pd.to_numeric(orders["order_amount"])
orders["quantity"] = pd.to_numeric(orders["quantity"])
orders["discount_pct"] = pd.to_numeric(orders["discount_pct"])
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

# Join tables for analysis.
data = (
    orders
    .merge(customers[["customer_id", "city"]], on="customer_id")
    .merge(products[["product_id", "category"]], on="product_id")
)

# 1. Cross-check Q2 with a pandas pivot table.
pivot = pd.pivot_table(
    data,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="All"
)

print("\n1. PANDAS PIVOT TABLE — REVENUE BY CITY AND CATEGORY")
print(pivot.round(2))
print("Pivot grand total INR:", round(pivot.loc["All", "All"], 2))

# 2. Per-city sum, mean and count.
city_stats = data.groupby("city").agg(
    revenue_sum=("order_amount", "sum"),
    average_order=("order_amount", "mean"),
    order_count=("order_amount", "count")
)

print("\n2. CITY STATISTICS")
print(city_stats.round(2))

# 3. Distribution statistics.
amount = data["order_amount"]
q1 = amount.quantile(0.25)
median = amount.median()
q3 = amount.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

print("\n3. COMPLETED ORDER AMOUNT DISTRIBUTION")
print("Count:", len(amount))
print("Mean INR:", round(amount.mean(), 2))
print("Median INR:", round(median, 2))
print("Skewness:", round(amount.skew(), 2))
print("Q1 INR:", round(q1, 2))
print("Q3 INR:", round(q3, 2))
print("IQR INR:", round(iqr, 2))
print("Lower fence INR:", round(lower_fence, 2))
print("Upper fence INR:", round(upper_fence, 2))
print("Outliers below lower fence:", int((amount < lower_fence).sum()))
print("Outliers above upper fence:", int((amount > upper_fence).sum()))

# 4. Correlation matrix.
columns = [
    "order_amount", "discount_pct",
    "quantity", "delivery_minutes"
]
correlation = data[columns].corr()

print("\n4. CORRELATION MATRIX")
print(correlation.round(3))

pairs = correlation.where(
    ~pd.DataFrame(
        [[i == j for j in range(len(columns))]
         for i in range(len(columns))],
        index=columns, columns=columns
    )
).stack()

strongest_pair = pairs.abs().idxmax()
print(
    "Largest absolute correlation:",
    strongest_pair,
    round(pairs.loc[strongest_pair], 3)
)

print("\nPANDAS ANALYSIS COMPLETED")
