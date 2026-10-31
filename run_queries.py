
from pathlib import Path
import sqlite3
import pandas as pd

sql_file = Path("sql/queries.sql")
db_file = Path("data/instamart.db")

# Read all seven SQL queries.
sql_text = sql_file.read_text(encoding="utf-8")
queries = [
    query.strip()
    for query in sql_text.split(";")
    if query.strip()
]

print("Number of queries found:", len(queries))

if len(queries) != 7:
    raise ValueError("Expected exactly 7 SQL queries.")

with sqlite3.connect(db_file) as connection:
    for number, query in enumerate(queries, start=1):
        result = pd.read_sql_query(query, connection)

        print(f"\n{'=' * 50}")
        print(f"QUERY Q{number}")
        print("Rows returned:", len(result))
        print(result.head(10).to_string(index=False))

        if number == 2:
            print(
                "Total revenue INR:",
                round(result["revenue_inr"].sum(), 2)
            )

        if number == 5 and not result.empty:
            print(
                "Average spend INR:",
                result["average_spend"].iloc[0]
            )

        if number == 6:
            top_customers = result[
                result["spend_rank"] == 1
            ]
            print("\nTop customer in each city:")
            print(top_customers.to_string(index=False))

        if number == 7 and not result.empty:
            print(
                "Final cumulative revenue INR:",
                result["cumulative_revenue_inr"].iloc[-1]
            )

print("\nALL 7 SQL QUERIES EXECUTED")
