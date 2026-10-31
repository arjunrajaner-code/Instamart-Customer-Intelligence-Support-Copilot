import sqlite3

conn = sqlite3.connect("data/instamart.db")

tables = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()

print("TABLES:")

for table in tables:
    table_name = table[0]
    print("\nTable:", table_name)

    columns = conn.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()

    for column in columns:
        print(column)

conn.close()