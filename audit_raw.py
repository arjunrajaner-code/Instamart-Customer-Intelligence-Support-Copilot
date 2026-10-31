
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

import pandas as pd

RAW = Path("data/raw")
OUTPUT = Path("outputs")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Columns referenced by the audit.
REQUIRED_COLUMNS = {
    "customers": {"age", "city", "gender"},
    "products": set(),
    "orders": {
        "order_id",
        "status",
        "order_amount",
        "delivery_minutes",
    },
}


def load_csv(name):
    """Load a CSV and validate the columns used by this audit."""
    path = RAW / f"{name}.csv"

    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    missing = REQUIRED_COLUMNS[name] - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    return df


def audit_table(name, df):
    print(f"\n{'=' * 60}")
    print(f"{name}: shape = {df.shape}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing-value audit:")
    for column in df.columns:
        missing = df[column].str.strip().eq("").sum()
        percentage = missing / len(df) * 100 if len(df) else 0

        print(
            f"{column}: {missing} missing "
            f"({percentage:.2f}%)"
        )


def run_audit():
    customers = load_csv("customers")
    products = load_csv("products")
    orders = load_csv("orders")

    audit_table("CUSTOMERS", customers)
    audit_table("PRODUCTS", products)
    audit_table("ORDERS", orders)

    # ORDER AUDIT
    amount_raw = orders["order_amount"].str.strip()
    delivery_raw = orders["delivery_minutes"].str.strip()

    amount = pd.to_numeric(amount_raw, errors="coerce")
    delivery = pd.to_numeric(delivery_raw, errors="coerce")

    print("\n" + "=" * 60)
    print("ORDER AUDIT CHECKPOINTS")

    print("Raw order rows:", len(orders))
    print("Exact duplicate rows:", orders.duplicated().sum())

    print(
        "Duplicate order IDs:",
        orders.loc[
            orders["order_id"].str.strip().ne(""), "order_id"
        ].duplicated().sum()
    )

    status = orders["status"].str.strip().str.lower()
    print("Pending rows:", status.eq("pending").sum())
    print("Distinct order IDs:", orders["order_id"].nunique())

    print("Blank amounts:", amount_raw.eq("").sum())
    print(
        "Invalid non-blank amounts:",
        (amount_raw.ne("") & amount.isna()).sum()
    )
    print("Negative amounts:", amount.lt(0).sum())

    print("Blank delivery times:", delivery_raw.eq("").sum())
    print(
        "Invalid non-blank delivery times:",
        (delivery_raw.ne("") & delivery.isna()).sum()
    )
    print("Negative delivery times:", delivery.lt(0).sum())

    # CUSTOMER AUDIT
    age_raw = customers["age"].str.strip()
    age = pd.to_numeric(age_raw, errors="coerce")

    print("\nCUSTOMER AUDIT CHECKPOINTS")
    print("Blank ages:", age_raw.eq("").sum())
    print(
        "Invalid non-blank ages:",
        (age_raw.ne("") & age.isna()).sum()
    )
    print(
        "Ages outside 10-100:",
        (age.notna() & ~age.between(10, 100)).sum()
    )

    print(
        "Distinct raw city spellings:",
        customers["city"].str.strip().nunique()
    )
    print(
        "Distinct non-blank raw gender spellings:",
        customers.loc[
            customers["gender"].str.strip().ne(""),
            "gender"
        ].str.strip().nunique()
    )
    print(
        "Blank genders:",
        customers["gender"].str.strip().eq("").sum()
    )


def main():
    # Capture the audit so it can be displayed and saved.
    report = StringIO()

    try:
        with redirect_stdout(report):
            run_audit()
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print("Audit failed:", exc)
        return

    output = report.getvalue()
    print(output)

    report_path = OUTPUT / "raw_data_audit.txt"
    report_path.write_text(output, encoding="utf-8")
    print("Audit report saved:", report_path)


if __name__ == "__main__":
    main()
