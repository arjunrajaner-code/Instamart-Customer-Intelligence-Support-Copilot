
import json
from pathlib import Path

import pandas as pd

# ============================================================
# PART E - REFUND AGENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ORDERS_PATH = BASE_DIR / "data" / "raw" / "orders.csv"
POLICY_PATH = BASE_DIR / "data" / "policies" / "refund_policy.txt"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "refund_agent_results.json"

REQUIRED_COLUMNS = {
    "order_id",
    "customer_id",
    "status",
    "order_amount",
    "delivery_minutes",
}


def load_orders():
    """Load orders and validate required columns."""

    if not ORDERS_PATH.exists():
        raise FileNotFoundError(
            f"Orders file not found: {ORDERS_PATH}"
        )

    orders = pd.read_csv(ORDERS_PATH)

    missing = REQUIRED_COLUMNS - set(orders.columns)

    if missing:
        raise ValueError(
            f"Missing order columns: {sorted(missing)}"
        )

    orders["order_id"] = (
        orders["order_id"]
        .astype("string")
        .str.strip()
    )

    orders["customer_id"] = (
        orders["customer_id"]
        .astype("string")
        .str.strip()
    )

    orders["status"] = (
        orders["status"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Convert invalid numeric values to NaN.
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )

    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    return orders


def load_policy():
    """Load and check the refund policy document."""

    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            f"Policy file not found: {POLICY_PATH}"
        )

    policy_text = POLICY_PATH.read_text(
        encoding="utf-8"
    )

    policy_lower = policy_text.lower()

    required_terms = [
        "completed",
        "45 minutes",
        "1000",
        "7 days",
    ]

    missing_terms = [
        term
        for term in required_terms
        if term not in policy_lower
    ]

    if missing_terms:
        raise ValueError(
            "Could not verify policy rules: "
            + ", ".join(missing_terms)
        )

    return policy_text


def get_order(order_id, orders):
    """Find exactly one order by order ID."""

    order_id = str(order_id).strip()

    matches = orders[
        orders["order_id"] == order_id
    ]

    if matches.empty:
        return None

    if len(matches) > 1:
        raise ValueError(
            f"Duplicate order ID found: {order_id}"
        )

    return matches.iloc[0]


def make_result(order_id, decision, reason, refund_inr=0.0):
    """Create a consistent result structure."""

    return {
        "order_id": str(order_id),
        "decision": decision,
        "reason": reason,
        "refund_inr": refund_inr,
    }


def assess_refund(
    order_id,
    orders,
    policy_text,
    delivery_date=None,
    request_date=None,
):
    """Assess refund eligibility using available data and policy."""

    order_id = str(order_id).strip()

    order = get_order(order_id, orders)

    if order is None:
        return make_result(
            order_id,
            "NOT_FOUND",
            "Order ID was not found.",
        )

    result = {
        "order_id": str(order["order_id"]),
        "customer_id": (
            None
            if pd.isna(order["customer_id"])
            else str(order["customer_id"])
        ),
    }

    def finish(decision, reason, refund_inr=0.0):
        return {
            **result,
            "decision": decision,
            "reason": reason,
            "refund_inr": refund_inr,
        }

    # Rule 1: Only completed orders qualify.
    if pd.isna(order["status"]):
        return finish(
            "NEEDS_REVIEW",
            "Order status is missing.",
            None,
        )

    if order["status"] != "completed":
        return finish(
            "DENIED",
            "Only completed orders qualify for a refund request.",
        )

    delivery = order["delivery_minutes"]
    amount = order["order_amount"]

    # Rule 2: Validate numeric data.
    if pd.isna(delivery) or pd.isna(amount):
        return finish(
            "NEEDS_REVIEW",
            "Delivery time or order amount is missing or invalid.",
            None,
        )

    if delivery < 0 or amount < 0:
        return finish(
            "NEEDS_REVIEW",
            "Negative delivery time or order amount requires review.",
            None,
        )

    result["delivery_minutes"] = float(delivery)

    # Rule 3: Delivery within 45 minutes is not eligible.
    if delivery <= 45:
        return finish(
            "NOT_ELIGIBLE",
            "Delivery was within 45 minutes.",
        )

    # Rule 4: The refund must be requested within 7 days
    # of delivery. Both dates must be supplied and verified.
    if not delivery_date or not request_date:
        return finish(
            "NEEDS_REVIEW",
            "Delivery date and refund request date are required "
            "to verify the 7-day policy.",
            None,
        )

    try:
        delivered = pd.to_datetime(
            delivery_date, format="%Y-%m-%d", errors="raise"
        ).date()

        requested = pd.to_datetime(
            request_date, format="%Y-%m-%d", errors="raise"
        ).date()

    except (ValueError, TypeError):
        return finish(
            "NEEDS_REVIEW",
            "Invalid date. Use YYYY-MM-DD format.",
            None,
        )

    if requested < delivered:
        return finish(
            "NEEDS_REVIEW",
            "Refund request date cannot be before delivery date.",
            None,
        )

    days_elapsed = (requested - delivered).days

    if days_elapsed > 7:
        return finish(
            "DENIED",
            "Refund request was raised more than 7 days "
            "after delivery.",
        )

    # Rule 5: Full order amount is refundable.
    refund_amount = round(float(amount), 2)

    result["delivery_date"] = delivered.isoformat()
    result["request_date"] = requested.isoformat()
    result["days_after_delivery"] = days_elapsed

    if refund_amount <= 1000:
        return finish(
            "AUTO_APPROVAL",
            "Eligible for a full refund; amount is up to INR 1000 "
            "and the 7-day condition is satisfied.",
            refund_amount,
        )

    return finish(
        "HUMAN_APPROVAL",
        "Eligible for a full refund; human approval is required "
        "above INR 1000.",
        refund_amount,
    )


def save_results(results):
    """Save audit results as JSON."""

    existing_results = []

    if OUTPUT_PATH.exists():
        try:
            existing_results = json.loads(
                OUTPUT_PATH.read_text(encoding="utf-8")
            )

            if not isinstance(existing_results, list):
                existing_results = []

        except (json.JSONDecodeError, OSError):
            existing_results = []

    # Append new results without deleting previous records.
    existing_results.extend(results)

    OUTPUT_PATH.write_text(
        json.dumps(
            existing_results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nResults saved: {OUTPUT_PATH}")


def main():
    try:
        orders = load_orders()
        policy_text = load_policy()

    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"Startup error: {exc}")
        return

    print("=" * 50)
    print("PART E - REFUND AGENT")
    print("=" * 50)

    print(f"Orders loaded: {len(orders)}")
    print(f"Policy loaded: {POLICY_PATH}")

    print("\nEnter an order ID, for example ORD-110227.")
    print("Type 'exit' to stop.\n")

    results = []

    while True:
        order_id = input("Order ID: ").strip()

        if order_id.lower() in {"exit", "quit"}:
            break

        if not order_id:
            print("Please enter an order ID.\n")
            continue

        try:
            order = get_order(order_id, orders)

            if order is None:
                result = make_result(
                    order_id,
                    "NOT_FOUND",
                    "Order ID was not found.",
                )
            elif (
                order["status"] == "completed"
                and pd.notna(order["delivery_minutes"])
                and pd.notna(order["order_amount"])
                and order["delivery_minutes"] > 45
                and order["delivery_minutes"] >= 0
                and order["order_amount"] >= 0
            ):
                print(
                    "\nEnter dates for the 7-day policy check."
                )
                print("Date format: YYYY-MM-DD")

                delivery_date = input(
                    "Delivery date: "
                ).strip()

                request_date = input(
                    "Refund request date: "
                ).strip()

                result = assess_refund(
                    order_id,
                    orders,
                    policy_text,
                    delivery_date=delivery_date or None,
                    request_date=request_date or None,
                )
            else:
                result = assess_refund(
                    order_id,
                    orders,
                    policy_text,
                )

            results.append(result)

            print("\nRefund assessment:")
            print(
                json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False,
                )
            )

        except ValueError as exc:
            print(f"Validation error: {exc}")

        print()

    if results:
        try:
            save_results(results)
        except OSError as exc:
            print(f"Could not save results: {exc}")

    print("\nPART E - REFUND AGENT STOPPED")


if __name__ == "__main__":
    main()
