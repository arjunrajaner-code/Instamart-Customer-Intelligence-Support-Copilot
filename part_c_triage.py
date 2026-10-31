
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import mock_llm
from sklearn.metrics import accuracy_score, confusion_matrix

RAW = Path("data/raw")
OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
CATEGORIES = ["billing", "delivery", "technical", "fraud"]

# Exact patterns and placeholders specified by the course.
PII_PATTERNS = {
    "email": (r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"),
    "phone": (r"\b[6-9]\d{9}\b", "[PHONE]"),
    "order_id": (r"ORD-\d{6}", "[ORDER_ID]"),
}

ZERO_SYSTEM = """
CLASSIFY
Classify the customer support ticket.
Allowed categories: billing, delivery, technical, fraud.
Return JSON only in exactly this form:
{"category": "...", "reason": "..."}
Do not add markdown or text outside the JSON.
"""

FEW_SYSTEM = """
CLASSIFY
Classify the customer support ticket.
Allowed categories: billing, delivery, technical, fraud.
Return JSON only in exactly this form:
{"category": "...", "reason": "..."}
Do not add markdown or text outside the JSON.
EXAMPLES:
"My account was hacked" -> {"category":"fraud","reason":"account compromised"}
"My app crashes" -> {"category":"technical","reason":"app problem"}
"My delivery is late" -> {"category":"delivery","reason":"late delivery"}
"I was charged twice" -> {"category":"billing","reason":"duplicate charge"}
"""

# --------------------------------------------------
# 1. Redaction
# --------------------------------------------------
def redact(text):
    text = str(text)
    counts = {}

    for name, (pattern, replacement) in PII_PATTERNS.items():
        text, count = re.subn(
            pattern, replacement, text, flags=re.IGNORECASE
        )
        counts[name] = count

    return text, counts


def contains_pii(text):
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern, _ in PII_PATTERNS.values()
    )


# --------------------------------------------------
# 2. Validate model output
# --------------------------------------------------
def validate_response(content):
    # First, try JSON.
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        data = None

    if isinstance(data, dict):
        category = data.get("category")
        reason = data.get("reason", "")

        # Valid JSON with an unsupported category must be flagged.
        if category in CATEGORIES:
            return "classified_json", category, str(reason)

        return "flagged_invalid_output", "", ""

    # Only non-JSON responses use keyword extraction.
    words = re.findall(r"[a-z]+", (content or "").lower())
    found = [word for word in words if word in CATEGORIES]

    if len(found) == 1:
        return "classified_keyword", found[0], "category extracted from text"

    return "flagged_invalid_output", "", ""


# --------------------------------------------------
# 3. Load input and audit PII
# --------------------------------------------------
tickets = pd.read_csv(
    RAW / "tickets.csv",
    dtype={"ticket_id": str, "customer_id": str, "order_id": str},
    keep_default_na=False,
)

required = {
    "ticket_id", "customer_id", "order_id", "text", "true_category"
}
if not required.issubset(tickets.columns):
    raise ValueError(f"Missing ticket columns: {required - set(tickets.columns)}")

redacted_texts = []
pii_totals = {name: 0 for name in PII_PATTERNS}

for original in tickets["text"]:
    safe_text, counts = redact(original)
    redacted_texts.append(safe_text)

    for name, count in counts.items():
        pii_totals[name] += count

# Confirm that reapplying all patterns finds no remaining PII.
assert all(not contains_pii(text) for text in redacted_texts)

print("PII SUBSTITUTIONS")
for name, count in pii_totals.items():
    print(f"{name}: {count}")
print("Total substitutions:", sum(pii_totals.values()))
print("PII remaining after redaction: 0")

# --------------------------------------------------
# 4. Input guardrails and resilient classification
# --------------------------------------------------
def run_pipeline(prompt_type, token_budget=None):
    # Required: reset immediately before every independent full run.
    mock_llm.reset()

    system = FEW_SYSTEM if prompt_type == "few" else ZERO_SYSTEM
    running_tokens = 0
    results = []
    retry_log = []

    for row_number, row in tickets.iterrows():
        ticket_id = row["ticket_id"]
        customer_id = row["customer_id"]
        original = row["text"]
        safe_text = redacted_texts[row_number]

        result = {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": "",
            "category": "",
            "reason": "",
            "tokens_used": 0,
            "attempts": 0,
            "error": "",
            "true_category": row["true_category"],
        }

        # Empty and long inputs must never trigger model calls.
        if not original.strip():
            result["status"] = "skipped_empty"
            results.append(result)
            continue

        if mock_llm.est_tokens(original) > MAX_INPUT_TOKENS:
            result["status"] = "flagged_long"
            results.append(result)
            continue

        # User message must be exactly the redacted ticket text.
        assert not contains_pii(safe_text)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": safe_text},
        ]
        assert messages[-1]["content"] == safe_text

        for retry_number in range(4):  # initial call + up to 3 retries
            # Check budget before every model call.
            if (
                token_budget is not None
                and running_tokens >= token_budget
            ):
                result["status"] = "budget_exhausted"
                result["error"] = "Token budget reached before model call"
                break

            result["attempts"] += 1

            try:
                response = mock_llm.chat(messages)
                used = response.usage.total_tokens
                running_tokens += used
                result["tokens_used"] += used

                content = response.choices[0].message.content or ""
                status, category, reason = validate_response(content)

                result["status"] = status
                result["category"] = category
                result["reason"] = reason

                if status == "flagged_invalid_output":
                    result["error"] = "Model output failed validation"

                break

            except mock_llm.TransientLLMError as exc:
                result["error"] = str(exc)

                if retry_number >= 3:
                    # A permanent failure is recorded, not hidden.
                    result["status"] = "flagged_invalid_output"
                    break

                wait = (
                    BASE_DELAY
                    * (2 ** retry_number)
                    * random.uniform(0.9, 1.15)
                )

                line = (
                    f"ticket_id={ticket_id} "
                    f"retry_number={retry_number} wait={wait:.5f}s"
                )
                retry_log.append(line)
                print("RETRY:", line)
                time.sleep(wait)

        if not result["status"]:
            result["status"] = "flagged_invalid_output"

        results.append(result)

    frame = pd.DataFrame(results)

    print(f"\n{prompt_type.upper()} RUN")
    print("Status counts:")
    print(frame["status"].value_counts().to_string())
    print("Total tokens:", running_tokens)

    valid = frame[
        frame["status"].isin(["classified_json", "classified_keyword"])
    ]
    print("Tickets classified:", len(valid))

    if len(valid):
        accuracy = accuracy_score(
            valid["true_category"], valid["category"]
        )
        print("Accuracy:", round(accuracy * 100, 2), "%")
        print(
    "Tokens per classified ticket:",
    round(running_tokens / len(valid), 2)
)

    if prompt_type == "zero":
        print("\nZERO-SHOT CONFUSION MATRIX")
        print(confusion_matrix(
            valid["true_category"],
            valid["category"],
            labels=CATEGORIES,
        ))
        print("Category order:", CATEGORIES)

    print("Retry count:", len(retry_log))
    return frame, retry_log


# --------------------------------------------------
# 5. Zero-shot and few-shot comparisons
# --------------------------------------------------
zero_df, zero_retries = run_pipeline("zero")
few_df, few_retries = run_pipeline("few")

zero_df.to_csv(OUT / "triage_zero_shot.csv", index=False)
few_df.to_csv(OUT / "triage_few_shot.csv", index=False)

# --------------------------------------------------
# 6. Zero-shot token-budget demonstration
# --------------------------------------------------
budget_df, budget_retries = run_pipeline(
    "zero", token_budget=6000
)
budget_df.to_csv(OUT / "triage_budget_demo.csv", index=False)

print(
    "\nBudget demo: budget=6000; tickets exhausted:",
    int((budget_df["status"] == "budget_exhausted").sum())
)

# --------------------------------------------------
# 7. Final few-shot run with segment-aware routing
# --------------------------------------------------
final_df, final_retries = run_pipeline(
    "few", token_budget=100000
)

segments = pd.read_csv(
    OUT / "customer_segments.csv",
    dtype={"customer_id": str},
)

segment_lookup = segments.set_index("customer_id")[
    ["segment_name", "is_premium"]
].to_dict("index")


def is_premium(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


final_rows = []

for row in final_df.to_dict(orient="records"):
    customer_id = row["customer_id"]
    segment = segment_lookup.get(customer_id, {})
    segment_name = segment.get("segment_name", "")
    premium = is_premium(segment.get("is_premium", False))

    status = row["status"]
    category = row["category"]

    if status in {
        "skipped_empty", "flagged_long",
        "flagged_invalid_output", "budget_exhausted"
    }:
        priority, route = "NORMAL", "human_review"
    elif category == "fraud":
        priority, route = "HIGH", "security_team"
    elif premium:
        priority, route = "HIGH", "premium_desk"
    else:
        priority, route = "NORMAL", "support_queue"

    final_rows.append({
        "ticket_id": row["ticket_id"],
        "customer_id": customer_id,
        "status": status,
        "category": category,
        "segment_name": segment_name,
        "priority": priority,
        "route": route,
        "tokens_used": row["tokens_used"],
    })

final_output = pd.DataFrame(final_rows)
final_output.to_csv(OUT / "triage_results.csv", index=False)

print("\nFINAL ROUTING COUNTS")
print(final_output["route"].value_counts().to_string())

print("\nFINAL STATUS COUNTS")
print(final_output["status"].value_counts().to_string())

print("\nSaved outputs/triage_results.csv")
print("Saved outputs/triage_budget_demo.csv")
print("PART C PIPELINE COMPLETED")
