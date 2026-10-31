import json
import sqlite3
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import mock_llm

from part_d_rag import retrieve


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "data" / "instamart.db"
OUTPUT_DIR = BASE_DIR / "outputs"
STATE_DIR = BASE_DIR / "state"

OUTPUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"
AGENT_RUNS_PATH = OUTPUT_DIR / "agent_runs.jsonl"


# ==========================================
# TOOL 1: GET ORDER
# ==========================================

def get_order(order_id):

    conn = sqlite3.connect(DB_PATH)

    try:
        row = conn.execute(
            """
            SELECT order_id, status,
                   order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        ).fetchone()

    finally:
        conn.close()

    if row is None:
        return {"error": "order not found"}

    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": (
            float(row[2]) if row[2] is not None else None
        ),
        "delivery_minutes": (
            float(row[3]) if row[3] is not None else None
        )
    }


# ==========================================
# TOOL 2: SEARCH POLICY
# ==========================================

def search_policy(query):

    results = retrieve(query, k=3)

    return {"results": results}


# ==========================================
# TOOL 3: CALCULATE REFUND
# ==========================================

def calculate_refund(order_amount, delivery_minutes):

    if order_amount is None or delivery_minutes is None:
        return {"error": "missing order information"}

    amount = float(order_amount)
    delivery = float(delivery_minutes)

    if amount < 0 or delivery < 0:
        return {"error": "invalid negative values"}

    refund = amount if delivery > 45 else 0.0

    return {
        "refund_inr": round(refund, 2)
    }


# ==========================================
# LEDGER
# ==========================================

def append_ledger(tool_name, order_id, refund_inr):

    record = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }

    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ==========================================
# TOOL 4: ISSUE REFUND
# ==========================================

def issue_refund(order_id, refund_inr):

    amount = float(refund_inr)

    if amount <= 0 or amount > 1000:
        return {"error": "refund amount not eligible"}

    append_ledger("issue_refund", order_id, amount)

    return {
        "refund_inr": round(amount, 2),
        "message": "Simulated refund recorded."
    }


# ==========================================
# TOOL 5: ESCALATE
# ==========================================

def escalate_to_human(order_id, refund_inr):

    amount = float(refund_inr)

    if amount <= 1000:
        return {"error": "human escalation not required"}

    append_ledger("escalate_to_human", order_id, amount)

    return {
        "refund_inr": round(amount, 2),
        "message": "Sent for human approval."
    }


# ==========================================
# TOOL SPECIFICATIONS
# ==========================================

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Get order details from SQLite",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Search refund policy",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate refund amount",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"}
            },
            "required": [
                "order_amount",
                "delivery_minutes"
            ]
        }
    },
    {
        "name": "issue_refund",
        "description": "Record simulated refund",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate refund above INR 1000",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


AGENT_SYSTEM_PROMPT = """
You are an Instamart Refund Agent.

Use tools to investigate refund requests.

Rules:
1. Retrieve order information first.
2. Search the refund policy.
3. Only completed orders are eligible.
4. Delivery above 45 minutes qualifies for refund.
5. Refunds up to INR 1000 can be processed automatically.
6. Refunds above INR 1000 require human approval.
7. Never refund cancelled orders.
8. Never claim actual money transfer.
9. Send unresolved cases to human support.
"""
# ==================================================
# CHECKPOINT SAVE
# ==================================================

def save_checkpoint(
    run_id,
    scenario_id,
    messages,
    tokens_used,
    step_number,
    tools_executed
):

    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_number": step_number,
        "tools_executed": tools_executed
    }

    path = STATE_DIR / f"{run_id}.json"
    temp_path = path.with_suffix(".tmp")

    temp_path.write_text(
        json.dumps(checkpoint, indent=2),
        encoding="utf-8"
    )

    temp_path.replace(path)


# ==================================================
# CHECKPOINT LOAD
# ==================================================

def load_checkpoint(run_id):

    path = STATE_DIR / f"{run_id}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {run_id}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


# ==================================================
# AGENT RUN LOGGER
# ==================================================

def log_agent_run(result):

    with AGENT_RUNS_PATH.open(
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(result, default=str) + "\n"
        )


# ==================================================
# TOOL EXECUTION
# ==================================================

def execute_tool(tool_name, arguments):

    if tool_name not in TOOL_MAP:
        return {
            "error": f"Unknown tool: {tool_name}"
        }

    try:
        return TOOL_MAP[tool_name](**arguments)

    except Exception as error:
        return {
            "error": str(error)
        }


# ==================================================
# FALLBACK RESPONSES
# ==================================================

def get_fallback(reason):

    fallbacks = {
        "max_iterations": (
            "FALLBACK_MAX_ITERATIONS: "
            "Maximum iterations reached. "
            "Please contact human support."
        ),

        "token_limit": (
            "FALLBACK_TOKEN_LIMIT: "
            "Token budget reached. "
            "Please contact human support."
        ),

        "timeout": (
            "FALLBACK_TIMEOUT: "
            "Execution time exceeded. "
            "Please contact human support."
        )
    }

    return fallbacks[reason]


# ==================================================
# REACT AGENT
# ==================================================

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tool=None
):

    if run_id is None:
        run_id = (
            f"{scenario_id}_"
            f"{int(time.time() * 1000)}"
        )

    messages = [
        {
            "role": "system",
            "content": AGENT_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    tokens_used = 0
    llm_calls = 0
    step_number = 0
    tools_executed = []

    start_time = time.monotonic()

    status = "RUNNING"
    final_answer = ""

    for iteration in range(max_iterations):

        # Timeout check
        if time.monotonic() - start_time >= time_limit_seconds:

            status = "FALLBACK_TIMEOUT"
            final_answer = get_fallback("timeout")
            break

        # Token check before LLM call
        if tokens_used >= token_limit:

            status = "FALLBACK_TOKEN_LIMIT"
            final_answer = get_fallback("token_limit")
            break

        # LLM call with retry
        response = None

        for attempt in range(2):

            try:
                response = mock_llm.chat(
                    messages,
                    tools=TOOLS_SPEC
                )
                break

            except mock_llm.TransientLLMError:
                if attempt == 1:
                    status = "ERROR"
                    final_answer = (
                        "LLM temporary failure. "
                        "Please contact human support."
                    )

        if response is None:
            break

        llm_calls += 1

        tokens_used += response.usage.total_tokens

        # Check token limit before tool execution
        if tokens_used >= token_limit:

            if response.choices[0].message.tool_calls:

                status = "FALLBACK_TOKEN_LIMIT"
                final_answer = get_fallback("token_limit")
                break

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        # --------------------------------------
        # TOOL CALL RESPONSE
        # --------------------------------------

        if tool_calls:

            assistant_tool_calls = []

            for tool_call in tool_calls:

                assistant_tool_calls.append({
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(
                            tool_call.arguments
                        )
                    }
                })

            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_tool_calls
            })

            for tool_call in tool_calls:

                tool_name = tool_call.name
                arguments = tool_call.arguments

                observation = execute_tool(
                    tool_name,
                    arguments
                )

                step_number += 1

                tools_executed.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "observation": observation
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(
                        observation,
                        default=str
                    )
                })

                # Save checkpoint after every tool
                save_checkpoint(
                    run_id,
                    scenario_id,
                    messages,
                    tokens_used,
                    step_number,
                    tools_executed
                )

                # Crash simulation
                if (
                    crash_after_tool is not None
                    and step_number == crash_after_tool
                ):

                    raise RuntimeError(
                        "SIMULATED_AGENT_CRASH"
                    )

            continue

        # --------------------------------------
        # FINAL RESPONSE
        # --------------------------------------

        final_answer = message.content or ""

        status = "COMPLETED"
        break

    else:

        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = get_fallback("max_iterations")

    # ------------------------------------------
    # FINAL RESULT
    # ------------------------------------------

    result = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "status": status,
        "final_answer": final_answer,
        "steps": step_number,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "tools_executed": tools_executed
    }

    save_checkpoint(
        run_id,
        scenario_id,
        messages,
        tokens_used,
        step_number,
        tools_executed
    )

    return result


# ==================================================
# RESUME AGENT
# ==================================================

def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):

    checkpoint = load_checkpoint(run_id)

    scenario_id = checkpoint["scenario_id"]
    messages = checkpoint["messages"]
    tokens_used = checkpoint["tokens_used"]
    step_number = checkpoint["step_number"]
    tools_executed = checkpoint["tools_executed"]

    start_time = time.monotonic()
    llm_calls = 0

    status = "RUNNING"
    final_answer = ""

    for iteration in range(max_iterations):

        if time.monotonic() - start_time >= time_limit_seconds:

            status = "FALLBACK_TIMEOUT"
            final_answer = get_fallback("timeout")
            break

        if tokens_used >= token_limit:

            status = "FALLBACK_TOKEN_LIMIT"
            final_answer = get_fallback("token_limit")
            break

        response = None

        for attempt in range(2):

            try:
                response = mock_llm.chat(
                    messages,
                    tools=TOOLS_SPEC
                )
                break

            except mock_llm.TransientLLMError:
                if attempt == 1:
                    status = "ERROR"
                    final_answer = "LLM temporary failure."

        if response is None:
            break

        llm_calls += 1
        tokens_used += response.usage.total_tokens

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if tool_calls:

            if tokens_used >= token_limit:

                status = "FALLBACK_TOKEN_LIMIT"
                final_answer = get_fallback("token_limit")
                break

            assistant_tool_calls = []

            for tool_call in tool_calls:

                assistant_tool_calls.append({
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(
                            tool_call.arguments
                        )
                    }
                })

            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_tool_calls
            })

            for tool_call in tool_calls:

                observation = execute_tool(
                    tool_call.name,
                    tool_call.arguments
                )

                step_number += 1

                tools_executed.append({
                    "tool": tool_call.name,
                    "arguments": tool_call.arguments,
                    "observation": observation
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "content": json.dumps(
                        observation,
                        default=str
                    )
                })

                save_checkpoint(
                    run_id,
                    scenario_id,
                    messages,
                    tokens_used,
                    step_number,
                    tools_executed
                )

            continue

        final_answer = message.content or ""
        status = "COMPLETED"
        break

    else:

        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = get_fallback("max_iterations")

    result = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "status": status,
        "final_answer": final_answer,
        "steps": step_number,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "tools_executed": tools_executed
    }

    save_checkpoint(
        run_id,
        scenario_id,
        messages,
        tokens_used,
        step_number,
        tools_executed
    )

    return result

# ==================================================
# SCENARIO RESULT CLASSIFICATION
# ==================================================

def classify_agent_result(result):

    answer = result["final_answer"].upper()

    if "REFUND:" in answer:
        return "REFUND"

    if "ESCALATE:" in answer:
        return "ESCALATE"

    if "NOT_FOUND:" in answer:
        return "NOT_FOUND"

    if "DENIED:" in answer:
        return "DENIED"

    if result["status"] == "FALLBACK_MAX_ITERATIONS":
        return "FALLBACK"

    return "ERROR"


# ==================================================
# SEVEN SCENARIO EVALUATION
# ==================================================

def evaluate_scenarios():

    scenario_file = BASE_DIR / "data" / "raw" / "agent_scenarios.csv"

    if not scenario_file.exists():
        raise FileNotFoundError(
            "agent_scenarios.csv not found"
        )

    scenarios = pd.read_csv(scenario_file)

    results = []

    # Keep scenario test ledger separate
    global LEDGER_PATH

    original_ledger = LEDGER_PATH

    LEDGER_PATH = OUTPUT_DIR / "d6_test_refund_ledger.jsonl"

    if LEDGER_PATH.exists():
        LEDGER_PATH.unlink()

    try:

        for _, row in scenarios.iterrows():

            mock_llm.reset()

            result = run_agent(
                user_message=str(row["message"]),
                scenario_id=str(row["scenario_id"]),
                max_iterations=6,
                token_limit=6000,
                time_limit_seconds=10
            )

            actual = classify_agent_result(result)

            expected = str(
                row["expected_outcome"]
            ).upper()

            expected_refund = float(
                row["expected_refund_inr"]
            )

            refund_amount = 0.0

            for step in result["tools_executed"]:

                observation = step["observation"]

                if "refund_inr" in observation:
                    refund_amount = float(
                        observation["refund_inr"]
                    )

            passed = actual == expected

            if expected in ("REFUND", "ESCALATE"):

                passed = (
                    passed
                    and abs(
                        refund_amount - expected_refund
                    ) < 0.01
                )

            result["expected_outcome"] = expected
            result["actual_outcome"] = actual
            result["expected_refund_inr"] = expected_refund
            result["actual_refund_inr"] = refund_amount
            result["passed"] = passed

            results.append(result)

            # One JSONL record per scenario
            log_agent_run(result)

    finally:

        LEDGER_PATH = original_ledger

    report = pd.DataFrame(results)

    report.to_csv(
        OUTPUT_DIR / "d6_scenario_results.csv",
        index=False
    )

    print("\nSEVEN SCENARIO RESULTS")

    print(
        report[
            [
                "scenario_id",
                "expected_outcome",
                "actual_outcome",
                "expected_refund_inr",
                "actual_refund_inr",
                "passed"
            ]
        ].to_string(index=False)
    )

    print("\nTOTAL:", len(report))
    print("PASSED:", int(report["passed"].sum()))
    print("FAILED:", int((~report["passed"]).sum()))

    return report


# ==================================================
# TOKEN LIMIT TEST
# ==================================================

def test_token_limit():

    mock_llm.reset()

    result = run_agent(
        user_message="Check refund for ORD-100048",
        scenario_id="TOKEN_LIMIT_TEST",
        token_limit=200
    )

    print("\nTOKEN LIMIT TEST")
    print(result)

    return result


# ==================================================
# TIMEOUT TEST
# ==================================================

def test_timeout():

    mock_llm.reset()

    result = run_agent(
        user_message="Check refund for ORD-100048",
        scenario_id="TIMEOUT_TEST",
        time_limit_seconds=0
    )

    print("\nTIMEOUT TEST")
    print(result)

    return result


# ==================================================
# CHECKPOINT AND RESUME TEST
# ==================================================

def test_checkpoint_resume():

    print("\nCHECKPOINT AND RESUME TEST")

    run_id = "D7_RESUME_TEST"

    checkpoint_path = STATE_DIR / f"{run_id}.json"

    # Remove old test checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Use a separate test ledger
    global LEDGER_PATH

    original_ledger = LEDGER_PATH

    LEDGER_PATH = OUTPUT_DIR / "d7_resume_test_ledger.jsonl"

    if LEDGER_PATH.exists():
        LEDGER_PATH.unlink()

    try:

        mock_llm.reset()

        # --------------------------------------
        # FIRST RUN: CRASH AFTER SECOND TOOL
        # --------------------------------------

        try:

            run_agent(
                user_message="Check refund for ORD-100048",
                scenario_id="D7_RESUME_TEST",
                run_id=run_id,
                crash_after_tool=2
            )

        except RuntimeError as error:

            if "SIMULATED_AGENT_CRASH" not in str(error):
                raise

            print("Simulated crash after second tool.")

        # --------------------------------------
        # VERIFY CHECKPOINT
        # --------------------------------------

        checkpoint = load_checkpoint(run_id)

        print(
            "Tools completed before crash:",
            [
                item["tool"]
                for item in checkpoint["tools_executed"]
            ]
        )

        # --------------------------------------
        # RESUME FROM SAVED STATE
        # --------------------------------------

        result = resume_agent(
            run_id,
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10
        )

        print("\nRESUMED RESULT")
        print(json.dumps(result, indent=2))

        # --------------------------------------
        # VERIFY EXACT TOOL SEQUENCE
        # --------------------------------------

        expected_tools = [
            "get_order",
            "search_policy",
            "calculate_refund",
            "issue_refund"
        ]

        actual_tools = [
            item["tool"]
            for item in result["tools_executed"]
        ]

        assert actual_tools == expected_tools, (
            f"Unexpected tool sequence: {actual_tools}"
        )

        # --------------------------------------
        # VERIFY SINGLE REFUND LEDGER ENTRY
        # --------------------------------------

        ledger_entries = []

        if LEDGER_PATH.exists():

            with LEDGER_PATH.open(
                "r",
                encoding="utf-8"
            ) as file:

                ledger_entries = [
                    json.loads(line)
                    for line in file
                    if line.strip()
                ]

        assert len(ledger_entries) == 1, (
            "Duplicate or missing refund ledger entry"
        )

        assert (
            ledger_entries[0]["refund_inr"] == 128.25
        )

        print("\nCHECKPOINT RESUME: PASS")
        print("Total tool executions:", len(actual_tools))
        print("Duplicate executions: 0")
        print("Refund ledger entries:", len(ledger_entries))

        return result

    finally:

        LEDGER_PATH = original_ledger


# ==================================================
# REACT VS PLAN AND EXECUTE
# ==================================================

def print_agent_comparison():

    print("\nREACT VS PLAN-AND-EXECUTE")

    comparison = """
    ReAct:
    - Executes one tool at a time.
    - Observes each result before deciding the next action.
    - Handles unexpected orders and missing information dynamically.
    - May require more model calls.

    Plan-and-Execute:
    - Creates a plan before executing the tools.
    - Can reduce repeated reasoning calls.
    - Works well when the workflow is predictable.
    - A fixed plan may need adjustment when unexpected results occur.

    Selected approach:
    ReAct is preferred for the refund workflow because
    order status, delivery time and refund eligibility
    must be checked before taking the next action.
    """

    print(comparison)


# ==================================================
# MAIN EXECUTION
# ==================================================

if __name__ == "__main__":

    print("=" * 55)
    print("PART D - REFUND REACT AGENT")
    print("=" * 55)

    # Reset mock model before scenario batch
    mock_llm.reset()

    # Seven scenario evaluation
    report = evaluate_scenarios()

    # Verify all seven scenarios
    assert len(report) == 7

    assert report["passed"].all(), (
        "One or more agent scenarios failed."
    )

    # Token limit test
    token_result = test_token_limit()

    assert (
        token_result["status"] == "FALLBACK_TOKEN_LIMIT"
    )

    # Timeout test
    timeout_result = test_timeout()

    assert (
        timeout_result["status"] == "FALLBACK_TIMEOUT"
    )

    # Checkpoint and resume
    resume_result = test_checkpoint_resume()

    assert (
        resume_result["status"] == "COMPLETED"
    )

    # Comparison
    print_agent_comparison()

    print("\n" + "=" * 55)
    print("PART D AGENT TESTS COMPLETED")
    print("=" * 55)