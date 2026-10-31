# Instamart Customer Intelligence & Support Copilot

An end-to-end data engineering, machine learning and AI project that transforms raw customer data into actionable customer insights, predicts customer lapse, automates support ticket triage, and provides policy-based question answering with a tool-using refund agent.

---

## 1. Overview

### Project Description

The Instamart Customer Intelligence & Support Copilot is a data science and AI capstone project developed using synthetic customer, product, order and support ticket datasets.

The project addresses four major business challenges:

1. Improving data quality and building a reliable analytical database.
2. Identifying customer segments and predicting customers likely to stop ordering.
3. Classifying support tickets while protecting personal information and routing them to the correct teams.
4. Building a policy-based RAG assistant and an AI refund agent with tool execution and safety guardrails.

The project is divided into four connected parts.

| Part | Module | Description | Marks |
|---|---|---|---:|
| A | Data Foundation | Data auditing, cleaning, validation, SQLite and SQL analysis | 22 |
| B | Customer Intelligence | RFM segmentation and lapse prediction | 28 |
| C | Support Ticket Triage | Classification, privacy protection and routing | 22 |
| D | Policy RAG & Refund Agent | Policy retrieval, question answering and ReAct agent | 28 |
| | **Total** | | **100** |

### Technology Stack

| Area | Technologies |
|---|---|
| Programming Language | Python 3.11+ |
| Data Processing | Pandas, NumPy |
| Database | SQLite |
| Machine Learning | Scikit-learn |
| Model Persistence | Joblib |
| NLP | TF-IDF, Cosine Similarity |
| LLM Integration | Supplied offline Mock LLM |
| Development Environment | Visual Studio Code |

All datasets are synthetic and fictional. This project does not use real Instamart customer information, policies, transactions or pricing.

No paid API, external LLM service or API key is required. All language-model calls use the supplied `mock_llm.py`.

---

## 2. Reproduce

Follow these commands in order to reproduce the project from a clean clone.

### Prerequisites

- Python 3.11 or above
- SQLite 3.25 or above
- Git
- Visual Studio Code (recommended)

### Step 1: Clone the repository

```bash
git clone <your-github-repository-url>
cd instamart-copilot
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### Step 4: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Verify the environment

```bash
python --version
python -c "import sqlite3; print(sqlite3.sqlite_version)"
pip check
```

### Step 6: Generate synthetic datasets

```bash
python generate_data.py
```

### Step 7: Part A – Data Foundation

```bash
python audit_raw.py
python clean_data.py
python validate_data.py
python check_db.py
python run_queries.py
python part_a_analysis.py
```

### Step 8: Part B – Customer Intelligence

```bash
python part_b_segments.py
python part_b_lapse.py
```

### Step 9: Part C – Support Ticket Triage

```bash
python part_c_triage.py
```

### Step 10: Part D – Policy RAG Assistant

```bash
python part_d_rag.py
```

### Step 11: Part D – Refund Agent

```bash
python part_d_react_agent.py
```

The scripts generate the datasets, database, models, evaluation results and agent checkpoints in their respective directories.

The supplied `generate_data.py` and `mock_llm.py` files must remain unmodified.

---

## 3. Part A – Data Foundation: Clean, Store and Query

### Objective

To audit, clean, validate and store raw customer, product and order datasets in a structured SQLite database.

### Task 1: Data Auditing

The raw datasets were inspected using Pandas.

The audit identified missing values, duplicate records, inconsistent categorical values, invalid amounts and impossible delivery times.

#### Raw Data Audit Results

| Audit Metric | Result |
|---|---:|
| Raw order rows | 13,084 |
| Exact duplicate order rows | 119 |
| Pending order rows | 656 |
| Distinct order IDs | 12,309 |
| Invalid non-numeric order amounts | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2,541 |
| Blank customer ages | 61 |
| Customer ages outside 10–100 | 6 |
| Distinct raw city spellings | 18 |
| Distinct non-blank gender spellings | 10 |
| Blank gender values | 30 |

### Task 2: Customer Data Cleaning

The customer dataset was cleaned using the following rules:

1. Removed leading and trailing spaces from city names.
2. Standardised city names using title case.
3. Mapped gender values into `male`, `female` and `unknown`.
4. Replaced blank gender values with `unknown`.
5. Treated ages outside 10–100 as missing.
6. Filled missing ages using the median age of the customer's city.
7. Converted `signup_date` into a datetime format.

#### Customer Cleaning Results

| Metric | Result |
|---|---:|
| Pune | 227 |
| Delhi | 210 |
| Bengaluru | 197 |
| Chennai | 191 |
| Hyderabad | 191 |
| Mumbai | 184 |
| Female | 610 |
| Male | 560 |
| Unknown | 30 |
| Missing ages after cleaning | 0 |

### Task 3: Order Data Cleaning

The order dataset was cleaned in the following order:

1. Converted `order_ts` into datetime.
2. Removed exact duplicate rows.
3. Kept the latest record for each `order_id`.
4. Converted `order_amount` into numeric values.
5. Reconstructed missing amounts using product price, quantity and discount.

The amount calculation used:

`order_amount = round(price_inr * quantity * (1 - discount_pct / 100), 2)`

6. Converted negative delivery times into missing values.
7. Filled missing delivery times for completed orders using the median delivery time of the corresponding product category.
8. Kept delivery times missing for cancelled orders because those orders were never delivered.

#### Final Order Cleaning Results

| Metric | Result |
|---|---:|
| Total cleaned orders | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |
| Pending orders | 0 |
| Missing order amounts | 0 |
| Missing completed delivery times | 0 |
| Missing cancelled delivery times | 1,199 |
| Completed order revenue | INR 29,98,668.80 |

### Task 4: Data Validation

The validation script checks record counts, unique IDs, missing values, order statuses, customer ages and completed revenue.

Validation was executed successfully.

**Result: ALL CHECKPOINTS PASSED**

### Task 5: SQLite Database

The cleaned datasets were stored as CSV files and in an SQLite database.

Database location:

`data/instamart.db`

Tables:

- `customers`
- `products`
- `orders`

### Task 6: SQL Analysis

Seven SQL queries were created and executed using `pandas.read_sql_query()`.

| Query | SQL Concept | Result |
|---|---|---:|
| Q1 | WHERE and HAVING | 30 customers |
| Q2 | Three-table JOIN | 30 city-category combinations |
| Q3 | Scalar subquery | 11 products |
| Q4 | LEFT JOIN | 25 customers |
| Q5 | Common Table Expression | 550 customers |
| Q6 | Ranking window functions | 8 rows where RANK differs from ROW_NUMBER |
| Q7 | Running total window function | 183 daily rows |

#### Important SQL Findings

- Completed revenue across all city and category combinations: INR 29,98,668.80.
- Highest revenue combination: Bengaluru – Electronics, INR 2,33,909.45.
- Average completed customer spend: INR 2,552.06.
- Final cumulative revenue in Q7: INR 29,98,668.80.

In Q6, ties can cause `RANK()` and `ROW_NUMBER()` to differ because tied customers receive the same rank, while `ROW_NUMBER()` assigns a unique sequential number.

### Task 7: Pandas Analysis

The SQL results were cross-checked using Pandas.

Exploratory analysis included:

- Revenue by city.
- Mean and count of completed order amounts.
- Order amount distribution.
- Mean, median and skewness.
- Quartiles and IQR-based outlier detection.
- Correlation analysis.

The largest absolute correlation observed was between `order_amount` and `discount_pct`, approximately -0.095.

This correlation is weak and does not establish causation.

### Part A Outputs

- `data/clean/customers.csv`
- `data/clean/products.csv`
- `data/clean/orders.csv`
- `data/instamart.db`
- `sql/queries.sql`
- `outputs/raw_data_audit.txt`

**Part A Result:** Data cleaning, validation, database creation and all seven SQL queries were executed successfully.

---

## 4. Part B – Customer Segmentation and Lapse Prediction

### Objective

To identify customer groups based on purchasing behaviour and predict customers who are likely to stop ordering.

### Task 1: RFM Analysis

The RFM approach was used to understand customer purchasing behaviour.

| Metric | Meaning |
|---|---|
| Recency | Days since the last completed order |
| Frequency | Number of distinct completed orders |
| Monetary | Total amount spent on completed orders |

The RFM metrics were calculated using the snapshot date `2025-10-01 00:00:00`.

### Task 2: Customer Segmentation

The following steps were implemented:

1. Calculated Recency, Frequency and Monetary values.
2. Standardised RFM features using `StandardScaler`.
3. Applied K-Means clustering.
4. Compared cluster counts from K=2 to K=8.
5. Evaluated inertia and silhouette scores.
6. Selected K=4.
7. Profiled and named the customer segments.
8. Applied PCA for dimensionality reduction.

#### K-Means Evaluation

| K | Inertia | Silhouette Score |
|---:|---:|---:|
| 2 | 1820.761 | 0.465 |
| 3 | 1268.235 | 0.445 |
| 4 | 743.071 | 0.526 |
| 5 | 617.367 | 0.480 |
| 6 | 503.171 | 0.482 |
| 7 | 408.460 | 0.416 |
| 8 | 366.503 | 0.412 |

K=4 was selected because it provided the highest silhouette score among the tested values and produced useful business-oriented customer groups.

Inertia generally decreases when the number of clusters increases because more clusters can represent the data more closely. However, lower inertia alone does not identify the best K because too many clusters can reduce business interpretability.

### Task 3: Segment Profiling

Four customer segments were identified.

| Segment | Customers | Mean Recency | Mean Frequency | Mean Monetary (INR) |
|---|---:|---:|---:|---:|
| Needs Attention | 436 | 28.86 | 3.77 | 1,057.62 |
| Premium | 138 | 24.12 | 6.77 | 7,129.70 |
| Frequent Loyal | 419 | 8.77 | 19.40 | 3,410.02 |
| At Risk | 182 | 109.04 | 2.20 | 685.99 |

The segment with the highest mean monetary value was identified as Premium.

Cluster numbers are only labels and do not represent an inherent ranking.

#### Business Actions

| Segment | Recommended Action |
|---|---|
| Needs Attention | Offer personalised promotions to increase engagement |
| Premium | Provide loyalty rewards and exclusive benefits |
| Frequent Loyal | Encourage repeat purchases and subscription benefits |
| At Risk | Launch win-back campaigns and targeted offers |

### Task 4: PCA Analysis

PCA was applied using two components.

| Component | Explained Variance |
|---|---:|
| Principal Component 1 | 0.6343 |
| Principal Component 2 | 0.2180 |
| Total | 0.8523 |

The first two components explain approximately 85.23% of the variance in the scaled RFM data.

Output:

`outputs/customer_segments.csv`

### Task 5: Lapse Prediction

A customer was defined as lapsed if they had no completed order between 1 September 2025 and 30 September 2025.

The cohort was created using completed orders before 1 September 2025.

#### Cohort Details

| Metric | Result |
|---|---:|
| Cohort size | 1,156 |
| Lapsed customers | 443 |
| Training rows | 924 |
| Test rows | 232 |
| Lapsed percentage | 38.32% |

Eleven features were used:

- Recency
- Frequency
- Monetary
- Average discount percentage
- Cancellation rate
- Average delivery time
- Favourite product category
- City
- Age
- Gender
- Customer tenure

No orders dated on or after 1 September 2025 were used in the model features.

Customer segment labels were not used as model features.

### Task 6: Model Training and Evaluation

Two models were trained and tuned:

1. Logistic Regression
2. Random Forest Classifier

GridSearchCV with five-fold cross-validation was used.

#### Model Comparison

| Model | Cross-Validation F1 | Test F1 |
|---|---:|---:|
| Logistic Regression | 0.639 | 0.724 |
| Random Forest | 0.670 | 0.706 |

**Selected Model: Random Forest**

Random Forest was selected because it achieved the higher cross-validation F1-score.

#### Selected Model Test Metrics

| Metric | Score |
|---|---:|
| Precision | 0.673 |
| Recall | 0.742 |
| F1-score | 0.706 |

The selected model achieved a test F1-score above the required minimum of 0.55.

The test F1-score is calculated for the positive lapsed class.

### Task 7: Class Balancing and Model Interpretation

Class balancing was performed by oversampling the minority lapsed class on training data only.

The experiment compared model performance before and after balancing on the unchanged test set.

Random Forest feature importance and Logistic Regression coefficients were also examined to understand the factors associated with customer lapse.

### Task 8: Model Persistence

The selected model was saved using Joblib:

`outputs/lapse_model.joblib`

The model was reloaded and its predictions were verified against the original test predictions.

The predictions matched exactly.

### Task 9: Premium Retention List

The top 10 Premium customers were ranked using predicted lapse probability.

Output:

`outputs/retention_list.csv`

- Retention list rows: 10
- Customers who actually lapsed in this list: 2

**Important limitation:** The Premium flag uses whole-window RFM data, including September 2025. Therefore, the retention list is illustrative and is not a leakage-free evaluation.

### Part B Outputs

- `outputs/customer_segments.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

**Part B Result:** Customer segmentation, PCA, model tuning, model persistence and retention-list generation were completed.

---

## 5. Part C – Support Ticket Triage with Guardrails

### Objective

To classify customer support tickets, protect personal information, validate model responses, manage token usage and route tickets to the correct support teams.

The project contains 164 synthetic support tickets.

### Task 1: Input Guardrails

Before making a model call:

- Empty or whitespace-only tickets were skipped.
- Tickets exceeding 400 estimated tokens were flagged as long.
- Empty and long tickets were not sent to the Mock LLM.

#### Input Guardrail Results

| Status | Count |
|---|---:|
| Empty tickets skipped | 4 |
| Long tickets flagged | 4 |

### Task 2: PII Redaction

Personal information was removed before ticket text was sent to the model.

| Information | Replacement |
|---|---|
| Email address | `[EMAIL]` |
| Phone number | `[PHONE]` |
| Order ID | `[ORDER_ID]` |

#### Redaction Results

| Type | Substitutions |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| **Total** | **278** |

The redaction patterns were applied again to verify that no matching personal information remained.

Zero matching PII values remained after redaction.

### Task 3: Classification Prompts

Two classification strategies were implemented.

**Zero-Shot Classification**

The model receives the ticket without labelled examples and identifies its category.

**Few-Shot Classification**

The model receives labelled examples before classifying the ticket.

The four allowed categories are:

- Billing
- Delivery
- Technical
- Fraud

The model is instructed to return JSON with `category` and `reason`.

### Task 4: Retry Mechanism

Transient LLM failures were handled using:

- Maximum three retries per ticket.
- Exponential backoff.
- Random jitter.
- `mock_llm.reset()` before each full run.

The full run produced 19 transient errors, and all were recovered through retries.

### Task 5: Output Validation

Every model response was validated before accepting a classification.

| Status | Count |
|---|---:|
| Classified using JSON | 135 |
| Classified using keyword fallback | 12 |
| Flagged invalid output | 9 |

Invalid outputs were flagged for human review instead of guessing a category.

### Task 6: Zero-Shot vs Few-Shot Evaluation

| Metric | Zero-Shot | Few-Shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Correct classifications | 126 | 147 |
| Accuracy | 85.7% | 100% |

The few-shot approach achieved higher accuracy in the evaluation run.

However, few-shot prompts use additional tokens because labelled examples are included in the prompt. This increases processing cost. The additional cost is worthwhile when higher classification accuracy is more important than minimising token usage.

### Task 7: Token Budget Management

A token budget was implemented to prevent excessive model usage.

Before each model call, the pipeline checks whether the accumulated token usage has reached the configured budget.

If the budget is exhausted, the ticket is routed to human review without making another model call.

The budget demonstration is saved in:

`outputs/triage_budget_demo.csv`

### Task 8: Ticket Routing

The final routing pipeline uses customer segment information from Part B.

| Condition | Route | Priority |
|---|---|---|
| Empty, long, invalid or budget-exhausted ticket | human_review | NORMAL |
| Fraud category | security_team | HIGH |
| Premium customer | premium_desk | HIGH |
| Other valid tickets | support_queue | NORMAL |

Customers without a matching segment record are treated as non-premium.

The final routing output contains ticket ID, customer ID, classification status, category, segment name, priority, route and tokens used.

### Part C Outputs

- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/tickets_zero_shot.csv`
- `outputs/tickets_few_shot.csv`
- `outputs/tickets_routed.csv`

**Part C Result:** Ticket guardrails, PII redaction, retry handling, classification, token budget management and customer-aware routing were implemented and tested.

---

## 6. Part D – Policy RAG Assistant and Refund Agent

### Objective

To build a policy question-answering assistant using Retrieval-Augmented Generation and a tool-using refund agent that retrieves order details, searches policies, calculates refunds and performs safe simulated actions.

### Task 1: Policy Document Chunking

Five policy documents were loaded from `data/policies/`.

The documents were divided into:

- Chunk size: 40 words
- Chunk overlap: 10 words
- Total chunks: 15

The source filename was stored with every chunk.

### Task 2: Policy Retrieval

A TF-IDF-based retrieval system was implemented.

Implementation:

1. Used `TfidfVectorizer()` to vectorise the policy chunks.
2. Transformed user questions using the same fitted vectorizer.
3. Used cosine similarity to rank relevant chunks.
4. Retrieved the top three chunks for each query.

TF measures how frequently a term appears in a document. IDF reduces the weight of terms that occur across many documents.

One limitation of TF-IDF is that it may not recognise two phrases with the same meaning if they use different words.

### Task 3: RAG Question Answering

The RAG pipeline follows these steps:

1. Receives a user question.
2. Retrieves the three most relevant policy chunks.
3. Adds the retrieved context to the prompt.
4. Sends the question and context to the offline Mock LLM.
5. Generates an answer based on the retrieved policy information.
6. Returns the answer and its source documents.

The assistant is instructed to answer only from the supplied policy context.

If the required information is unavailable, it returns a fallback response instead of inventing an answer.

### Task 4: RAG Evaluation

The assistant was evaluated using 14 policy questions.

| Evaluation Metric | Result |
|---|---:|
| Answerable questions with expected document retrieved | 12/12 |
| Answers containing expected facts | 12/12 |
| Unanswerable questions with correct fallback | 2/2 |

A second evaluation was performed using zero chunk overlap.

At least one answerable question failed in the zero-overlap evaluation because relevant information was divided across chunk boundaries.

### Task 5: Refund Agent Tools

The refund agent uses five tools.

| Tool | Purpose |
|---|---|
| `get_order` | Retrieves order details from SQLite |
| `search_policy` | Retrieves relevant policy chunks |
| `calculate_refund` | Calculates the eligible refund amount |
| `issue_refund` | Records a simulated refund |
| `escalate_to_human` | Escalates high-value refund cases |

The tools are stored in `TOOLS_SPEC` and mapped to Python functions using `TOOL_MAP`.

### Refund Rules

The agent follows these rules:

1. Only completed orders are eligible for refunds.
2. Completed orders delivered in more than 45 minutes receive a full refund.
3. Completed orders delivered in 45 minutes or less receive no late-delivery refund.
4. Refunds above INR 1,000 are escalated to a human.
5. Refunds up to INR 1,000 can be processed automatically.
6. All simulated requests are treated as being within the policy's seven-day request window.

The refund agent does not process real payments. Refund and escalation actions are recorded in a simulated ledger.

### Task 6: ReAct Agent Workflow

The ReAct agent uses a reasoning and action loop.

The workflow is:

1. Receive the user request.
2. Ask the Mock LLM for the next action.
3. Identify the requested tool and its arguments.
4. Execute the tool.
5. Append the tool observation to the conversation.
6. Ask the model for the next action.
7. Return the final result or a safe fallback.

The default limits are:

| Limit | Value |
|---|---:|
| Maximum iterations | 6 |
| Token limit | 6,000 |
| Time limit | 10 seconds |

The agent records its tool calls, arguments, observations, model calls and token usage.

### Task 7: Seven-Scenario Evaluation

The agent was evaluated against seven predefined scenarios.

| Scenario | Expected Result | Expected Refund (INR) |
|---|---|---:|
| S1 | REFUND | 128.25 |
| S2 | REFUND | 135.00 |
| S3 | ESCALATE | 1,999.00 |
| S4 | DENIED | 0.00 |
| S5 | DENIED | 0.00 |
| S6 | NOT_FOUND | 0.00 |
| S7 | FALLBACK_MAX_ITERATIONS | — |

The scenario execution details, including actual status, model calls, tokens and tool observations, are recorded in:

`outputs/agent_runs.jsonl`

S1 completed with five model calls. The run log contains the detailed per-scenario execution values.

### Task 8: Safe Stop Demonstrations

Three safe stopping conditions were implemented.

| Demonstration | Configuration | Expected Status |
|---|---|---|
| Maximum iterations | Maximum six iterations | FALLBACK_MAX_ITERATIONS |
| Token limit | 200 tokens | FALLBACK_TOKEN_LIMIT |
| Timeout | 0 seconds | FALLBACK_TIMEOUT |

When a limit is reached, the agent stops safely and returns the completed steps with a human-handoff summary.

### Task 9: Checkpoint and Resume

The agent supports checkpoint saving and resuming.

After every tool observation:

- The complete message history is saved.
- Token usage is saved.
- The step number is saved.
- The list of executed tools is saved.
- The checkpoint is stored in the `state/` directory.

The `resume_agent(run_id)` function loads a checkpoint and continues the workflow.

#### Crash and Resume Demonstration

The S1 scenario was interrupted immediately after the second tool execution, after its checkpoint had been saved.

The agent was then resumed using the saved checkpoint.

**Tool execution order across the crash and resume:**

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`

#### Checkpoint Resume Results

| Metric | Result |
|---|---:|
| Total tool executions | 4 |
| Duplicate tool executions | 0 |
| Refund ledger entries | 1 |
| Final status | REFUND |
| Final refund amount | INR 128.25 |

The demonstration completed successfully without repeating previously executed tools.

### Task 10: ReAct vs Plan-and-Execute

**ReAct**

- Executes one tool at a time.
- Observes each tool result before choosing the next action.
- Handles unexpected results such as missing orders or cancelled orders dynamically.
- May require more model calls because reasoning continues after each observation.

**Plan-and-Execute**

- Creates a plan before executing the tools.
- Can reduce repeated reasoning calls when the workflow is predictable.
- May require replanning when an unexpected observation occurs.
- A fixed plan can be less flexible when order details or policy checks change the next action.

**Selected Approach: ReAct**

ReAct was selected for the refund workflow because order status, delivery time, refund eligibility and refund amount must be checked before taking the next action. Observing each tool result makes the workflow safer and more adaptable to missing orders, cancelled orders and escalation conditions.

### Part D Outputs

- `outputs/policy_chunks.json`
- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `state/` checkpoint files

**Part D Result:** Policy retrieval, grounded question answering, tool execution, safe stopping, scenario evaluation and checkpoint-resume functionality were implemented and tested.

---

## 7. Limitations and Next Steps

### Limitations

1. All datasets and policies are synthetic and fictional.
2. The project does not connect to real Instamart systems or process real refunds.
3. TF-IDF retrieval relies on word similarity and may miss semantically similar phrases.
4. Mock LLM behaviour is deterministic and does not represent the full capabilities of a production LLM.
5. The Premium customer flag uses whole-window RFM data, including September 2025. Therefore, the retention list is illustrative and is not a leakage-free evaluation.
6. The project is designed as an offline capstone implementation rather than a production deployment.

### Future Improvements

- Introduce semantic embedding-based policy retrieval.
- Improve customer lapse prediction using additional historical data.
- Add more advanced model explainability.
- Improve ticket classification with larger labelled datasets.
- Add monitoring for agent failures and tool execution.
- Integrate the system with a real customer support platform after appropriate security and validation.
- Develop a production-ready interface for customer support teams.

---

## 8. Submission

This project is submitted as one public GitHub repository.

The repository must:

- Be publicly accessible without login.
- Use a single branch named `main`.
- Contain the complete source code for Parts A–D.
- Include the supplied, unmodified `generate_data.py` and `mock_llm.py`.
- Include `requirements.txt` and `.gitignore`.
- Include the required generated data, database, SQL, model and output files.
- Include this `README.md`.

No screenshots, images, PDFs, presentations, videos or audio files are required for submission.

All project explanations, evaluation results and findings are documented in this README.

---

**Project: Instamart Customer Intelligence & Support Copilot**  
**Type: AI & Machine Learning Capstone Project**  
**Total Marks: 100**