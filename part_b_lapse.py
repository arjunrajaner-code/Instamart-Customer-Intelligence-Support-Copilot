
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

warnings.filterwarnings("ignore", category=FutureWarning)

CLEAN = Path("data/clean")
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2025-09-01")
SEPT_END = pd.Timestamp("2025-10-01")


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

customers = pd.read_csv(CLEAN / "customers.csv")
orders = pd.read_csv(CLEAN / "orders.csv")
products = pd.read_csv(CLEAN / "products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

orders = orders.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

completed = orders[
    orders["status"] == "completed"
].copy()


# --------------------------------------------------
# 2. CREATE LAPSE LABEL
# --------------------------------------------------

before = completed[
    completed["order_ts"] < CUTOFF
]

september = completed[
    (completed["order_ts"] >= CUTOFF)
    & (completed["order_ts"] < SEPT_END)
]

cohort_ids = before["customer_id"].unique()

lapsed_ids = (
    set(cohort_ids) - set(september["customer_id"])
)

labels = pd.DataFrame({
    "customer_id": cohort_ids
})

labels["lapsed"] = (
    labels["customer_id"]
    .isin(lapsed_ids)
    .astype(int)
)

print("=" * 55)
print("COHORT CHECKPOINTS")
print("=" * 55)

print("Cohort customers:", len(labels))
print("Lapsed customers:", int(labels["lapsed"].sum()))
print(
    "Lapse rate:",
    round(labels["lapsed"].mean() * 100, 2),
    "%"
)

assert len(labels) == 1156, "Unexpected cohort size"
assert int(labels["lapsed"].sum()) == 443, "Unexpected lapsed count"


# --------------------------------------------------
# 3. BUILD FEATURES USING PRE-CUTOFF DATA ONLY
# --------------------------------------------------

pre_orders = orders[
    orders["order_ts"] < CUTOFF
].copy()

pre_completed = pre_orders[
    pre_orders["status"] == "completed"
].copy()

rfm = (
    pre_completed.groupby("customer_id")
    .agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_delivery=("delivery_minutes", "mean"),
        favourite_category=(
            "category",
            lambda x: x.mode().iloc[0]
        )
    )
    .reset_index()
)

rfm["recency"] = (
    CUTOFF - rfm["last_order"]
).dt.days

rfm = rfm.drop(columns="last_order")


# Cancellation rate:
# Cancelled orders / all final-status orders before cutoff.

final_orders = pre_orders[
    pre_orders["status"].isin(["completed", "cancelled"])
]

total_final = (
    final_orders.groupby("customer_id").size()
)

cancelled_count = (
    final_orders[
        final_orders["status"] == "cancelled"
    ]
    .groupby("customer_id")
    .size()
)

cancel_rate = (
    cancelled_count
    .reindex(total_final.index, fill_value=0)
    / total_final
).rename("cancellation_rate").reset_index()


# Merge all features.

features = (
    labels[["customer_id"]]
    .merge(
        rfm,
        on="customer_id",
        how="left",
        validate="one_to_one"
    )
    .merge(
        cancel_rate,
        on="customer_id",
        how="left"
    )
    .merge(
        customers[
            [
                "customer_id",
                "city",
                "age",
                "gender",
                "signup_date"
            ]
        ],
        on="customer_id",
        how="left",
        validate="one_to_one"
    )
)

features["tenure_days"] = (
    CUTOFF - features["signup_date"]
).dt.days

features = features.drop(columns="signup_date")


# Feature columns.

numeric_cols = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery",
    "age",
    "tenure_days"
]

categorical_cols = [
    "favourite_category",
    "city",
    "gender"
]


# Confirm no missing feature values.

if features[numeric_cols + categorical_cols].isna().any().any():

    missing = (
        features[numeric_cols + categorical_cols]
        .isna()
        .sum()
    )

    raise ValueError(
        f"Missing feature values:\n{missing[missing > 0]}"
    )

X = features[numeric_cols + categorical_cols]
y = labels["lapsed"]

assert not X.isna().any().any(), "Missing values found in features"

print("\nFEATURE CHECKPOINTS")
print("Feature rows:", len(X))
print("Feature columns:", len(X.columns))
print("Missing feature values:", int(X.isna().sum().sum()))


# --------------------------------------------------
# 4. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test, id_train, id_test = (
    train_test_split(
        X,
        y,
        features["customer_id"],
        test_size=0.2,
        stratify=y,
        random_state=42
    )
)

print("\n" + "=" * 55)
print("SPLIT CHECKPOINTS")
print("=" * 55)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

baseline_predictions = np.zeros(len(y_test), dtype=int)

baseline_accuracy = (
    y_test == 0
).mean()

baseline_f1 = f1_score(
    y_test,
    baseline_predictions,
    zero_division=0
)

print("Baseline accuracy:", round(baseline_accuracy, 3))
print("Baseline lapsed-class F1:", round(baseline_f1, 3))

assert len(X_train) == 924, "Unexpected training size"
assert len(X_test) == 232, "Unexpected test size"


# --------------------------------------------------
# 5. PREPROCESSING WITHOUT DATA LEAKAGE
# --------------------------------------------------

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "encoder",
        OneHotEncoder(
            drop="first",
            handle_unknown="ignore"
        )
    )
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols)
])


# --------------------------------------------------
# 6. LOGISTIC REGRESSION GRID SEARCH
# --------------------------------------------------

lr_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

lr_search = GridSearchCV(
    lr_pipeline,
    {
        "model__C": [0.01, 0.1, 1, 10, 100]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("\nTraining Logistic Regression...")

lr_search.fit(X_train, y_train)


# --------------------------------------------------
# 7. RANDOM FOREST GRID SEARCH
# --------------------------------------------------

rf_pipeline = Pipeline([
    ("preprocess", preprocessor),
    (
        "model",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )
])

rf_search = GridSearchCV(
    rf_pipeline,
    {
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("Training Random Forest...")

rf_search.fit(X_train, y_train)


# --------------------------------------------------
# 8. EVALUATION FUNCTION
# --------------------------------------------------

def evaluate(name, model, X_eval, y_eval):

    predictions = model.predict(X_eval)

    precision = precision_score(
        y_eval,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_eval,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_eval,
        predictions,
        zero_division=0
    )

    print(f"\n{name}")
    print(
        "Confusion matrix:\n",
        confusion_matrix(y_eval, predictions)
    )
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1:", round(f1, 3))

    return predictions


# Logistic Regression results.

print("\n" + "=" * 55)
print("LOGISTIC REGRESSION")
print("=" * 55)

print("Best parameters:", lr_search.best_params_)
print("Best CV F1:", round(lr_search.best_score_, 3))

lr_predictions = evaluate(
    "Logistic Regression Test",
    lr_search.best_estimator_,
    X_test,
    y_test
)


# Random Forest results.

print("\n" + "=" * 55)
print("RANDOM FOREST")
print("=" * 55)

print("Best parameters:", rf_search.best_params_)
print("Best CV F1:", round(rf_search.best_score_, 3))

rf_predictions = evaluate(
    "Random Forest Test",
    rf_search.best_estimator_,
    X_test,
    y_test
)


# --------------------------------------------------
# 9. TRAINING SCALER VS FULL COHORT MEAN
# --------------------------------------------------

fitted_scaler = (
    lr_search.best_estimator_
    .named_steps["preprocess"]
    .named_transformers_["numeric"]
    .named_steps["scaler"]
)

frequency_index = numeric_cols.index("frequency")

training_frequency_mean = (
    fitted_scaler.mean_[frequency_index]
)

full_cohort_frequency_mean = X["frequency"].mean()

print("\n" + "=" * 55)
print("FREQUENCY MEAN COMPARISON")
print("=" * 55)

print(
    "Training scaler frequency mean:",
    round(training_frequency_mean, 2)
)

print(
    "Full cohort frequency mean:",
    round(full_cohort_frequency_mean, 2)
)

print(
    "The means differ because the scaler is fitted only "
    "on training rows, while the full-cohort mean includes "
    "both training and test rows."
)


# --------------------------------------------------
# 10. CLASS-BALANCE EXPERIMENT
# --------------------------------------------------

train_frame = X_train.reset_index(drop=True).copy()

train_frame["target"] = (
    y_train.reset_index(drop=True)
)

majority_count = (
    train_frame["target"].value_counts().max()
)

minority = train_frame[
    train_frame["target"] == 1
]

extra = minority.sample(
    n=majority_count - len(minority),
    replace=True,
    random_state=42
)

balanced = pd.concat(
    [train_frame, extra],
    ignore_index=True
)

balanced = (
    balanced.sample(
        frac=1,
        random_state=42
    )
    .reset_index(drop=True)
)

balanced_lr = clone(
    lr_search.best_estimator_
)

balanced_lr.fit(
    balanced.drop(columns="target"),
    balanced["target"]
)

balanced_predictions = evaluate(
    "Balanced Logistic Regression Test",
    balanced_lr,
    X_test,
    y_test
)

print("\n" + "=" * 55)
print("CLASS BALANCE EXPERIMENT")
print("=" * 55)

print(
    "Original training class counts:",
    y_train.value_counts().to_dict()
)

print(
    "Balanced training class counts:",
    balanced["target"].value_counts().to_dict()
)


# Compare original and balanced metrics.

original_precision = precision_score(
    y_test,
    lr_predictions,
    zero_division=0
)

original_recall = recall_score(
    y_test,
    lr_predictions,
    zero_division=0
)

original_f1 = f1_score(
    y_test,
    lr_predictions,
    zero_division=0
)

balanced_precision = precision_score(
    y_test,
    balanced_predictions,
    zero_division=0
)

balanced_recall = recall_score(
    y_test,
    balanced_predictions,
    zero_division=0
)

balanced_f1 = f1_score(
    y_test,
    balanced_predictions,
    zero_division=0
)

comparison = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1"],
    "Original": [
        original_precision,
        original_recall,
        original_f1
    ],
    "Balanced": [
        balanced_precision,
        balanced_recall,
        balanced_f1
    ]
})

comparison["Change"] = (
    comparison["Balanced"] - comparison["Original"]
)

print("\nCLASS BALANCE METRIC COMPARISON")
print(comparison.round(3).to_string(index=False))

largest_change = comparison.loc[
    comparison["Change"].abs().idxmax()
]

print(
    "\nMetric with largest absolute change:",
    largest_change["Metric"]
)


# --------------------------------------------------
# 11. FEATURE IMPORTANCE
# --------------------------------------------------

best_rf = rf_search.best_estimator_

feature_names = (
    best_rf.named_steps["preprocess"]
    .get_feature_names_out()
)

importances = (
    best_rf.named_steps["model"]
    .feature_importances_
)

importance = (
    pd.Series(importances, index=feature_names)
    .sort_values(ascending=False)
)

print("\n" + "=" * 55)
print("TOP FIVE RANDOM FOREST FEATURES")
print("=" * 55)

print(importance.head(5))


# Logistic Regression coefficients.

best_lr = lr_search.best_estimator_

lr_names = (
    best_lr.named_steps["preprocess"]
    .get_feature_names_out()
)

coefficients = (
    best_lr.named_steps["model"].coef_[0]
)

coefficient_table = pd.DataFrame({
    "Feature": lr_names,
    "Coefficient": coefficients
})

coefficient_table["Direction"] = np.where(
    coefficient_table["Coefficient"] > 0,
    "Positive",
    "Negative"
)

coefficient_table["Absolute Value"] = (
    coefficient_table["Coefficient"].abs()
)

coefficient_table = (
    coefficient_table
    .sort_values("Absolute Value", ascending=False)
    .head(3)
)

print("\nTOP THREE LOGISTIC REGRESSION COEFFICIENTS")
print(
    coefficient_table[
        ["Feature", "Coefficient", "Direction"]
    ].round(4).to_string(index=False)
)


# --------------------------------------------------
# 12. SELECT AND SAVE BEST MODEL
# --------------------------------------------------

lr_test_f1 = f1_score(
    y_test,
    lr_predictions,
    zero_division=0
)

rf_test_f1 = f1_score(
    y_test,
    rf_predictions,
    zero_division=0
)

# Select model using Cross-Validation F1, not Test F1.
if rf_search.best_score_ > lr_search.best_score_:
    selected_model = rf_search.best_estimator_
    selected_model_name = "Random Forest"
    selected_f1 = rf_search.best_score_
else:
    selected_model = lr_search.best_estimator_
    selected_model_name = "Logistic Regression"
    selected_f1 = lr_search.best_score_

if selected_model_name == "Random Forest":
    selected_test_f1 = rf_test_f1
else:
    selected_test_f1 = lr_test_f1    

print("\n" + "=" * 55)
print("SELECTED MODEL")
print("=" * 55)

print("Selected model:", selected_model_name)
print("Selected CV F1:", round(selected_f1, 3))
print("Selected test F1:", round(selected_test_f1, 3))

if selected_test_f1 < 0.55:
    print(
        "WARNING: Selected model is below the PDF "
        "minimum test F1 of 0.55."
    )
else:
    print("Selected model meets the minimum test F1 requirement.")


# Save model.

model_path = OUTPUTS / "lapse_model.joblib"

joblib.dump(selected_model, model_path)

print("Model saved:", model_path)


# Reload model and verify predictions.

reloaded = joblib.load(model_path)

original_predictions = selected_model.predict(X_test)
reloaded_predictions = reloaded.predict(X_test)

assert np.array_equal(
    original_predictions,
    reloaded_predictions
), "Reloaded model predictions do not match"

print(
    "Reloaded model reproduces test predictions exactly."
)


# --------------------------------------------------
# 13. PREMIUM CUSTOMER RETENTION LIST
# --------------------------------------------------

segments = pd.read_csv(
    OUTPUTS / "customer_segments.csv"
)

premium_ids = set(
    segments.loc[
        segments["is_premium"].astype(str).str.lower().isin(
            ["true", "1"]
        ),
        "customer_id"
    ]
)

probabilities = (
    selected_model.predict_proba(X_test)[:, 1]
)

retention = pd.DataFrame({
    "customer_id": id_test.to_numpy(),
    "lapse_probability": probabilities,
    "actually_lapsed": y_test.to_numpy()
})

retention = (
    retention[
        retention["customer_id"].isin(premium_ids)
    ]
    .sort_values(
        "lapse_probability",
        ascending=False
    )
    .head(10)
)

retention.to_csv(
    OUTPUTS / "retention_list.csv",
    index=False
)

print("\n" + "=" * 55)
print("PREMIUM RETENTION LIST")
print("=" * 55)

print(retention.to_string(index=False))

print(
    "Actually lapsed in retention list:",
    int(retention["actually_lapsed"].sum())
)

print(
    "Retention list rows:",
    len(retention)
)

assert list(retention.columns) == [
    "customer_id",
    "lapse_probability",
    "actually_lapsed"
]


# --------------------------------------------------
# 14. FINAL VALIDATION
# --------------------------------------------------

print("\n" + "=" * 55)
print("PART B FINAL VALIDATION")
print("=" * 55)

print("Cohort size:", len(labels))
print("Lapsed customers:", int(y.sum()))
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Selected model:", selected_model_name)
print("Selected test F1:", round(selected_f1, 3))

print(
    "Customer segments file exists:",
    (OUTPUTS / "customer_segments.csv").exists()
)

print(
    "Lapse model exists:",
    model_path.exists()
)

print(
    "Retention list exists:",
    (OUTPUTS / "retention_list.csv").exists()
)

print(
    "\nLIMITATION: The Premium flag uses whole-window RFM, "
    "including September. Therefore, the retention list is "
    "illustrative and is not a leakage-free evaluation."
)

print("\nPART B LAPSE PREDICTION COMPLETED")
