
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Paths and fixed snapshot date
CLEAN = Path("data/clean")
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(parents=True, exist_ok=True)

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")

# 1. Load cleaned data
customers = pd.read_csv(CLEAN / "customers.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
orders["order_amount"] = pd.to_numeric(orders["order_amount"])

# Use completed orders only
completed = orders[orders["status"] == "completed"].copy()

# 2. Build RFM: Recency, Frequency, Monetary
rfm = (
    completed.groupby("customer_id")
    .agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)

rfm["recency"] = (SNAPSHOT - rfm["last_order"]).dt.days

rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
].copy()

print("\nRFM CHECKPOINTS")
print("Customers:", len(rfm))
print("Recency median:", rfm["recency"].median())
print("Recency min:", rfm["recency"].min())
print("Recency max:", rfm["recency"].max())
print("Frequency mean:", round(rfm["frequency"].mean(), 2))
print("Frequency max:", rfm["frequency"].max())
print("Monetary mean:", round(rfm["monetary"].mean(), 2))
print("Monetary max:", round(rfm["monetary"].max(), 2))

# 3. Scale the RFM features
features = ["recency", "frequency", "monetary"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[features])

# 4. Evaluate K-means for K = 2 to 8
print("\nK-MEANS COMPARISON")
results = []

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )
    labels = model.fit_predict(X_scaled)

    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    results.append({
        "K": k,
        "inertia": inertia,
        "silhouette": silhouette
    })

results_df = pd.DataFrame(results)
print(results_df.round(3).to_string(index=False))

# The course permits choosing K from {3, 4, 5}.
# Choose K=4 for interpretable business segments.
FINAL_K = 4

final_model = KMeans(
    n_clusters=FINAL_K,
    n_init=10,
    random_state=42
)
rfm["segment_id"] = final_model.fit_predict(X_scaled)

# 5. Profile the segments
profile = (
    rfm.groupby("segment_id")
    .agg(
        customers=("customer_id", "size"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    )
)

print("\nSEGMENT PROFILE")
print(profile.round(2).to_string())

# Highest mean monetary segment is Premium.
premium_id = profile["mean_monetary"].idxmax()

def name_segment(segment_id):
    row = profile.loc[segment_id]

    if segment_id == premium_id:
        return "Premium"

    if (
        row["mean_frequency"] >= profile["mean_frequency"].median()
        and row["mean_recency"] <= profile["mean_recency"].median()
    ):
        return "Frequent Loyal"

    if row["mean_recency"] > profile["mean_recency"].median():
        if row["mean_recency"] > 60:
            return "At Risk"
    return "Needs Attention"

    return "Regular"

rfm["segment_name"] = rfm["segment_id"].map(
    lambda sid: name_segment(sid)
)
rfm["is_premium"] = rfm["segment_id"].eq(premium_id)

print("\nSEGMENT NAMES")
print(
    rfm.groupby(["segment_id", "segment_name"])
    .size()
    .to_string()
)
print("Premium segment ID:", premium_id)

# 6. PCA: two components from scaled RFM
pca = PCA(n_components=2)
pca.fit(X_scaled)

print("\nPCA EXPLAINED VARIANCE RATIO")
print(pca.explained_variance_ratio_)
print("Total:", pca.explained_variance_ratio_.sum())

# 7. Save the required segment file
output_columns = [
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "segment_id",
    "segment_name",
    "is_premium",
]

rfm[output_columns].to_csv(
    OUTPUTS / "customer_segments.csv",
    index=False
)

print("\nSaved outputs/customer_segments.csv")
print("PART B SEGMENTATION COMPLETED")
