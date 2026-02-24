import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", None)

df = pd.read_csv("olist_phase1_clean_encoded.csv")
print("Loaded:", df.shape)


# Select numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[numeric_cols].copy()

# scaling data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Running K-Means for k = 2 to 10
sse = []
sil = []
k_range = range(2, 11)
print("\nRunning KMeans...")

for k in k_range:
    km = KMeans(n_clusters=k, random_state=5805, n_init=10)
    labels = km.fit_predict(X_scaled)
    sse.append(km.inertia_)
    sil.append(silhouette_score(X_scaled, labels))
    print(f"k={k}: SSE={km.inertia_:.2f}, Sil={sil[-1]:.4f}")

# SSE plot
plt.figure(figsize=(7,4))
plt.plot(k_range, sse, marker="o")
plt.title("Elbow Plot (SSE)")
plt.xlabel("k")
plt.ylabel("SSE")
plt.grid(True)
plt.show()

# Silhouette plot
plt.figure(figsize=(7,4))
plt.plot(k_range, sil, marker="o")
plt.title("Silhouette Scores")
plt.xlabel("k")
plt.ylabel("Silhouette")
plt.grid(True)
plt.show()

# Best K determined by silhouette
best_k = k_range[np.argmax(sil)]
print("\nBest K =", best_k)

# Final KMeans
kmeans = KMeans(n_clusters=best_k, random_state=5805, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster Counts:")
print(df["kmeans_cluster"].value_counts())

# Cluster summary – only meaningful columns
summary_cols = [
    "payment_value", "price", "freight_value",
    "order_volume", "order_month", "order_weekday"
]

cluster_summary = (
    df.groupby("kmeans_cluster")[summary_cols]
    .agg(["mean", "median"])
    .round(2)
)

print("\nCluster Summary:")
print(cluster_summary)

# PCA for DBSCAN visualization
pca = PCA(n_components=2, random_state=5805)
X_pca = pca.fit_transform(X_scaled)

db = DBSCAN(eps=1.8, min_samples=25)
df["dbscan_cluster"] = db.fit_predict(X_scaled)

n_clusters = len(set(df["dbscan_cluster"])) - (1 if -1 in df["dbscan_cluster"].values else 0)
n_noise = list(df["dbscan_cluster"]).count(-1)

print("\nDBSCAN Clusters =", n_clusters)
print("Noise Points =", n_noise)

# DBSCAN clusters plot
plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df["dbscan_cluster"], cmap="tab10", s=5)
plt.title("DBSCAN Clusters (via PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# Apriori model
cat_cols = [c for c in df.columns if "product_category" in c]
basket = df[cat_cols].copy()
basket = basket.astype(bool)

# Frequent itemsets
freq = apriori(basket, min_support=0.03, use_colnames=True)
freq = freq.sort_values("support", ascending=False)

print("\nFrequent Itemsets:")
print(freq.head(10))

# Association rule mining - rules
rules = association_rules(freq, metric="lift", min_threshold=1.05)
rules = rules.sort_values("lift", ascending=False)

print("\nTop 10 Association Rules:")
print(rules.head(10))

# Saving results
df.to_csv("phase4_output_with_clusters.csv", index=False)
cluster_summary.to_csv("phase4_cluster_summary.csv")
rules.to_csv("phase4_association_rules.csv", index=False)

print("\nPhase 4 completed!")
