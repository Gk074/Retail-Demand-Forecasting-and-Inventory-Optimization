import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# base raw URL for your repo
base_url = "https://raw.githubusercontent.com/Gk074/Machine-Learning-1-Final-Term-Project/refs/heads/main/"

files = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "category_translation": "product_category_name_translation.csv"
}

dfs = {}
for name, fname in files.items():
    url = base_url + fname
    df_temp = pd.read_csv(url)
    dfs[name] = df_temp
    print(f"{name:20s} -> shape: {df_temp.shape}")

orders = dfs["orders"]
order_items = dfs["order_items"]
customers = dfs["customers"]
sellers = dfs["sellers"]
products = dfs["products"]
geolocation = dfs["geolocation"]
payments = dfs["payments"]
reviews = dfs["reviews"]
cat_trans = dfs["category_translation"]

print("\nOrders columns:\n", orders.columns)
print("\nOrder items columns:\n", order_items.columns)
print("\nCustomers columns:\n", customers.columns)
print("\nSellers columns:\n", sellers.columns)
print("\nProducts columns:\n", products.columns)
print("\nPayments columns:\n", payments.columns)
print("\nReviews columns:\n", reviews.columns)
print("\nCategory translation columns:\n", cat_trans.columns)

# group by zip code prefix and take mean lat/lng
geo_agg = (
    geolocation
    .groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
    .mean()
    .reset_index()
)

print("Aggregated geolocation shape:", geo_agg.shape)
print(geo_agg.head())

# merge customer zips with aggregated geolocation
customers_geo = customers.merge(
    geo_agg,
    left_on="customer_zip_code_prefix",
    right_on="geolocation_zip_code_prefix",
    how="left"
)

# rename columns so they are clear after full merge
customers_geo = customers_geo.rename(columns={
    "geolocation_lat": "cust_lat",
    "geolocation_lng": "cust_lng"
}).drop(columns=["geolocation_zip_code_prefix"])

print("\ncustomers_geo columns:\n", customers_geo.columns)

# same idea for sellers
sellers_geo = sellers.merge(
    geo_agg,
    left_on="seller_zip_code_prefix",
    right_on="geolocation_zip_code_prefix",
    how="left"
)

sellers_geo = sellers_geo.rename(columns={
    "geolocation_lat": "seller_lat",
    "geolocation_lng": "seller_lng"
}).drop(columns=["geolocation_zip_code_prefix"])

print("\nsellers_geo columns:\n", sellers_geo.columns)

products_full = products.merge(
    cat_trans,
    on="product_category_name",
    how="left"
)

print("\nproducts_full columns:\n", products_full.columns[:10])

# merge orders with customers (each order has one customer)
orders_customers = orders.merge(
    customers_geo,
    on="customer_id",
    how="left"
)

print("orders_customers shape:", orders_customers.shape)

# merge order_items with products and sellers
items_prod = order_items.merge(
    products_full,
    on="product_id",
    how="left"
)

items_prod_sellers = items_prod.merge(
    sellers_geo,
    on="seller_id",
    how="left"
)

print("items_prod_sellers shape:", items_prod_sellers.shape)

# bring in payments and reviews (one row per payment/review per order)
orders_pay = payments.merge(
    orders_customers,
    on="order_id",
    how="right"   # keep all orders even if payment info is missing
)

orders_pay_rev = orders_pay.merge(
    reviews,
    on="order_id",
    how="left"
)

print("orders_pay_rev shape:", orders_pay_rev.shape)

# finally, join the per-order info with per-item info
df_full = items_prod_sellers.merge(
    orders_pay_rev,
    on="order_id",
    how="left"
)

print("\nFinal merged df_full shape:", df_full.shape)
print(df_full.head())

print("\n # df_full.info:")
print(df_full.info())

print("\n # Missing values (top 30 columns):")
print(df_full.isna().sum().sort_values(ascending=False).head(30))

# Convert timestamps to datetime
time_cols = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for col in time_cols:
    df_full[col] = pd.to_datetime(df_full[col], errors="coerce")

# Keep delivered orders with required fields
required_cols = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "price",
    "freight_value",
    "payment_value",
    "review_score"
]

df_del = df_full[df_full["order_status"] == "delivered"].copy()
df_del = df_del.dropna(subset=required_cols)

print("After filtering delivered + required fields:", df_del.shape)

# Regression target: delivery_time_days
df_del["delivery_time_days"] = (
    df_del["order_delivered_customer_date"] - df_del["order_purchase_timestamp"]
).dt.days

# Binary lateness flag
df_del["is_late_delivery"] = (
    df_del["order_delivered_customer_date"] > df_del["order_estimated_delivery_date"]
).astype(int)

# Time features
df_del["order_weekday"] = df_del["order_purchase_timestamp"].dt.weekday
df_del["order_month"] = df_del["order_purchase_timestamp"].dt.month
df_del["order_year"] = df_del["order_purchase_timestamp"].dt.year

# Order volume per order
order_volume = (
    dfs["order_items"]
    .groupby("order_id")["order_item_id"]
    .count()
    .reset_index()
    .rename(columns={"order_item_id": "order_volume"})
)

df_del = df_del.merge(order_volume, on="order_id", how="left")

# Volume buckets for association rules
def volume_bucket(v):
    if v == 1:
        return "low"
    elif v <= 3:
        return "medium"
    else:
        return "high"

df_del["order_volume_category"] = df_del["order_volume"].apply(volume_bucket)

# One row per order (keep highest priced item)
df_del = df_del.sort_values(["order_id", "price"], ascending=[True, False])
df_order = df_del.drop_duplicates(subset="order_id").copy()

print("After collapsing to 1 row per order:", df_order.shape)

# Build modeling dataframe with candidate columns
df_order["product_category_name_english"] = df_order["product_category_name_english"].fillna("unknown")

df_model = df_order[[
    "order_id",
    "product_category_name_english",
    "seller_state",
    "customer_state",
    "price",
    "freight_value",
    "payment_value",
    "review_score",          # Phase 3 target
    "delivery_time_days",    # Phase 2 target
    "order_weekday",
    "order_month",
    "order_year",
    "order_volume",
    "order_volume_category", # Phase 4 target
    "is_late_delivery"
]].copy()

df_model = df_model.rename(columns={
    "product_category_name_english": "product_category",
    "seller_state": "seller_region",
    "customer_state": "customer_region"
})

print("Candidate modeling set shape:", df_model.shape)
print(df_model.head())

# Handle missing values and duplicates
df_model["product_category"] = df_model["product_category"].fillna("unknown")
df_model["seller_region"] = df_model["seller_region"].fillna("unknown")
df_model["customer_region"] = df_model["customer_region"].fillna("unknown")

numeric_cols_raw = df_model.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols_raw:
    df_model[col] = df_model[col].fillna(df_model[col].median())

dup_count = df_model.duplicated().sum()
print("Duplicate rows in df_model:", dup_count)
if dup_count > 0:
    df_model = df_model.drop_duplicates()
    print("After drop_duplicates:", df_model.shape)

print("\nMissing values after cleaning:")
print(df_model.isna().sum())

# One-hot encode categoricals (avoid dummy trap)
df_ml = df_model.copy()

cat_cols = ["product_category", "seller_region", "customer_region", "order_volume_category"]

df_ml = pd.get_dummies(df_ml, columns=cat_cols, drop_first=True)

# Build feature matrix (X) for DR and outlier detection
feature_cols = [c for c in df_ml.columns if c not in [
    "order_id", "delivery_time_days", "review_score",
    "order_volume_category", "is_late_delivery"
]]

X = df_ml[feature_cols].values

# Standardize features (needed for PCA, LDA, IsolationForest)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Encoded + scaled feature matrix shape:", X_scaled.shape)

# Outlier detection using Isolation Forest
iso = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=5805
)

outlier_flags = iso.fit_predict(X_scaled)   # -1 = outlier, 1 = normal
mask_inliers = (outlier_flags == 1)

print("Outliers removed by IsolationForest:", (~mask_inliers).sum())

df_ml_clean = df_ml.loc[mask_inliers].reset_index(drop=True)
print("Shape after outlier removal:", df_ml_clean.shape)

X_clean = df_ml_clean[feature_cols].values
X_clean_scaled = scaler.fit_transform(X_clean)

# keep original-type columns (no dummies) for interpretation
df_clean_original = df_model.loc[mask_inliers].reset_index(drop=True)

# Covariance and correlation (numeric variables)
num_cols = [
    "price", "freight_value", "payment_value",
    "review_score", "delivery_time_days",
    "order_weekday", "order_month", "order_year",
    "order_volume", "is_late_delivery"
]

cov_matrix = df_clean_original[num_cols].cov()
corr_matrix = df_clean_original[num_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, cmap="coolwarm")
plt.title("Sample Covariance Matrix (Numeric Features)")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Sample Pearson Correlation Matrix (Numeric Features)")
plt.show()

print("\nCorrelation with delivery_time_days:")
print(corr_matrix["delivery_time_days"].sort_values(ascending=False))

# Random Forest feature importance (regression target)
y_reg = df_ml_clean["delivery_time_days"].values

rf_reg = RandomForestRegressor(
    n_estimators=200,
    random_state=5805
)
rf_reg.fit(X_clean, y_reg)

rf_importances = pd.Series(
    rf_reg.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("\nRandom Forest Feature Importance (delivery_time_days):")
print(rf_importances.head(20))

plt.figure(figsize=(10, 5))
rf_importances.head(15).plot(kind="bar")
plt.title("Top RF Feature Importances – delivery_time_days")
plt.tight_layout()
plt.show()

# PCA and condition number
pca = PCA(n_components=0.95, random_state=5805)
pca.fit(X_clean_scaled)

print("\nPCA summary:")
print("Components for 95% variance:", pca.n_components_)
print("Explained variance ratios:", pca.explained_variance_ratio_)

sing_vals = pca.singular_values_
cond_number = sing_vals.max() / sing_vals.min()
print("Approximate condition number from PCA singular values:", cond_number)

plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_, marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Explained Variance")
plt.tight_layout()
plt.show()

# SVD and condition number
u, s, vh = np.linalg.svd(X_clean_scaled, full_matrices=False)
svd_cond = s.max() / s.min()
print("\nSVD summary:")
print("First 10 singular values:", s[:10])
print("Condition number from SVD:", svd_cond)

# VIF on selected numeric features
vif_features = df_clean_original[[
    "price", "freight_value", "payment_value",
    "delivery_time_days", "order_volume"
]].copy()

vif_features = sm.add_constant(vif_features)

vif_data = pd.DataFrame()
vif_data["feature"] = vif_features.columns
vif_data["VIF"] = [
    variance_inflation_factor(vif_features.values, i)
    for i in range(vif_features.shape[1])
]

print("\nVIF (numeric subset):")
print(vif_data)

# LDA for classification target (review_score)
y_cls = df_ml_clean["review_score"].astype(int).values

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_clean_scaled, y_cls)

print("\nLDA explained variance ratio:", lda.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1],
                      c=y_cls, cmap="viridis", s=5, alpha=0.6)
plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")
plt.title("LDA Projection (review_score)")
plt.colorbar(scatter, label="review_score")
plt.tight_layout()
plt.show()

# Class balance checks
print("\nReview_score distribution (cleaned):")
print(df_ml_clean["review_score"].value_counts(normalize=True).sort_index())

print("\nLate_delivery distribution (cleaned):")
print(df_ml_clean["is_late_delivery"].value_counts(normalize=True))

# Final dataset for later phases and downsampling
# save the full dataset before any downsampling
df_ml_clean.to_csv("olist_phase1_clean_encoded_full.csv", index=False)
print("Saved full encoded dataset:", df_ml_clean.shape)

# create a copy for downsampling
df_phase1_final = df_ml_clean.copy()

# downsample to exactly 60k rows
max_rows = 60000
if len(df_phase1_final) > max_rows:
    df_phase1_final = (
        df_phase1_final
        .sample(n=max_rows, random_state=5805)
        .reset_index(drop=True)
    )
    print("Downsampled df_phase1_final to:", df_phase1_final.shape)
else:
    print("No downsampling needed. df_phase1_final size:", df_phase1_final.shape)

# save the 60k dataset
df_phase1_final.to_csv("olist_phase1_clean_encoded.csv", index=False)
print("Saved 60k dataset as olist_phase1_clean_encoded.csv")

