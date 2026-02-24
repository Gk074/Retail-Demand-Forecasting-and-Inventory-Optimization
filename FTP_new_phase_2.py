import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
pd.set_option("display.max_columns", None)

# Load full Phase I dataset
df = pd.read_csv("olist_phase1_clean_encoded_full.csv")
print("Loaded dataset:", df.shape)

# Choosing regression target and predictors
target = "order_volume"

numeric_cols = [
    "price", "freight_value", "payment_value", "review_score",
    "delivery_time_days", "order_weekday", "order_month", "order_year"
]

region_cols = [c for c in df.columns if c.startswith("customer_region_") or c.startswith("seller_region_")]

features = numeric_cols + region_cols
print("Number of regression predictors:", len(features))

X = df[features].astype(float)
y = df[target].astype(float)

# Train-test split
X_train_base, X_test_base, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=5805
)
print("Train size:", X_train_base.shape, " Test size:", X_test_base.shape)

# Custom accuracy function
def reg_accuracy(y_true, y_pred):
    y_round = np.rint(y_pred)
    exact = np.mean(y_round == y_true)
    within1 = np.mean(np.abs(y_round - y_true) <= 1)
    return exact, within1

# OLS regression
X_train_ols = sm.add_constant(X_train_base)
X_test_ols = sm.add_constant(X_test_base)

ols = sm.OLS(y_train, X_train_ols).fit()
print("\nOLS SUMMARY:")
print(ols.summary())

# OLS predictions and metrics
y_train_pred = ols.predict(X_train_ols)
y_test_pred = ols.predict(X_test_ols)

print("\nOLS Train MSE:", mean_squared_error(y_train, y_train_pred))
print("OLS Test MSE:", mean_squared_error(y_test, y_test_pred))
print("OLS Train R2:", r2_score(y_train, y_train_pred))
print("OLS Test R2:", r2_score(y_test, y_test_pred))

train_e, train_w1 = reg_accuracy(y_train, y_train_pred)
test_e, test_w1 = reg_accuracy(y_test, y_test_pred)

print("\nOLS Train Accuracy (exact):", train_e * 100)
print("OLS Test Accuracy (exact):", test_e * 100)
print("OLS Train Accuracy (±1 item):", train_w1 * 100)
print("OLS Test Accuracy (±1 item):", test_w1 * 100)

# Coefficient confidence intervals
print("\nOLS 95% CI:")
print(ols.conf_int())

# T-test and F-test
print("\nSignificant coefficients (p<0.05):")
print(ols.pvalues[ols.pvalues < 0.05])

print("\nF-statistic:", ols.fvalue)
print("F-test p-value:", ols.f_pvalue)

print("\nAIC:", ols.aic)
print("BIC:", ols.bic)

# Forward stepwise AIC selection
def stepwise_forward(X_df, y_vec):
    remaining = list(X_df.columns)
    selected = []
    current_aic = np.inf

    while remaining:
        scores = []
        for col in remaining:
            cols = selected + [col]
            X_temp = sm.add_constant(X_df[cols])
            model = sm.OLS(y_vec, X_temp).fit()
            scores.append((model.aic, col))
        scores.sort()
        best_aic, best_col = scores[0]

        if best_aic < current_aic:
            selected.append(best_col)
            remaining.remove(best_col)
            current_aic = best_aic
            print("added:", best_col, "AIC:", best_aic)
        else:
            break
    return selected

print("\nRunning stepwise regression...")
step_cols = stepwise_forward(X_train_base, y_train)
print("\nSelected stepwise features:", step_cols)

# Final stepwise model
X_train_sw = sm.add_constant(X_train_base[step_cols])
X_test_sw = sm.add_constant(X_test_base[step_cols])

sw_model = sm.OLS(y_train, X_train_sw).fit()
print("\nSTEPWISE SUMMARY:")
print(sw_model.summary())

y_train_sw = sw_model.predict(X_train_sw)
y_test_sw = sw_model.predict(X_test_sw)

sw_te = mean_squared_error(y_test, y_test_sw)
sw_tr = mean_squared_error(y_train, y_train_sw)

print("\nStepwise Train MSE:", sw_tr)
print("Stepwise Test MSE:", sw_te)
print("Stepwise Train R2:", r2_score(y_train, y_train_sw))
print("Stepwise Test R2:", r2_score(y_test, y_test_sw))

sw_e_tr, sw_w1_tr = reg_accuracy(y_train, y_train_sw)
sw_e_te, sw_w1_te = reg_accuracy(y_test, y_test_sw)

print("\nStepwise Train Accuracy (exact):", sw_e_tr * 100)
print("Stepwise Test Accuracy (exact):", sw_e_te * 100)
print("Stepwise Train Accuracy (±1 item):", sw_w1_tr * 100)
print("Stepwise Test Accuracy (±1 item):", sw_w1_te * 100)

# Sklearn regression models
models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=5805, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=5805),
    "KNN": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=15))]),
    "SVR_RBF": Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.5))])
}

results = []

for name, model in models.items():
    model.fit(X_train_base, y_train)

    yp_tr = model.predict(X_train_base)
    yp_te = model.predict(X_test_base)

    mse_tr = mean_squared_error(y_train, yp_tr)
    mse_te = mean_squared_error(y_test, yp_te)

    r2_tr = r2_score(y_train, yp_tr)
    r2_te = r2_score(y_test, yp_te)

    acc_e_tr, acc_w1_tr = reg_accuracy(y_train, yp_tr)
    acc_e_te, acc_w1_te = reg_accuracy(y_test, yp_te)

    results.append([name, mse_tr, mse_te, r2_tr, r2_te, acc_e_tr, acc_e_te, acc_w1_tr, acc_w1_te])

results_df = pd.DataFrame(results, columns=[
    "Model", "Train_MSE", "Test_MSE", "Train_R2", "Test_R2",
    "Train_Acc_exact", "Test_Acc_exact", "Train_Acc±1", "Test_Acc±1"
])

print("\nMODEL COMPARISON: ")
print(results_df)

# Plot actual vs predicted for OLS
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:300], label="Actual", alpha=0.7)
plt.plot(y_test_pred.values[:300], label="OLS Predicted", alpha=0.7)
plt.title("OLS Prediction vs Actual (first 300 test samples)")
plt.xlabel("Sample index")
plt.ylabel("Order Volume")
plt.legend()
plt.show()

print("\nPhase 2 completed!")
