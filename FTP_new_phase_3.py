import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, accuracy_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", None)

print("new code running")
pd.set_option("display.max_columns", None)
import warnings; warnings.filterwarnings("ignore")

# gets the encoded data
df = pd.read_csv("olist_phase1_clean_encoded.csv")
print("Loaded dataset:", df.shape)


# Building 3 class category
def map_review_to_class(x):
    if x <= 2: return 0
    if x == 3: return 1
    return 2

df = df[df["review_score"].notna()].copy()
df["review_3class"] = df["review_score"].apply(map_review_to_class)

print("\nClass distribution (0,1,2):")
print(df["review_3class"].value_counts(normalize=True).sort_index())


# Removing region dummy fearures
region_cols = [c for c in df.columns
               if c.startswith("customer_region_") or c.startswith("seller_region_")]

df = df.drop(columns=region_cols)
print("\nRemoved region dummy columns:", len(region_cols))
print("Shape after region drop:", df.shape)


# Feature matrix selection
drop_cols = ["order_id", "review_score", "review_3class",
             "delivery_time_days", "order_volume_category", "is_late_delivery"]

feature_cols = [c for c in df.columns if c not in drop_cols]

X_full = df[feature_cols].astype(float)
y = df["review_3class"].astype(int)

print("\nCandidate features:", X_full.shape[1])


# Train and Test split
X_train_full, X_test_full, y_train_full, y_test = train_test_split(
    X_full, y, test_size=0.30, random_state=5805, stratify=y
)
print("Train shape:", X_train_full.shape, " Test shape:", X_test_full.shape)


# Downsampling
train_df = X_train_full.copy()
train_df["target"] = y_train_full.values

major = train_df[train_df["target"] == 2]
minor = train_df[train_df["target"] != 2]

target_n = min(len(major), 2 * len(minor))
major_down = resample(major, replace=False, n_samples=target_n, random_state=5805)

train_bal = pd.concat([minor, major_down]).sample(frac=1, random_state=5805)

X_train_full = train_bal.drop(columns=["target"])
y_train_full = train_bal["target"]

print("\nBalanced training shape:", X_train_full.shape)
print("Balanced class distribution:")
print(y_train_full.value_counts(normalize=True))


# Feature selection
rf_fs = RandomForestClassifier(n_estimators=120, random_state=5805, n_jobs=-1)
rf_fs.fit(X_train_full, y_train_full)

importances = pd.Series(rf_fs.feature_importances_, index=X_train_full.columns)
top_features = importances.sort_values(ascending=False).head(25).index.tolist()

print("\nSelected top 25 features:")
for f in top_features: print("  -", f)

X_train = X_train_full[top_features]
X_test = X_test_full[top_features]


# micro specificity
def macro_specificity(cm):
    spec = []
    total = cm.sum()
    for i in range(3):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - (tp + fn + fp)
        spec.append(tn / (tn + fp) if tn + fp else 0)
    return np.mean(spec)


# Model evaluation
def evaluate_model(name, model):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    acc_tr = accuracy_score(y_train_full, y_pred_train)
    acc_te = accuracy_score(y_test, y_pred_test)

    cm = confusion_matrix(y_test, y_pred_test, labels=[0,1,2])
    prec = precision_score(y_test, y_pred_test, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred_test, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
    spec = macro_specificity(cm)

    # ROC AUC (class 2 vs others)
    scores = None
    auc = np.nan
    try:
        proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else model.decision_function(X_test)
        idx = list(model.classes_).index(2)
        scores = proba[:, idx]
        auc = roc_auc_score((y_test == 2).astype(int), scores)
    except: pass

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
    cv_acc = cross_val_score(model, X_train, y_train_full, cv=cv, scoring="accuracy").mean()

    print(f"\n{name}:")
    print("Train Acc:", round(acc_tr*100,2))
    print("Test Acc :", round(acc_te*100,2))
    print("Precision:", round(prec,3))
    print("Recall   :", round(rec,3))
    print("Specificity:", round(spec,3))
    print("F1 Score :", round(f1,3))
    print("ROC AUC  :", round(auc,3))
    print("CV Acc   :", round(cv_acc*100,2))
    print("Confusion:\n", cm)

    return { "Scores": scores }


# Models
scaler = StandardScaler()
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)

results = {}


# LDA
lda = Pipeline([
    ("scaler", scaler),
    ("clf", LinearDiscriminantAnalysis())
])
lda.fit(X_train, y_train_full)
results["LDA"] = evaluate_model("LDA", lda)


# Logistic regression model
log = Pipeline([
    ("scaler", scaler),
    ("clf", LogisticRegression(max_iter=800, multi_class="multinomial",
                              class_weight="balanced"))
])
log_grid = GridSearchCV(log, {"clf__C":[1.0]}, cv=cv, n_jobs=-1)
log_grid.fit(X_train, y_train_full)
results["LogReg"] = evaluate_model("Logistic Regression", log_grid.best_estimator_)


# K- Nearest Neighbour
k_vals = [3,5,7,9,11]  # trimmed due to high calculation time
knn_scores = []
for k in k_vals:
    knn = Pipeline([("scaler", scaler), ("clf", KNeighborsClassifier(n_neighbors=k))])
    knn_scores.append(cross_val_score(knn, X_train, y_train_full, cv=cv, scoring="accuracy").mean())

best_k = k_vals[np.argmax(knn_scores)]
knn = Pipeline([("scaler", scaler), ("clf", KNeighborsClassifier(best_k))])
knn.fit(X_train, y_train_full)
results["KNN"] = evaluate_model(f"KNN (k={best_k})", knn)


# Decision tree model - base
dt_base = DecisionTreeClassifier(class_weight="balanced", random_state=5805)
dt_base.fit(X_train, y_train_full)
results["DT_Base"] = evaluate_model("Decision Tree Base", dt_base)


# Decision tree (pre pruned)
dt_grid = GridSearchCV(
    DecisionTreeClassifier(class_weight="balanced", random_state=5805),
    {"max_depth":[8], "min_samples_split":[20], "criterion":["gini"]},
    cv=cv, n_jobs=-1)
dt_grid.fit(X_train, y_train_full)
dt_best = dt_grid.best_estimator_
results["DT_Pre"] = evaluate_model("Decision Tree Pre-Pruned", dt_best)


# Decision tree (post pruned)
path = dt_base.cost_complexity_pruning_path(X_train, y_train_full)
alphas = path.ccp_alphas
alphas = np.maximum(alphas, 0.0)

alpha_candidates = np.linspace(alphas.min(), alphas.max(), 10)

best_alpha = 0.0
best_score = -np.inf

for a in alpha_candidates:
    dt_tmp = DecisionTreeClassifier(
        ccp_alpha=float(a),
        class_weight="balanced",
        random_state=5805
    )
    score = cross_val_score(
        dt_tmp, X_train, y_train_full,
        cv=cv, scoring="accuracy"
    ).mean()

    if score > best_score:
        best_score = score
        best_alpha = float(a)

print("\nBest ccp_alpha for post-pruning:", best_alpha)

dt_post = DecisionTreeClassifier(
    ccp_alpha=best_alpha,
    class_weight="balanced",
    random_state=5805
)
dt_post.fit(X_train, y_train_full)
results["DT_Post"] = evaluate_model("Decision Tree Post-Pruned", dt_post)



# SVM linear model
svm_lin = Pipeline([
    ("scaler", scaler),
    ("clf", SVC(kernel="linear", probability=True, class_weight="balanced"))
])
svm_lin.fit(X_train, y_train_full)
results["SVM_Linear"] = evaluate_model("SVM Linear", svm_lin)


# SVM polynomial model
svm_poly = Pipeline([
    ("scaler", scaler),
    ("clf", SVC(kernel="poly", degree=3, probability=True, class_weight="balanced"))
])
svm_poly.fit(X_train, y_train_full)
results["SVM_Poly"] = evaluate_model("SVM Polynomial", svm_poly)


# SVM RBF model
svm_rbf = Pipeline([
    ("scaler", scaler),
    ("clf", SVC(kernel="rbf", gamma="scale", probability=True, class_weight="balanced"))
])
svm_rbf.fit(X_train, y_train_full)
results["SVM_RBF"] = evaluate_model("SVM RBF", svm_rbf)


# Naive bayes model
nb = Pipeline([("scaler", scaler), ("clf", GaussianNB())])
nb.fit(X_train, y_train_full)
results["NB"] = evaluate_model("Naive Bayes", nb)


# Random forest model
rf = RandomForestClassifier(
    n_estimators=120, max_depth=None,
    class_weight="balanced", n_jobs=-1, random_state=5805
)
rf.fit(X_train, y_train_full)
results["RF"] = evaluate_model("Random Forest", rf)


# Gradient boosting model
gb = GradientBoostingClassifier(
    n_estimators=150, learning_rate=0.08,
    random_state=5805
)
gb.fit(X_train, y_train_full)
results["GB"] = evaluate_model("Gradient Boosting", gb)


# Neural Network
mlp = Pipeline([
    ("scaler", scaler),
    ("clf", MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=5805))
])
mlp.fit(X_train, y_train_full)
results["MLP"] = evaluate_model("Neural Network - MLP Classifier", mlp)


# ROC curves from all models
plt.figure(figsize=(8,6))
y_bin = (y_test == 2).astype(int)

for name, res in results.items():
    scores = res["Scores"]
    if scores is None: continue
    try:
        fpr, tpr, _ = roc_curve(y_bin, scores)
        auc = roc_auc_score(y_bin, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
    except:
        continue

plt.plot([0,1],[0,1],"k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (Class 2 vs others)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ROC_Phase3_AllModels.png", dpi=300)

# Comparison table of all models
comparison_rows = []

for name, res in results.items():
    model = name

    # Extract metrics stored in evaluate_model()
    # (We already printed, now we recompute summary row)
    y_pred_test = None
    try:
        # Get model predictions again
        y_pred_test = locals()[name.replace(" ", "_").replace("-", "_").lower()].predict(X_test)
    except:
        pass

    if y_pred_test is None:
        continue

    acc_test = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred_test, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)

    # ROC AUC
    auc = np.nan
    if res["Scores"] is not None:
        auc = roc_auc_score((y_test == 2).astype(int), res["Scores"])

    comparison_rows.append([
        model,
        round(acc_test*100, 2),
        round(prec, 3),
        round(rec, 3),
        round(f1, 3),
        round(auc, 3) if not np.isnan(auc) else "NA"
    ])

# Convert to DataFrame
comparison_df = pd.DataFrame(
    comparison_rows,
    columns=["Model", "Test Accuracy (%)", "Precision", "Recall", "F1 Score", "ROC AUC"]
)

# Sort by Test Accuracy (best at top)
comparison_df = comparison_df.sort_values(by="Test Accuracy (%)", ascending=False)

print("\n\nPHASE 3 MODEL COMPARISON TABLE: \n")
print(comparison_df.to_string(index=False))


comparison_df.to_csv("Phase3_Model_Comparison_Table.csv", index=False)

# === SAVE SEPARATE ROC CURVES FOR EACH MODEL ===
import os

roc_folder = "ROC_Models"
os.makedirs(roc_folder, exist_ok=True)

y_bin = (y_test == 2).astype(int)

print("\nSaving separate ROC curves for each model...")

for model_name, res in results.items():
    scores = res["Scores"]

    if scores is None:
        print(f"Skipping {model_name} (no probability scores).")
        continue

    try:
        fpr, tpr, _ = roc_curve(y_bin, scores)
        auc_value = roc_auc_score(y_bin, scores)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.grid(True)

        save_path = f"{roc_folder}/ROC_{model_name.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved: {save_path}")

    except Exception as e:
        print(f"Could not generate ROC for {model_name}: {e}")


print("\nPhase 3 is completed")
