import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# 0. Configuration
# ============================================================

SEED = 42

PROJECT_DIR = Path(__file__).resolve().parent
TRAINING_PATH = PROJECT_DIR / "training_random_oversampled.xlsx"
ORIGINAL_WORKBOOK_PATH = PROJECT_DIR / "default of credit card clients.xls"

TARGET_COL = "default payment next month"
ID_COL = "ID"

OUTPUT_PREDICTIONS_PATH = PROJECT_DIR / "logreg_reduced_fe_resampled_test_predictions.csv"

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

BILL_COLS = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
]

PAY_AMT_COLS = [
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

CATEGORICAL_COLS = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]

SELECTION_METRIC = "f1"


# ============================================================
# 1. Read data
# ============================================================

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df.dropna(how="all")


def read_resampled_training(path):
    df = pd.read_excel(path)
    return normalize_columns(df)


def read_original_sheet(path, sheet_name):
    df = pd.read_excel(
        path,
        sheet_name=sheet_name,
        header=0,
        skiprows=[1],
        engine="xlrd",
    )
    return normalize_columns(df)


train_df = read_resampled_training(TRAINING_PATH)
val_df = read_original_sheet(ORIGINAL_WORKBOOK_PATH, "Validation")
test_df = read_original_sheet(ORIGINAL_WORKBOOK_PATH, "Test")

print("Training source:", TRAINING_PATH)
print("Validation/Test source:", ORIGINAL_WORKBOOK_PATH)
print("Validation sheet: Validation")
print("Test sheet: Test")
print("Training shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)


# ============================================================
# 2. Data cleaning + feature engineering
# ============================================================

def validate_required_columns(df, dataset_name):
    required_cols = [ID_COL, TARGET_COL] + PAY_COLS + BILL_COLS + PAY_AMT_COLS + [
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{dataset_name} is missing columns: {missing_cols}")


def preprocess_raw_data(df):
    df = df.copy()

    for col in PAY_COLS:
        df[col] = df[col].replace({-2: 0, -1: 0})

    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # Reduced feature engineering: keep only compact, interpretable summary features.
    df["HAS_DELAY"] = (df[PAY_COLS].max(axis=1) > 0).astype(int)
    df["NUM_DELAY_MONTHS"] = (df[PAY_COLS] > 0).sum(axis=1)
    df["MAX_DELAY"] = df[PAY_COLS].max(axis=1)
    recent_delay = df[["PAY_0", "PAY_2", "PAY_3"]].mean(axis=1)
    old_delay = df[["PAY_4", "PAY_5", "PAY_6"]].mean(axis=1)
    df["AVG_DELAY_RECENT3"] = recent_delay
    df["DELAY_CHANGE"] = recent_delay - old_delay

    avg_bill_amt = df[BILL_COLS].mean(axis=1)
    avg_pay_amt = df[PAY_AMT_COLS].mean(axis=1)

    df["AVG_BILL_LIMIT_RATIO"] = avg_bill_amt / (df["LIMIT_BAL"] + 1)
    df["PAY1_BILL1_RATIO"] = df["PAY_AMT1"] / (np.abs(df["BILL_AMT1"]) + 1)
    df["AVG_PAY_BILL_RATIO"] = avg_pay_amt / (np.abs(avg_bill_amt) + 1)
    df["TOTAL_PAY_BILL_RATIO"] = (
        df[PAY_AMT_COLS].sum(axis=1) / (np.abs(df[BILL_COLS].sum(axis=1)) + 1)
    )

    df["BILL_TREND"] = (
        df[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3"]].mean(axis=1)
        - df[["BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]].mean(axis=1)
    )
    df["PAY_TREND"] = (
        df[["PAY_AMT1", "PAY_AMT2", "PAY_AMT3"]].mean(axis=1)
        - df[["PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]].mean(axis=1)
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.fillna(0)


for name, data in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Test", test_df),
]:
    validate_required_columns(data, name)


train_df = preprocess_raw_data(train_df)
val_df = preprocess_raw_data(val_df)
test_df = preprocess_raw_data(test_df)


# ============================================================
# 3. Split X and y
# ============================================================

feature_cols = [col for col in train_df.columns if col not in [ID_COL, TARGET_COL]]
numeric_cols = [col for col in feature_cols if col not in CATEGORICAL_COLS]

X_train = train_df[feature_cols]
y_train = train_df[TARGET_COL].astype(int)

X_val = val_df[feature_cols]
y_val = val_df[TARGET_COL].astype(int)

X_test = test_df[feature_cols]
y_test = test_df[TARGET_COL].astype(int)


# ============================================================
# 4. Preprocessor
# ============================================================

try:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", onehot, CATEGORICAL_COLS),
        ("numeric", StandardScaler(), numeric_cols),
    ]
)


# ============================================================
# 5. Threshold selection
# ============================================================

def find_best_threshold(y_true, y_prob, metric="f1"):
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.5
    best_score = -1
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError("metric must be 'f1' or 'balanced_accuracy'")
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold, best_score


# ============================================================
# 6. Evaluation function
# ============================================================

def evaluate_dataset(name, X, y, model, threshold):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    balanced_acc = balanced_accuracy_score(y, pred)
    auc = roc_auc_score(y, prob)

    print(f"\n========== {name} Results ==========")
    print(f"Threshold: {threshold:.2f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"ROC AUC: {auc:.4f}")

    return {
        "prob": prob, "pred": pred,
        "precision": precision, "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
        "auc": auc,
    }


# ============================================================
# 7. Hyperparameter grid 
# ============================================================


param_grid = [
    {"penalty": "l2", "C": 0.001},
    {"penalty": "l2", "C": 0.01},
    {"penalty": "l2", "C": 0.1},
    {"penalty": "l2", "C": 1.0},
    {"penalty": "l2", "C": 10.0},
    {"penalty": "l2", "C": 100.0},
    {"penalty": "l1", "C": 0.01},
    {"penalty": "l1", "C": 0.1},
    {"penalty": "l1", "C": 1.0},
    {"penalty": "l1", "C": 10.0},
]


# ============================================================
# 8. Train LR with given params
# ============================================================

def train_lr(params, X_train, y_train):
    solver = "liblinear" if params["penalty"] == "l1" else "lbfgs"

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("lr", LogisticRegression(
            penalty=params["penalty"],
            C=params["C"],
            solver=solver,
            max_iter=2000,
            random_state=SEED,
        )),
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, y_train)
    return model


# ============================================================
# 9. Model selection using validation set
# ============================================================

best_model = None
best_params = None
best_threshold = 0.5
best_val_score = -1



for params in param_grid:
    print("\n================================================")
    print("Trying parameters:", params)
    print("================================================")

    model = train_lr(params, X_train, y_train)
    val_prob = model.predict_proba(X_val)[:, 1]

    threshold, val_score = find_best_threshold(
        y_true=y_val, y_prob=val_prob, metric=SELECTION_METRIC
    )
    val_pred = (val_prob >= threshold).astype(int)

    val_precision = precision_score(y_val, val_pred, zero_division=0)
    val_recall = recall_score(y_val, val_pred, zero_division=0)
    val_f1 = f1_score(y_val, val_pred, zero_division=0)
    val_bal_acc = balanced_accuracy_score(y_val, val_pred)
    val_auc = roc_auc_score(y_val, val_prob)

    print(f"Validation threshold: {threshold:.2f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    print(f"Validation F1: {val_f1:.4f}")
    print(f"Validation Balanced Accuracy: {val_bal_acc:.4f}")
    print(f"Validation ROC AUC: {val_auc:.4f}")

    if val_score > best_val_score:
        best_val_score = val_score
        best_model = model
        best_params = params
        best_threshold = threshold


print("\n==============================")
print("Best parameters:", best_params)
print("Best validation threshold:", best_threshold)
print("Best validation score:", best_val_score)
print("Selection metric:", SELECTION_METRIC)
print("==============================")


# ============================================================
# 10. Evaluate Training / Validation / Test
# ============================================================

train_results = evaluate_dataset("Training", X_train, y_train, best_model, best_threshold)
val_results = evaluate_dataset("Validation", X_val, y_val, best_model, best_threshold)
test_results = evaluate_dataset("Test", X_test, y_test, best_model, best_threshold)


# ============================================================
# 11. Final test results
# ============================================================

test_prob = test_results["prob"]
test_pred = test_results["pred"]
cm = confusion_matrix(y_test, test_pred)

print("\n========== Final Test Results ==========")
print(f"Threshold: {best_threshold:.2f}")
print(f"Precision: {test_results['precision']:.4f}")
print(f"Recall: {test_results['recall']:.4f}")
print(f"F1 Score: {test_results['f1']:.4f}")
print(f"Balanced Accuracy: {test_results['balanced_accuracy']:.4f}")
print(f"ROC AUC: {test_results['auc']:.4f}")

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 12. ROC Curve
# ============================================================

fpr, tpr, _ = roc_curve(y_test, test_prob)
auc = test_results["auc"]

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression, AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve on Test Set")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 13. Overfitting diagnosis
# ============================================================

print("\n========== Overfitting Diagnosis ==========")
print(f"Training F1: {train_results['f1']:.4f}")
print(f"Validation F1: {val_results['f1']:.4f}")
print(f"Test F1: {test_results['f1']:.4f}")
print(f"Training AUC: {train_results['auc']:.4f}")
print(f"Validation AUC: {val_results['auc']:.4f}")
print(f"Test AUC: {test_results['auc']:.4f}")

f1_gap = train_results["f1"] - test_results["f1"]
auc_gap = train_results["auc"] - test_results["auc"]
print(f"F1 gap between training and test: {f1_gap:.4f}")
print(f"AUC gap between training and test: {auc_gap:.4f}")

if f1_gap > 0.15 or auc_gap > 0.15:
    print("Diagnosis: The model is likely overfitting.")
elif train_results["f1"] < 0.45 and test_results["f1"] < 0.45:
    print("Diagnosis: The model may be underfitting.")
else:
    print("Diagnosis: The model has reasonable generalization performance.")


# ============================================================
# 14. Save prediction results
# ============================================================

output_df = test_df[[ID_COL, TARGET_COL]].copy()
output_df["Predicted_Probability"] = test_prob
output_df["Predicted_Class"] = test_pred

output_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)

print("\nPrediction results saved to:")
print(OUTPUT_PREDICTIONS_PATH)
