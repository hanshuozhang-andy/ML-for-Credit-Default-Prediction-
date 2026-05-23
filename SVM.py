from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# 0. Configuration
# ============================================================

SEED = 42

PROJECT_DIR = Path(__file__).resolve().parent
TRAINING_PATH = PROJECT_DIR / "training_random_oversampled.xlsx"
ORIGINAL_WORKBOOK_PATH = PROJECT_DIR / "default of credit card clients.xls"

OUTPUT_PREDICTIONS_PATH = PROJECT_DIR / "svm_test_predictions.csv"
VALIDATION_RESULTS_PATH = PROJECT_DIR / "svm_validation_results.csv"


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
# 2. Basic columns
# ============================================================

target_col = "default payment next month"
id_col = "ID"

pay_cols = [
    "PAY_0", "PAY_2", "PAY_3",
    "PAY_4", "PAY_5", "PAY_6"
]

bill_cols = [
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"
]

pay_amt_cols = [
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]


# ============================================================
# 3. Data cleaning + feature engineering
# Same feature engineering as the neural network code.
# ============================================================

def validate_required_columns(df, dataset_name):
    """
    Check whether all columns needed for the reduced feature engineering exist.
    This follows the same idea as the MLP reduced-feature-engineering code.
    """
    required_cols = [id_col, target_col] + pay_cols + bill_cols + pay_amt_cols + [
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
    """
    Reduced feature engineering version from the MLP code.

    Compared with the previous SVM code, this version keeps only compact and
    interpretable summary features. It removes the large number of extra
    aggregate and signed-log transformed variables.
    """
    df = df.copy()

    # --------------------------------------------------------
    # 3.1 Clean PAY variables
    # -2, -1, and 0 are treated as no repayment delay, coded as 0.
    # --------------------------------------------------------
    for col in pay_cols:
        df[col] = df[col].replace({-2: 0, -1: 0})

    # --------------------------------------------------------
    # 3.2 Clean categorical variables
    # EDUCATION: 0, 5, 6 are merged into 4: others.
    # MARRIAGE: 0 is merged into 3: others.
    # --------------------------------------------------------
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # --------------------------------------------------------
    # 3.3 Reduced repayment status features
    # Keep compact features describing delay existence, frequency, severity,
    # and recent change in repayment behavior.
    # --------------------------------------------------------
    df["HAS_DELAY"] = (df[pay_cols].max(axis=1) > 0).astype(int)
    df["NUM_DELAY_MONTHS"] = (df[pay_cols] > 0).sum(axis=1)
    df["MAX_DELAY"] = df[pay_cols].max(axis=1)

    recent_delay = df[["PAY_0", "PAY_2", "PAY_3"]].mean(axis=1)
    old_delay = df[["PAY_4", "PAY_5", "PAY_6"]].mean(axis=1)
    df["AVG_DELAY_RECENT3"] = recent_delay
    df["DELAY_CHANGE"] = recent_delay - old_delay

    # --------------------------------------------------------
    # 3.4 Reduced financial ratio features
    # Use average bill and average payment only as intermediate variables.
    # These variables are not saved as independent features.
    # --------------------------------------------------------
    avg_bill_amt = df[bill_cols].mean(axis=1)
    avg_pay_amt = df[pay_amt_cols].mean(axis=1)

    df["AVG_BILL_LIMIT_RATIO"] = avg_bill_amt / (df["LIMIT_BAL"] + 1)
    df["PAY1_BILL1_RATIO"] = df["PAY_AMT1"] / (np.abs(df["BILL_AMT1"]) + 1)
    df["AVG_PAY_BILL_RATIO"] = avg_pay_amt / (np.abs(avg_bill_amt) + 1)
    df["TOTAL_PAY_BILL_RATIO"] = (
        df[pay_amt_cols].sum(axis=1) / (np.abs(df[bill_cols].sum(axis=1)) + 1)
    )

    # --------------------------------------------------------
    # 3.5 Reduced trend features
    # Compare recent three-month average with earlier three-month average.
    # --------------------------------------------------------
    df["BILL_TREND"] = (
        df[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3"]].mean(axis=1)
        - df[["BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]].mean(axis=1)
    )

    df["PAY_TREND"] = (
        df[["PAY_AMT1", "PAY_AMT2", "PAY_AMT3"]].mean(axis=1)
        - df[["PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]].mean(axis=1)
    )

    # Replace infinite values caused by ratios and fill missing values.
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
# 4. Split X and y
# ============================================================

feature_cols = [
    col for col in train_df.columns
    if col not in [id_col, target_col]
]

X_train = train_df[feature_cols]
y_train = train_df[target_col].astype(int)

X_val = val_df[feature_cols]
y_val = val_df[target_col].astype(int)

X_test = test_df[feature_cols]
y_test = test_df[target_col].astype(int)

print("\nClass distribution:")
print("Training:\n", y_train.value_counts(normalize=True).sort_index())
print("Validation:\n", y_val.value_counts(normalize=True).sort_index())
print("Test:\n", y_test.value_counts(normalize=True).sort_index())


# ============================================================
# 5. Categorical and numerical columns
# Same definition as the neural network code.
# PAY_0 to PAY_6 are treated as categorical repayment-status variables.
# ============================================================

categorical_cols = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]

numeric_cols = [
    col for col in feature_cols
    if col not in categorical_cols
]


# ============================================================
# 6. Preprocessor
# One-hot encoding for categorical variables.
# StandardScaler for numerical variables because SVM is scale-sensitive.
# ============================================================

try:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", onehot, categorical_cols),
        ("numeric", StandardScaler(), numeric_cols)
    ]
)


# ============================================================
# 7. Threshold selection
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
# 8. Evaluation function
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
        "prob": prob,
        "pred": pred,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
        "auc": auc
    }


# ============================================================
# 9. Build SVM models
# ============================================================

def build_svm_model(params):
    """
    Build an SVM pipeline.

    Linear SVM is recommended for this dataset because it is much faster.
    LinearSVC does not naturally provide predict_proba(), so we wrap it with
    CalibratedClassifierCV to obtain probability estimates for threshold tuning
    and ROC AUC calculation.

    RBF SVM is also provided as an optional nonlinear model, but it can be slow
    on 30,000 samples.
    """
    kernel = params["kernel"]

    if kernel == "linear":
        base_svm = LinearSVC(
            C=params["C"],
            class_weight=params["class_weight"],
            max_iter=10000,
            dual=False,
            random_state=SEED
        )
        svm_clf = CalibratedClassifierCV(
            estimator=base_svm,
            method="sigmoid",
            cv=3
        )

    elif kernel == "rbf":
        svm_clf = SVC(
            kernel="rbf",
            C=params["C"],
            gamma=params["gamma"],
            class_weight=params["class_weight"],
            probability=True,
            cache_size=1000,
            random_state=SEED
        )

    else:
        raise ValueError("kernel must be 'linear' or 'rbf'")

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("svm", svm_clf)
    ])

    return model


# ============================================================
# 10. Hyperparameter grid
# ============================================================
# To keep training time reasonable, this grid focuses on Linear SVM.
# You can uncomment the RBF settings if your computer can train them.
# ============================================================

param_grid = []

for C in [0.001, 0.01, 0.1, 1, 10]:
    for class_weight in [None, "balanced"]:
        param_grid.append({
            "kernel": "linear",
            "C": C,
            "class_weight": class_weight
        })

# Optional nonlinear RBF SVM. This may take much longer.
# for C in [0.1, 1, 10]:
#     for gamma in ["scale", 0.01, 0.1]:
#         for class_weight in [None, "balanced"]:
#             param_grid.append({
#                 "kernel": "rbf",
#                 "C": C,
#                 "gamma": gamma,
#                 "class_weight": class_weight
#             })


# ============================================================
# 11. Model selection using validation set
# ============================================================

best_model = None
best_params = None
best_threshold = 0.5
best_val_score = -1
validation_records = []

# Choose one of: "f1" or "balanced_accuracy".
# f1 balances precision and recall.
selection_metric = "f1"

for params in param_grid:

    print("\nTrying parameters:", params)

    model = build_svm_model(params)
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]

    threshold, val_score = find_best_threshold(
        y_true=y_val,
        y_prob=val_prob,
        metric=selection_metric
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

    record = params.copy()
    record.update({
        "threshold": threshold,
        "precision": val_precision,
        "recall": val_recall,
        "f1": val_f1,
        "balanced_accuracy": val_bal_acc,
        "auc": val_auc,
        "selection_score": val_score
    })
    validation_records.append(record)

    if val_score > best_val_score:
        best_val_score = val_score
        best_model = model
        best_params = params
        best_threshold = threshold


validation_results_df = pd.DataFrame(validation_records)
print("\n========== Validation Summary ==========")
print(validation_results_df.sort_values(by="selection_score", ascending=False))

print("\n==============================")
print("Best parameters:", best_params)
print("Best validation threshold:", best_threshold)
print("Best validation score:", best_val_score)
print("Selection metric:", selection_metric)
print("==============================")


# ============================================================
# 12. Evaluate Training / Validation / Test
# ============================================================

train_results = evaluate_dataset(
    name="Training",
    X=X_train,
    y=y_train,
    model=best_model,
    threshold=best_threshold
)

val_results = evaluate_dataset(
    name="Validation",
    X=X_val,
    y=y_val,
    model=best_model,
    threshold=best_threshold
)

test_results = evaluate_dataset(
    name="Test",
    X=X_test,
    y=y_test,
    model=best_model,
    threshold=best_threshold
)


# ============================================================
# 13. Final test results
# ============================================================

test_prob = test_results["prob"]
test_pred = test_results["pred"]

cm = confusion_matrix(y_test, test_pred)

print("\n========== Final Test Results ==========")
print(f"Best SVM parameters: {best_params}")
print(f"Threshold: {best_threshold:.2f}")
print(f"Precision: {test_results['precision']:.4f}")
print(f"Recall: {test_results['recall']:.4f}")
print(f"F1 Score: {test_results['f1']:.4f}")
print(f"Balanced Accuracy: {test_results['balanced_accuracy']:.4f}")
print(f"ROC AUC: {test_results['auc']:.4f}")

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 14. ROC Curve
# ============================================================

fpr, tpr, thresholds = roc_curve(y_test, test_prob)
auc = test_results["auc"]

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"SVM, AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SVM ROC Curve on Test Set")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 15. Overfitting diagnosis
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
# 16. Save prediction results and validation summary
# ============================================================

output_df = test_df[[id_col, target_col]].copy()
output_df["Predicted_Probability"] = test_prob
output_df["Predicted_Class"] = test_pred

output_path = OUTPUT_PREDICTIONS_PATH
validation_output_path = VALIDATION_RESULTS_PATH

output_df.to_csv(output_path, index=False)
validation_results_df.to_csv(validation_output_path, index=False)

print("\nPrediction results saved to:")
print(output_path)
print("\nValidation results saved to:")
print(validation_output_path)
