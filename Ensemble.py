import copy
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 0. Configuration
# ============================================================

SEED = 42

PROJECT_DIR = Path(__file__).resolve().parent
TRAINING_PATH = PROJECT_DIR / "training_random_oversampled.xlsx"
ORIGINAL_WORKBOOK_PATH = PROJECT_DIR / "default of credit card clients.xls"

TARGET_COL = "default payment next month"
ID_COL = "ID"

OUTPUT_PREDICTIONS_PATH = PROJECT_DIR / "ensemble_reduced_fe_resampled_test_predictions.csv"
OUTPUT_WEIGHT_SEARCH_PATH = PROJECT_DIR / "ensemble_weight_search_validation_results.csv"

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

SELECTION_METRIC = "balanced_accuracy"
THRESHOLDS = np.arange(0.05, 0.96, 0.01)

# The weight search uses a validation-only probability blend.
# 0.05 means weights are tried in 5% increments and sum to 1.
WEIGHT_STEP = 0.05

LR_PARAM_GRID = [
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

SVM_PARAM_GRID = [
    {"C": C, "class_weight": class_weight}
    for C in [0.001, 0.01, 0.1, 1, 10]
    for class_weight in [None, "balanced"]
]

# Keep the MLP architecture and optimizer settings aligned with the current
# resampled reduced-feature MLP script.
HIDDEN_LAYER_SIZES = (16,8,4)
LEARNING_RATE = 0.0002
WEIGHT_DECAY = 0.01
BATCH_SIZE = 512
MAX_EPOCHS = 1000
PATIENCE = 20
MIN_DELTA = 1e-5
DROPOUT_RATE = 0
MLP_LOG_EVERY_EPOCHS = 25


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


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
# 2. Shared cleaning and feature engineering
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
# 3. Split and preprocess once for all three models
# ============================================================

feature_cols = [col for col in train_df.columns if col not in [ID_COL, TARGET_COL]]
numeric_cols = [col for col in feature_cols if col not in CATEGORICAL_COLS]

X_train = train_df[feature_cols]
y_train = train_df[TARGET_COL].astype(int)

X_val = val_df[feature_cols]
y_val = val_df[TARGET_COL].astype(int)

X_test = test_df[feature_cols]
y_test = test_df[TARGET_COL].astype(int)

print("\nClass distribution:")
print("Training:\n", y_train.value_counts(normalize=True).sort_index())
print("Validation:\n", y_val.value_counts(normalize=True).sort_index())
print("Test:\n", y_test.value_counts(normalize=True).sort_index())

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

X_train_processed = np.asarray(preprocessor.fit_transform(X_train), dtype=np.float32)
X_val_processed = np.asarray(preprocessor.transform(X_val), dtype=np.float32)
X_test_processed = np.asarray(preprocessor.transform(X_test), dtype=np.float32)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed validation shape:", X_val_processed.shape)
print("Processed test shape:", X_test_processed.shape)


# ============================================================
# 4. Shared scoring helpers
# ============================================================

def find_best_threshold(y_true, y_prob, metric=SELECTION_METRIC):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in THRESHOLDS:
        y_pred = (y_prob >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError("metric must be 'f1' or 'balanced_accuracy'")

        if score > best_score:
            best_threshold = threshold
            best_score = score

    return best_threshold, best_score


def calculate_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "pred": y_pred,
        "prob": y_prob,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
    }


def print_metrics(name, results, threshold):
    print(f"\n========== {name} Results ==========")
    print(f"Threshold: {threshold:.2f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"ROC AUC: {results['auc']:.4f}")


def evaluate_probability(name, y_true, y_prob, threshold):
    results = calculate_metrics(y_true, y_prob, threshold)
    print_metrics(name, results, threshold)
    return results


def record_validation_metrics(params, threshold, val_score, results):
    record = params.copy()
    record.update({
        "threshold": threshold,
        "precision": results["precision"],
        "recall": results["recall"],
        "f1": results["f1"],
        "balanced_accuracy": results["balanced_accuracy"],
        "auc": results["auc"],
        "selection_score": val_score,
    })
    return record


# ============================================================
# 5. MLP model
# ============================================================

class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, dropout_rate):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_mlp(X_train_np, y_train_np, X_val_np, y_val_np):
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np.reshape(-1, 1), dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_np.reshape(-1, 1), dtype=torch.float32)

    model = MLPBinaryClassifier(
        input_dim=X_train_np.shape[1],
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        dropout_rate=DROPOUT_RATE,
    ).to(DEVICE)

    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    X_val_device = X_val_tensor.to(DEVICE)
    y_val_device = y_val_tensor.to(DEVICE)

    best_state = None
    best_val_loss = np.inf
    no_improve_count = 0
    history = {
        "train_losses": [],
        "val_losses": [],
    }

    print("\nTraining MLP...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_count += xb.size(0)

        train_loss = total_loss / total_count

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_device)
            val_loss = loss_fn(val_logits, y_val_device).item()

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)

        if epoch == 1 or epoch % MLP_LOG_EVERY_EPOCHS == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train loss: {train_loss:.5f} | "
                f"Validation loss: {val_loss:.5f}"
            )

        if val_loss < best_val_loss - MIN_DELTA:
            best_state = copy.deepcopy(model.state_dict())
            best_val_loss = val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= PATIENCE:
            print(f"Early stopping MLP at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_val_loss"] = best_val_loss
    return model, history


def predict_mlp_probability(model, X_np, batch_size=1024):
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
    probabilities = []

    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(DEVICE))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())

    return np.vstack(probabilities).ravel()


# ============================================================
# 6. Logistic regression and SVM selection
# ============================================================

def build_lr_model(params):
    solver = "liblinear" if params["penalty"] == "l1" else "lbfgs"
    return LogisticRegression(
        penalty=params["penalty"],
        C=params["C"],
        solver=solver,
        max_iter=2000,
        random_state=SEED,
    )


def build_svm_model(params):
    base_svm = LinearSVC(
        C=params["C"],
        class_weight=params["class_weight"],
        max_iter=10000,
        dual=False,
        random_state=SEED,
    )
    return CalibratedClassifierCV(
        estimator=base_svm,
        method="sigmoid",
        cv=3,
    )


def select_sklearn_model(model_name, build_model, param_grid):
    best_model = None
    best_params = None
    best_threshold = 0.5
    best_val_score = -1.0
    validation_records = []

    print(f"\nSelecting {model_name}...")
    for params in param_grid:
        print(f"\nTrying {model_name} parameters:", params)
        model = build_model(params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_train_processed, y_train)

        val_prob = model.predict_proba(X_val_processed)[:, 1]
        threshold, val_score = find_best_threshold(y_val, val_prob)
        val_results = calculate_metrics(y_val, val_prob, threshold)
        print_metrics(f"{model_name} Validation", val_results, threshold)

        validation_records.append(
            record_validation_metrics(params, threshold, val_score, val_results)
        )

        if val_score > best_val_score:
            best_model = model
            best_params = params
            best_threshold = threshold
            best_val_score = val_score

    validation_df = pd.DataFrame(validation_records)
    print(f"\n========== {model_name} Validation Summary ==========")
    print(validation_df.sort_values(by="selection_score", ascending=False))
    print(f"\nBest {model_name} parameters:", best_params)
    print(f"Best {model_name} validation threshold:", best_threshold)
    print(f"Best {model_name} validation score:", best_val_score)
    return best_model, best_params, best_threshold, validation_df


# ============================================================
# 7. Train and score the three base models
# ============================================================

mlp_model, mlp_history = train_mlp(
    X_train_np=X_train_processed,
    y_train_np=y_train.to_numpy(dtype=np.float32),
    X_val_np=X_val_processed,
    y_val_np=y_val.to_numpy(dtype=np.float32),
)

mlp_probabilities = {
    "train": predict_mlp_probability(mlp_model, X_train_processed),
    "val": predict_mlp_probability(mlp_model, X_val_processed),
    "test": predict_mlp_probability(mlp_model, X_test_processed),
}
mlp_threshold, mlp_val_score = find_best_threshold(
    y_val,
    mlp_probabilities["val"],
)
print("\nBest MLP validation threshold:", mlp_threshold)
print("Best MLP validation score:", mlp_val_score)
print("Best MLP validation loss:", mlp_history["best_val_loss"])

lr_model, lr_params, lr_threshold, lr_validation_df = select_sklearn_model(
    model_name="Logistic Regression",
    build_model=build_lr_model,
    param_grid=LR_PARAM_GRID,
)
lr_probabilities = {
    "train": lr_model.predict_proba(X_train_processed)[:, 1],
    "val": lr_model.predict_proba(X_val_processed)[:, 1],
    "test": lr_model.predict_proba(X_test_processed)[:, 1],
}

svm_model, svm_params, svm_threshold, svm_validation_df = select_sklearn_model(
    model_name="SVM",
    build_model=build_svm_model,
    param_grid=SVM_PARAM_GRID,
)
svm_probabilities = {
    "train": svm_model.predict_proba(X_train_processed)[:, 1],
    "val": svm_model.predict_proba(X_val_processed)[:, 1],
    "test": svm_model.predict_proba(X_test_processed)[:, 1],
}

base_models = {
    "mlp": {
        "label": "MLP",
        "threshold": mlp_threshold,
        "probabilities": mlp_probabilities,
    },
    "logreg": {
        "label": "Logistic Regression",
        "threshold": lr_threshold,
        "probabilities": lr_probabilities,
    },
    "svm": {
        "label": "SVM",
        "threshold": svm_threshold,
        "probabilities": svm_probabilities,
    },
}

print("\n========== Base Model Results ==========")
for base_model in base_models.values():
    label = base_model["label"]
    threshold = base_model["threshold"]
    probabilities = base_model["probabilities"]
    train_result = evaluate_probability(
        f"{label} Training",
        y_train,
        probabilities["train"],
        threshold,
    )
    val_result = evaluate_probability(
        f"{label} Validation",
        y_val,
        probabilities["val"],
        threshold,
    )
    test_result = evaluate_probability(
        f"{label} Test",
        y_test,
        probabilities["test"],
        threshold,
    )
    base_model["results"] = {
        "train": train_result,
        "val": val_result,
        "test": test_result,
    }

if base_models["mlp"]["results"]["val"]["auc"] <= 0.55:
    print(
        "\nWarning: MLP validation AUC is near random. "
        "The validation weight search may reduce its ensemble weight."
    )


# ============================================================
# 8. Validation weight search for soft voting
# ============================================================

def weighted_probability(probabilities, weights, split_name):
    return sum(
        weights[model_name] * probabilities[model_name][split_name]
        for model_name in weights
    )


def search_soft_voting_weights(validation_probabilities):
    weight_units = int(round(1 / WEIGHT_STEP))
    best_record = None
    records = []

    for mlp_units in range(weight_units + 1):
        for lr_units in range(weight_units - mlp_units + 1):
            svm_units = weight_units - mlp_units - lr_units
            weights = {
                "mlp": mlp_units / weight_units,
                "logreg": lr_units / weight_units,
                "svm": svm_units / weight_units,
            }
            val_prob = weighted_probability(
                validation_probabilities,
                weights,
                split_name="val",
            )
            threshold, selection_score = find_best_threshold(y_val, val_prob)
            results = calculate_metrics(y_val, val_prob, threshold)

            record = {
                "w_mlp": weights["mlp"],
                "w_logreg": weights["logreg"],
                "w_svm": weights["svm"],
                "threshold": threshold,
                "precision": results["precision"],
                "recall": results["recall"],
                "f1": results["f1"],
                "balanced_accuracy": results["balanced_accuracy"],
                "auc": results["auc"],
                "selection_score": selection_score,
            }
            records.append(record)

            is_better_score = (
                best_record is None
                or record["selection_score"] > best_record["selection_score"]
            )
            is_better_auc_tie = (
                best_record is not None
                and np.isclose(record["selection_score"], best_record["selection_score"])
                and record["auc"] > best_record["auc"]
            )
            if is_better_score or is_better_auc_tie:
                best_record = record

    return best_record, pd.DataFrame(records)


probabilities_by_model = {
    model_name: model_data["probabilities"]
    for model_name, model_data in base_models.items()
}

equal_weights = {
    "mlp": 1 / 3,
    "logreg": 1 / 3,
    "svm": 1 / 3,
}
equal_probabilities = {
    split_name: weighted_probability(probabilities_by_model, equal_weights, split_name)
    for split_name in ["train", "val", "test"]
}
equal_threshold, equal_val_score = find_best_threshold(
    y_val,
    equal_probabilities["val"],
)

best_weight_record, weight_search_df = search_soft_voting_weights(
    probabilities_by_model
)
best_weights = {
    "mlp": best_weight_record["w_mlp"],
    "logreg": best_weight_record["w_logreg"],
    "svm": best_weight_record["w_svm"],
}
ensemble_threshold = best_weight_record["threshold"]
ensemble_probabilities = {
    split_name: weighted_probability(probabilities_by_model, best_weights, split_name)
    for split_name in ["train", "val", "test"]
}

print("\n========== Soft Voting Validation Search ==========")
print("Equal weights:", equal_weights)
print("Equal-weight validation threshold:", equal_threshold)
print("Equal-weight validation score:", equal_val_score)
print("\nTop validation weight settings:")
print(
    weight_search_df.sort_values(
        by=["selection_score", "auc"],
        ascending=False,
    ).head(10)
)
print("\nSelected weights:", best_weights)
print("Selected ensemble threshold:", ensemble_threshold)
print("Selection metric:", SELECTION_METRIC)


# ============================================================
# 9. Evaluate equal soft voting and tuned soft voting
# ============================================================

print("\n========== Equal-Weight Soft Voting Results ==========")
equal_train_results = evaluate_probability(
    "Equal-Weight Ensemble Training",
    y_train,
    equal_probabilities["train"],
    equal_threshold,
)
equal_val_results = evaluate_probability(
    "Equal-Weight Ensemble Validation",
    y_val,
    equal_probabilities["val"],
    equal_threshold,
)
equal_test_results = evaluate_probability(
    "Equal-Weight Ensemble Test",
    y_test,
    equal_probabilities["test"],
    equal_threshold,
)

print("\n========== Tuned Soft Voting Results ==========")
ensemble_train_results = evaluate_probability(
    "Tuned Ensemble Training",
    y_train,
    ensemble_probabilities["train"],
    ensemble_threshold,
)
ensemble_val_results = evaluate_probability(
    "Tuned Ensemble Validation",
    y_val,
    ensemble_probabilities["val"],
    ensemble_threshold,
)
ensemble_test_results = evaluate_probability(
    "Tuned Ensemble Test",
    y_test,
    ensemble_probabilities["test"],
    ensemble_threshold,
)

ensemble_cm = confusion_matrix(y_test, ensemble_test_results["pred"])
print("\n========== Final Tuned Ensemble Test Results ==========")
print("Selected weights:", best_weights)
print(f"Threshold: {ensemble_threshold:.2f}")
print(f"Precision: {ensemble_test_results['precision']:.4f}")
print(f"Recall: {ensemble_test_results['recall']:.4f}")
print(f"F1 Score: {ensemble_test_results['f1']:.4f}")
print(f"Balanced Accuracy: {ensemble_test_results['balanced_accuracy']:.4f}")
print(f"ROC AUC: {ensemble_test_results['auc']:.4f}")
print("\nConfusion Matrix:")
print(ensemble_cm)


# ============================================================
# 10. Generalization check
# ============================================================

print("\n========== Tuned Ensemble Generalization Check ==========")
print(f"Training F1: {ensemble_train_results['f1']:.4f}")
print(f"Validation F1: {ensemble_val_results['f1']:.4f}")
print(f"Test F1: {ensemble_test_results['f1']:.4f}")
print(f"Training AUC: {ensemble_train_results['auc']:.4f}")
print(f"Validation AUC: {ensemble_val_results['auc']:.4f}")
print(f"Test AUC: {ensemble_test_results['auc']:.4f}")

f1_gap = ensemble_train_results["f1"] - ensemble_test_results["f1"]
auc_gap = ensemble_train_results["auc"] - ensemble_test_results["auc"]
print(f"F1 gap between training and test: {f1_gap:.4f}")
print(f"AUC gap between training and test: {auc_gap:.4f}")

if (
    ensemble_train_results["auc"] <= 0.55
    and ensemble_val_results["auc"] <= 0.55
    and ensemble_test_results["auc"] <= 0.55
):
    print("Diagnosis: The ensemble has not learned meaningful discrimination.")
elif f1_gap > 0.15 or auc_gap > 0.15:
    print("Diagnosis: The ensemble is likely overfitting.")
elif ensemble_train_results["f1"] < 0.45 and ensemble_test_results["f1"] < 0.45:
    print("Diagnosis: The ensemble may be underfitting.")
else:
    print("Diagnosis: The ensemble has reasonable generalization performance.")


# ============================================================
# 11. Plot ROC curves
# ============================================================

plt.figure(figsize=(8, 6))
for model_name, model_data in base_models.items():
    test_prob = model_data["probabilities"]["test"]
    test_auc = model_data["results"]["test"]["auc"]
    fpr, tpr, _ = roc_curve(y_test, test_prob)
    plt.plot(fpr, tpr, label=f"{model_data['label']}, AUC = {test_auc:.4f}")

equal_fpr, equal_tpr, _ = roc_curve(y_test, equal_probabilities["test"])
plt.plot(
    equal_fpr,
    equal_tpr,
    label=f"Equal Ensemble, AUC = {equal_test_results['auc']:.4f}",
)

ensemble_fpr, ensemble_tpr, _ = roc_curve(y_test, ensemble_probabilities["test"])
plt.plot(
    ensemble_fpr,
    ensemble_tpr,
    linewidth=2.5,
    label=f"Tuned Ensemble, AUC = {ensemble_test_results['auc']:.4f}",
)

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves on Test Set")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 12. Save outputs
# ============================================================

output_df = test_df[[ID_COL, TARGET_COL]].copy()
output_df["MLP_Probability"] = mlp_probabilities["test"]
output_df["Logistic_Regression_Probability"] = lr_probabilities["test"]
output_df["SVM_Probability"] = svm_probabilities["test"]
output_df["Equal_Weight_Ensemble_Probability"] = equal_probabilities["test"]
output_df["Predicted_Probability"] = ensemble_probabilities["test"]
output_df["Predicted_Class"] = ensemble_test_results["pred"]
output_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)
weight_search_df.to_csv(OUTPUT_WEIGHT_SEARCH_PATH, index=False)

print("\nPrediction results saved to:")
print(OUTPUT_PREDICTIONS_PATH)
print("\nValidation weight search saved to:")
print(OUTPUT_WEIGHT_SEARCH_PATH)
