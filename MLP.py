import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
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

OUTPUT_PREDICTIONS_PATH = PROJECT_DIR / "mlp_reduced_fe_resampled_test_predictions.csv"

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

HIDDEN_LAYER_SIZES = (16,8,4)
LEARNING_RATE = 0.0002
WEIGHT_DECAY = 0.01
BATCH_SIZE = 512
MAX_EPOCHS = 1000
PATIENCE = 20
MIN_DELTA = 1e-5
DROPOUT_RATE = 0
SELECTION_METRIC = "f1"
PLOT_EVERY_N_BATCHES = 10
USE_RANDOM_VALIDATION_BATCH = False
VALIDATION_BATCH_SIZE = BATCH_SIZE


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
if USE_RANDOM_VALIDATION_BATCH:
    print("Validation loss mode: random mini-batch")
    print("Validation batch size:", VALIDATION_BATCH_SIZE)
else:
    print("Validation loss mode: full validation set")


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

print("\nClass distribution:")
print("Training:\n", y_train.value_counts(normalize=True).sort_index())
print("Validation:\n", y_val.value_counts(normalize=True).sort_index())
print("Test:\n", y_test.value_counts(normalize=True).sort_index())


# ============================================================
# 4. Preprocessing
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

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
X_val_processed = np.asarray(X_val_processed, dtype=np.float32)
X_test_processed = np.asarray(X_test_processed, dtype=np.float32)

y_train_np = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
y_val_np = y_val.to_numpy(dtype=np.float32).reshape(-1, 1)
y_test_np = y_test.to_numpy(dtype=np.float32).reshape(-1, 1)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed validation shape:", X_val_processed.shape)
print("Processed test shape:", X_test_processed.shape)


# ============================================================
# 5. Convert to torch tensors
# ============================================================

X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)


# ============================================================
# 6. Define MLP model
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


# ============================================================
# 7. Training and prediction helpers
# ============================================================

def train_mlp(
    X_train_tensor,
    y_train_tensor,
    X_val_tensor,
    y_val_tensor,
    input_dim,
):
    model = MLPBinaryClassifier(
        input_dim=input_dim,
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        dropout_rate=DROPOUT_RATE,
    ).to(DEVICE)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    val_generator = torch.Generator()
    val_generator.manual_seed(SEED + 1)

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
    batch_steps = []
    batch_train_losses = []
    train_losses = []
    val_steps = []
    val_losses = []
    global_step = 0
    batch_loss_buffer = []

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
            global_step += 1
            batch_loss_buffer.append(loss.item())

            if global_step % PLOT_EVERY_N_BATCHES == 0:
                batch_steps.append(global_step)
                batch_train_losses.append(float(np.mean(batch_loss_buffer)))
                batch_loss_buffer.clear()

        train_loss = total_loss / total_count

        model.eval()
        with torch.no_grad():
            if USE_RANDOM_VALIDATION_BATCH:
                n_val = X_val_tensor.size(0)
                val_batch_size = min(VALIDATION_BATCH_SIZE, n_val)
                val_idx = torch.randperm(n_val, generator=val_generator)[:val_batch_size]
                val_xb = X_val_tensor[val_idx].to(DEVICE)
                val_yb = y_val_tensor[val_idx].to(DEVICE)
                val_logits = model(val_xb)
                val_loss = loss_fn(val_logits, val_yb).item()
            else:
                val_logits = model(X_val_device)
                val_loss = loss_fn(val_logits, y_val_device).item()

        train_losses.append(train_loss)
        val_steps.append(global_step)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss:.5f} | "
            f"Validation loss: {val_loss:.5f}"
        )

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    if batch_loss_buffer:
        batch_steps.append(global_step)
        batch_train_losses.append(float(np.mean(batch_loss_buffer)))

    if best_state is not None:
        model.load_state_dict(best_state)

    history = {
        "batch_steps": batch_steps,
        "batch_train_losses": batch_train_losses,
        "train_losses": train_losses,
        "val_steps": val_steps,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
    }
    return model, history


def predict_probability(model, X_tensor, batch_size=1024):
    model.eval()
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
    all_probs = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs).ravel()


def find_best_threshold(y_true, y_prob, metric="f1"):
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.5
    best_score = -1.0

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


def evaluate_dataset(name, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    results = {
        "pred": y_pred,
        "prob": y_prob,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
    }

    print(f"\n========== {name} Results ==========")
    print(f"Threshold: {threshold:.2f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"ROC AUC: {results['auc']:.4f}")

    return results


def plot_roc_and_loss_curves(y_test, test_prob, test_auc, history):
    fpr, tpr, thresholds = roc_curve(y_test, test_prob)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(fpr, tpr, label=f"Neural Network, AUC = {test_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve on Test Set")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        history["batch_steps"],
        history["batch_train_losses"],
        label=f"Training Loss (avg every {PLOT_EVERY_N_BATCHES} batches)",
        linewidth=1,
        alpha=0.65,
    )
    axes[1].plot(
        history["val_steps"],
        history["val_losses"],
        label="Validation Loss",
        linewidth=2,
    )
    axes[1].set_xlabel("Gradient Descent Step (Mini-Batch)")
    axes[1].set_ylabel("Binary Cross-Entropy Loss")
    axes[1].set_title("MLP Training and Validation Loss")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    plt.show()


# ============================================================
# 8. Train, select threshold, evaluate
# ============================================================

input_dim = X_train_processed.shape[1]

model, history = train_mlp(
    X_train_tensor=X_train_tensor,
    y_train_tensor=y_train_tensor,
    X_val_tensor=X_val_tensor,
    y_val_tensor=y_val_tensor,
    input_dim=input_dim,
)

val_prob = predict_probability(model, X_val_tensor)
best_threshold, best_val_score = find_best_threshold(
    y_true=y_val.to_numpy(),
    y_prob=val_prob,
    metric=SELECTION_METRIC,
)

print("\n==============================")
print("Best parameters:", {
    "hidden_layer_sizes": HIDDEN_LAYER_SIZES,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "batch_size": BATCH_SIZE,
    "dropout_rate": DROPOUT_RATE,
})
print("Best validation threshold:", best_threshold)
print("Best validation score:", best_val_score)
print("Selection metric:", SELECTION_METRIC)
print("Best validation loss:", history["best_val_loss"])
print("==============================")

train_prob = predict_probability(model, X_train_tensor)
val_prob = predict_probability(model, X_val_tensor)
test_prob = predict_probability(model, X_test_tensor)

train_results = evaluate_dataset(
    name="Training",
    y_true=y_train.to_numpy(),
    y_prob=train_prob,
    threshold=best_threshold,
)
val_results = evaluate_dataset(
    name="Validation",
    y_true=y_val.to_numpy(),
    y_prob=val_prob,
    threshold=best_threshold,
)
test_results = evaluate_dataset(
    name="Test",
    y_true=y_test.to_numpy(),
    y_prob=test_prob,
    threshold=best_threshold,
)

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
# 9. ROC Curve and Loss Curve
# ============================================================

plot_roc_and_loss_curves(
    y_test=y_test.to_numpy(),
    test_prob=test_prob,
    test_auc=test_results["auc"],
    history=history,
)


# ============================================================
# 10. Overfitting diagnosis
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
# 11. Save outputs
# ============================================================

output_df = test_df[[ID_COL, TARGET_COL]].copy()
output_df["Predicted_Probability"] = test_prob
output_df["Predicted_Class"] = test_pred
output_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)

print("\nPrediction results saved to:")
print(OUTPUT_PREDICTIONS_PATH)
