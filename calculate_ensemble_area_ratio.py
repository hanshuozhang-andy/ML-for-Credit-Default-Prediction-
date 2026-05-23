from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS_PATH = PROJECT_DIR / "ensemble_reduced_fe_resampled_test_predictions.csv"
TARGET_COL = "default payment next month"


def cumulative_gain_area(values):
    return np.sum((values[:-1] + values[1:]) / 2.0)


def lift_chart_area_ratio(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    n_samples = y_true.size
    n_defaults = int(y_true.sum())
    if n_defaults == 0 or n_defaults == n_samples:
        raise ValueError("Area ratio needs both default and non-default cases.")

    ranked_labels = y_true[np.argsort(-y_prob, kind="mergesort")]
    customer_counts = np.arange(n_samples + 1)

    model_curve = np.concatenate(([0], np.cumsum(ranked_labels)))
    baseline_curve = customer_counts * n_defaults / n_samples
    best_curve = np.minimum(customer_counts, n_defaults)

    model_area = cumulative_gain_area(model_curve) - cumulative_gain_area(baseline_curve)
    best_area = cumulative_gain_area(best_curve) - cumulative_gain_area(baseline_curve)
    return model_area / best_area


def main():
    predictions = pd.read_csv(DEFAULT_PREDICTIONS_PATH)

    if TARGET_COL not in predictions.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    probability_cols = [
        col for col in predictions.columns
        if col.endswith("Probability") or col == "Predicted_Probability"
    ]
    if not probability_cols:
        raise ValueError("No probability columns found in the prediction CSV.")

    y_true = predictions[TARGET_COL].to_numpy()

    print("Prediction file:", DEFAULT_PREDICTIONS_PATH)
    print("Number of samples:", len(predictions))
    print("Number of defaults:", int(np.sum(y_true)))
    print("\n========== Lift-Chart Area Ratios ==========")

    for col in probability_cols:
        area_ratio = lift_chart_area_ratio(y_true, predictions[col].to_numpy())
        print(f"{col}: {area_ratio:.6f}")


if __name__ == "__main__":
    main()
