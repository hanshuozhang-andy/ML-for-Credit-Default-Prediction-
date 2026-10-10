import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
TARGET_COL = "default payment next month"
FIGURE_NAME = "roc_test_lr_0_7802"


def source_hashes():
    suffixes = {".py", ".csv", ".xls", ".xlsx", ".pptx"}
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in PROJECT_DIR.iterdir()
        if path.is_file() and path.suffix in suffixes
    }


def main():
    original_hashes = source_hashes()
    original = pd.read_csv(PROJECT_DIR / "ensemble_reduced_fe_resampled_test_predictions.csv")
    logreg = pd.read_csv(PROJECT_DIR / "logreg_reduced_fe_resampled_test_predictions.csv")
    combined = original.merge(
        logreg[["ID", TARGET_COL, "Predicted_Probability"]],
        on="ID",
        how="left",
        sort=False,
        suffixes=("", "_standalone_lr"),
        validate="one_to_one",
        indicator=True,
    )
    assert len(combined) == len(original) == len(logreg) == 6000
    assert combined["_merge"].eq("both").all()
    assert np.array_equal(combined["ID"], original["ID"])
    assert np.array_equal(combined[TARGET_COL], combined[TARGET_COL + "_standalone_lr"])

    curves = [
        ("MLP", "MLP_Probability", 0.7935),
        ("Logistic Regression", "Predicted_Probability_standalone_lr", 0.7802),
        ("SVM", "SVM_Probability", 0.7785),
        ("Equal Ensemble", "Equal_Weight_Ensemble_Probability", 0.7916),
        ("Tuned Ensemble", "Predicted_Probability", 0.7908),
    ]
    y_true = combined[TARGET_COL].to_numpy()
    plt.rcParams["pdf.fonttype"] = 42
    figure, axes = plt.subplots(figsize=(8, 6))
    metrics = {}
    for label, column, expected_auc in curves:
        probability = combined[column].to_numpy()
        assert np.isfinite(probability).all()
        assert ((probability >= 0) & (probability <= 1)).all()
        score = roc_auc_score(y_true, probability)
        fpr, tpr, _ = roc_curve(y_true, probability)
        assert f"{score:.4f}" == f"{expected_auc:.4f}"
        assert np.isclose(auc(fpr, tpr), score, rtol=0, atol=1e-12)
        linewidth = 2.5 if label == "Tuned Ensemble" else 1.5
        axes.plot(fpr, tpr, linewidth=linewidth, label=f"{label}, AUC = {score:.4f}")
        metrics[label] = float(score)

    axes.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    axes.set_xlabel("False Positive Rate")
    axes.set_ylabel("True Positive Rate")
    axes.set_title("ROC Curves on Test Set")
    axes.legend()
    axes.grid(True)
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / f"{FIGURE_NAME}.png", dpi=600)
    figure.savefig(OUTPUT_DIR / f"{FIGURE_NAME}.pdf")
    plt.close(figure)

    assert source_hashes() == original_hashes, "An original project file changed."
    verification = {
        "samples": len(combined),
        "auc": metrics,
        "original_files_unchanged": True,
        "source_sha256": original_hashes,
        "ensemble_note": "Original ensemble probabilities are preserved; no retuning was performed.",
    }
    (PROJECT_DIR / "work" / "roc_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in verification.items() if key != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
