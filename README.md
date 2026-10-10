# Credit Card Default Prediction

This repository contains a group course project comparing logistic regression, a calibrated linear SVM, a multilayer perceptron (MLP), and soft-voting ensembles for credit card default prediction.

## Dataset and evaluation

The data originate from the [Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), provided by I-Cheng Yeh through the UCI Machine Learning Repository.

Use the workbook supplied in this repository, which contains the project's predefined split:

- Training: 17,999 records
- Validation: 6,001 records
- Test: 6,000 records

Random oversampling was applied only to the training set, producing 28,036 records, to deal with class imbalance. Validation and test sets remained their original distributions.

## Main files

| File | Purpose |
| --- | --- |
| `oversample.py` | Generate the randomly oversampled training workbook |
| `Logistic_Regression.py` | Train and evaluate standalone logistic regression |
| `SVM.py` | Train and evaluate the calibrated linear SVM |
| `MLP.py` | Train and evaluate the neural network |
| `Ensemble.py` | Train ensemble components and select voting weights |
| `PCA.py` | Explore the data using PCA visualisation |
| `calculate_ensemble_area_ratio.py` | Calculate Area Ratios from the saved ensemble prediction file |
| `work/redraw_roc.py` | Reproduce the combined ROC figure from saved predictions |

PCA is used for exploratory analysis and is not part of the supervised models' input pipeline.

## Environment and usage

The current local environment uses Python 3.13.7. From the repository root, create an environment and install the following package versions:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.3.3 pandas==2.3.3 scikit-learn==1.7.2 matplotlib==3.10.8 torch==2.9.1 openpyxl==3.1.5 xlrd==2.0.2
```

To redraw the ROC figure without retraining:

```bash
python work/redraw_roc.py
```

This uses standalone LR probabilities (ROC-AUC 0.7802) and the remaining curves from the ensemble prediction file. The PNG and PDF are saved in `outputs/`.

To rerun the model experiments:

```bash
python oversample.py
python Logistic_Regression.py
python SVM.py
python MLP.py
python Ensemble.py
```

Running the model scripts retrains the models and overwrites their corresponding prediction CSVs. The ensemble selects its LR and SVM hyperparameters separately using validation balanced accuracy.

## Saved test results

| Model | F1-score | Balanced accuracy | ROC-AUC |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.5483 | 0.7057 | 0.7802 |
| SVM | 0.5464 | 0.7073 | 0.7785 |
| MLP | 0.5375 | 0.6978 | 0.7935 |
| Tuned ensemble | 0.5470 | 0.7123 | 0.7908 |

## Contributions

- **Hanshuo Zhang:** feature engineering, neural networks, ensemble learning, and the report's conclusion.
- **Yu Chen:** logistic regression, abstract, introduction, and motivation.
- **Zhenghao Zeng:** data resampling, PCA, and SVM.
