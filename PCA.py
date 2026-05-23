# ============================================================
# PCA Analysis for Credit Card Default Dataset
# Raw Features + One-Hot Encoding Version
# No Feature Engineering
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA


# ============================================================
# 1. Read Excel File
# ============================================================

file_path = "default of credit card clients.xls"

def read_sheet(sheet_name):
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=0,
        skiprows=[1],
        engine="xlrd"
    )
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    return df


train_df = read_sheet("Training(Non-resampling)")
val_df = read_sheet("Validation")
test_df = read_sheet("Test")

print("Training shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)


# ============================================================
# 2. Define Basic Columns
# ============================================================

target_col = "default payment next month"
id_col = "ID"


# ============================================================
# 3. Combine Dataset for PCA Visualization
# ============================================================

all_df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)

print("Combined raw data shape:", all_df.shape)


# ============================================================
# 4. Use Original Features Only
# ============================================================

feature_cols = [
    col for col in all_df.columns
    if col not in [id_col, target_col]
]

X = all_df[feature_cols]
y = all_df[target_col].astype(int)

print("Number of original features:", len(feature_cols))
print("Original feature columns:")
print(feature_cols)


# ============================================================
# 5. Define Categorical and Numerical Columns
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


categorical_cols = [
    col for col in categorical_cols
    if col in X.columns
]

numeric_cols = [
    col for col in feature_cols
    if col not in categorical_cols
]

print("\nCategorical columns:")
print(categorical_cols)

print("\nNumerical columns:")
print(numeric_cols)


# ============================================================
# 6. Basic Missing Value Handling
# ============================================================

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)


# ============================================================
# 7. One-Hot Encoding + Standardization
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

X_processed = preprocessor.fit_transform(X)
X_processed = np.asarray(X_processed, dtype=np.float32)

print("\nProcessed data shape:", X_processed.shape)


# ============================================================
# 8. PCA
# ============================================================

pca = PCA(n_components=2, random_state=42)

X_pca = pca.fit_transform(X_processed)

print("\nExplained variance ratio:")
print("PC1:", round(pca.explained_variance_ratio_[0], 4))
print("PC2:", round(pca.explained_variance_ratio_[1], 4))
print("Total:", round(pca.explained_variance_ratio_.sum(), 4))


# ============================================================
# 9. Create PCA DataFrame
# ============================================================

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Default": y.values
})

print("\nPCA DataFrame preview:")
print(pca_df.head())


# ============================================================
# 10. Plot PCA Result
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    pca_df[pca_df["Default"] == 0]["PC1"],
    pca_df[pca_df["Default"] == 0]["PC2"],
    alpha=0.4,
    s=10,
    label="Non-default"
)

plt.scatter(
    pca_df[pca_df["Default"] == 1]["PC1"],
    pca_df[pca_df["Default"] == 1]["PC2"],
    alpha=0.4,
    s=10,
    label="Default"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization Using Raw Features with One-Hot Encoding")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()