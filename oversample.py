from pathlib import Path

import pandas as pd


# 1. Read source workbook from the current project folder.
PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = PROJECT_DIR / "default of credit card clients.xls"
OUTPUT_PATH = PROJECT_DIR / "training_random_oversampled.xlsx"

# Target variable.
TARGET_COL = "default payment next month"


# 2. Clean worksheet data.
def clean_sheet(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)

    # The first row in the split sheets contains X1, X2, ..., Y labels.
    if str(df.iloc[0, -1]) == "Y":
        df = df.iloc[1:].copy()

    # Convert values to numeric type.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove fully empty rows.
    df = df.dropna(how="all").reset_index(drop=True)

    # Convert complete numeric columns to integer type.
    for col in df.columns:
        if df[col].isna().sum() == 0:
            df[col] = df[col].astype(int)

    return df


# 3. Read the original non-resampled training set.
train = clean_sheet(SOURCE_PATH, "Training(Non-resampling)")

print("Original training class distribution:")
print(train[TARGET_COL].value_counts())


# 4. Split majority and minority classes.
majority_class = train[TARGET_COL].value_counts().idxmax()
minority_class = train[TARGET_COL].value_counts().idxmin()

majority = train[train[TARGET_COL] == majority_class]
minority = train[train[TARGET_COL] == minority_class]

print("\nMajority class:", majority_class)
print("Minority class:", minority_class)


# 5. Random oversampling.
# Sample the minority class with replacement until it matches the majority class.
minority_oversampled = minority.sample(
    n=len(majority),
    replace=True,
    random_state=42,
)

# Combine the majority class and oversampled minority class.
train_oversampled = pd.concat(
    [majority, minority_oversampled],
    axis=0,
)

# Shuffle the final training set.
train_oversampled = train_oversampled.sample(
    frac=1,
    random_state=42,
).reset_index(drop=True)


# 6. Check class distribution after oversampling.
print("\nAfter Random Oversampling:")
print(train_oversampled[TARGET_COL].value_counts())


# 7. Save the oversampled training data.
train_oversampled.to_excel(OUTPUT_PATH, index=False)

print("\nFile saved successfully:")
print(OUTPUT_PATH)
