import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

try:
    from . import config
except ImportError:
    import config


def load_data(file_path=config.DATA_FILE):
    print(f"[1] Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"    Loaded {len(df):,} transactions and {df.shape[1]} columns.")
    return df


def basic_checks(df):
    print("\n[2] Basic data-quality checks")
    missing = df.isnull().sum().sum()
    print(f"    Total missing values: {missing}")

    n_fraud = int(df[config.TARGET_COLUMN].sum())
    n_total = len(df)
    fraud_pct = 100 * n_fraud / n_total
    print(f"    Fraud transactions : {n_fraud:,} ({fraud_pct:.3f}%)")
    print(f"    Normal transactions: {n_total - n_fraud:,} ({100 - fraud_pct:.3f}%)")
    print(f"    Imbalance ratio    : about 1 fraud for every "
          f"{(n_total - n_fraud) // max(n_fraud, 1):,} normal transactions")


def scale_features(df, scaler=None):
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df[config.COLUMNS_TO_SCALE] = scaler.fit_transform(df[config.COLUMNS_TO_SCALE])
    else:
        df[config.COLUMNS_TO_SCALE] = scaler.transform(df[config.COLUMNS_TO_SCALE])
    return df, scaler


def split_data(df):
    print("\n[3] Splitting into train and test sets (stratified)")
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    print(f"    Training set: {len(X_train):,} rows ({int(y_train.sum())} frauds)")
    print(f"    Test set    : {len(X_test):,} rows ({int(y_test.sum())} frauds)")
    return X_train, X_test, y_train, y_test


def balance_with_smote(X_train, y_train):
    print("\n[4] Balancing the training set with SMOTE")
    print(f"    Before: {len(y_train):,} rows ({int(y_train.sum())} frauds)")

    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"    After : {len(y_res):,} rows ({int(y_res.sum())} frauds)")
    return X_res, y_res


def preprocess(file_path=config.DATA_FILE, use_smote=True):
    df = load_data(file_path)
    basic_checks(df)
    df, scaler = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    if use_smote:
        X_train, y_train = balance_with_smote(X_train, y_train)

    feature_names = list(X_test.columns)
    return X_train, X_test, y_train, y_test, scaler, feature_names


if __name__ == "__main__":
    preprocess()
