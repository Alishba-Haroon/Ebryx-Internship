"""Dataset loading + preprocessing (Iris, Breast Cancer, Titanic)."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(name: str):
    """Returns X (DataFrame/ndarray), y (ndarray of ints), class_names, task."""
    if name == "iris":                       # multi-class (3 classes)
        d = load_iris()
        return d.data, d.target, list(d.target_names), "multiclass"
    if name == "cancer":                     # binary (malignant / benign)
        d = load_breast_cancer()
        return d.data, d.target, list(d.target_names), "binary"
    if name == "titanic":                    # binary (survived)
        df = fetch_openml("titanic", version=1, as_frame=True).frame
        df = df[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "survived"]].copy()
        df["age"] = df["age"].fillna(df["age"].median())
        df["fare"] = df["fare"].fillna(df["fare"].median())
        df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
        df["pclass"] = df["pclass"].astype(int)
        df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True, dtype=float)
        y = df.pop("survived").astype(int).values
        return df.values.astype(float), y, ["died", "survived"], "binary"
    raise ValueError(f"Unknown dataset: {name}")


def make_splits(X, y, seed=42):
    """60/20/20 stratified train/val/test split + standardization (fit on train only!)."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    sc = StandardScaler().fit(X_tr)
    return sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te), y_tr, y_va, y_te
