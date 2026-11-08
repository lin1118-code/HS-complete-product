import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score

def eligible_subset(df, min_class_count=5, min_unique_groups=4):
    counts = df["HS Code"].value_counts()
    eligible = counts[counts >= min_class_count].index
    sub = df[df["HS Code"].isin(eligible)].copy()

    group_counts = sub.groupby("HS Code")["duplicate_group"].nunique()
    eligible_after_group_check = group_counts[group_counts >= min_unique_groups].index
    sub = sub[sub["HS Code"].isin(eligible_after_group_check)].reset_index(drop=True)

    return sub

def evaluate_dummy_cv(df, n_splits=4, random_state=42):
    sub = eligible_subset(df)
    X = sub["model_text"]
    y = sub["HS Code"]
    groups = sub["duplicate_group"]

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X.iloc[train_idx].to_frame(), y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx].to_frame())

        rows.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "accuracy": accuracy_score(y.iloc[test_idx], pred),
            "macro_f1": f1_score(y.iloc[test_idx], pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y.iloc[test_idx], pred, average="weighted", zero_division=0),
        })

    result = pd.DataFrame(rows)
    summary = result[["accuracy", "macro_f1", "weighted_f1"]].agg(["mean", "std"])
    return sub, result, summary
