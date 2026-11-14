"""Load exported Parquet files and build evaluation targets."""
from pathlib import Path

import pandas as pd


def load_split(data_dir="data/processed"):
    """Return train, test and items DataFrames."""
    data_dir = Path(data_dir)
    train = pd.read_parquet(data_dir / "train_interactions.parquet")
    test = pd.read_parquet(data_dir / "test_interactions.parquet")
    items = pd.read_parquet(data_dir / "dim_items.parquet")
    return train, test, items


def ground_truth(test, known_only=True):
    """Map visitor_id -> set of item_ids the visitor interacted with in test."""
    if known_only:
        test = test[test["is_known_visitor"]]
    return test.groupby("visitor_id")["item_id"].agg(set).to_dict()


def seen_items(train):
    """Map visitor_id -> set of item_ids the visitor interacted with in train."""
    return train.groupby("visitor_id")["item_id"].agg(set).to_dict()
