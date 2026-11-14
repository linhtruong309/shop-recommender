"""Popularity baselines: recommend the same top-k items to every visitor.

Usage:
    python ml/baseline.py --data-dir data/processed
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from data import ground_truth, load_split
from metrics import evaluate


def top_items(train, k, since=None):
    """Top-k items by total interaction score, optionally only after a timestamp."""
    if since is not None:
        train = train[train["last_interaction_at"] >= since]
    return (
        train.groupby("item_id")["interaction_score"].sum()
        .nlargest(k).index.tolist()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--recent-days", type=int, default=7)
    parser.add_argument("--out", default="ml/results/baseline.json", type=Path)
    args = parser.parse_args()

    train, test, _ = load_split(args.data_dir)
    train_end = train["last_interaction_at"].max()
    recent_since = train_end - pd.Timedelta(days=args.recent_days)

    baselines = {
        "popular_all_time": top_items(train, args.k),
        f"popular_last_{args.recent_days}d": top_items(train, args.k, since=recent_since),
    }

    results = {}
    for audience, known_only in [("known_visitors", True), ("all_visitors", False)]:
        truth = ground_truth(test, known_only=known_only)
        for name, items in baselines.items():
            recs = {visitor: items for visitor in truth}
            results[f"{name}/{audience}"] = evaluate(recs, truth, args.k)

    for name, metrics in results.items():
        scores = "  ".join(f"{m}={v:.4f}" for m, v in metrics.items() if m != "n_users")
        print(f"{name:<40} {scores}  users={metrics['n_users']:,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
