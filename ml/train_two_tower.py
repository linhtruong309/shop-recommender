"""Train the Two-Tower model, evaluate on test, export embeddings for serving.

Usage:
    python ml/train_two_tower.py --data-dir data/processed
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline import top_items
from data import ground_truth, load_split
from metrics import evaluate
from two_tower import (
    ItemIndex, TwoTower, build_training_samples, build_user_sequences, in_batch_loss, pad_left,
)


@torch.no_grad()
def all_item_vectors(model, n_items, batch=65536):
    model.eval()
    idx = torch.arange(n_items)
    return torch.cat([model.item_vectors(idx[i:i + batch]) for i in range(0, n_items, batch)])


@torch.no_grad()
def recommend(model, histories, item_vecs, k, batch=2048):
    """Top-k item indices per history row (padding item excluded)."""
    model.eval()
    item_vecs = item_vecs.clone()
    item_vecs[0] = 0
    out = []
    for i in range(0, len(histories), batch):
        scores = model.user_vectors(histories[i:i + batch]) @ item_vecs.T
        scores[:, 0] = float("-inf")
        out.append(scores.topk(k, dim=1).indices)
    return torch.cat(out)


def export_artifacts(out_dir, index, item_vecs, user_ids, user_vecs, popular_ids):
    """Raw binaries for the C++ service: ids as int64, vectors as float32 row-major."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index.item_ids.astype(np.int64).tofile(out_dir / "item_ids.bin")
    item_vecs[1:].numpy().astype(np.float32).tofile(out_dir / "item_vecs.bin")
    np.asarray(user_ids, dtype=np.int64).tofile(out_dir / "user_ids.bin")
    user_vecs.numpy().astype(np.float32).tofile(out_dir / "user_vecs.bin")
    np.asarray(popular_ids, dtype=np.int64).tofile(out_dir / "popular_ids.bin")
    meta = {"dim": int(item_vecs.shape[1]), "n_items": len(index.item_ids), "n_users": len(user_ids)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--max-targets-per-user", type=int, default=50)
    parser.add_argument("--no-logq", action="store_true", help="Disable popularity correction")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--recent-days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", default="ml/results/two_tower.json", type=Path)
    parser.add_argument("--artifacts", default="ml/artifacts", type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data
    train, test, items = load_split(args.data_dir)
    index = ItemIndex(items)
    sequences = build_user_sequences(train, index)
    histories, targets = build_training_samples(
        sequences, args.max_history, args.max_targets_per_user, args.seed)
    print(f"Items: {index.n_items - 1:,}  users: {len(sequences):,}  training samples: {len(targets):,}")

    # Popularity correction
    log_q = None
    if not args.no_logq:
        counts = torch.bincount(targets, minlength=index.n_items).float() + 1
        log_q = torch.log(counts / counts.sum())

    # Training
    model = TwoTower(index, dim=args.dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total, steps = time.time(), 0.0, 0
        for batch in torch.randperm(len(targets)).split(args.batch_size):
            h, t = histories[batch], targets[batch]
            loss = in_batch_loss(model.user_vectors(h), model.item_vectors(t), t, args.temperature, log_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, steps = total + loss.item(), steps + 1
        print(f"Epoch {epoch}/{args.epochs}  loss={total / steps:.4f}  ({time.time() - start:.0f}s)")

    # Recommendations for known visitors that appear in test
    item_vecs = all_item_vectors(model, index.n_items)
    eval_ids = [v for v in ground_truth(test, known_only=True)]
    eval_hist = torch.tensor(np.stack([pad_left(sequences[v], args.max_history) for v in eval_ids]))
    top_idx = recommend(model, eval_hist, item_vecs, args.k).numpy()
    model_recs = {v: index.item_ids[row - 1].tolist() for v, row in zip(eval_ids, top_idx)}

    # Hybrid: model for known visitors, recent popularity for new visitors
    recent_since = train["last_interaction_at"].max() - pd.Timedelta(days=args.recent_days)
    popular = top_items(train, args.k, since=recent_since)

    results = {}
    for audience, known_only in [("known_visitors", True), ("all_visitors", False)]:
        truth = ground_truth(test, known_only=known_only)
        recs = {v: model_recs.get(v, popular) for v in truth}
        results[f"two_tower/{audience}"] = evaluate(recs, truth, args.k)

    for name, metrics in results.items():
        scores = "  ".join(f"{m}={v:.4f}" for m, v in metrics.items() if m != "n_users")
        print(f"{name:<30} {scores}  users={metrics['n_users']:,}")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    args.results.write_text(json.dumps({"config": config, "results": results}, indent=2))
    print(f"Saved -> {args.results}")

    # Serving artifacts: vectors for every visitor seen in train
    known_ids = list(sequences.keys())
    with torch.no_grad():
        user_vecs = torch.cat([
            model.user_vectors(torch.tensor(np.stack(
                [pad_left(sequences[v], args.max_history) for v in known_ids[i:i + 65536]])))
            for i in range(0, len(known_ids), 65536)
        ])
    export_artifacts(args.artifacts, index, item_vecs, known_ids, user_vecs, popular)
    torch.save(model.state_dict(), args.artifacts / "two_tower.pt")
    print(f"Artifacts -> {args.artifacts}")


if __name__ == "__main__":
    main()
