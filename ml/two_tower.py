"""Two-Tower retrieval model and training data preparation."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemIndex:
    """Map raw item / category ids to contiguous indices (0 = padding / unknown)."""

    def __init__(self, items):
        self.item_ids = items["item_id"].to_numpy()
        self.item2idx = pd.Series(np.arange(1, len(items) + 1), index=self.item_ids)

        def encode(col):
            codes, _ = pd.factorize(items[col])
            return torch.tensor(codes + 1, dtype=torch.long)  # NaN -> -1 -> 0

        pad = torch.zeros(1, dtype=torch.long)
        self.category = torch.cat([pad, encode("category_id")])
        self.parent = torch.cat([pad, encode("parent_category_id")])

    @property
    def n_items(self):
        return len(self.item_ids) + 1

    def encode_items(self, item_ids):
        return self.item2idx.reindex(item_ids).fillna(0).astype(np.int64).to_numpy()


def build_user_sequences(train, index):
    """visitor_id -> item indices ordered by last interaction time."""
    df = train[["visitor_id", "item_id", "last_interaction_at"]].copy()
    df["idx"] = index.encode_items(df["item_id"])
    df = df.sort_values(["visitor_id", "last_interaction_at"])
    visitors, starts = np.unique(df["visitor_id"].to_numpy(), return_index=True)
    return dict(zip(visitors.tolist(), np.split(df["idx"].to_numpy(), starts[1:])))


def pad_left(seq, length):
    """Keep the last `length` items, left-pad with 0."""
    seq = seq[-length:]
    out = np.zeros(length, dtype=np.int64)
    if len(seq):
        out[-len(seq):] = seq
    return out


def build_training_samples(sequences, max_history, max_targets_per_user, seed=0):
    """Leave-one-out samples: (history without target, target) for users with >= 2 items."""
    rng = np.random.default_rng(seed)
    histories, targets = [], []
    for seq in sequences.values():
        if len(seq) < 2:
            continue
        positions = np.arange(len(seq))
        if len(positions) > max_targets_per_user:
            positions = rng.choice(positions, max_targets_per_user, replace=False)
        for pos in positions:
            histories.append(pad_left(np.delete(seq, pos), max_history))
            targets.append(seq[pos])
    return (
        torch.tensor(np.stack(histories), dtype=torch.long),
        torch.tensor(np.array(targets), dtype=torch.long),
    )


class TwoTower(nn.Module):
    """User tower: mean of history item embeddings -> MLP.
    Item tower: item id + category + parent category embeddings -> MLP."""

    def __init__(self, index, dim=64):
        super().__init__()
        n_cat = int(index.category.max()) + 1
        n_parent = int(index.parent.max()) + 1
        self.register_buffer("category", index.category)
        self.register_buffer("parent", index.parent)

        self.item_emb = nn.Embedding(index.n_items, dim, padding_idx=0)
        self.cat_emb = nn.Embedding(n_cat, dim // 2, padding_idx=0)
        self.parent_emb = nn.Embedding(n_parent, dim // 2, padding_idx=0)

        self.item_mlp = nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))
        self.user_mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))

    def item_vectors(self, idx):
        x = torch.cat([
            self.item_emb(idx),
            self.cat_emb(self.category[idx]),
            self.parent_emb(self.parent[idx]),
        ], dim=-1)
        return F.normalize(self.item_mlp(x), dim=-1)

    def user_vectors(self, history):
        mask = (history > 0).unsqueeze(-1).float()
        pooled = (self.item_emb(history) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(self.user_mlp(pooled), dim=-1)


def in_batch_loss(user_vecs, item_vecs, targets, temperature, log_q=None):
    """Softmax over in-batch items; masks duplicate targets; optional logQ popularity correction."""
    logits = user_vecs @ item_vecs.T / temperature
    if log_q is not None:
        logits = logits - log_q[targets].unsqueeze(0)
    same = targets.unsqueeze(0) == targets.unsqueeze(1)
    same.fill_diagonal_(False)
    logits = logits.masked_fill(same, float("-inf"))
    labels = torch.arange(len(targets), device=logits.device)
    return F.cross_entropy(logits, labels)
