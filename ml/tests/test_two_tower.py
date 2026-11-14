import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from two_tower import (  # noqa: E402
    ItemIndex, TwoTower, build_training_samples, in_batch_loss, pad_left,
)


def make_index():
    items = pd.DataFrame({
        "item_id": [10, 20, 30],
        "category_id": [1, 1, None],
        "parent_category_id": [5, 5, None],
    })
    return ItemIndex(items)


def test_item_index_reserves_zero_for_padding():
    index = make_index()
    assert index.encode_items([10, 30, 999]).tolist() == [1, 3, 0]
    assert index.category[3] == 0  # missing category -> 0


def test_pad_left_keeps_most_recent():
    assert pad_left(np.array([1, 2, 3]), 2).tolist() == [2, 3]
    assert pad_left(np.array([7]), 3).tolist() == [0, 0, 7]


def test_training_samples_leave_one_out():
    histories, targets = build_training_samples({"a": np.array([1, 2]), "b": np.array([3])}, 3, 50)
    assert len(targets) == 2  # user b skipped: single item
    for h, t in zip(histories, targets):
        assert t.item() not in h.tolist()


def test_vectors_are_normalized():
    model = TwoTower(make_index(), dim=8)
    item_vecs = model.item_vectors(torch.tensor([1, 2, 3]))
    user_vecs = model.user_vectors(torch.tensor([[0, 1, 2]]))
    assert torch.allclose(item_vecs.norm(dim=1), torch.ones(3), atol=1e-5)
    assert torch.allclose(user_vecs.norm(dim=1), torch.ones(1), atol=1e-5)


def test_loss_is_finite_with_duplicate_targets():
    model = TwoTower(make_index(), dim=8)
    targets = torch.tensor([1, 1, 2])
    histories = torch.tensor([[0, 2], [0, 3], [1, 3]])
    loss = in_batch_loss(model.user_vectors(histories), model.item_vectors(targets), targets, 0.1)
    assert torch.isfinite(loss)
