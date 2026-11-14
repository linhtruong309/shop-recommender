"""Shrink serving artifacts for small hosts: keep users with enough history, pack as tar.gz.

Usage:
    python ml/slim_artifacts.py --min-interactions 2
"""
import argparse
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="ml/artifacts", type=Path)
    parser.add_argument("--data-dir", default="data/processed", type=Path)
    parser.add_argument("--min-interactions", type=int, default=2)
    parser.add_argument("--out-dir", default="ml/artifacts_slim", type=Path)
    args = parser.parse_args()

    meta = json.loads((args.artifacts / "meta.json").read_text())
    dim = meta["dim"]
    user_ids = np.fromfile(args.artifacts / "user_ids.bin", dtype=np.int64)
    user_vecs = np.fromfile(args.artifacts / "user_vecs.bin", dtype=np.float32).reshape(-1, dim)

    # Keep users with at least N train interactions
    train = pd.read_parquet(args.data_dir / "train_interactions.parquet", columns=["visitor_id"])
    counts = train["visitor_id"].value_counts()
    active = counts[counts >= args.min_interactions].index.to_numpy()
    keep = np.isin(user_ids, active)

    # Write slim artifacts (items and popular list unchanged)
    out = args.out_dir / "artifacts"
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    out.mkdir(parents=True)
    for name in ["item_ids.bin", "item_vecs.bin", "popular_ids.bin"]:
        shutil.copy(args.artifacts / name, out / name)
    user_ids[keep].tofile(out / "user_ids.bin")
    user_vecs[keep].tofile(out / "user_vecs.bin")
    meta_out = meta | {"n_users": int(keep.sum()), "min_interactions": args.min_interactions}
    (out / "meta.json").write_text(json.dumps(meta_out, indent=2))

    # Pack for upload as a GitHub Release asset
    archive = args.out_dir / "artifacts.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for f in sorted(out.iterdir()):
            tar.add(f, arcname=f.name)

    size_mb = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"Users kept: {keep.sum():,} of {len(user_ids):,} (>= {args.min_interactions} interactions)")
    print(f"Artifacts: {size_mb:.0f} MB in memory, archive {archive.stat().st_size / 1e6:.0f} MB -> {archive}")


if __name__ == "__main__":
    main()
