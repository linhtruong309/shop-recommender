"""NumPy latency baseline for the same top-k search as the C++ service.

Usage:
    python ml/bench_numpy.py --artifacts ml/artifacts --requests 1000
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="ml/artifacts", type=Path)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    dim = json.loads((args.artifacts / "meta.json").read_text())["dim"]
    item_vecs = np.fromfile(args.artifacts / "item_vecs.bin", dtype=np.float32).reshape(-1, dim)
    user_vecs = np.fromfile(args.artifacts / "user_vecs.bin", dtype=np.float32).reshape(-1, dim)

    rng = np.random.default_rng(42)
    rows = rng.integers(0, len(user_vecs), args.requests + 20)

    def recommend(row):
        scores = item_vecs @ user_vecs[row]
        top = np.argpartition(-scores, args.k)[:args.k]
        return top[np.argsort(-scores[top])]

    for row in rows[:20]:  # warm-up
        recommend(row)

    ms = []
    for row in rows[20:]:
        t0 = time.perf_counter()
        recommend(row)
        ms.append((time.perf_counter() - t0) * 1000)

    p50, p95, p99 = np.percentile(ms, [50, 95, 99])
    print(f"NumPy top-{args.k} over {len(item_vecs):,} items, {args.requests} requests")
    print(f"  p50 = {p50:.3f} ms\n  p95 = {p95:.3f} ms\n  p99 = {p99:.3f} ms")


if __name__ == "__main__":
    main()
