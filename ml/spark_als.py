"""Spark MLlib ALS baseline (implicit feedback), evaluated with the same metrics.

Usage:
    python ml/spark_als.py --data-dir data/processed
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from baseline import top_items
from data import ground_truth, load_split
from metrics import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--reg", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0, help="Implicit confidence weight")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--recent-days", type=int, default=7)
    parser.add_argument("--driver-memory", default="4g")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", default="ml/results/spark_als.json", type=Path)
    args = parser.parse_args()

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("shop-recommender-als")
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoints")

    # Data: Spark reads Parquet directly; ALS needs integer user / item ids
    data_dir = Path(args.data_dir)
    train_sdf = (
        spark.read.parquet(str(data_dir / "train_interactions.parquet"))
        .select(
            F.col("visitor_id").cast("int").alias("user"),
            F.col("item_id").cast("int").alias("item"),
            F.col("interaction_score").cast("float").alias("rating"),
        )
    )
    print(f"Training rows: {train_sdf.count():,}")

    # Train
    als = ALS(
        rank=args.rank, maxIter=args.max_iter, regParam=args.reg, alpha=args.alpha,
        implicitPrefs=True, userCol="user", itemCol="item", ratingCol="rating",
        coldStartStrategy="drop", seed=args.seed, checkpointInterval=5,
    )
    start = time.time()
    model = als.fit(train_sdf)
    print(f"ALS trained in {time.time() - start:.0f}s")

    # Recommend for known visitors that appear in test
    train, test, _ = load_split(args.data_dir)
    truth_known = ground_truth(test, known_only=True)
    users = spark.createDataFrame([(int(v),) for v in truth_known], "user int")
    recs_sdf = model.recommendForUserSubset(users, args.k)
    model_recs = {
        row["user"]: [r["item"] for r in row["recommendations"]]
        for row in recs_sdf.collect()
    }

    # Hybrid: ALS for known visitors, recent popularity for new visitors
    recent_since = train["last_interaction_at"].max() - pd.Timedelta(days=args.recent_days)
    popular = top_items(train, args.k, since=recent_since)

    results = {}
    for audience, known_only in [("known_visitors", True), ("all_visitors", False)]:
        truth = ground_truth(test, known_only=known_only)
        recs = {v: model_recs.get(v, popular) for v in truth}
        results[f"spark_als/{audience}"] = evaluate(recs, truth, args.k)

    for name, metrics in results.items():
        scores = "  ".join(f"{m}={v:.4f}" for m, v in metrics.items() if m != "n_users")
        print(f"{name:<30} {scores}  users={metrics['n_users']:,}")

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps({"config": config, "results": results}, indent=2))
    print(f"Saved -> {args.results}")
    spark.stop()


if __name__ == "__main__":
    main()
