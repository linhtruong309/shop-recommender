Recommendation models and evaluation.

| File | Purpose |
|---|---|
| `export_parquet.py` | Export dbt marts from Snowflake to Parquet |
| `data.py` | Load Parquet files, build ground truth per visitor |
| `metrics.py` | Recall@K, NDCG@K, HitRate@K |
| `baseline.py` | Popularity baselines (all-time, last N days) |
| `two_tower.py` | Two-Tower model, training samples, in-batch loss with logQ correction |
| `train_two_tower.py` | Train, evaluate, export embeddings to `ml/artifacts/` |
| `tests/` | Unit tests for metrics and model |

Results are written to `ml/results/`.
