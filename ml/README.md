Recommendation models and evaluation.

| File | Purpose |
|---|---|
| `export_parquet.py` | Export dbt marts from Snowflake to Parquet |
| `data.py` | Load Parquet files, build ground truth per visitor |
| `metrics.py` | Recall@K, NDCG@K, HitRate@K |
| `baseline.py` | Popularity baselines (all-time, last N days) |
| `tests/` | Unit tests for metrics |

Results are written to `ml/results/`.
