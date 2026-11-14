# shop-recommender

E-commerce product recommender built on the RetailRocket dataset.
Stack: Snowflake + dbt (data pipeline) → PyTorch Two-Tower and Spark ALS (models) → C++ (REST inference service, deployed with Docker on Render).

```
RetailRocket CSV ──PUT/COPY──► Snowflake RAW
                                   │ dbt
                                   ▼
             STAGING → INTERMEDIATE (sessions, categories) → MARTS (fct_user_item_interactions)
                                   │
                                   ▼
                     PyTorch Two-Tower ──export embeddings──► C++ service
                                                               GET /recommendations
```

## Live demo

The C++ API is deployed on Render (free plan; the first request after 15 idle minutes takes ~1 minute to wake up):

- Health: https://shop-recommender-api.onrender.com/health
- Personalized (visitor with history): https://shop-recommender-api.onrender.com/recommendations?user_id=2&k=5
- Cold start (unknown visitor → popular items): https://shop-recommender-api.onrender.com/recommendations?user_id=999999999&k=5

The deployed build serves 260,764 visitors with 2+ interactions to fit the 512 MB free instance; embeddings are downloaded from the [v1.0.0 release](https://github.com/linhtruong309/shop-recommender/releases/tag/v1.0.0) at build time.

## Setup

1. Create a Snowflake account and run `snowflake/setup.sql` as ACCOUNTADMIN
   (set the `GRANT ROLE ... TO USER` line to your Snowflake user).
2. Download RetailRocket from Kaggle into `data/` (see `data/README.md`).
3. Load the raw data:
   ```bash
   cp .env.example .env   # fill in your Snowflake credentials
   pip install -r ingestion/requirements.txt
   python ingestion/load_retailrocket.py --data-dir data
   ```
4. Run dbt:
   ```bash
   pip install dbt-snowflake
   cp dbt_reco/profiles.yml.example ~/.dbt/profiles.yml
   set -a; source .env; set +a
   cd dbt_reco
   dbt deps && dbt build
   ```
5. Export marts to Parquet:
   ```bash
   pip install -r ml/requirements.txt
   python ml/export_parquet.py --out-dir data/processed
   ```
6. Run baselines and tests:
   ```bash
   python ml/baseline.py --data-dir data/processed
   python ml/train_two_tower.py --data-dir data/processed
   python ml/spark_als.py --data-dir data/processed   # needs Java 17+
   pytest ml/tests
   ```
7. Build and run the C++ service:
   ```bash
   cd inference_cpp
   cmake -S . -B build && cmake --build build -j
   ctest --test-dir build --output-on-failure
   ./build/reco_server ../ml/artifacts 8080
   curl "localhost:8080/recommendations?user_id=1&k=10"
   ```

## dbt models

| Model | Description |
|---|---|
| `stg_events` | View / add-to-cart / transaction events, epoch ms converted to timestamps |
| `stg_item_properties`, `stg_category_tree` | Item properties and category hierarchy |
| `int_item_current_category` | Latest category per item |
| `int_sessions` | Sessions split on 30-minute inactivity (`var: session_timeout_minutes`) |
| `fct_user_item_interactions` | Full-period visitor–item interaction score (view=1, cart=3, purchase=5) |
| `int_events_split` | Events labeled `train` / `test`; last 14 days = test (`var: test_window_days`) |
| `train_interactions` | Visitor–item pairs before the cutoff (model training) |
| `test_interactions` | Visitor–item pairs after the cutoff, with `is_known_visitor` flag (evaluation) |
| `dim_items` | Item category, parent category and train popularity |

## Results

Time-based split: last 14 days are the test set. Metrics at K = 10.

| Model | Audience | Recall@10 | NDCG@10 | HitRate@10 |
|---|---|---|---|---|
| Popularity (all time) | known visitors (14,774) | 0.0052 | 0.0035 | 0.0083 |
| Popularity (last 7 days) | known visitors | 0.0055 | 0.0041 | 0.0090 |
| Two-Tower (PyTorch) | known visitors | 0.0393 | 0.0231 | 0.0532 |
| **ALS (Spark MLlib)** | known visitors | **0.0428** | **0.0312** | **0.0583** |
| Popularity (all time) | all visitors (142,025) | 0.0081 | 0.0048 | 0.0096 |
| Two-Tower + popularity fallback | all visitors | 0.0101 | 0.0063 | 0.0128 |
| **ALS + popularity fallback** | all visitors | **0.0105** | **0.0072** | **0.0133** |

- Both learned models beat the best popularity baseline by ~7–8x for visitors with history.
- ALS (implicit feedback, rank 64, 10 iterations) edges out the untuned Two-Tower (5 epochs) on this sparse dataset; Two-Tower adds category features and serves from the C++ service.
- ~90% of test visitors are new (cold start); they fall back to recent popularity.

## Serving

The C++ service loads 235,061 item vectors and 1,280,329 user vectors into memory and scores all items per request with multi-threaded SIMD dot products. Unknown visitors fall back to recent popularity.

```
GET /health
GET /recommendations?user_id=1&k=10
→ {"user_id":1,"source":"two_tower","latency_ms":1.2,"items":[{"item_id":219512,"score":0.55}, ...]}
```

Latency of exact top-10 search over all items, 1,000 random users, MacBook Air (Apple Silicon):

| Implementation | p50 | p95 | p99 |
|---|---|---|---|
| NumPy (`ml/bench_numpy.py`) | 3.18 ms | 4.33 ms | 8.65 ms |
| **C++ (`inference_cpp/reco_bench`)** | **1.17 ms** | **1.29 ms** | **1.38 ms** |

C++ is 2.7x faster at p50 and 6.3x faster at p99, with a much tighter tail.

## Roadmap

- [x] Snowflake setup, raw data ingestion, dbt staging/intermediate/marts + tests
- [x] Time-based train/test split, export to Parquet
- [x] Popularity baseline and Recall@10 evaluation
- [x] Two-Tower model in PyTorch
- [x] Spark MLlib ALS baseline, compared with Popularity and Two-Tower
- [x] C++ REST inference service
- [x] Latency benchmark, results in README
- [x] Public deployment (Docker + Render)
