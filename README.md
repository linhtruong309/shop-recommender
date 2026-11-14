# shop-recommender

E-commerce product recommender built on the RetailRocket dataset.
Stack: Snowflake + dbt (data pipeline) → PyTorch (Two-Tower model) → C++ (REST inference service).

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

## Roadmap

- [x] Snowflake setup, raw data ingestion, dbt staging/intermediate/marts + tests
- [x] Time-based train/test split, export to Parquet
- [ ] Popularity baseline and Recall@10 evaluation
- [ ] Two-Tower model in PyTorch
- [ ] C++ REST inference service
- [ ] Latency benchmark, results in README
