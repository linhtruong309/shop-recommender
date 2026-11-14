"""Export dbt mart tables from Snowflake to local Parquet files.

Usage:
    python ml/export_parquet.py --out-dir data/processed
"""
import argparse
import os
from pathlib import Path

import snowflake.connector
from dotenv import load_dotenv

# Snowflake table -> output file
TABLES = {
    "MARTS.TRAIN_INTERACTIONS": "train_interactions.parquet",
    "MARTS.TEST_INTERACTIONS": "test_interactions.parquet",
    "MARTS.DIM_ITEMS": "dim_items.parquet",
}


def connect():
    load_dotenv()
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        role=os.environ.get("SNOWFLAKE_ROLE", "RECO_ROLE"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "RECO_WH"),
        database=os.environ.get("SNOWFLAKE_DATABASE", "RECO"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/processed", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    conn = connect()
    cur = conn.cursor()
    try:
        for table, filename in TABLES.items():
            print(f"Exporting {table} ...")
            cur.execute(f"SELECT * FROM {table}")
            df = cur.fetch_pandas_all()
            df.columns = [c.lower() for c in df.columns]
            path = args.out_dir / filename
            df.to_parquet(path, index=False)
            print(f"    {len(df):,} rows -> {path}")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
