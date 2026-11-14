"""Nạp bộ dữ liệu RetailRocket từ thư mục data/ vào Snowflake (schema RAW).

Cách dùng:
    python ingestion/load_retailrocket.py --data-dir data
"""
import argparse
import os
from pathlib import Path

import snowflake.connector
from dotenv import load_dotenv

# file -> bảng đích
FILES = {
    "events.csv": "EVENTS",
    "item_properties_part1.csv": "ITEM_PROPERTIES",
    "item_properties_part2.csv": "ITEM_PROPERTIES",
    "category_tree.csv": "CATEGORY_TREE",
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
        schema="RAW",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--truncate", action="store_true", help="Xóa dữ liệu cũ trước khi nạp")
    args = parser.parse_args()

    missing = [f for f in FILES if not (args.data_dir / f).exists()]
    if missing:
        raise SystemExit(f"Thiếu file trong {args.data_dir}: {', '.join(missing)}")

    conn = connect()
    cur = conn.cursor()
    try:
        if args.truncate:
            for table in set(FILES.values()):
                cur.execute(f"TRUNCATE TABLE {table}")

        for filename in FILES:
            path = (args.data_dir / filename).resolve()
            print(f"Upload {filename} ...")
            # PUT tự nén gzip; OVERWRITE để chạy lại được
            cur.execute(f"PUT 'file://{path.as_posix()}' @RETAILROCKET_STAGE AUTO_COMPRESS=TRUE OVERWRITE=TRUE")

        for filename, table in FILES.items():
            print(f"COPY {filename} -> {table} ...")
            cur.execute(
                f"COPY INTO {table} FROM @RETAILROCKET_STAGE/{filename}.gz "
                f"FILE_FORMAT = (FORMAT_NAME = CSV_HEADER) ON_ERROR = 'ABORT_STATEMENT'"
            )
            for row in cur.fetchall():
                print("   ", row[:4])

        for table in sorted(set(FILES.values())):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            print(f"{table}: {cur.fetchone()[0]:,} dòng")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
