# shop-recommender

Hệ thống gợi ý sản phẩm thương mại điện tử trên bộ dữ liệu RetailRocket.
Stack: Snowflake + dbt (pipeline dữ liệu) → PyTorch (Two-Tower) → C++ (ONNX Runtime + FAISS) mở REST API.

```
RetailRocket CSV ──PUT/COPY──► Snowflake RAW
                                   │ dbt
                                   ▼
             STAGING → INTERMEDIATE (sessions, category) → MARTS (fct_user_item_interactions)
                                   │
                                   ▼
                     PyTorch Two-Tower ──export ONNX + embedding──► C++ service
                                                                     GET /recommendations
```

## Tuần 1: nạp dữ liệu và dựng dbt

1. Tạo tài khoản Snowflake (bản dùng thử miễn phí), chạy `snowflake/setup.sql`
   (nhớ thay `YOUR_USER`).
2. Tải RetailRocket từ Kaggle vào `data/` (xem `data/README.md`).
3. Nạp dữ liệu:
   ```bash
   cp .env.example .env   # điền thông tin Snowflake
   pip install -r ingestion/requirements.txt
   python ingestion/load_retailrocket.py --data-dir data
   ```
4. Chạy dbt:
   ```bash
   pip install dbt-snowflake
   cp dbt_reco/profiles.yml.example ~/.dbt/profiles.yml
   export $(grep -v '^#' .env | xargs)
   cd dbt_reco
   dbt deps && dbt build
   ```

## Các model dbt

| Model | Mô tả |
|---|---|
| `stg_events` | Sự kiện view/addtocart/transaction, đổi epoch ms sang timestamp |
| `stg_item_properties`, `stg_category_tree` | Thuộc tính và cây danh mục |
| `int_item_current_category` | Danh mục mới nhất của mỗi sản phẩm |
| `int_sessions` | Chia session theo khoảng nghỉ 30 phút (`var: session_timeout_minutes`) |
| `fct_user_item_interactions` | Bảng huấn luyện: điểm tương tác visitor–item (view=1, cart=3, mua=5) |

## Lộ trình

- [x] Tuần 1: Snowflake setup, nạp dữ liệu, dbt staging/intermediate/marts + test
- [ ] Tuần 2: Chia train/test theo thời gian (dbt), bảng đặc trưng item & user
- [ ] Tuần 3–4: Two-Tower bằng PyTorch, đo Recall@10, NDCG@10 so với baseline "phổ biến nhất"
- [ ] Tuần 5–6: Dịch vụ C++ (ONNX Runtime + FAISS + Crow), endpoint REST
- [ ] Tuần 7: Benchmark độ trễ p50/p99, so sánh với bản Python
- [ ] Tuần 8: CI (dbt build trên PR, build C++), README, demo
