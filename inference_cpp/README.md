C++ inference service: loads embeddings exported by `ml/train_two_tower.py`
and serves top-k recommendations over REST.

| File | Purpose |
|---|---|
| `include/recommender.hpp`, `src/recommender.cpp` | Load artifacts, multi-threaded top-k by dot product, popularity fallback |
| `src/server.cpp` | REST API (cpp-httplib) |
| `src/bench.cpp` | Latency benchmark (p50 / p95 / p99) |
| `tests/test_recommender.cpp` | Unit tests |

Build and run (from `inference_cpp/`):

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/reco_bench ../ml/artifacts 1000
./build/reco_server ../ml/artifacts 8080
```

Endpoints:

- `GET /health`
- `GET /recommendations?user_id=<id>&k=10` → `{"user_id", "source": "two_tower" | "popular", "latency_ms", "items": [{"item_id", "score"}]}`

Environment variables (used when deployed):

- `PORT`: listen port (default 8080)
- `RECO_THREADS`: worker threads for scoring (default: all cores, max 8)

Docker (embeddings are downloaded from the GitHub Release at build time):

```bash
docker build -t shop-recommender-api .
docker run -p 8080:8080 shop-recommender-api
```
