#include "recommender.hpp"

#include <algorithm>
#include <fstream>
#include <queue>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace reco {
namespace {

// Read a raw binary file into a vector of T
template <typename T>
std::vector<T> read_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("Cannot open " + path);
    const auto bytes = static_cast<size_t>(in.tellg());
    if (bytes % sizeof(T) != 0) throw std::runtime_error("Unexpected size: " + path);
    std::vector<T> data(bytes / sizeof(T));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
    return data;
}

// Read "dim" from meta.json
int read_dim(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    std::smatch m;
    const std::string text = ss.str();
    if (!std::regex_search(text, m, std::regex("\"dim\"\\s*:\\s*(\\d+)")))
        throw std::runtime_error("No dim in " + path);
    return std::stoi(m[1]);
}

// Min-heap ordering: smallest score on top
struct MinScore {
    bool operator()(const Scored& a, const Scored& b) const { return a.score > b.score; }
};
using TopHeap = std::priority_queue<Scored, std::vector<Scored>, MinScore>;

void push_top(TopHeap& heap, Scored s, size_t k) {
    if (heap.size() < k) heap.push(s);
    else if (s.score > heap.top().score) { heap.pop(); heap.push(s); }
}

}  // namespace

Recommender Recommender::load(const std::string& dir) {
    const int dim = read_dim(dir + "/meta.json");
    return Recommender(dim,
                       read_binary<int64_t>(dir + "/item_ids.bin"),
                       read_binary<float>(dir + "/item_vecs.bin"),
                       read_binary<int64_t>(dir + "/user_ids.bin"),
                       read_binary<float>(dir + "/user_vecs.bin"),
                       read_binary<int64_t>(dir + "/popular_ids.bin"));
}

Recommender::Recommender(int dim,
                         std::vector<int64_t> item_ids, std::vector<float> item_vecs,
                         std::vector<int64_t> user_ids, std::vector<float> user_vecs,
                         std::vector<int64_t> popular_ids, unsigned n_threads)
    : dim_(dim),
      item_ids_(std::move(item_ids)), item_vecs_(std::move(item_vecs)),
      user_ids_(std::move(user_ids)), user_vecs_(std::move(user_vecs)),
      popular_ids_(std::move(popular_ids)) {
    if (item_vecs_.size() != item_ids_.size() * dim_)
        throw std::runtime_error("item_vecs does not match item_ids x dim");
    if (user_vecs_.size() != user_ids_.size() * dim_)
        throw std::runtime_error("user_vecs does not match user_ids x dim");

    user_row_.reserve(user_ids_.size());
    for (size_t i = 0; i < user_ids_.size(); ++i) user_row_[user_ids_[i]] = i;

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    n_threads_ = n_threads ? n_threads : std::min(hw, 8u);
}

std::vector<Scored> Recommender::top_k(const float* query, int k) const {
    const size_t n = item_ids_.size();
    const size_t kk = std::min<size_t>(static_cast<size_t>(std::max(k, 0)), n);
    if (kk == 0) return {};

    // Each thread scans a slice of items and keeps its own top-k
    const unsigned t = static_cast<unsigned>(std::min<size_t>(n_threads_, n));
    std::vector<TopHeap> heaps(t);
    auto scan = [&](unsigned w) {
        const size_t begin = n * w / t, end = n * (w + 1) / t;
        for (size_t i = begin; i < end; ++i) {
            const float* v = &item_vecs_[i * dim_];
            float s = 0.f;
#pragma omp simd reduction(+ : s)
            for (int d = 0; d < dim_; ++d) s += query[d] * v[d];
            push_top(heaps[w], {item_ids_[i], s}, kk);
        }
    };
    std::vector<std::thread> workers;
    for (unsigned w = 1; w < t; ++w) workers.emplace_back(scan, w);
    scan(0);
    for (auto& th : workers) th.join();

    // Merge partial results
    TopHeap merged;
    for (auto& h : heaps)
        while (!h.empty()) { push_top(merged, h.top(), kk); h.pop(); }
    std::vector<Scored> out;
    while (!merged.empty()) { out.push_back(merged.top()); merged.pop(); }
    std::reverse(out.begin(), out.end());
    return out;
}

Result Recommender::recommend(int64_t user_id, int k) const {
    const auto it = user_row_.find(user_id);
    if (it != user_row_.end()) return {top_k(&user_vecs_[it->second * dim_], k), true};

    Result fallback{{}, false};
    for (size_t i = 0; i < popular_ids_.size() && static_cast<int>(i) < k; ++i)
        fallback.items.push_back({popular_ids_[i], 0.f});
    return fallback;
}

}  // namespace reco
