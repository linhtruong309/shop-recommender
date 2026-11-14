#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace reco {

struct Scored {
    int64_t item_id;
    float score;
};

struct Result {
    std::vector<Scored> items;
    bool personalized;  // false = popularity fallback
};

// Embeddings exported by ml/train_two_tower.py
class Recommender {
public:
    // Load from an artifacts directory; n_threads = 0 uses all cores (max 8)
    static Recommender load(const std::string& dir, unsigned n_threads = 0);

    // Build from in-memory data (used by tests)
    Recommender(int dim,
                std::vector<int64_t> item_ids, std::vector<float> item_vecs,
                std::vector<int64_t> user_ids, std::vector<float> user_vecs,
                std::vector<int64_t> popular_ids, unsigned n_threads = 0);

    // Top-k items by dot product; popularity fallback for unknown users
    Result recommend(int64_t user_id, int k) const;

    // Top-k for an arbitrary query vector
    std::vector<Scored> top_k(const float* query, int k) const;

    bool has_user(int64_t user_id) const { return user_row_.count(user_id) > 0; }
    size_t n_items() const { return item_ids_.size(); }
    size_t n_users() const { return user_ids_.size(); }
    int dim() const { return dim_; }
    const std::vector<int64_t>& user_ids() const { return user_ids_; }

private:
    int dim_;
    std::vector<int64_t> item_ids_;
    std::vector<float> item_vecs_;  // n_items x dim, row-major
    std::vector<int64_t> user_ids_;
    std::vector<float> user_vecs_;  // n_users x dim, row-major
    std::vector<int64_t> popular_ids_;
    std::unordered_map<int64_t, size_t> user_row_;
    unsigned n_threads_;
};

}  // namespace reco
