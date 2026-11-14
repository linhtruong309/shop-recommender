// Latency benchmark: recommend() for random known users, report p50 / p95 / p99
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "recommender.hpp"

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "../ml/artifacts";
    const int n_requests = argc > 2 ? std::atoi(argv[2]) : 1000;
    const int k = 10;

    const auto model = reco::Recommender::load(dir);
    const auto& users = model.user_ids();
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> pick(0, users.size() - 1);

    for (int i = 0; i < 20; ++i) model.recommend(users[pick(rng)], k);  // warm-up

    std::vector<double> ms;
    ms.reserve(n_requests);
    for (int i = 0; i < n_requests; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        model.recommend(users[pick(rng)], k);
        ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    auto pct = [&](double p) { return ms[static_cast<size_t>(p * (ms.size() - 1))]; };

    std::cout << "C++ top-" << k << " over " << model.n_items() << " items, "
              << n_requests << " requests\n"
              << "  p50 = " << pct(0.50) << " ms\n"
              << "  p95 = " << pct(0.95) << " ms\n"
              << "  p99 = " << pct(0.99) << " ms\n";
    return 0;
}
