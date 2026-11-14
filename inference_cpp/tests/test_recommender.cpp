// Minimal unit tests (no framework): exit code 0 = pass
#include <cmath>
#include <iostream>

#include "recommender.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* name) {
    std::cout << (ok ? "PASS " : "FAIL ") << name << "\n";
    if (!ok) ++failures;
}

reco::Recommender make_model(unsigned threads) {
    // 2-d vectors: items point in different directions
    return reco::Recommender(
        2,
        {101, 102, 103, 104}, {1, 0,  0, 1,  0.7f, 0.7f,  -1, 0},
        {7}, {1, 0},
        {104, 103}, threads);
}

}  // namespace

int main() {
    for (unsigned threads : {1u, 3u}) {
        const auto model = make_model(threads);

        auto r = model.recommend(7, 2);
        check(r.personalized, "known user is personalized");
        check(r.items.size() == 2, "returns k items");
        check(r.items[0].item_id == 101, "best match ranked first");
        check(r.items[1].item_id == 103, "second best ranked second");
        check(r.items[0].score >= r.items[1].score, "scores are descending");

        auto all = model.recommend(7, 10);
        check(all.items.size() == 4, "k larger than catalog is capped");

        auto cold = model.recommend(999, 2);
        check(!cold.personalized, "unknown user falls back");
        check(cold.items.size() == 2 && cold.items[0].item_id == 104, "fallback uses popular order");
    }

    bool threw = false;
    try { reco::Recommender(2, {1}, {1, 0, 0}, {}, {}, {}); } catch (...) { threw = true; }
    check(threw, "rejects mismatched vector sizes");

    std::cout << (failures ? "FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}
