// REST service: GET /health, GET /recommendations?user_id=<id>&k=<n>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>

#include <httplib.h>

#include "recommender.hpp"

namespace {

std::string error_json(const std::string& message) {
    return "{\"error\":\"" + message + "\"}";
}

}  // namespace

int main(int argc, char** argv) {
    // Config: argv first, then environment (PORT is set by hosting platforms)
    const char* env_port = std::getenv("PORT");
    const char* env_threads = std::getenv("RECO_THREADS");
    const std::string dir = argc > 1 ? argv[1] : "../ml/artifacts";
    const int port = argc > 2 ? std::atoi(argv[2]) : (env_port ? std::atoi(env_port) : 8080);
    const unsigned threads = env_threads ? static_cast<unsigned>(std::atoi(env_threads)) : 0;
    const char* env_web = std::getenv("RECO_WEB_DIR");
    const std::string web_dir = env_web ? env_web : "web";

    std::cout << "Loading artifacts from " << dir << " ..." << std::endl;
    const auto start = std::chrono::steady_clock::now();
    const auto model = reco::Recommender::load(dir, threads);
    const auto load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "Loaded " << model.n_items() << " items, " << model.n_users()
              << " users, dim " << model.dim() << " in " << load_s << "s" << std::endl;

    httplib::Server server;

    // Allow the API to be called from other origins (e.g. a portfolio page)
    server.set_default_headers({{"Access-Control-Allow-Origin", "*"}});

    // Demo page: static files served at /
    if (server.set_mount_point("/", web_dir))
        std::cout << "Serving demo page from " << web_dir << std::endl;

    server.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        std::ostringstream body;
        body << "{\"status\":\"ok\",\"items\":" << model.n_items()
             << ",\"users\":" << model.n_users() << "}";
        res.set_content(body.str(), "application/json");
    });

    // Random known users, for the demo page
    server.Get("/sample_users", [&](const httplib::Request& req, httplib::Response& res) {
        int n = 6;
        try {
            if (req.has_param("n")) n = std::stoi(req.get_param_value("n"));
        } catch (...) {}
        n = std::clamp(n, 1, 20);
        const auto& ids = model.user_ids();
        static thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<size_t> pick(0, ids.size() - 1);
        std::ostringstream body;
        body << "{\"user_ids\":[";
        for (int i = 0; i < n && !ids.empty(); ++i) body << (i ? "," : "") << ids[pick(rng)];
        body << "]}";
        res.set_content(body.str(), "application/json");
    });

    server.Get("/recommendations", [&](const httplib::Request& req, httplib::Response& res) {
        // Validate input
        if (!req.has_param("user_id")) {
            res.status = 400;
            res.set_content(error_json("user_id is required"), "application/json");
            return;
        }
        int64_t user_id;
        int k = 10;
        try {
            user_id = std::stoll(req.get_param_value("user_id"));
            if (req.has_param("k")) k = std::stoi(req.get_param_value("k"));
        } catch (...) {
            res.status = 400;
            res.set_content(error_json("user_id and k must be integers"), "application/json");
            return;
        }
        if (k < 1 || k > 100) {
            res.status = 400;
            res.set_content(error_json("k must be between 1 and 100"), "application/json");
            return;
        }

        // Recommend and time it
        const auto t0 = std::chrono::steady_clock::now();
        const auto result = model.recommend(user_id, k);
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

        std::ostringstream body;
        body << "{\"user_id\":" << user_id
             << ",\"source\":\"" << (result.personalized ? "two_tower" : "popular") << "\""
             << ",\"latency_ms\":" << ms << ",\"items\":[";
        for (size_t i = 0; i < result.items.size(); ++i) {
            if (i) body << ",";
            body << "{\"item_id\":" << result.items[i].item_id
                 << ",\"score\":" << result.items[i].score << "}";
        }
        body << "]}";
        res.set_content(body.str(), "application/json");
    });

    std::cout << "Listening on http://localhost:" << port << std::endl;
    server.listen("0.0.0.0", port);
    return 0;
}
