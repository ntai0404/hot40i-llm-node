#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Config {
    std::filesystem::path file;
    std::string pattern = "random";
    std::uint64_t block_bytes = 8ULL * 1024ULL * 1024ULL;
    std::uint64_t reads = 64;
    std::uint64_t seed = 0x4040;
};

void usage() {
    std::cout
        << "usage: h40_io_bench --file PATH [--pattern sequential|random] "
           "[--block-bytes N] [--reads N] [--seed N]\n";
}

std::uint64_t parse_u64(const std::string& text) {
    std::size_t consumed = 0;
    const auto value = std::stoull(text, &consumed, 0);
    if (consumed != text.size()) {
        throw std::invalid_argument("invalid integer: " + text);
    }
    return value;
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            usage();
            std::exit(0);
        } else if (arg == "--file") {
            cfg.file = require_value("--file");
        } else if (arg == "--pattern") {
            cfg.pattern = require_value("--pattern");
        } else if (arg == "--block-bytes") {
            cfg.block_bytes = parse_u64(require_value("--block-bytes"));
        } else if (arg == "--reads") {
            cfg.reads = parse_u64(require_value("--reads"));
        } else if (arg == "--seed") {
            cfg.seed = parse_u64(require_value("--seed"));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (cfg.file.empty()) {
        throw std::invalid_argument("--file is required");
    }
    if (cfg.pattern != "sequential" && cfg.pattern != "random") {
        throw std::invalid_argument("--pattern must be sequential or random");
    }
    if (cfg.block_bytes == 0 || cfg.reads == 0) {
        throw std::invalid_argument("--block-bytes and --reads must be positive");
    }
    return cfg;
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double rank = q * static_cast<double>(values.size() - 1);
    const auto low = static_cast<std::size_t>(rank);
    const auto high = std::min(low + 1, values.size() - 1);
    const double frac = rank - static_cast<double>(low);
    return values[low] * (1.0 - frac) + values[high] * frac;
}

std::string json_escape(const std::string& input) {
    std::string out;
    for (const char ch : input) {
        if (ch == '\\' || ch == '"') {
            out.push_back('\\');
        }
        out.push_back(ch);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);
        const auto file_size = std::filesystem::file_size(cfg.file);
        if (file_size < cfg.block_bytes) {
            throw std::runtime_error("file is smaller than one benchmark block");
        }

        std::ifstream input(cfg.file, std::ios::binary);
        if (!input) {
            throw std::runtime_error("failed to open file");
        }
        std::vector<char> buffer(static_cast<std::size_t>(cfg.block_bytes));
        const std::uint64_t slots = file_size / cfg.block_bytes;
        std::mt19937_64 rng(cfg.seed);
        std::uniform_int_distribution<std::uint64_t> dist(0, slots - 1);
        std::vector<double> latency_ms;
        latency_ms.reserve(static_cast<std::size_t>(cfg.reads));

        const auto wall_start = std::chrono::steady_clock::now();
        for (std::uint64_t i = 0; i < cfg.reads; ++i) {
            const std::uint64_t slot = cfg.pattern == "sequential" ? (i % slots) : dist(rng);
            const auto offset = static_cast<std::streamoff>(slot * cfg.block_bytes);
            const auto op_start = std::chrono::steady_clock::now();
            input.clear();
            input.seekg(offset);
            input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            if (input.gcount() != static_cast<std::streamsize>(buffer.size())) {
                throw std::runtime_error("short read");
            }
            const auto op_wall = std::chrono::steady_clock::now() - op_start;
            latency_ms.push_back(std::chrono::duration<double, std::milli>(op_wall).count());
        }
        const auto wall = std::chrono::steady_clock::now() - wall_start;
        const double seconds = std::chrono::duration<double>(wall).count();
        const auto bytes = cfg.block_bytes * cfg.reads;
        const double mib = static_cast<double>(bytes) / 1024.0 / 1024.0;

        std::cout << "{\n"
                  << "  \"schema_version\": 1,\n"
                  << "  \"file\": \"" << json_escape(cfg.file.string()) << "\",\n"
                  << "  \"file_size_bytes\": " << file_size << ",\n"
                  << "  \"pattern\": \"" << cfg.pattern << "\",\n"
                  << "  \"block_bytes\": " << cfg.block_bytes << ",\n"
                  << "  \"reads\": " << cfg.reads << ",\n"
                  << "  \"seed\": " << cfg.seed << ",\n"
                  << "  \"bytes\": " << bytes << ",\n"
                  << "  \"seconds\": " << seconds << ",\n"
                  << "  \"mib_per_second\": " << (mib / seconds) << ",\n"
                  << "  \"iops\": " << (static_cast<double>(cfg.reads) / seconds) << ",\n"
                  << "  \"latency_ms_p50\": " << percentile(latency_ms, 0.50) << ",\n"
                  << "  \"latency_ms_p95\": " << percentile(latency_ms, 0.95) << ",\n"
                  << "  \"latency_ms_p99\": " << percentile(latency_ms, 0.99) << "\n"
                  << "}\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "h40_io_bench: " << exc.what() << "\n";
        usage();
        return 2;
    }
}
