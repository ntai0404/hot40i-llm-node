#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Config {
    int threads = 1;
    std::vector<int> cpus;
    double seconds = 0.25;
    std::size_t mem_bytes = 32ULL * 1024ULL * 1024ULL;
    int mat_m = 256;
    int mat_k = 1024;
};

void usage() {
    std::cout << "usage: h40_cpu_memory_bench --threads N --cpus LIST "
                 "[--seconds S] [--mem-bytes N] [--mat-m N] [--mat-k N]\n";
}

std::uint64_t parse_u64(const std::string& text) {
    std::size_t consumed = 0;
    const auto value = std::stoull(text, &consumed, 0);
    if (consumed != text.size()) {
        throw std::invalid_argument("invalid integer: " + text);
    }
    return value;
}

std::vector<int> parse_cpus(const std::string& text) {
    std::vector<int> cpus;
    std::size_t start = 0;
    while (start < text.size()) {
        const auto comma = text.find(',', start);
        const auto token = text.substr(start, comma == std::string::npos ? comma : comma - start);
        if (token.empty()) {
            throw std::invalid_argument("empty cpu id in --cpus");
        }
        cpus.push_back(static_cast<int>(parse_u64(token)));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (cpus.empty()) {
        throw std::invalid_argument("--cpus must not be empty");
    }
    return cpus;
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
        } else if (arg == "--threads") {
            cfg.threads = static_cast<int>(parse_u64(require_value("--threads")));
        } else if (arg == "--cpus") {
            cfg.cpus = parse_cpus(require_value("--cpus"));
        } else if (arg == "--seconds") {
            cfg.seconds = std::stod(require_value("--seconds"));
        } else if (arg == "--mem-bytes") {
            cfg.mem_bytes = static_cast<std::size_t>(parse_u64(require_value("--mem-bytes")));
        } else if (arg == "--mat-m") {
            cfg.mat_m = static_cast<int>(parse_u64(require_value("--mat-m")));
        } else if (arg == "--mat-k") {
            cfg.mat_k = static_cast<int>(parse_u64(require_value("--mat-k")));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (cfg.threads < 1 || cfg.threads > 16) {
        throw std::invalid_argument("--threads must be 1..16");
    }
    if (cfg.cpus.empty()) {
        throw std::invalid_argument("--cpus is required");
    }
    if (cfg.seconds <= 0.0 || cfg.mem_bytes < 1024 || cfg.mat_m < 1 || cfg.mat_k < 16) {
        throw std::invalid_argument("invalid benchmark dimensions");
    }
    return cfg;
}

void pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

template <typename Fn>
double run_parallel(const Config& cfg, Fn fn) {
    std::atomic<std::uint64_t> units{0};
    std::atomic<std::uint64_t> checksum{0};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(cfg.seconds);
    std::vector<std::thread> workers;
    for (int tid = 0; tid < cfg.threads; ++tid) {
        workers.emplace_back([&, tid]() {
            pin_to_cpu(cfg.cpus[static_cast<std::size_t>(tid) % cfg.cpus.size()]);
            std::uint64_t local_units = 0;
            std::uint64_t local_sum = 0;
            while (std::chrono::steady_clock::now() < deadline) {
                local_sum += fn(tid);
                ++local_units;
            }
            units.fetch_add(local_units, std::memory_order_relaxed);
            checksum.fetch_add(local_sum, std::memory_order_relaxed);
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    return static_cast<double>(units.load(std::memory_order_relaxed));
}

std::string cpu_csv(const std::vector<int>& cpus) {
    std::string out;
    for (std::size_t i = 0; i < cpus.size(); ++i) {
        if (i) out += ",";
        out += std::to_string(cpus[i]);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);
        std::vector<std::uint8_t> src(cfg.mem_bytes);
        std::vector<std::uint8_t> dst(cfg.mem_bytes);
        for (std::size_t i = 0; i < src.size(); ++i) {
            src[i] = static_cast<std::uint8_t>(i * 131U + 7U);
        }

        std::vector<std::int8_t> vec(static_cast<std::size_t>(cfg.mat_k));
        std::vector<std::int8_t> mat8(static_cast<std::size_t>(cfg.mat_m * cfg.mat_k));
        std::vector<std::uint8_t> mat4(static_cast<std::size_t>(cfg.mat_m * cfg.mat_k / 2));
        for (std::size_t i = 0; i < vec.size(); ++i) vec[i] = static_cast<std::int8_t>((i % 31) - 15);
        for (std::size_t i = 0; i < mat8.size(); ++i) mat8[i] = static_cast<std::int8_t>((i % 17) - 8);
        for (std::size_t i = 0; i < mat4.size(); ++i) mat4[i] = static_cast<std::uint8_t>(((i & 15) << 4) | ((i + 3) & 15));

        const auto memcpy_units = run_parallel(cfg, [&](int) {
            std::memcpy(dst.data(), src.data(), src.size());
            return static_cast<std::uint64_t>(dst[0]);
        });
        const auto read_units = run_parallel(cfg, [&](int) {
            std::uint64_t sum = 0;
            const auto* words = reinterpret_cast<const std::uint64_t*>(src.data());
            for (std::size_t i = 0; i < src.size() / sizeof(std::uint64_t); ++i) sum += words[i];
            return sum;
        });
        const auto int8_units = run_parallel(cfg, [&](int) {
            std::int64_t total = 0;
            for (int m = 0; m < cfg.mat_m; ++m) {
                std::int32_t acc = 0;
                const auto base = static_cast<std::size_t>(m * cfg.mat_k);
                for (int k = 0; k < cfg.mat_k; ++k) acc += mat8[base + k] * vec[static_cast<std::size_t>(k)];
                total += acc;
            }
            return static_cast<std::uint64_t>(total);
        });
        const auto int4_units = run_parallel(cfg, [&](int) {
            std::int64_t total = 0;
            for (int m = 0; m < cfg.mat_m; ++m) {
                std::int32_t acc = 0;
                const auto base = static_cast<std::size_t>(m * cfg.mat_k / 2);
                for (int k = 0; k < cfg.mat_k; k += 2) {
                    const auto packed = mat4[base + static_cast<std::size_t>(k / 2)];
                    const auto lo = static_cast<std::int8_t>((packed & 0x0f) - 8);
                    const auto hi = static_cast<std::int8_t>(((packed >> 4) & 0x0f) - 8);
                    acc += lo * vec[static_cast<std::size_t>(k)];
                    acc += hi * vec[static_cast<std::size_t>(k + 1)];
                }
                total += acc;
            }
            return static_cast<std::uint64_t>(total);
        });

        const double mib_per_copy = static_cast<double>(cfg.mem_bytes) / 1024.0 / 1024.0;
        const double read_mib_per_iter = static_cast<double>(cfg.mem_bytes) / 1024.0 / 1024.0;
        const double mat_ops = static_cast<double>(cfg.mat_m) * static_cast<double>(cfg.mat_k) * 2.0;
        std::cout << "{\n"
                  << "  \"schema_version\": 1,\n"
                  << "  \"threads\": " << cfg.threads << ",\n"
                  << "  \"cpus\": \"" << cpu_csv(cfg.cpus) << "\",\n"
                  << "  \"seconds_per_kernel\": " << cfg.seconds << ",\n"
                  << "  \"mem_bytes\": " << cfg.mem_bytes << ",\n"
                  << "  \"mat_m\": " << cfg.mat_m << ",\n"
                  << "  \"mat_k\": " << cfg.mat_k << ",\n"
                  << "  \"memcpy_mib_per_second\": " << (memcpy_units * mib_per_copy / cfg.seconds) << ",\n"
                  << "  \"read_mib_per_second\": " << (read_units * read_mib_per_iter / cfg.seconds) << ",\n"
                  << "  \"int8_matvec_gops\": " << (int8_units * mat_ops / cfg.seconds / 1.0e9) << ",\n"
                  << "  \"int4_matvec_gops\": " << (int4_units * mat_ops / cfg.seconds / 1.0e9) << "\n"
                  << "}\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "h40_cpu_memory_bench: " << exc.what() << "\n";
        usage();
        return 2;
    }
}
