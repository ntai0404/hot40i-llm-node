#include "h40/flash_tensor_provider.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ProbeRange {
    std::uint64_t offset{};
    std::uint64_t length{};
};

struct Config {
    std::filesystem::path file;
    bool make_fixture{false};
};

void usage() {
    std::cout << "usage: h40_flash_provider_probe --file PATH [--make-fixture]\n";
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage();
            std::exit(0);
        }
        if (arg == "--file") {
            if (++i >= argc) throw std::invalid_argument("missing value for --file");
            cfg.file = argv[i];
        } else if (arg == "--make-fixture") {
            cfg.make_fixture = true;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (cfg.file.empty()) throw std::invalid_argument("--file is required");
    return cfg;
}

std::byte fixture_byte(std::uint64_t index) {
    return std::byte((index * 1315423911ULL + (index >> 7U) + 0x5aU) & 0xffU);
}

void write_fixture(const std::filesystem::path& path) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("failed to create fixture");
    std::vector<char> buffer(64 * 1024);
    constexpr std::uint64_t file_bytes = 8ULL * 1024ULL * 1024ULL;
    for (std::uint64_t offset = 0; offset < file_bytes; offset += buffer.size()) {
        const auto chunk = std::min<std::uint64_t>(buffer.size(), file_bytes - offset);
        for (std::uint64_t i = 0; i < chunk; ++i) {
            buffer[static_cast<std::size_t>(i)] = static_cast<char>(fixture_byte(offset + i));
        }
        output.write(buffer.data(), static_cast<std::streamsize>(chunk));
    }
}

std::uint64_t fnv1a(std::span<const std::byte> data) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto value : data) {
        hash ^= static_cast<std::uint64_t>(value);
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::uint64_t expected_hash(std::uint64_t offset, std::uint64_t length) {
    std::vector<std::byte> expected(static_cast<std::size_t>(length));
    for (std::uint64_t i = 0; i < length; ++i) {
        expected[static_cast<std::size_t>(i)] = fixture_byte(offset + i);
    }
    return fnv1a(expected);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto cfg = parse_args(argc, argv);
        if (cfg.make_fixture) write_fixture(cfg.file);

        const std::vector<ProbeRange> ranges{
            {0, 4096},
            {12345, 65536},
            {1024ULL * 1024ULL - 17ULL, 131072},
            {7ULL * 1024ULL * 1024ULL + 333ULL, 262144},
            {8ULL * 1024ULL * 1024ULL - 8192ULL, 8192},
        };

        h40::FlashTensorProvider provider(cfg.file);
        std::vector<double> trace_ms;
        provider.set_trace_sink([&](const h40::FlashReadTrace& trace) {
            trace_ms.push_back(static_cast<double>(trace.nanoseconds) / 1000000.0);
        });

        std::uint64_t bytes = 0;
        bool hashes_match = true;
        const auto wall_start = std::chrono::steady_clock::now();
        for (const auto& range : ranges) {
            std::vector<std::byte> buffer(static_cast<std::size_t>(range.length));
            provider.read({range.offset, range.length}, buffer);
            bytes += range.length;
            hashes_match = hashes_match && (fnv1a(buffer) == expected_hash(range.offset, range.length));
        }
        const auto elapsed = std::chrono::steady_clock::now() - wall_start;
        const double seconds = std::chrono::duration<double>(elapsed).count();
        const auto stats = provider.stats();

        std::cout << "{\n"
                  << "  \"schema_version\": 1,\n"
                  << "  \"fixture_bytes\": " << provider.file_size() << ",\n"
                  << "  \"range_count\": " << ranges.size() << ",\n"
                  << "  \"bytes_read\": " << bytes << ",\n"
                  << "  \"seconds\": " << seconds << ",\n"
                  << "  \"mib_per_second\": " << (static_cast<double>(bytes) / 1048576.0 / seconds) << ",\n"
                  << "  \"provider_operations\": " << stats.operations << ",\n"
                  << "  \"provider_bytes\": " << stats.bytes << ",\n"
                  << "  \"provider_nanoseconds\": " << stats.nanoseconds << ",\n"
                  << "  \"trace_count\": " << trace_ms.size() << ",\n"
                  << "  \"hashes_match\": " << (hashes_match ? "true" : "false") << "\n"
                  << "}\n";
        return hashes_match ? 0 : 3;
    } catch (const std::exception& exc) {
        std::cerr << "h40_flash_provider_probe: " << exc.what() << "\n";
        usage();
        return 2;
    }
}
