#include "h40/expert_loader.hpp"
#include "h40/flash_tensor_provider.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Config {
    std::string file;
    std::uint32_t layer{};
    std::uint32_t expert{};
    std::uint64_t offset{};
    std::uint64_t length{};
    std::string sha256;
};

std::uint64_t parse_u64(const std::string& text) {
    std::size_t consumed = 0;
    const auto value = std::stoull(text, &consumed, 0);
    if (consumed != text.size()) throw std::invalid_argument("invalid integer: " + text);
    return value;
}

void usage() {
    std::cout << "usage: h40_expert_loader_probe --file PATH --layer N --expert N --offset N --length N --sha256 HEX\n";
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char* name) -> std::string {
            if (++i >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
            return argv[i];
        };
        if (arg == "--help" || arg == "-h") {
            usage();
            std::exit(0);
        } else if (arg == "--file") {
            cfg.file = value("--file");
        } else if (arg == "--layer") {
            cfg.layer = static_cast<std::uint32_t>(parse_u64(value("--layer")));
        } else if (arg == "--expert") {
            cfg.expert = static_cast<std::uint32_t>(parse_u64(value("--expert")));
        } else if (arg == "--offset") {
            cfg.offset = parse_u64(value("--offset"));
        } else if (arg == "--length") {
            cfg.length = parse_u64(value("--length"));
        } else if (arg == "--sha256") {
            cfg.sha256 = value("--sha256");
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (cfg.file.empty() || cfg.sha256.size() != 64 || cfg.length == 0) {
        throw std::invalid_argument("missing required probe arguments");
    }
    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto cfg = parse_args(argc, argv);
        h40::ModelIndex index;
        index.put({cfg.layer, cfg.expert}, {cfg.offset, cfg.length}, cfg.sha256);
        h40::FlashTensorProvider provider(cfg.file);
        h40::ExpertLoader loader(index, provider);
        std::vector<std::byte> slot(static_cast<std::size_t>(cfg.length));
        const auto result = loader.load({cfg.layer, cfg.expert}, slot, true);
        const auto stats = provider.stats();
        std::cout << "{\n"
                  << "  \"schema_version\": 1,\n"
                  << "  \"layer\": " << cfg.layer << ",\n"
                  << "  \"expert\": " << cfg.expert << ",\n"
                  << "  \"offset\": " << cfg.offset << ",\n"
                  << "  \"length\": " << cfg.length << ",\n"
                  << "  \"loaded_bytes\": " << result.bytes.size() << ",\n"
                  << "  \"checksum_verified\": " << (result.checksum_verified ? "true" : "false") << ",\n"
                  << "  \"provider_operations\": " << stats.operations << ",\n"
                  << "  \"provider_bytes\": " << stats.bytes << "\n"
                  << "}\n";
        return result.checksum_verified && stats.operations == 1 && stats.bytes == cfg.length ? 0 : 3;
    } catch (const std::exception& exc) {
        std::cerr << "h40_expert_loader_probe: " << exc.what() << "\n";
        usage();
        return 2;
    }
}
