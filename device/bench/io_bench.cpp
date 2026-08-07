#include "h40/flash_tensor_provider.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <span>
#include <vector>

using namespace h40;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: h40_io_bench <file> [block_mib=8] [reads=64]\n";
        return 2;
    }
    const std::filesystem::path path = argv[1];
    const std::size_t block_mib = argc > 2 ? std::stoull(argv[2]) : 8;
    const std::size_t reads = argc > 3 ? std::stoull(argv[3]) : 64;
    const std::size_t block = block_mib * 1024ULL * 1024ULL;
    const auto file_size = std::filesystem::file_size(path);
    if (file_size < block) {
        std::cerr << "file is smaller than one benchmark block\n";
        return 2;
    }

    FlashTensorProvider provider(path);
    std::vector<std::byte> buffer(block);
    std::mt19937_64 rng(0x4040);
    const std::uint64_t slots = file_size / block;
    std::uniform_int_distribution<std::uint64_t> dist(0, slots - 1);

    const auto wall_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < reads; ++i) {
        const auto slot = dist(rng);
        provider.read({slot * block, block}, buffer);
    }
    const auto wall = std::chrono::steady_clock::now() - wall_start;
    const double seconds = std::chrono::duration<double>(wall).count();
    const double mib = static_cast<double>(block * reads) / 1024.0 / 1024.0;
    const auto stats = provider.stats();

    std::cout << "{\n"
              << "  \"file\": \"" << path.string() << "\",\n"
              << "  \"pattern\": \"random_fixed_block\",\n"
              << "  \"block_mib\": " << block_mib << ",\n"
              << "  \"reads\": " << reads << ",\n"
              << "  \"mib_per_second\": " << (mib / seconds) << ",\n"
              << "  \"operations\": " << stats.operations << ",\n"
              << "  \"bytes\": " << stats.bytes << "\n"
              << "}\n";
    return 0;
}
