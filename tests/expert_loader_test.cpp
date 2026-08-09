#include "h40/expert_loader.hpp"
#include "h40/flash_tensor_provider.hpp"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <new>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

std::atomic<std::uint64_t> g_allocations{0};
bool g_count_allocations = false;

std::filesystem::path make_fixture() {
    const auto path = std::filesystem::temp_directory_path() / "h40_expert_loader_test.bin";
    std::vector<std::byte> data(128, std::byte{0});
    data[16] = std::byte{'a'};
    data[17] = std::byte{'b'};
    data[18] = std::byte{'c'};
    const char* hello = "hello world";
    for (std::size_t i = 0; i < 11; ++i) data[32 + i] = std::byte{static_cast<unsigned char>(hello[i])};
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    assert(output);
    output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return path;
}

template <typename Exception, typename Fn>
void expect_throws(Fn&& fn) {
    bool thrown = false;
    try {
        fn();
    } catch (const Exception&) {
        thrown = true;
    }
    assert(thrown);
}

}  // namespace

void* operator new(std::size_t size) {
    if (g_count_allocations) ++g_allocations;
    if (void* ptr = std::malloc(size)) return ptr;
    throw std::bad_alloc();
}

void operator delete(void* ptr) noexcept { std::free(ptr); }

void operator delete(void* ptr, std::size_t) noexcept { std::free(ptr); }

int main() {
    const auto path = make_fixture();
    {
        h40::ModelIndex index;
        index.put({0, 0}, {16, 3}, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        index.put({0, 1}, {32, 11}, "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
        index.put({0, 2}, {16, 3}, "0000000000000000000000000000000000000000000000000000000000000000");

        h40::FlashTensorProvider provider(path);
        h40::ExpertLoader loader(index, provider);
        std::vector<std::byte> slot(64);

        const auto before_allocations = g_allocations.load();
        g_count_allocations = true;
        const auto abc = loader.load({0, 0}, slot, true);
        g_count_allocations = false;
        assert(g_allocations.load() == before_allocations);
        assert(abc.bytes.size() == 3);
        assert(abc.checksum_verified);
        assert(abc.bytes[0] == std::byte{'a'});
        assert(abc.bytes[1] == std::byte{'b'});
        assert(abc.bytes[2] == std::byte{'c'});

        const auto hello = loader.load({0, 1}, slot, true);
        assert(hello.bytes.size() == 11);
        assert(hello.checksum_verified);

        expect_throws<std::out_of_range>([&] { (void)loader.load({9, 9}, slot, false); });
        expect_throws<std::invalid_argument>([&] {
            std::vector<std::byte> tiny(2);
            (void)loader.load({0, 0}, tiny, false);
        });
        expect_throws<std::runtime_error>([&] { (void)loader.load({0, 2}, slot, true); });

        const auto stats = provider.stats();
        assert(stats.operations == 3);
        assert(stats.bytes == 17);
    }
    std::filesystem::remove(path);
    return 0;
}
