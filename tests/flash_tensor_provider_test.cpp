#include "h40/flash_tensor_provider.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::filesystem::path make_fixture() {
    const auto path = std::filesystem::temp_directory_path() / "h40_flash_tensor_provider_test.bin";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    assert(output);
    for (std::uint32_t i = 0; i < 4096; ++i) {
        const auto value = static_cast<unsigned char>((i * 17U + 29U) & 0xffU);
        output.write(reinterpret_cast<const char*>(&value), 1);
    }
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

std::vector<std::byte> expected_bytes(std::uint64_t offset, std::size_t length) {
    std::vector<std::byte> expected(length);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expected[i] = std::byte(((offset + i) * 17U + 29U) & 0xffU);
    }
    return expected;
}

}  // namespace

int main() {
    const auto path = make_fixture();
    {
        h40::FlashTensorProvider provider(path);
        assert(provider.file_size() == 4096);

        std::vector<h40::FlashReadTrace> traces;
        provider.set_trace_sink([&](const h40::FlashReadTrace& trace) { traces.push_back(trace); });

        std::vector<std::byte> first(32);
        provider.read({0, first.size()}, first);
        assert(first == expected_bytes(0, first.size()));

        std::vector<std::byte> middle(257);
        provider.read({123, middle.size()}, middle);
        assert(middle == expected_bytes(123, middle.size()));

        std::vector<std::byte> tail(64);
        provider.read({4096 - tail.size(), tail.size()}, tail);
        assert(tail == expected_bytes(4096 - tail.size(), tail.size()));

        expect_throws<std::invalid_argument>([&] {
            std::vector<std::byte> wrong(2);
            provider.read({0, 3}, wrong);
        });
        expect_throws<std::out_of_range>([&] {
            std::vector<std::byte> out(1);
            provider.read({4096, 1}, out);
        });
        expect_throws<std::out_of_range>([&] {
            std::vector<std::byte> out(16);
            provider.read({4090, out.size()}, out);
        });

        const auto stats = provider.stats();
        assert(stats.operations == 3);
        assert(stats.bytes == first.size() + middle.size() + tail.size());
        assert(stats.nanoseconds > 0);
        assert(traces.size() == 3);
        assert(traces[1].slice.offset == 123);
        assert(traces[1].slice.length == middle.size());
        assert(traces[1].nanoseconds > 0);
    }

    std::filesystem::remove(path);
    return 0;
}
