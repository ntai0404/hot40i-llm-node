#include "h40/embedding_lookup.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <stdexcept>

int main() {
    const h40::EmbeddingTable table{0, 4096, 10, 16};
    const auto row = h40::resolve_embedding_row(table, 3);
    assert(row.file_id == 0);
    assert(row.offset == 4096 + 3 * 16);
    assert(row.length == 16);

    bool threw = false;
    try {
        (void)h40::resolve_embedding_row(table, 10);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);

    h40::EmbeddingRowCache cache(32);
    const std::array<std::byte, 16> a{};
    const std::array<std::byte, 16> b{std::byte{1}};
    const std::array<std::byte, 16> c{std::byte{2}};
    cache.put(1, a);
    cache.put(2, b);
    assert(cache.used_bytes() == 32);
    assert(cache.get(1).size() == 16);
    cache.put(3, c);
    assert(cache.used_bytes() == 32);
    assert(cache.get(2).size() == 16 || cache.get(1).size() == 16);
    assert(cache.get(3).size() == 16);
    cache.clear();
    assert(cache.used_bytes() == 0);
    return 0;
}
