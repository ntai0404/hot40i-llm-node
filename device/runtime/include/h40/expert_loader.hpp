#pragma once

#include "h40/model_index.hpp"
#include "h40/tensor_provider.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace h40 {

struct ExpertLoadResult {
    std::span<const std::byte> bytes;
    bool checksum_verified{};
};

namespace detail {

inline std::uint32_t sha256_ch(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return (x & y) ^ (~x & z);
}

inline std::uint32_t sha256_maj(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

inline std::uint32_t sha256_big0(std::uint32_t x) {
    return std::rotr(x, 2) ^ std::rotr(x, 13) ^ std::rotr(x, 22);
}

inline std::uint32_t sha256_big1(std::uint32_t x) {
    return std::rotr(x, 6) ^ std::rotr(x, 11) ^ std::rotr(x, 25);
}

inline std::uint32_t sha256_small0(std::uint32_t x) {
    return std::rotr(x, 7) ^ std::rotr(x, 18) ^ (x >> 3U);
}

inline std::uint32_t sha256_small1(std::uint32_t x) {
    return std::rotr(x, 17) ^ std::rotr(x, 19) ^ (x >> 10U);
}

inline std::uint32_t load_be32(const std::byte* data) {
    return (std::to_integer<std::uint32_t>(data[0]) << 24U) |
           (std::to_integer<std::uint32_t>(data[1]) << 16U) |
           (std::to_integer<std::uint32_t>(data[2]) << 8U) |
           std::to_integer<std::uint32_t>(data[3]);
}

inline void sha256_compress(std::array<std::uint32_t, 8>& state, const std::byte* block) {
    static constexpr std::array<std::uint32_t, 64> k{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::array<std::uint32_t, 64> w{};
    for (std::size_t i = 0; i < 16; ++i) w[i] = load_be32(block + i * 4);
    for (std::size_t i = 16; i < 64; ++i) {
        w[i] = sha256_small1(w[i - 2]) + w[i - 7] + sha256_small0(w[i - 15]) + w[i - 16];
    }
    auto a = state[0];
    auto b = state[1];
    auto c = state[2];
    auto d = state[3];
    auto e = state[4];
    auto f = state[5];
    auto g = state[6];
    auto h = state[7];
    for (std::size_t i = 0; i < 64; ++i) {
        const auto t1 = h + sha256_big1(e) + sha256_ch(e, f, g) + k[i] + w[i];
        const auto t2 = sha256_big0(a) + sha256_maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

inline std::array<std::byte, 32> sha256(std::span<const std::byte> data) {
    std::array<std::uint32_t, 8> state{
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    std::size_t offset = 0;
    while (data.size() - offset >= 64) {
        sha256_compress(state, data.data() + offset);
        offset += 64;
    }

    std::array<std::byte, 128> tail{};
    const auto tail_bytes = data.size() - offset;
    for (std::size_t i = 0; i < tail_bytes; ++i) tail[i] = data[offset + i];
    tail[tail_bytes] = std::byte{0x80};
    const auto total_bits = static_cast<std::uint64_t>(data.size()) * 8ULL;
    const auto final_block_bytes = tail_bytes + 1 + 8 <= 64 ? 64 : 128;
    for (std::size_t i = 0; i < 8; ++i) {
        tail[final_block_bytes - 1 - i] = std::byte((total_bits >> (i * 8U)) & 0xffU);
    }
    sha256_compress(state, tail.data());
    if (final_block_bytes == 128) sha256_compress(state, tail.data() + 64);

    std::array<std::byte, 32> digest{};
    for (std::size_t i = 0; i < state.size(); ++i) {
        digest[i * 4 + 0] = std::byte((state[i] >> 24U) & 0xffU);
        digest[i * 4 + 1] = std::byte((state[i] >> 16U) & 0xffU);
        digest[i * 4 + 2] = std::byte((state[i] >> 8U) & 0xffU);
        digest[i * 4 + 3] = std::byte(state[i] & 0xffU);
    }
    return digest;
}

inline char hex_digit(unsigned value) { return static_cast<char>(value < 10 ? '0' + value : 'a' + value - 10); }

inline bool sha256_hex_equals(std::span<const std::byte> data, const std::array<char, 64>& expected) {
    const auto digest = sha256(data);
    for (std::size_t i = 0; i < digest.size(); ++i) {
        const auto value = std::to_integer<unsigned>(digest[i]);
        if (expected[i * 2] != hex_digit(value >> 4U)) return false;
        if (expected[i * 2 + 1] != hex_digit(value & 0x0fU)) return false;
    }
    return true;
}

}  // namespace detail

class ExpertLoader {
public:
    ExpertLoader(const ModelIndex& index, TensorProvider& provider) : index_(index), provider_(provider) {}
    [[nodiscard]] const ModelIndex& index() const noexcept { return index_; }

    [[nodiscard]] ExpertLoadResult load(ExpertKey key, std::span<std::byte> destination, bool verify_checksum) const {
        const auto record = index_.find_record(key);
        if (!record.has_value()) throw std::out_of_range("expert key missing from model index");
        const auto bytes = static_cast<std::size_t>(record->slice.length);
        if (record->slice.length > destination.size()) throw std::invalid_argument("destination span too small");
        const auto target = destination.first(bytes);
        provider_.read(record->slice, target);
        bool verified = false;
        if (verify_checksum && record->has_sha256) {
            if (!detail::sha256_hex_equals(target, record->sha256)) throw std::runtime_error("expert checksum mismatch");
            verified = true;
        }
        return {target, verified};
    }

private:
    const ModelIndex& index_;
    TensorProvider& provider_;
};

}  // namespace h40
