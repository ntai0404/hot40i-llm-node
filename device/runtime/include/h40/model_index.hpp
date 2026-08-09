#pragma once

#include "h40/tensor_provider.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace h40 {

struct ExpertKey {
    std::uint32_t layer{};
    std::uint32_t expert{};

    bool operator==(const ExpertKey&) const = default;
};

struct ExpertKeyHash {
    std::size_t operator()(const ExpertKey& key) const noexcept {
        return (static_cast<std::size_t>(key.layer) << 32U) ^ key.expert;
    }
};

struct ExpertRecord {
    TensorSlice slice{};
    std::array<char, 64> sha256{};
    bool has_sha256{};
};

class ModelIndex {
public:
    void put(ExpertKey key, TensorSlice slice);
    void put(ExpertKey key, TensorSlice slice, std::string_view sha256);
    [[nodiscard]] std::optional<TensorSlice> find(ExpertKey key) const;
    [[nodiscard]] std::optional<ExpertRecord> find_record(ExpertKey key) const;
    [[nodiscard]] std::size_t size() const noexcept { return experts_.size(); }

private:
    std::unordered_map<ExpertKey, ExpertRecord, ExpertKeyHash> experts_;
};

} // namespace h40
