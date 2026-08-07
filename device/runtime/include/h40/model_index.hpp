#pragma once

#include "h40/tensor_provider.hpp"

#include <cstdint>
#include <optional>
#include <string>
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

class ModelIndex {
public:
    void put(ExpertKey key, TensorSlice slice);
    [[nodiscard]] std::optional<TensorSlice> find(ExpertKey key) const;
    [[nodiscard]] std::size_t size() const noexcept { return experts_.size(); }

private:
    std::unordered_map<ExpertKey, TensorSlice, ExpertKeyHash> experts_;
};

} // namespace h40
