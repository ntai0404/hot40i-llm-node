#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace h40 {

struct RouterSelection {
    std::vector<std::uint32_t> expert_ids;
    std::vector<float> weights;
};

[[nodiscard]] RouterSelection select_top_k_experts(
    std::span<const float> logits,
    std::size_t top_k
);

} // namespace h40
