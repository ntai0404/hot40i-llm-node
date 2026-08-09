#include "h40/router.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace h40 {

namespace {

bool better_logit(std::span<const float> logits, std::uint32_t lhs, std::uint32_t rhs) noexcept {
    const float a = logits[lhs];
    const float b = logits[rhs];
    if (a == b) {
        return lhs < rhs;
    }
    return a > b;
}

} // namespace

RouterSelection select_top_k_experts(std::span<const float> logits, std::size_t top_k) {
    if (top_k == 0) {
        throw std::invalid_argument("top_k must be greater than zero");
    }
    if (top_k > logits.size()) {
        throw std::invalid_argument("top_k cannot exceed expert count");
    }
    if (logits.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::invalid_argument("expert count exceeds uint32 id range");
    }
    for (const float logit : logits) {
        if (!std::isfinite(logit)) {
            throw std::invalid_argument("router logits must be finite");
        }
    }

    std::vector<std::uint32_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0U);
    std::stable_sort(indices.begin(), indices.end(), [&](std::uint32_t lhs, std::uint32_t rhs) {
        return better_logit(logits, lhs, rhs);
    });
    indices.resize(top_k);

    float max_selected = logits[indices.front()];
    for (const auto id : indices) {
        max_selected = std::max(max_selected, logits[id]);
    }

    RouterSelection out;
    out.expert_ids = std::move(indices);
    out.weights.resize(top_k);

    double normalizer = 0.0;
    for (std::size_t i = 0; i < top_k; ++i) {
        const double value = std::exp(static_cast<double>(logits[out.expert_ids[i]] - max_selected));
        out.weights[i] = static_cast<float>(value);
        normalizer += value;
    }
    for (auto& weight : out.weights) {
        weight = static_cast<float>(static_cast<double>(weight) / normalizer);
    }
    return out;
}

} // namespace h40
