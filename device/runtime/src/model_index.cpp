#include "h40/model_index.hpp"

namespace h40 {

void ModelIndex::put(ExpertKey key, TensorSlice slice) { experts_[key] = slice; }

std::optional<TensorSlice> ModelIndex::find(ExpertKey key) const {
    const auto it = experts_.find(key);
    if (it == experts_.end()) return std::nullopt;
    return it->second;
}

} // namespace h40
