#include "h40/model_index.hpp"

#include <algorithm>
#include <stdexcept>

namespace h40 {

void ModelIndex::put(ExpertKey key, TensorSlice slice) { experts_[key] = {slice, {}, false}; }

void ModelIndex::put(ExpertKey key, TensorSlice slice, std::string_view sha256) {
    if (sha256.size() != 64) throw std::invalid_argument("expert sha256 must be 64 hex characters");
    ExpertRecord record;
    record.slice = slice;
    std::copy(sha256.begin(), sha256.end(), record.sha256.begin());
    record.has_sha256 = true;
    experts_[key] = record;
}

std::optional<TensorSlice> ModelIndex::find(ExpertKey key) const {
    const auto it = experts_.find(key);
    if (it == experts_.end()) return std::nullopt;
    return it->second.slice;
}

std::optional<ExpertRecord> ModelIndex::find_record(ExpertKey key) const {
    const auto it = experts_.find(key);
    if (it == experts_.end()) return std::nullopt;
    return it->second;
}

} // namespace h40
