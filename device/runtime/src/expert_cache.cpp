#include "h40/expert_cache.hpp"

#include <stdexcept>

namespace h40 {

ExpertCache::ExpertCache(std::size_t budget_bytes) : budget_bytes_(budget_bytes) {
    if (budget_bytes == 0) throw std::invalid_argument("cache budget must be > 0");
}

void ExpertCache::touch(ExpertKey key, Entry& entry) {
    lru_.erase(entry.lru_it);
    lru_.push_front(key);
    entry.lru_it = lru_.begin();
}

void ExpertCache::evict_until(std::size_t required) {
    if (required > budget_bytes_) throw std::bad_alloc();
    while (used_bytes_ + required > budget_bytes_) {
        if (lru_.empty()) throw std::logic_error("cache accounting mismatch");
        const auto key = lru_.back();
        lru_.pop_back();
        const auto it = entries_.find(key);
        if (it == entries_.end()) throw std::logic_error("LRU key missing from cache");
        used_bytes_ -= it->second.bytes.size();
        entries_.erase(it);
        ++stats_.evictions;
    }
}

std::span<const std::byte> ExpertCache::get_or_load(
    ExpertKey key,
    const TensorSlice& slice,
    TensorProvider& provider) {
    if (auto it = entries_.find(key); it != entries_.end()) {
        ++stats_.hits;
        touch(key, it->second);
        return it->second.bytes;
    }

    ++stats_.misses;
    evict_until(static_cast<std::size_t>(slice.length));
    Entry entry;
    entry.bytes.resize(static_cast<std::size_t>(slice.length));
    provider.read(slice, entry.bytes);
    stats_.bytes_loaded += entry.bytes.size();
    used_bytes_ += entry.bytes.size();
    lru_.push_front(key);
    entry.lru_it = lru_.begin();
    const auto [it, inserted] = entries_.emplace(key, std::move(entry));
    if (!inserted) throw std::logic_error("cache insertion failed");
    return it->second.bytes;
}

} // namespace h40
