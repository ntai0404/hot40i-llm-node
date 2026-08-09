#include "h40/expert_cache.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace h40 {

ExpertCache::ExpertCache(std::size_t budget_bytes) : ExpertCache(budget_bytes, budget_bytes) {}

ExpertCache::ExpertCache(std::size_t budget_bytes, std::size_t slot_bytes)
    : owned_storage_(std::make_unique<std::byte[]>(budget_bytes)),
      storage_(owned_storage_.get(), budget_bytes),
      budget_bytes_(budget_bytes),
      slot_bytes_(slot_bytes) {
    if (budget_bytes == 0) throw std::invalid_argument("cache budget must be > 0");
    if (slot_bytes == 0) throw std::invalid_argument("cache slot bytes must be > 0");
    slot_count_ = budget_bytes_ / slot_bytes_;
    if (slot_count_ == 0) throw std::invalid_argument("cache budget must contain at least one slot");
    free_slots_.reserve(slot_count_);
    for (std::size_t i = 0; i < slot_count_; ++i) free_slots_.push_back(slot_count_ - i - 1);
}

ExpertCache::ExpertCache(std::span<std::byte> storage, std::size_t slot_bytes)
    : storage_(storage), budget_bytes_(storage.size()), slot_bytes_(slot_bytes) {
    if (storage.empty()) throw std::invalid_argument("cache storage must be non-empty");
    if (slot_bytes == 0) throw std::invalid_argument("cache slot bytes must be > 0");
    slot_count_ = budget_bytes_ / slot_bytes_;
    if (slot_count_ == 0) throw std::invalid_argument("cache storage must contain at least one slot");
    free_slots_.reserve(slot_count_);
    for (std::size_t i = 0; i < slot_count_; ++i) free_slots_.push_back(slot_count_ - i - 1);
}

void ExpertCache::touch(ExpertKey key, Entry& entry) {
    lru_.erase(entry.lru_it);
    lru_.push_front(key);
    entry.lru_it = lru_.begin();
}

void ExpertCache::evict_one() {
    if (lru_.empty()) throw std::logic_error("cache accounting mismatch");
    const auto key = lru_.back();
    lru_.pop_back();
    const auto it = entries_.find(key);
    if (it == entries_.end()) throw std::logic_error("LRU key missing from cache");
    used_bytes_ -= it->second.bytes;
    free_slots_.push_back(it->second.slot);
    entries_.erase(it);
    ++stats_.evictions;
}

std::span<std::byte> ExpertCache::slot_span(std::size_t slot, std::size_t bytes) {
    if (slot >= slot_count_ || bytes > slot_bytes_) throw std::out_of_range("cache slot span out of range");
    const auto offset = slot * slot_bytes_;
    return storage_.subspan(offset, bytes);
}

std::span<const std::byte> ExpertCache::slot_span(std::size_t slot, std::size_t bytes) const {
    if (slot >= slot_count_ || bytes > slot_bytes_) throw std::out_of_range("cache slot span out of range");
    const auto offset = slot * slot_bytes_;
    return storage_.subspan(offset, bytes);
}

std::span<const std::byte> ExpertCache::get_or_load(
    ExpertKey key,
    const TensorSlice& slice,
    TensorProvider& provider) {
    if (auto it = entries_.find(key); it != entries_.end()) {
        ++stats_.hits;
        touch(key, it->second);
        return slot_span(it->second.slot, it->second.bytes);
    }

    ++stats_.misses;
    const auto bytes = static_cast<std::size_t>(slice.length);
    if (bytes > slot_bytes_) throw std::bad_alloc();
    while (free_slots_.empty()) evict_one();
    const auto slot = free_slots_.back();
    free_slots_.pop_back();

    Entry entry;
    entry.slot = slot;
    entry.bytes = bytes;
    auto destination = slot_span(slot, bytes);
    provider.read(slice, destination);
    stats_.bytes_loaded += bytes;
    used_bytes_ += bytes;
    lru_.push_front(key);
    entry.lru_it = lru_.begin();
    const auto [it, inserted] = entries_.emplace(key, std::move(entry));
    if (!inserted) throw std::logic_error("cache insertion failed");
    return slot_span(it->second.slot, it->second.bytes);
}

} // namespace h40
