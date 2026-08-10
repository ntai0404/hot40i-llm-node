#include "h40/expert_cache.hpp"

#include "h40/expert_loader.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace h40 {

namespace {

std::size_t align_up(std::size_t value, std::size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("slot alignment must be a non-zero power of two");
    }
    const auto mask = alignment - 1;
    if (value > static_cast<std::size_t>(-1) - mask) throw std::bad_alloc();
    return (value + mask) & ~mask;
}

}  // namespace

ExpertCache::ExpertCache(std::size_t budget_bytes) : ExpertCache(budget_bytes, budget_bytes, 1) {}

ExpertCache::ExpertCache(std::size_t budget_bytes, std::size_t slot_bytes)
    : ExpertCache(budget_bytes, slot_bytes, 1) {}

ExpertCache::ExpertCache(std::size_t budget_bytes, std::size_t slot_bytes, std::size_t slot_alignment)
    : owned_storage_(std::make_unique<std::byte[]>(budget_bytes)),
      storage_(owned_storage_.get(), budget_bytes),
      budget_bytes_(budget_bytes),
      slot_bytes_(slot_bytes),
      slot_stride_bytes_(align_up(slot_bytes, slot_alignment)),
      slot_alignment_(slot_alignment) {
    if (budget_bytes == 0) throw std::invalid_argument("cache budget must be > 0");
    if (slot_bytes == 0) throw std::invalid_argument("cache slot bytes must be > 0");
    slot_count_ = budget_bytes_ / slot_stride_bytes_;
    if (slot_count_ == 0) throw std::invalid_argument("cache budget must contain at least one slot");
    free_slots_.reserve(slot_count_);
    for (std::size_t i = 0; i < slot_count_; ++i) free_slots_.push_back(slot_count_ - i - 1);
}

ExpertCache::ExpertCache(std::span<std::byte> storage, std::size_t slot_bytes)
    : ExpertCache(storage, slot_bytes, 1) {}

ExpertCache::ExpertCache(std::span<std::byte> storage, std::size_t slot_bytes, std::size_t slot_alignment)
    : storage_(storage),
      budget_bytes_(storage.size()),
      slot_bytes_(slot_bytes),
      slot_stride_bytes_(align_up(slot_bytes, slot_alignment)),
      slot_alignment_(slot_alignment) {
    if (storage.empty()) throw std::invalid_argument("cache storage must be non-empty");
    if (slot_bytes == 0) throw std::invalid_argument("cache slot bytes must be > 0");
    slot_count_ = budget_bytes_ / slot_stride_bytes_;
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

bool ExpertCache::contains(ExpertKey key) const noexcept {
    return entries_.find(key) != entries_.end();
}

std::span<std::byte> ExpertCache::slot_span(std::size_t slot, std::size_t bytes) {
    if (slot >= slot_count_ || bytes > slot_bytes_) throw std::out_of_range("cache slot span out of range");
    const auto offset = slot * slot_stride_bytes_;
    return storage_.subspan(offset, bytes);
}

std::span<const std::byte> ExpertCache::slot_span(std::size_t slot, std::size_t bytes) const {
    if (slot >= slot_count_ || bytes > slot_bytes_) throw std::out_of_range("cache slot span out of range");
    const auto offset = slot * slot_stride_bytes_;
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
    stats_.peak_used_bytes = std::max<std::uint64_t>(
        stats_.peak_used_bytes,
        static_cast<std::uint64_t>(used_bytes_));
    lru_.push_front(key);
    entry.lru_it = lru_.begin();
    const auto [it, inserted] = entries_.emplace(key, std::move(entry));
    if (!inserted) throw std::logic_error("cache insertion failed");
    return slot_span(it->second.slot, it->second.bytes);
}

CacheLoadResult ExpertCache::get_or_load(
    ExpertKey key,
    const ExpertLoader& loader,
    bool verify_checksum) {
    if (auto it = entries_.find(key); it != entries_.end()) {
        ++stats_.hits;
        touch(key, it->second);
        return {slot_span(it->second.slot, it->second.bytes), true};
    }

    const auto record = loader.index().find_record(key);
    if (!record.has_value()) throw std::out_of_range("expert key missing from model index");

    ++stats_.misses;
    const auto bytes = static_cast<std::size_t>(record->slice.length);
    if (bytes > slot_bytes_) throw std::bad_alloc();
    while (free_slots_.empty()) evict_one();
    const auto slot = free_slots_.back();
    free_slots_.pop_back();

    Entry entry;
    entry.slot = slot;
    entry.bytes = bytes;
    auto destination = slot_span(slot, bytes);
    const auto loaded = loader.load(key, destination, verify_checksum);
    if (loaded.bytes.size() != bytes) throw std::logic_error("expert loader returned unexpected byte count");
    stats_.bytes_loaded += bytes;
    used_bytes_ += bytes;
    stats_.peak_used_bytes = std::max<std::uint64_t>(
        stats_.peak_used_bytes,
        static_cast<std::uint64_t>(used_bytes_));
    lru_.push_front(key);
    entry.lru_it = lru_.begin();
    const auto [it, inserted] = entries_.emplace(key, std::move(entry));
    if (!inserted) throw std::logic_error("cache insertion failed");
    return {slot_span(it->second.slot, it->second.bytes), false};
}

CacheLoadResult ExpertCache::insert_loaded(
    ExpertKey key,
    std::span<const std::byte> bytes) {
    if (entries_.find(key) != entries_.end()) {
        throw std::logic_error("cannot insert an expert that is already cached");
    }
    if (bytes.empty() || bytes.size() > slot_bytes_) throw std::bad_alloc();
    while (free_slots_.empty()) evict_one();
    const auto slot = free_slots_.back();
    free_slots_.pop_back();

    auto destination = slot_span(slot, bytes.size());
    std::copy(bytes.begin(), bytes.end(), destination.begin());
    Entry entry;
    entry.slot = slot;
    entry.bytes = bytes.size();
    ++stats_.misses;
    stats_.bytes_loaded += bytes.size();
    used_bytes_ += bytes.size();
    stats_.peak_used_bytes = std::max<std::uint64_t>(
        stats_.peak_used_bytes,
        static_cast<std::uint64_t>(used_bytes_));
    lru_.push_front(key);
    entry.lru_it = lru_.begin();
    const auto [it, inserted] = entries_.emplace(key, std::move(entry));
    if (!inserted) throw std::logic_error("cache insertion failed");
    return {slot_span(it->second.slot, it->second.bytes), false};
}

} // namespace h40
