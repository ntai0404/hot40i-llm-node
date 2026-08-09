#pragma once

#include "h40/expert_cache.hpp"
#include "h40/ram_arena.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace h40 {

struct BoundedRuntimeGraphConfig {
    std::size_t safe_rss_budget_bytes{};
    std::size_t resident_bytes{};
    std::size_t embedding_cache_bytes{};
    std::size_t output_head_chunk_bytes{};
    std::size_t attention_stream_bytes{};
    std::size_t expert_cache_bytes{};
    std::size_t kv_state_bytes{};
    std::size_t io_bytes{};
    std::size_t scratch_bytes{};
    std::size_t expert_slot_bytes{};
    std::size_t expert_slot_alignment{64};
    std::size_t full_checkpoint_bytes{};
    bool allow_full_checkpoint_residency{};
    std::string backend{"cpu_sync"};
};

struct BoundedRuntimeGraphMetrics {
    std::size_t safe_rss_budget_bytes{};
    std::size_t committed_bytes{};
    std::size_t headroom_bytes{};
    std::size_t peak_allocation_bytes{};
};

class BoundedRuntimeGraph {
public:
    explicit BoundedRuntimeGraph(BoundedRuntimeGraphConfig config) : config_(std::move(config)) {
        validate_config();
    }

    void start() {
        if (arenas_) throw std::logic_error("runtime graph is already started");
        arenas_ = std::make_unique<FixedMemoryArenas>(FixedMemoryPlan{
            config_.safe_rss_budget_bytes,
            {
                {ArenaRegionKind::resident, "resident_dense_small", config_.resident_bytes, 64},
                {ArenaRegionKind::embedding, "embedding_lru_rows", config_.embedding_cache_bytes, 64},
                {ArenaRegionKind::output_head, "output_head_chunk", config_.output_head_chunk_bytes, 64},
                {ArenaRegionKind::scratch, "attention_layer_stream", config_.attention_stream_bytes, 64},
                {ArenaRegionKind::expert_cache, "expert_cache_slots", config_.expert_cache_bytes, config_.expert_slot_alignment},
                {ArenaRegionKind::kv_state, "kv_state", config_.kv_state_bytes, 64},
                {ArenaRegionKind::io, "io_double_buffer", config_.io_bytes, 64},
                {ArenaRegionKind::scratch, "activation_and_logits_scratch", config_.scratch_bytes, 64},
            },
        });
        if (arenas_->committed_bytes() > config_.safe_rss_budget_bytes) {
            arenas_.reset();
            throw std::bad_alloc();
        }
    }

    void stop() noexcept { arenas_.reset(); }

    [[nodiscard]] bool started() const noexcept { return static_cast<bool>(arenas_); }
    [[nodiscard]] const BoundedRuntimeGraphConfig& config() const noexcept { return config_; }

    [[nodiscard]] BoundedRuntimeGraphMetrics metrics() const {
        require_started();
        return {
            config_.safe_rss_budget_bytes,
            arenas_->committed_bytes(),
            arenas_->headroom_bytes(),
            arenas_->committed_bytes(),
        };
    }

    [[nodiscard]] std::span<std::byte> resident() { return region(ArenaRegionKind::resident); }
    [[nodiscard]] std::span<std::byte> embedding_cache() { return region(ArenaRegionKind::embedding); }
    [[nodiscard]] std::span<std::byte> output_head_chunk() { return region(ArenaRegionKind::output_head); }
    [[nodiscard]] std::span<std::byte> expert_cache_storage() { return region(ArenaRegionKind::expert_cache); }
    [[nodiscard]] std::span<std::byte> kv_state() { return region(ArenaRegionKind::kv_state); }
    [[nodiscard]] std::span<std::byte> io() { return region(ArenaRegionKind::io); }
    [[nodiscard]] std::span<std::byte> attention_stream() { return named_region("attention_layer_stream"); }
    [[nodiscard]] std::span<std::byte> scratch() { return named_region("activation_and_logits_scratch"); }

    [[nodiscard]] ExpertCache make_expert_cache() {
        return ExpertCache(expert_cache_storage(), config_.expert_slot_bytes, config_.expert_slot_alignment);
    }

private:
    void validate_config() const {
        if (config_.safe_rss_budget_bytes == 0) throw std::invalid_argument("safe RSS budget must be > 0");
        if (config_.expert_cache_bytes == 0) throw std::invalid_argument("expert cache budget must be > 0");
        if (config_.expert_slot_bytes == 0) throw std::invalid_argument("expert slot bytes must be > 0");
        if (config_.expert_cache_bytes < config_.expert_slot_bytes) {
            throw std::invalid_argument("expert cache budget must fit at least one expert slot");
        }
        if (!config_.allow_full_checkpoint_residency && config_.full_checkpoint_bytes > config_.resident_bytes) {
            throw std::invalid_argument("full checkpoint residency is not allowed by bounded graph config");
        }
        const std::size_t requested =
            config_.resident_bytes +
            config_.embedding_cache_bytes +
            config_.output_head_chunk_bytes +
            config_.attention_stream_bytes +
            config_.expert_cache_bytes +
            config_.kv_state_bytes +
            config_.io_bytes +
            config_.scratch_bytes;
        if (requested > config_.safe_rss_budget_bytes) throw std::bad_alloc();
    }

    void require_started() const {
        if (!arenas_) throw std::logic_error("runtime graph is not started");
    }

    [[nodiscard]] std::span<std::byte> region(ArenaRegionKind kind) {
        require_started();
        return arenas_->region(kind);
    }

    [[nodiscard]] std::span<std::byte> named_region(const char* name) {
        require_started();
        return arenas_->region(name);
    }

    BoundedRuntimeGraphConfig config_;
    std::unique_ptr<FixedMemoryArenas> arenas_;
};

}  // namespace h40
