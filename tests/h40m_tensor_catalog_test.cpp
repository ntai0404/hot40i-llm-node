#include "h40/h40m_tensor_catalog.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <vector>

int main() {
    const auto catalog = h40::H40mTensorCatalog::load_tsv("artifacts/model/h40m/tensor_catalog.tsv");
    assert(catalog.size() == 459);
    const auto embedding = catalog.find("model.embed_tokens.weight");
    assert(embedding.has_value());
    assert(embedding->dtype == "BF16");
    assert(embedding->shape.size() == 2);
    assert(embedding->shape[0] == 201088);
    assert(embedding->shape[1] == 2880);

    h40::FileTensorReader reader("artifacts/model/source");
    std::vector<float> row(embedding->shape[1]);
    reader.read_bf16_row(*embedding, 12194, row);
    float checksum = 0.0F;
    for (std::size_t i = 0; i < row.size(); i += 97) checksum += row[i];
    assert(std::isfinite(checksum));

    const auto norm = catalog.find("model.norm.weight");
    assert(norm.has_value());
    assert(norm->dtype == "BF16");
    assert(norm->shape.size() == 1);
    assert(norm->shape[0] == 2880);
    std::vector<float> norm_values(norm->shape[0]);
    reader.read_bf16_vector(*norm, norm_values);
    assert(std::isfinite(norm_values[0]));

    const auto router = catalog.find("model.layers.0.mlp.router.weight");
    assert(router.has_value());
    assert(router->shape.size() == 2);
    assert(router->shape[0] == 32);
    assert(router->shape[1] == 2880);
    std::vector<float> router_logits(router->shape[0]);
    std::vector<std::uint16_t> row_buffer(router->shape[1]);
    reader.bf16_matvec(*router, row, router_logits, row_buffer);
    for (float value : router_logits) assert(std::isfinite(value));
    std::vector<float> router_logits_range(4);
    reader.bf16_matvec_rows(*router, 3, row, router_logits_range, row_buffer);
    for (std::size_t i = 0; i < router_logits_range.size(); ++i) {
        assert(std::fabs(router_logits_range[i] - router_logits[i + 3]) <= 1.0e-5F);
    }

    const auto lm_head = catalog.find("lm_head.weight");
    assert(lm_head.has_value());
    assert(lm_head->shape.size() == 2);
    assert(lm_head->shape[0] == 201088);
    assert(lm_head->shape[1] == 2880);
    std::vector<float> lm_head_logits(8);
    reader.bf16_matvec_rows(*lm_head, 0, row, lm_head_logits, row_buffer);
    for (float value : lm_head_logits) assert(std::isfinite(value));
    return 0;
}
