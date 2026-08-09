#include "h40/router.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

void assert_close(float actual, double expected, double tolerance = 1.0e-6) {
    assert(std::fabs(static_cast<double>(actual) - expected) <= tolerance);
}

void assert_selection(
    const h40::RouterSelection& actual,
    const std::vector<std::uint32_t>& expected_ids,
    const std::vector<double>& expected_weights
) {
    assert(actual.expert_ids == expected_ids);
    assert(actual.weights.size() == expected_weights.size());
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < actual.weights.size(); ++i) {
        assert_close(actual.weights[i], expected_weights[i]);
        weight_sum += actual.weights[i];
    }
    assert_close(static_cast<float>(weight_sum), 1.0);
}

void fixture_rows_match_goldens() {
    const std::array<std::array<float, 4>, 3> logits{{
        {0.3627933470022995F, -0.1833364829537863F, -0.30944980951579043F, 0.28450300256251676F},
        {0.1803379988112045F, 0.7144753831688232F, -0.04457992819531151F, -0.7212508381399503F},
        {0.22105828960567428F, -0.13105131881174045F, -0.8036626588180752F, -0.114650184793299F},
    }};

    assert_selection(
        h40::select_top_k_experts(logits[0], 2),
        {0, 3},
        {0.5195625949189345, 0.48043740508106547}
    );
    assert_selection(
        h40::select_top_k_experts(logits[1], 2),
        {1, 0},
        {0.6304475731242347, 0.36955242687576534}
    );
    assert_selection(
        h40::select_top_k_experts(logits[2], 2),
        {0, 3},
        {0.5831476848751792, 0.41685231512482085}
    );
}

void top_four_and_ties_are_deterministic() {
    const std::array<float, 5> logits{1.0F, 3.0F, 3.0F, -1.0F, 2.0F};
    const auto selected = h40::select_top_k_experts(logits, 4);
    assert((selected.expert_ids == std::vector<std::uint32_t>{1, 2, 4, 0}));
    assert_close(selected.weights[0], selected.weights[1]);
    assert(selected.weights[0] > selected.weights[2]);
    assert(selected.weights[2] > selected.weights[3]);
}

void invalid_inputs_throw() {
    const std::array<float, 2> logits{0.0F, 1.0F};
    bool threw_zero = false;
    try {
        (void)h40::select_top_k_experts(logits, 0);
    } catch (const std::invalid_argument&) {
        threw_zero = true;
    }
    assert(threw_zero);

    bool threw_large = false;
    try {
        (void)h40::select_top_k_experts(logits, 3);
    } catch (const std::invalid_argument&) {
        threw_large = true;
    }
    assert(threw_large);

    const std::array<float, 1> bad{NAN};
    bool threw_nan = false;
    try {
        (void)h40::select_top_k_experts(bad, 1);
    } catch (const std::invalid_argument&) {
        threw_nan = true;
    }
    assert(threw_nan);
}

} // namespace

int main() {
    fixture_rows_match_goldens();
    top_four_and_ties_are_deterministic();
    invalid_inputs_throw();
    return 0;
}
