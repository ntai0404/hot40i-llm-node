#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace h40 {

struct HttpRequest {
    std::string method;
    std::string target;
    std::string body;
};

inline HttpRequest parse_http_request(std::string_view input, std::size_t max_body_bytes) {
    const auto header_end = input.find("\r\n\r\n");
    if (header_end == std::string_view::npos) throw std::invalid_argument("incomplete HTTP headers");
    const auto first_end = input.find("\r\n");
    if (first_end == std::string_view::npos) throw std::invalid_argument("missing request line");
    const auto request_line = input.substr(0, first_end);
    const auto first_space = request_line.find(' ');
    const auto second_space = request_line.find(' ', first_space + 1);
    if (first_space == std::string_view::npos || second_space == std::string_view::npos) {
        throw std::invalid_argument("invalid request line");
    }
    HttpRequest request{
        std::string(request_line.substr(0, first_space)),
        std::string(request_line.substr(first_space + 1, second_space - first_space - 1)),
        std::string(input.substr(header_end + 4)),
    };
    if (request.body.size() > max_body_bytes) throw std::length_error("request body exceeds configured limit");
    return request;
}

inline bool valid_token_csv(std::string_view input) {
    if (input.empty()) return false;
    bool digit = false;
    for (const char c : input) {
        if (c >= '0' && c <= '9') {
            digit = true;
        } else if (c == ',' && digit) {
            digit = false;
        } else if (c != '\r' && c != '\n' && c != ' ' && c != '\t') {
            return false;
        }
    }
    return digit;
}

}  // namespace h40
