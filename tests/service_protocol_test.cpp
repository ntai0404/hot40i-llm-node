#include "h40/service_protocol.hpp"

#include <cassert>
#include <stdexcept>

int main() {
    const auto get = h40::parse_http_request("GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n", 32);
    assert(get.method == "GET");
    assert(get.target == "/health");
    const auto post = h40::parse_http_request("POST /infer HTTP/1.1\r\nContent-Length: 8\r\n\r\n1,2,300", 32);
    assert(post.body == "1,2,300");
    assert(h40::valid_token_csv(post.body));
    assert(!h40::valid_token_csv("1,two,3"));
    bool bounded = false;
    try {
        static_cast<void>(h40::parse_http_request("POST /infer HTTP/1.1\r\n\r\n12345", 4));
    } catch (const std::length_error&) {
        bounded = true;
    }
    assert(bounded);
    return 0;
}
