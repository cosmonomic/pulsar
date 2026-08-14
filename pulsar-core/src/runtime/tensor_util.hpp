#pragma once

#include <ATen/ATen.h>

#include <string>

// Internal to pulsar-core/src/runtime; not part of the installed headers.

namespace pulsar {

inline at::ScalarType parse_dtype(const std::string& s, const char* who) {
    if (s == "float16" || s == "half") {
        return at::kHalf;
    }
    if (s == "bfloat16") {
        return at::kBFloat16;
    }
    if (s == "float32" || s == "float") {
        return at::kFloat;
    }
    TORCH_CHECK(false, who, ": unsupported dtype '", s, "' (float16/bfloat16/float32)");
}

}  // namespace pulsar
