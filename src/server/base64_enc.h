#pragma once
// Base64 encoder — server-internal only.
// Encodes raw byte blocks (e.g. float32 arrays) to base64 strings.

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace rope::server::detail {

inline std::string base64_encode(const void* data, std::size_t n_bytes) {
    static constexpr char kAlpha[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const auto* p = static_cast<const std::uint8_t*>(data);
    std::string out;
    out.reserve((n_bytes + 2) / 3 * 4);
    for (std::size_t i = 0; i < n_bytes; i += 3) {
        std::uint32_t b = static_cast<std::uint32_t>(p[i]) << 16;
        if (i + 1 < n_bytes) b |= static_cast<std::uint32_t>(p[i + 1]) << 8;
        if (i + 2 < n_bytes) b |= static_cast<std::uint32_t>(p[i + 2]);
        out.push_back(kAlpha[(b >> 18) & 0x3F]);
        out.push_back(kAlpha[(b >> 12) & 0x3F]);
        out.push_back((i + 1 < n_bytes) ? kAlpha[(b >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < n_bytes) ? kAlpha[b & 0x3F]        : '=');
    }
    return out;
}

template<typename T>
inline std::string base64_encode(const std::vector<T>& v) {
    return base64_encode(v.data(), v.size() * sizeof(T));
}

} // namespace rope::server::detail
