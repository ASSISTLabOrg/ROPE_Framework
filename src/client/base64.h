#pragma once
// Minimal base64 decoder used only by client.cpp.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rope::client::detail {

inline std::vector<std::uint8_t> base64_decode(std::string_view in) {
    static constexpr unsigned char kTable[256] = {
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64, // 0-15
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64, // 16-31
        64,64,64,64,64,64,64,64,64,64,64,62,64,64,64,63, // 32-47  (+, /)
        52,53,54,55,56,57,58,59,60,61,64,64,64,255,64,64, // 48-63  (0-9, =)
        64, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14, // 64-79  (A-O)
        15,16,17,18,19,20,21,22,23,24,25,64,64,64,64,64, // 80-95  (P-Z)
        64,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40, // 96-111 (a-o)
        41,42,43,44,45,46,47,48,49,50,51,64,64,64,64,64, // 112-127 (p-z)
    };

    std::vector<std::uint8_t> out;
    out.reserve(in.size() * 3 / 4);

    unsigned int buf = 0;
    int bits = 0;
    for (unsigned char c : in) {
        if (c == '=') break;
        if (c >= 128) throw std::runtime_error("base64_decode: invalid char");
        unsigned char v = kTable[c];
        if (v == 64) continue;  // whitespace
        buf = (buf << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<std::uint8_t>((buf >> bits) & 0xFF));
        }
    }
    return out;
}

} // namespace rope::client::detail
