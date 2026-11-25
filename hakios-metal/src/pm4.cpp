#include "hakios-metal/pm4.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace hakios::metal::pm4 {

Header decode(uint32_t word) {
    Header h;
    switch (word >> 28) {
    case 4:
        h.count = word & 0x7f;
        h.reg = (word >> 8) & 0x3ffff;
        if (((word >> 7) & 1) == oddParity(h.count) && ((word >> 27) & 1) == oddParity(h.reg)) {
            h.kind = Header::kRegisters;
        }
        break;
    case 7:
        h.count = word & 0x3fff;
        h.opcode = (word >> 16) & 0x7f;
        if (((word >> 15) & 1) == oddParity(h.count) && ((word >> 23) & 1) == oddParity(h.opcode)) {
            h.kind = Header::kCommand;
        }
        break;
    default:
        break;
    }
    return h;
}

size_t Stream::lastPacket(size_t from, uint32_t opcodeA, uint32_t opcodeB) const {
    size_t found = SIZE_MAX;
    for (size_t at = from; at < words_.size();) {
        const Header h = decode(words_[at]);
        if (h.kind == Header::kBad) break;   // (not a header: the walk stops, what was found stands)
        if (h.kind == Header::kCommand && (h.opcode == opcodeA || h.opcode == opcodeB)) found = at;
        at += 1 + size_t(h.count);
    }
    return found;
}

bool Stream::nopPacket(size_t at, uint32_t opcodeA, uint32_t opcodeB) {
    if (at >= words_.size()) return false;
    const Header h = decode(words_[at]);
    if (h.kind != Header::kCommand || (h.opcode != opcodeA && h.opcode != opcodeB) || at + h.count >= words_.size()) return false;
    words_[at] = type7(kNop, h.count);
    return true;
}

bool Stream::patch(size_t at, const uint32_t* words, size_t n) {
    if (at > words_.size() || n > words_.size() - at) return false;
    std::copy(words, words + n, words_.begin() + at);
    return true;
}

void Stream::command(uint32_t opcode, uint32_t count) {
    assert(count <= kMaxType7Count);
    words_.push_back(type7(opcode, count));
}

void Stream::nop(uint32_t words) {
    command(kNop, words);
    std::fill_n(words_.extend(words), words, 0u);
}

void Stream::waitForIdle() { command(kWaitForIdle, 0); }
void Stream::waitMemWrites() { command(kWaitMemWrites, 0); }
void Stream::waitForMe() { command(kWaitForMe, 0); }

void Stream::memWrite(uint64_t a, const uint32_t* values, uint32_t count) {
    while (count > 0) {
        uint32_t n = count < kMaxType7Count - 2 ? count : kMaxType7Count - 2;
        command(kMemWrite, 2 + n);
        address(a);
        words_.append(values, n);
        a += 4ull * n;
        values += n;
        count -= n;
    }
}

void Stream::memcpy(uint64_t destination, uint64_t source, uint32_t words) {
    command(kMemcpy, 5);
    words_.push_back(words);
    address(source);
    address(destination);
}

void Stream::indirect(uint64_t a, uint32_t words) {
    command(kIndirectBuffer, 3);
    address(a);
    words_.push_back(words);
}

uint32_t* Stream::open(uint32_t opcode, uint32_t count) {
    command(opcode, count);
    uint32_t* at = words_.extend(count);
    std::fill_n(at, count, 0u);
    return at;
}

}  // namespace hakios::metal::pm4
