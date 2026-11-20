#include "hakios-metal/pm4.hpp"

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

void Stream::command(uint32_t opcode, uint32_t count) {
    assert(count <= kMaxType7Count);
    words_.push_back(type7(opcode, count));
}

void Stream::registers(uint32_t reg, const uint32_t* values, uint32_t count) {
    // more than a header can count goes as several packets, each starting where the one before stopped
    while (count > 0) {
        uint32_t n = count < kMaxType4Count ? count : kMaxType4Count;
        words_.push_back(type4(reg, n));
        words_.insert(words_.end(), values, values + n);
        reg += n;
        values += n;
        count -= n;
    }
}

void Stream::nop(uint32_t words) {
    command(kNop, words);
    words_.insert(words_.end(), words, 0u);
}

void Stream::waitForIdle() { command(kWaitForIdle, 0); }
void Stream::waitMemWrites() { command(kWaitMemWrites, 0); }
void Stream::waitForMe() { command(kWaitForMe, 0); }

void Stream::memWrite(uint64_t a, const uint32_t* values, uint32_t count) {
    while (count > 0) {
        uint32_t n = count < kMaxType7Count - 2 ? count : kMaxType7Count - 2;
        command(kMemWrite, 2 + n);
        address(a);
        words_.insert(words_.end(), values, values + n);
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

void Stream::packet(uint32_t opcode, const uint32_t* words, uint32_t count) {
    command(opcode, count);
    words_.insert(words_.end(), words, words + count);
}

uint32_t* Stream::open(uint32_t opcode, uint32_t count) {
    command(opcode, count);
    const size_t at = words_.size();
    words_.resize(at + count);
    return words_.data() + at;
}

}  // namespace hakios::metal::pm4
