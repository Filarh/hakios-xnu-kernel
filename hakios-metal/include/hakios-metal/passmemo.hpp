// Render-pass memoization (metal_passmemo): a pass that is provably the one rendered at the same place of the frame
// before -- the same draws, pipelines, bound bytes, sampled textures at the same versions, the same attachments and
// actions -- and whose result does not depend on what its target held before it, is not run again: its words in the
// command stream become a copy of the result kept from an earlier run (the 2D engine, cache -> target, over the
// rectangle the pass can write) followed by NOPs. App-independent: what changes nothing is found from the stream.
//
// What is compared, and when (CommandBuffer::memoDraw, memoClose, memoResolve):
//   - while recording, the STRUCTURE: the pass (attachment formats and sizes, actions, clear colours), each draw's
//     program (its uid), fixed-function state (Draw's fields, render::Options' bytes), the samplers' words, each
//     sampled texture's identity (resourceId) and the write epoch of its memory (Buffer::writes: every GPU write this
//     process records and every Texture::replace bumps it);
//   - at commit, the CONTENTS the GPU will read that a program can still change after encoding: each bound buffer's
//     bytes (the bound range: Binding::bytes when the caller knows it, else to the buffer's end up to a cap), the index
//     ranges, and the rows of textures whose memory is written where the driver does not see it (a texture over a
//     buffer's rows, an IOSurface) -- unless the queue's version hook (setMemoVersion) answers for them.
// The words are replaced at commit (CommandQueue::submit), never before: Metal lets the program write a shared buffer
// until then.
//
// When the result does not depend on the target's prior contents:
//   - colour attachment 0's load action is Clear (the clear covers the whole level: what is kept and copied is the
//     whole level) or DontCare (what no draw writes is undefined anyway: the union of the draws' areas is copied);
//   - Load, only in mode 2 and only when the caller vouches (memoHint) that the pass's first draw is an axis-aligned
//     rectangle and the core sees that its pipeline overwrites it (no blending, every channel written, no discard, no
//     framebuffer fetch): that rectangle is the COVER, and every later draw that writes attachment 0 must lie inside it
//     (its scissor and viewport area, or the bound the caller vouched for it). CoreAnimation's screen pass is such a
//     pass (load=load, a first opaque quad over the damaged rectangle); the vouching is the plug-in's reading of
//     CoreAnimation's vertex convention, as metal_hsr's is -- a convention, not a contract, hence a mode of its own.
// Refused (the pass runs as it is): a depth or stencil attachment, multisampling, attachments past 0 that are not
// memoryless (or that the next pass loads), a draw that writes device memory (ssboWrites), counts samples, is
// indirect, samples a texture it renders into, or binds more bytes than the cap.
//
// Cost model: every candidate pass is recorded with a copy target -> cache after it, NOPped at commit unless the pass
// is the same (structure and contents) as at the last frame -- a pass seen twice in a row is kept, so a pass that
// changes every frame costs nothing on the GPU; on a hit the pass is one 2D copy of its rectangle and the store is
// NOPped. One cache texture a slot (the target's format and size), made the first time a candidate is closed there.
// At commit the processor hashes the bound bytes and the volatile textures' rows (memo::Stats::hashedBytes, hashNs).
#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace hakios::metal {

class Buffer;
class Texture;

namespace memo {

// Off, Clear/DontCare passes alone, and those plus Load passes with a vouched cover
enum Mode : uint32_t { kOff = 0, kStrict = 1, kCover = 2 };

// 128 bits of a non-cryptographic hash, two independent 64-bit lanes (a multiply-xorshift each), eight bytes a step:
// what tells one frame's pass from another's. A collision of both lanes between two different passes is the one way
// this is wrong; 2^-128 a comparison for inputs nobody chose to collide.
struct Hash {
    uint64_t a = 0x243f6a8885a308d3ull, b = 0x13198a2e03707344ull;
    static uint64_t mixA(uint64_t x) {
        x *= 0x9e3779b97f4a7c15ull;
        return x ^ (x >> 29);
    }
    static uint64_t mixB(uint64_t x) {
        x *= 0xc2b2ae3d27d4eb4full;
        return (x << 31) | (x >> 33);
    }
    void u64(uint64_t v) {
        a = mixA(a ^ v);
        b = mixB(b + v);
    }
    void bytes(const void* p, size_t n) {
        const uint8_t* c = static_cast<const uint8_t*>(p);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            uint64_t v;
            std::memcpy(&v, c + i, 8);
            u64(v);
        }
        uint64_t tail = 0;
        if (i < n) std::memcpy(&tail, c + i, n - i);
        u64(tail ^ (uint64_t(n) << 56));
    }
    template <class T>
    void pod(const T& v) { bytes(&v, sizeof v); }
    bool operator==(const Hash& o) const { return a == o.a && b == o.b; }
    bool operator!=(const Hash& o) const { return !(*this == o); }
};

// What the queue counts (CommandQueue::memoStats): the passes looked at and what became of them
struct Stats {
    uint64_t candidates = 0;   // passes that could be kept (eligible, enough draws)
    uint64_t hits = 0;         // ...run as a copy of the kept result
    uint64_t misses = 0;       // ...run as they are
    uint64_t stores = 0;       // ...after which the result was kept (a copy into the cache)
    uint64_t refused = 0;      // passes that could not be (memo::Pass::why)
    uint64_t skippedDraws = 0; // the draws of the passes hit
    uint64_t hashedBytes = 0;  // contents hashed at commit
    uint64_t hashNs = 0;       // ...and the time it took (the processor's)
    uint64_t cacheBytes = 0;   // the cache textures held
    std::string lastRefusal;   // why the last refused pass was
    std::string lastMiss;      // what differed in the last miss of a pass that had been kept ("contents", "structure", ...)
};

// A buffer range whose bytes are hashed at commit
struct Range {
    const Buffer* buffer;
    uint64_t offset, length;
};

// A texture whose memory the driver does not see written: its rows hashed at commit, or its version asked of the hook
struct Volatile {
    const Texture* texture;
};

// What is gathered of the open pass (CommandBuffer::OpenPass::memo)
struct Pass {
    bool on = false;          // the queue's mode is not off: this pass is looked at
    bool ok = false;          // ...and nothing refused it so far
    std::string why;          // what refused it
    Hash structure;
    size_t begin = 0;         // the stream's words before the pass's own (after its start stamp)
    int profiled = -1;        // its entry among the profiled passes, -1 none
    uint32_t draws = 0;
    uint32_t area[4] = {~0u, ~0u, 0, 0};   // what the draws can write of attachment 0 (X0 Y0 X1 Y1, inclusive)
    bool covered = false;     // a Load pass whose first draw's cover was vouched for and judged opaque
    uint32_t cover[4] = {0, 0, 0, 0};
    std::vector<Range> ranges;
    std::vector<Volatile> volatiles;
    uint64_t budget = 0;      // the bytes the ranges and volatile textures come to (bounded: CommandQueue::setMemo)
    // the index of range R among the pass's ranges, added if new: one range bound by many draws (a matrix every draw of
    // CoreAnimation binds at the same offset) is hashed once, its index going into the structure where it is bound
    uint32_t rangeIndex(const Range& r) {
        for (size_t i = ranges.size(); i-- > 0 && i + 64 >= ranges.size();) {
            if (ranges[i].buffer == r.buffer && ranges[i].offset == r.offset && ranges[i].length == r.length) return uint32_t(i);
        }
        ranges.push_back(r);
        budget += r.length;
        return uint32_t(ranges.size() - 1);
    }
    uint32_t volatileIndex(const Texture* t, uint64_t bytes) {
        for (size_t i = 0; i < volatiles.size(); i++) {
            if (volatiles[i].texture == t) return uint32_t(i);
        }
        volatiles.push_back(Volatile{t});
        budget += bytes;
        return uint32_t(volatiles.size() - 1);
    }
    void refuse(const char* reason) {
        if (ok) why = reason;
        ok = false;
    }
};

// The caller's word for the next draw (CommandBuffer::memoHint): where its vertices land, in pixels (X0 Y0 X1 Y1,
// not clipped), and whether they are exactly that axis-aligned rectangle (two triangles over its four corners)
struct Hint {
    bool set = false;
    bool rect = false;
    float box[4] = {0, 0, 0, 0};
};

}  // namespace memo
}  // namespace hakios::metal
