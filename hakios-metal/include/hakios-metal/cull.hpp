// Where a draw's vertices land, worked out on this processor, and what that says about covering: the geometry half of
// CommandBuffer::setCull (command.hpp), which leaves out a draw that later opaque rectangles of its pass cover whole.
// dev-notes/investigacion/anexos/33-panel-lateral/10-oclusion-exacta.md has the argument; this is its arithmetic.
//
// Nothing here is a convention of whoever draws. A draw's positions are what its vertex stage's own machine code (the
// ir3 words the GPU runs) makes of the bytes the GPU will fetch, run here by an interpreter of the few instructions a
// position that is a matrix times an attribute compiles to (mul.f, add.f, mad.f32, mov, nop, end). A stage with any
// instruction whose effect on the registers this does not know (a branch, a texture or memory read, a write to an
// address register) has no model: its draws neither cover nor are left out. An instruction of categories 2 to 4 that
// is not run only makes the registers it writes unknown, and a position that reads one is unknown.
//
// Every value carries a bound on how far the GPU's own result may lie from it: 0 for what is read from memory and for
// an operation whose exact result a float holds whatever the rounding (a product by 0, a sum with 0, small whole
// numbers), kUlps units in the last place otherwise, for each rounding the GPU may do (a mad.f32 fused or not). The
// bound goes through the division by w and the viewport, and kMargin of a pixel is added for what the hardware does
// after the vertex stage (its divide, its viewport transform in floats, the snap to its subpixel grid: 8 bits on this
// chip, turnip's subPixelPrecisionBits, tu_device.cc). A covering rectangle is taken that much INSIDE its corners and
// a covered draw's box that much OUTSIDE its vertices, so no answer rests on how the rasterizer breaks a tie on an
// edge: only on a pixel being drawn when its centre (x + 0.5, y + 0.5) lies inside a triangle by more than the snap.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "hakios-metal/render.hpp"

namespace hakios::metal {

class Buffer;
struct Bindings;

namespace cull {

// pixels, both ends inside (x0 > x1 or y0 > y1: none)
struct Rect {
    int32_t x0 = 0, y0 = 0, x1 = -1, y1 = -1;
    bool empty() const { return x0 > x1 || y0 > y1; }
    uint64_t pixels() const { return empty() ? 0 : uint64_t(x1 - x0 + 1) * uint64_t(y1 - y0 + 1); }
    bool holds(const Rect& o) const { return o.x0 >= x0 && o.y0 >= y0 && o.x1 <= x1 && o.y1 <= y1; }
};

constexpr uint32_t kMaxVertices = 64;   // of a draw whose box is worked out (a draw of more is neither covered nor covering)
constexpr uint32_t kMaxSources = 6;     // the buffers a stage's position may come from: its attributes' and its constants'
constexpr uint32_t kMaxRegion = 8;      // rectangles a draw's still uncovered part is kept as (Region)
constexpr uint32_t kMaxRangeDwords = 64;   // a pushed constant range read whole (CoreAnimation's are 16)
// A covering rectangle smaller than this on a side is not held against the draws before it: working out where each
// of them lands costs more than a small rectangle's worth of pixels saves. Chromium's compositor has the same rule
// for the same reason (viz's occlusion culler takes no occluder under its minimum size). Nothing of exactness in it.
constexpr int32_t kMinCoverer = 32;
// what the hardware may move a vertex by after the vertex stage, in pixels: 32 steps of its 1/256 grid
constexpr double kMargin = 0.125;
// units in the last place a GPU operation's result may be off by (1 is any rounding of the exact result)
constexpr double kUlps = 2.0;

// why a draw's positions are not known, or why they make no covering rectangle
enum Why : uint8_t {
    kOk = 0,
    kNoModel,    // the vertex stage has an instruction outside the subset, or no position
    kShape,      // the draw itself: indirect, instanced, too many vertices, not triangles, a restart index in a quad
    kNoData,     // a buffer not bound, not readable here, written by the GPU, or too short for what is fetched
    kUnknown,    // the position reads something the interpreter does not follow
    kW,          // a vertex's w is not provably above 0
    kRange,      // not finite, or the bound on the error too wide
    kNotRect,    // (a coverer) its two triangles are not the two halves of an axis-aligned rectangle
    kClip,       // (a coverer) a vertex not provably inside the near and far planes and the guard band
    kSmall,      // (a coverer) a rectangle under kMinCoverer pixels on a side
    kWhyCount,
};
const char* whyName(Why w);

// A vertex stage as the interpreter takes it, copied out of its Program (which the draw's caller may let go of before
// the pass ends): the instructions to its `end`, where each attribute comes from, the constant ranges, the immediates.
struct VsModel {
    bool usable = false;
    std::vector<uint32_t> words;   // (lo, hi) of each instruction, the `end` included
    uint32_t registers = 0;        // the full registers it uses, components counted (maxReg + 1) * 4
    uint32_t positionRegid = 0;
    struct Input {
        uint8_t regid, compmask;
        uint8_t comps;             // the floats fetched (1 to 4), 0: a format the interpreter does not read
        uint8_t source;            // which of the sources its buffer is
        uint32_t offset, stride;
    };
    std::vector<Input> inputs;
    struct Range {
        uint8_t source;
        uint32_t offsetVec4, startDword, sizeDwords;
    };
    std::vector<Range> ranges;
    std::vector<uint32_t> immediates;
    uint32_t immediatesAt = 0;     // the const file's dword the first lands at
    uint32_t sources = 0;
    uint8_t sourceIndex[kMaxSources] = {};   // each source's [[buffer(n)]]
};
// PROGRAM's vertex stage (never null: `usable` says). One made for a uid is kept and found again by it.
std::shared_ptr<const VsModel> modelOf(const Program& program);
// for the tests: the registers' components an instruction pair writes or reads as the interpreter decodes it, one line
// ("mul.f r2.x, c0.x, r0.x"), or "?" for one it does not run
std::string disassemble(uint32_t lo, uint32_t hi);

struct Source {
    const Buffer* buffer = nullptr;
    uint64_t offset = 0;
};
// what of a draw its positions depend on, taken as it is encoded
struct Geometry {
    std::shared_ptr<const VsModel> vs;
    Source src[kMaxSources];
    Primitive primitive = Primitive::kTriangle;
    uint32_t vertexStart = 0, vertexCount = 0;
    const Buffer* indexBuffer = nullptr;
    uint64_t indexOffset = 0;
    uint32_t indexType = 0;
    int32_t baseVertex = 0;
    float xoffset = 0, xscale = 0, yoffset = 0, yscale = 0;   // GRAS_CL_VPORT's, as a6xx::viewportWords makes them
    double guardX = 0, guardY = 0;   // the NDC extent inside which nothing is clipped, with room to spare (0: none known)
    Rect area;                       // render::drawArea's: the viewport and the scissor in the target
};
// kOk and *OUT, or why DRAW's positions will not be worked out (kShape, kNoModel, kNoData for a buffer not bound)
Why capture(const Program& program, const Draw& draw, const Bindings& vsBindings, const uint32_t area[4], Geometry* out);

// The bytes the answers were worked out from, kept so that they can be held against what the buffers hold when the
// command buffer is committed (Metal reads a buffer when the GPU runs the draw: a program may write it until then).
class Snapshots {
public:
    // LENGTH bytes at OFFSET of B into OUT and into the record; false when B cannot be read here, the GPU has written
    // it (its bytes here are then not what the draw will fetch), or the bytes are not all inside it
    bool take(const Buffer& b, uint64_t offset, uint32_t length, void* out);
    uint32_t mark() const { return uint32_t(records_.size()); }
    // whether records [FIRST, FIRST + COUNT) still read as they did
    bool same(uint32_t first, uint32_t count) const;
    // OTHER's records after this one's (a command buffer appended to another); the number the first of them now has
    uint32_t absorb(Snapshots& other);
    void clear() {
        bytes_.clear();
        records_.clear();
    }
    size_t bytes() const { return bytes_.size(); }

private:
    struct Record {
        const Buffer* buffer;
        uint64_t offset;
        uint32_t length, at;
    };
    std::vector<uint8_t> bytes_;
    std::vector<Record> records_;
};

// a draw's vertices in the target's pixels (x right, y down, a pixel's centre at +0.5)
struct Positions {
    uint32_t count = 0;
    bool restart = false;   // an index was the restart index (left out)
    struct P {
        double x, y;        // where the interpreter puts it
        double u;           // how far from there the hardware may put it, kMargin included
        bool inside;        // provably inside the near and far planes and the guard band
    } p[kMaxVertices];
};
Why evaluate(const Geometry& g, Snapshots* snapshots, Positions* out);
// every pixel a triangle of the draw can touch, inside its area (empty: none); the draw must be of triangles
Rect coverBox(const Geometry& g, const Positions& p);
// kOk and the pixels the draw's two triangles are sure to cover, inside its area; or why they are not such a rectangle
Why occluderRect(const Geometry& g, const Positions& p, Rect* out);

// what is left of a rectangle as rectangles are taken out of it
struct Region {
    uint32_t n = 0;
    Rect r[kMaxRegion];
    void set(const Rect& a) {
        n = a.empty() ? 0 : 1;
        r[0] = a;
    }
    bool empty() const { return n == 0; }
    // R taken out; false, nothing changed, when what is left would be more than kMaxRegion rectangles
    bool subtract(const Rect& o);
};

}  // namespace cull
}  // namespace hakios::metal
