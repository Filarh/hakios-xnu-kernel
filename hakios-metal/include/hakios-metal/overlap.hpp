// Whether the primitives of one draw can cover a pixel twice. SPDX-License-Identifier: MIT
//
// A draw whose fragment stage reads its target (framebuffer fetch: Metal's programmable blending) has to read, at a
// pixel an earlier primitive of the SAME draw wrote, what that primitive left. In a sysmem pass the rasterizer keeps
// that order when GRAS_SC_CNTL.SINGLE_PRIM_MODE is FLUSH_PER_OVERLAP_AND_OVERWRITE (a6xx.xml:3169-3193 the enum,
// :3212 the field; fd6_emit.cc:564-570 and tu_pipeline.cc:3373-3376 set it for a sysmem draw that reads its target).
// The order of a fetching draw after the draws BEFORE it is not the mode's but the barrier's (render.cpp: the CCU
// flush and CACHE_INVALIDATE before each fetching draw of a sysmem pass). So, as far as the ORDER goes, a draw none
// of whose primitives shares a pixel with another of its primitives is drawn the same with NO_FLUSH -- and on the
// Adreno 610 the mode is dear for a draw of many primitives: CoreAnimation's plus-darker text, 359 to 768 lines a
// draw, took 448 us a draw with it and 28.5 without (the phone, 26 November 2026, one set of traces of 160 command
// buffers each way); its quads, two triangles each, 20.6 against 18.6.
//
// What NO_FLUSH can cost besides the order is the mode's other half, and it is not about overlap: on a compressed
// (UBWC) target "there's the possibility that the flags for the current pixel are flushed before the data or
// vice-versa, leading to texture fetches via UCHE getting out of sync values" (a6xx.xml:3171-3181, of this very mode;
// tu_pipeline.cc:3365-3372 gives it as the reason turnip sets it in sysmem). Nothing proved here answers that, so a
// draw that reads a compressed attachment keeps the mode whatever its primitives are (render.cpp: the draw's
// exemption), unless the caller asks otherwise for a measurement (CommandBuffer::kSysmemFetchUnorderedUbwc).
//
// primitivesOverlap answers for one draw, and says never only when that is proved:
//
//   1. A draw of at most ONE primitive, one instance, its counts in the Draw (not indirect). One primitive gives each
//      sample at most one fragment, whatever the vertex stage computes.
//
//   2. A draw of exactly TWO triangles, one instance, that are the two halves of a quad: they share an edge and lie on
//      opposite sides of it, by a margin, as the GPU will place them. What that takes:
//
//      a. Where the GPU puts each vertex. The vertex stage's machine code (ir3, the words the GPU runs) is walked
//         with the attributes as unknowns and the constant file as the numbers it will hold (overlap.cpp: Walk).
//         Taken: a straight line of mov, add.f, mul.f, absneg.f and mad.f32 on full registers, nop, stp and end --
//         every register then an affine function of the attributes, the product of two that both depend on an
//         attribute refused. Anything else that writes a register this follows makes that register unknown; anything
//         it cannot follow at all (a branch, a load, another category) ends the walk without an answer. A position
//         register left unknown is no answer. So the position is a function of the attributes and constants alone --
//         not of the vertex's index, which is a register the walk never knows -- and the same attribute values give
//         the same position, bit for bit, since the GPU runs the same instructions on the same inputs.
//      b. The shared edge. Two vertices are the same when every attribute component the position depends on has the
//         same value in both (or they are one index). Exactly two of the first triangle's vertices must be the same
//         as two of the second's: A and C. The GPU's positions of A and C are then identical in both triangles.
//      c. No clipping. A clipped triangle gets new vertices, computed per triangle, and nothing says the two
//         triangles' new vertices on the shared edge agree. So for all four vertices: w above zero by more than its
//         error; z either the same value for all four (every attribute component z depends on the same in the
//         four: every vertex is then on the same side of the near plane, CoreAnimation's z = 0 among them) or above
//         zero by more than its error, and likewise w - z; x and y inside the guardband the driver programs
//         (GRAS_CL_GUARDBAND_CLIP_ADJ, a6xx.hpp: calcGuardband), inside which the rasterizer does not clip but
//         scissors (fd_calc_guardband, freedreno_guardband.h:17-75).
//      d. Opposite sides, robustly. In window coordinates, with B and D the two vertices that are not shared:
//         cross(C - A, B - A) and cross(C - A, D - A) of opposite signs, each larger than what moving every vertex by
//         DELTA can change it by, 2 DELTA (|C - A| + |B - A|) + 4 DELTA^2. DELTA is the rasterizer's snap to its
//         subpixel grid (kSubpixelSnap) plus the bound on the difference between the position computed here in
//         double and the GPU's in float32 (gamma_n times the sum of the terms' magnitudes, n the operations on the
//         path, with one ulp an operation -- twice what a correctly rounded operation loses -- and a mad counted as
//         two). After the snap B' and D' are then still strictly on opposite sides of the line A'C', the two
//         triangles' interiors are disjoint, and the rasterizer's fill rule gives a sample on the shared edge to
//         exactly one of them (the top-left rule Metal, Vulkan and Direct3D all require of a rasterizer, which is
//         what makes a mesh of triangles sharing edges draw each sample once; that the Adreno keeps it is taken from
//         its conformance, not measured here).
//
//   3. A draw of LINES (a list, kMaxLines at most, one instance) no two of which can reach the same pixel: each
//      line's pixels are those whose centres lie in its segment's bounding box grown by sqrt(2) times the half width
//      plus DELTA (overlap.cpp says why that holds every pixel a line can cover), and the lines' pixel sets are
//      disjoint. Taken when every line's set is within one pixel row -- CoreAnimation's coverage spans, horizontal
//      at pixel centres, which are where a plus-darker text's time went: 359 to 768 lines a draw, 448 us a draw with
//      the mode and 28.5 without. Runs on a row must have a clear pixel between them.
//
// Everything else may overlap: three triangles or more (CoreAnimation's nine-patch pieces, 12 and 24 indices), points
// and line strips of more than one primitive, strips of more than two triangles, instances, indirect draws, primitive
// restart indices, a vertex stage the walk cannot follow, an attribute in a format other than floats and halfs.
//
// Metal lets the program write a buffer until the command buffer is committed, and the GPU reads what is there when it
// runs. So an answer of cases 2 and 3 comes with its WITNESS: a copy of every byte it was decided from (the indices,
// the vertices' attributes, the constant ranges), each with where it was read. The answer holds for the bytes as
// copied; CommandBuffer keeps the witness and looks again at commit (OverlapWitness::holds), and puts the ordering
// mode back in the stream for a draw whose bytes changed (render.cpp: settleFetches).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hakios-metal/render.hpp"

namespace hakios::metal::render {

// The bytes the answers of a command buffer's draws were decided from, one after another.
struct OverlapWitness {
    static constexpr size_t kMaxBytes = 64 * 1024;   // a draw's, at most (1024 lines of 16-byte vertices are 32 KiB)
    struct Span {
        const uint8_t* live;   // where the bytes were read, in a buffer the command buffer keeps until it completes
        size_t at, length;     // their copy in BYTES
    };
    std::vector<Span> spans;
    std::vector<uint8_t> bytes;
    size_t first = 0;          // where the draw being answered began in BYTES

    // LENGTH bytes at LIVE copied: where in BYTES, or SIZE_MAX when the draw's would pass kMaxBytes
    size_t take(const uint8_t* live, size_t length);
    // spans FROM to TO (not included) are still what was copied
    bool holds(size_t from, size_t to) const;
    void clear() {
        spans.clear();
        bytes.clear();
        first = 0;
    }
};

enum class Overlap : uint8_t {
    kMay = 0,     // not proved otherwise: the ordering mode
    kNever = 1,   // proved; for the bytes W took, when it took any
};

// DRAW's primitives against each other, drawn with PROGRAM over VS_BINDINGS (the vertex stage's buffers: its
// attributes' and its constants'). What the answer was decided from is added to W (nothing for kMay).
Overlap primitivesOverlap(const Program& program, const Draw& draw, const Bindings& vsBindings, OverlapWitness* w);

// The rasterizer's snap, in pixels: what a vertex can move by when its window coordinates are put on the subpixel
// grid. The grid is 2^-8 pixel (tu_device.cc:1028: subPixelPrecisionBits = 8, the Vulkan limit turnip reports for
// every a6xx); rounded to nearest a coordinate moves by half a step, truncated by a whole one, so a vertex by at most
// sqrt(2) 2^-8 either way: 2^-7 covers both.
constexpr double kSubpixelSnap = 1.0 / 128.0;
// A line's half width in pixels as the driver programs it: GRAS_SU_CNTL.LINEHALFWIDTH, two quarters (a6xx.hpp:
// grasSuCntl's default, turnip's for a width of 1.0, tu_pipeline.cc:3132)
constexpr double kLineHalfWidth = 0.5;

}  // namespace hakios::metal::render
