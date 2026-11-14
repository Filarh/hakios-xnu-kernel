// A tiled pass's draws as draw-state groups. SPDX-License-Identifier: MIT
//
// A GMEM pass's draws are recorded once and run once a tile (render.hpp: encodeGmem). Recorded as a plain stream -- each
// draw the registers it changes, its loads, its draw packet -- every tile's command processor reads and runs the state
// of every draw, whether the tile's visibility streams let the draw through or not: the streams skip the draw packet
// and nothing else. CoreAnimation's main pass on the Adreno 610, 108 draws in 102 tiles, was 490 138 words read a
// frame against 4 939 for the same draws in sysmem.
//
// The command processor has the means for it, which both of Mesa's drivers are built on (docs/drivers/freedreno.rst,
// "Draw states"): CP_SET_DRAW_STATE names a fragment of packets by a group id, and "at draw time, if the draw call is
// going to actually execute (some primitive is visible in the current tile), the SQE goes through the GROUP_IDs and
// for any with an update since the last time they were executed, it executes the corresponding fragment". A draw the
// tile skips costs its CP_SET_DRAW_STATE (3 words a group) and its draw packet. The price is the documentation's next
// sentence: "the generated draw state has to always update all of the state you have chosen to pack into that
// GROUP_ID, since any of your previous state changes in a previous draw state command may have been skipped".
//
// drawStates makes that form out of the pass's recorded stream, at the pass's end, without the draws being encoded any
// differently: it walks the stream draw by draw, keeps what every register written so far holds, and for each group a
// draw touches writes a fragment with ALL of the group's registers as they stand at that draw (not only the ones the
// draw changed) followed by the draw's loads of that group. Which register goes into which group decides only how much
// is run again when something changes: any split is correct, because every fragment is whole.
//
// Every group is set DIRTY. The command processor does not run again a group set to the address and length it last
// ran with -- the microcode's compare, against what its hardware kept of that run -- and whether a disable, a reset
// or another context's work clears what it kept is the hardware's part, which no source here shows. A pass's
// fragments lie in memory the command buffers take turns with, so the next frame's fragment for a group is often at
// the address the last frame's was, with other words in it or with the registers written over since by the passes
// drawn from the stream itself. DIRTY makes the group run whatever the processor remembers: the draws here set a
// group only when its fragment changed, so nothing runs twice for it (turnip sets it on the states a PROGRAM_CONFIG
// invalidates, tu_cmd_buffer.cc:1275-1313: the same bit, there for the same kind of reason).
#pragma once

#include <cstdint>
#include <vector>

namespace hakios::metal::render {

// The groups, in the order the command processor runs the pending ones (lowest id first: the phone's microcode, and
// the order both witnesses rely on -- turnip's TU_DRAW_STATE_PROGRAM_CONFIG and fd6's FD6_GROUP_PROG_CONFIG are 0):
//   kDsBarrier   a framebuffer fetch's barrier (below): CP_WAIT_FOR_IDLE, CACHE_INVALIDATE, CP_WAIT_FOR_IDLE. No state:
//                an act, run before everything else of the draw it runs at, in tiles alone
//   kDsPass      what the pass's attachments decide: RB_MRT's buffer registers, the depth buffer's, MSAA, RB_RENDER_CNTL, LRZ
//   kDsProgram   HLSQ_INVALIDATE_CMD first (turnip's and fd6's program config: tu6_emit_program_config, setup_config_stateobj),
//                then every register the two stages and their link decide, then the two ST6_SHADER preloads
//   kDsVertex    the vertex fetch: VFD_*, the restart index
//   kDsLoads     what a draw loads besides its programs: constants, UBO tables, the texture and sampler tables with
//                their registers, the IBO table. Always sent DIRTY, as turnip sends its states that hold CP_LOAD_STATE
//                (tu_cs_emit_draw_state, tu_cmd_buffer.cc:1275-1313: they "are invalidated by the PROGRAM_CONFIG draw
//                state"), and sent again with every kDsProgram: after the invalidate the loads must run whatever they are
//   kDsRaster    the rasterizer: culling, polygon mode, the primitive mode of a framebuffer fetch
//   kDsViewport  the viewport, the guardband, the Z clamps, the two scissors (a group of their own: 21 words of the
//                rasterizer's 49, and the ones a draw with a viewport of its own changes)
//   kDsBlend     blending, depth and stencil
enum DrawStateGroup : uint32_t {
    kDsBarrier = 0, kDsPass = 1, kDsProgram = 2, kDsVertex = 3, kDsLoads = 4, kDsRaster = 5, kDsViewport = 6, kDsBlend = 7, kDsGroups = 8,
};

// A framebuffer fetch's barrier. A draw whose fragment stage reads its target in a tile reads the bin through the
// texture caches, in raster order by GRAS_SC_CNTL.SINGLE_PRIM_MODE = FLUSH_PER_OVERLAP as both witnesses order it
// (fd6 build_prim_mode, fd6_emit.cc:533-555; turnip, tu_pipeline.cc:4468-4500). The rasterizer orders it after the
// fetching draws before it and knows nothing of draws that ran without the mode, so the first fetching draw after
// one that did not fetch is recorded with a barrier before it: wait for idle, drop the texture caches, wait
// (render.cpp: encodeDraw). In the plain stream that barrier is three packets every tile runs as it walks past,
// whether the tile draws any of the fetching draws or not: 102 tiles x 27 such places were 5 568 waits for idle a
// frame.
//   kDsFetchGroup    the barrier is a draw-state group of its own, set DIRTY by the draw it was recorded before and
//                    let go by the first draw after that run of fetching draws that does not fetch. Pending from the
//                    one to the other, it runs at the first draw of the run that a tile draws -- which is where a
//                    tile needs it -- and in no tile that draws none of them. (The microcode runs a group's packets
//                    through its one packet table and waits for the waits a group left pending before the draw.)
//                    The barrier each tile had before its draws, for the caches the tile before left, goes: the
//                    first fetch a tile draws has this one.
//   kDsFetchInline   in the stream itself, before the draw's groups are set, as recorded
//   kDsFetchOrdered  no barrier at all between draws: every draw of the pass runs FLUSH_PER_OVERLAP, fetching or not,
//                    so the rasterizer has every primitive to order a fetch after (GRAS_SC_CNTL rewritten in the
//                    recorded draws); each tile keeps the barrier before its draws. Not what either witness does, and
//                    what it costs the draws that do not fetch is unmeasured: the bisect's alternative.
enum DrawStateFetch : uint32_t { kDsFetchGroup = 0, kDsFetchInline = 1, kDsFetchOrdered = 2 };

struct DrawStates {
    // Packets to run once, before the binning pass: the copies of constants bound off a 16-byte boundary into the
    // draws' tables (CP_MEMCPY), which the plain stream ran before each draw's loads in every tile -- the source and
    // the destination are the same every time -- and one wait for them all
    std::vector<uint32_t> prologue;
    // every fragment's words, one after another; LIST's CP_SET_DRAW_STATE entries name them by byte offset until
    // place() makes those the GPU addresses
    std::vector<uint32_t> blobs;
    // the draws: for each, CP_SET_DRAW_STATE with the groups it changes (none when it changes none) and its draw packet,
    // with whatever a draw must do in the stream itself before them (a framebuffer fetch's barrier)
    std::vector<uint32_t> list;
    std::vector<uint32_t> relocs;   // the indices in LIST of each entry's ADDR_LO word
    uint32_t draws = 0, entries = 0, fragments = 0;   // draws listed, groups set over them all, fragments written
    uint32_t barriers = 0;          // barriers made a group's (kDsFetchGroup): the tiles then need none before their draws
    // the fragments now at GPU address BLOBS_ADDRESS: LIST's entries point there
    void place(uint64_t blobsAddress);
};

// BODY's WORDS as draw-state groups, each for the modes MODES names (CP_SET_DRAW_STATE__0_BINNING | _GMEM | _SYSMEM);
// false, with *OUT unspecified, when the stream holds something this does not know how to move into a group (the pass
// is then drawn from the stream as it is). FETCH: a DrawStateFetch. TWIN: the fake's twins (Options::gmemTwin).
bool drawStates(const uint32_t* body, size_t words, uint32_t modes, DrawStates* out, uint8_t twin = 0, uint32_t fetch = kDsFetchGroup);

}  // namespace hakios::metal::render
