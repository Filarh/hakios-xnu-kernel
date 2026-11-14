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
#pragma once

#include <cstdint>
#include <vector>

namespace hakios::metal::render {

// The groups, in the order the command processor runs the pending ones (lowest id first: the phone's microcode, and
// the order both witnesses rely on -- turnip's TU_DRAW_STATE_PROGRAM_CONFIG and fd6's FD6_GROUP_PROG_CONFIG are 0):
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
enum DrawStateGroup : uint32_t { kDsPass = 0, kDsProgram = 1, kDsVertex = 2, kDsLoads = 3, kDsRaster = 4, kDsViewport = 5, kDsBlend = 6, kDsGroups = 7 };

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
    // the fragments now at GPU address BLOBS_ADDRESS: LIST's entries point there
    void place(uint64_t blobsAddress);
};

// BODY's WORDS as draw-state groups, each for the modes MODES names (CP_SET_DRAW_STATE__0_BINNING | _GMEM | _SYSMEM);
// false, with *OUT unspecified, when the stream holds something this does not know how to move into a group (the pass
// is then drawn from the stream as it is). TWIN: the fake's twins (Options::gmemTwin).
bool drawStates(const uint32_t* body, size_t words, uint32_t modes, DrawStates* out, uint8_t twin = 0);

}  // namespace hakios::metal::render
