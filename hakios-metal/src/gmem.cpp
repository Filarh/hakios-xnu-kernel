// A tiled pass's draws as draw-state groups (gmem.hpp). SPDX-License-Identifier: MIT
#include "hakios-metal/gmem.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

#include "hakios-metal/a6xx.hpp"
#include "hakios-metal/pm4.hpp"

namespace hakios::metal::render {

namespace {

using namespace a6xx;

// Which group a register is kept in. Only how much is run again when something changes hangs on this -- a fragment
// always holds every register of its group -- so the rule is by what changes together in a compositor's draws: the
// tables a draw binds, the vertex fetch, the rasterizer and viewport, the blend, what the attachments decide; whatever
// is none of those is the program's (the two stages, their link, the fragment stage's inputs and outputs), and so is
// any register this does not know.
DrawStateGroup groupOf(uint32_t r) {
    static_assert(REG_A6XX_VFD_CONTROL_0 == 0xa000 && REG_A6XX_VFD_POWER_CNTL == 0xa0f8 && REG_A6XX_PC_RESTART_INDEX == 0x9803, "the vertex fetch's registers");
    static_assert(REG_A6XX_SP_VS_TEX_COUNT == 0xa822 && REG_A6XX_SP_VS_TEX_SAMP == 0xa8a0 && REG_A6XX_SP_VS_TEX_CONST == 0xa8a8 && REG_A6XX_SP_FS_TEX_COUNT == 0xa9a7 &&
                      REG_A6XX_SP_FS_TEX_SAMP == 0xa9e0 && REG_A6XX_SP_FS_TEX_CONST == 0xa9e4 && REG_A6XX_SP_IBO == 0xab1a && REG_A6XX_SP_IBO_COUNT == 0xab20,
                  "the tables' registers");
    static_assert(REG_A6XX_GRAS_CL_CNTL == 0x8000 && REG_A6XX_GRAS_CL_GUARDBAND_CLIP_ADJ == 0x8006 && REG_A6XX_GRAS_CL_VPORT_XOFFSET == 0x8010 &&
                      REG_A6XX_GRAS_CL_Z_CLAMP_MIN == 0x8070 && REG_A6XX_GRAS_SU_CNTL == 0x8090 && REG_A6XX_GRAS_SU_POINT_SIZE == 0x8092 &&
                      REG_A6XX_GRAS_SU_POLY_OFFSET_SCALE == 0x8095 && REG_A6XX_GRAS_SC_CNTL == 0x80a0 && REG_A6XX_GRAS_SC_SCREEN_SCISSOR_TL == 0x80b0 &&
                      REG_A6XX_GRAS_SC_VIEWPORT_SCISSOR_TL == 0x80d0 && REG_A6XX_RB_Z_CLAMP_MIN == 0x88c0 && REG_A6XX_VPC_UNKNOWN_9107 == 0x9107 &&
                      REG_A6XX_VPC_POLYGON_MODE == 0x9108 && REG_A6XX_PC_RASTER_CNTL == 0x9980 && REG_A6XX_PC_POLYGON_MODE == 0x9981 &&
                      REG_A6XX_PC_PRIMITIVE_CNTL_0 == 0x9b00 && REG_A6XX_RB_UNKNOWN_8A00 == 0x8a00 && REG_A6XX_RB_UNKNOWN_8A30 == 0x8a30,
                  "the rasterizer's registers");
    static_assert(REG_A6XX_RB_MRT_CONTROL == 0x8820 && REG_A6XX_RB_MRT_BUF_INFO == 0x8822 && REG_A6XX_RB_MRT_BASE_GMEM == 0x8827 &&
                      REG_A6XX_RB_BLEND_RED_F32 == 0x8860 && REG_A6XX_RB_BLEND_CNTL == 0x8865 && REG_A6XX_SP_BLEND_CNTL == 0xa989 &&
                      REG_A6XX_RB_DEPTH_PLANE_CNTL == 0x8870 && REG_A6XX_RB_DEPTH_BUFFER_INFO == 0x8872 && REG_A6XX_RB_Z_BOUNDS_MIN == 0x8878 &&
                      REG_A6XX_RB_STENCIL_CONTROL == 0x8880 && REG_A6XX_RB_STENCIL_INFO == 0x8881 && REG_A6XX_RB_STENCILREF == 0x8887 &&
                      REG_A6XX_GRAS_SU_DEPTH_PLANE_CNTL == 0x8094 && REG_A6XX_GRAS_SU_DEPTH_CNTL == 0x8114 && REG_A6XX_GRAS_SU_STENCIL_CNTL == 0x8115,
                  "the blend's and the depth test's registers");
    static_assert(REG_A6XX_GRAS_SU_DEPTH_BUFFER_INFO == 0x8098 && REG_A6XX_RB_DEPTH_FLAG_BUFFER_BASE == 0x8900 && REG_A6XX_RB_MRT_FLAG_BUFFER == 0x8903 &&
                      REG_A6XX_SP_FS_MRT_REG == 0xa996 && REG_A6XX_GRAS_LRZ_MRT_BUF_INFO_0 == 0x8102 && REG_A6XX_RB_DITHER_CNTL == 0x880e &&
                      REG_A6XX_RB_SRGB_CNTL == 0x880f && REG_A6XX_SP_SRGB_CNTL == 0xa98a && REG_A6XX_GRAS_MAX_LAYER_INDEX == 0x8004 &&
                      REG_A6XX_SP_TP_RAS_MSAA_CNTL == 0xb300 && REG_A6XX_SP_TP_SAMPLE_CONFIG == 0xb304 && REG_A6XX_GRAS_RAS_MSAA_CNTL == 0x80a2 &&
                      REG_A6XX_RB_RAS_MSAA_CNTL == 0x8802 && REG_A6XX_RB_BLIT_GMEM_MSAA_CNTL == 0x88d5 && REG_A6XX_RB_RENDER_CNTL == 0x8801 &&
                      REG_A6XX_GRAS_LRZ_CNTL == 0x8100 && REG_A6XX_RB_LRZ_CNTL == 0x8898 && REG_A6XX_GRAS_LRZ_BUFFER_BASE == 0x8103,
                  "the pass's registers");
    // the vertex fetch: VFD_CONTROL_0..6, the offsets, the 32 fetches, the decodes and destinations; the restart index
    if ((r >= 0xa000 && r < 0xa0f8) || r == 0x9803) return kDsVertex;
    // the texture and sampler tables of the two stages and the IBO table: their pointers and counts (the loads follow)
    if (r == 0xa822 || r == 0xa9a7 || (r >= 0xa8a0 && r < 0xa8a2) || (r >= 0xa8a8 && r < 0xa8aa) || (r >= 0xa9e0 && r < 0xa9e2) || (r >= 0xa9e4 && r < 0xa9e6) ||
        (r >= 0xab1a && r < 0xab1c) || r == 0xab20) {
        return kDsLoads;
    }
    // the guardband, the 16 viewports and Z clamps, the screen and viewport scissors, RB_Z_CLAMP
    if (r == 0x8006 || (r >= 0x8010 && r < 0x8090) || (r >= 0x80b0 && r < 0x80f0) || r == 0x88c0 || r == 0x88c1) return kDsViewport;
    // GRAS_CL_CNTL, GRAS_SU_CNTL, the point and polygon offset registers, GRAS_SC_CNTL, the polygon modes,
    // PC_RASTER_CNTL, PC_PRIMITIVE_CNTL_0, the four shading-rate registers
    if (r == 0x8000 || (r >= 0x8090 && r < 0x8094) || (r >= 0x8095 && r < 0x8098) || r == 0x80a0 || r == 0x9107 || r == 0x9108 || r == 0x9980 || r == 0x9981 ||
        r == 0x9b00 || r == 0x8a00 || r == 0x8a10 || r == 0x8a20 || r == 0x8a30) {
        return kDsRaster;
    }
    // RB_MRT[i]: CONTROL and BLEND_CONTROL are the blend's, the six after them the buffer's
    if (r >= 0x8820 && r < 0x8860) return ((r - 0x8820) & 7) < 2 ? kDsBlend : kDsPass;
    // the blend constants, RB_ALPHA_CONTROL, RB_BLEND_CNTL, SP_BLEND_CNTL; the depth plane and test, the Z bounds, the
    // stencil test and its reference and masks, and GRAS's copies of the enables
    if ((r >= 0x8860 && r < 0x8866) || r == 0xa989 || r == 0x8870 || r == 0x8871 || r == 0x8878 || r == 0x8879 || r == 0x8880 || (r >= 0x8887 && r < 0x888a) ||
        r == 0x8094 || r == 0x8114 || r == 0x8115) {
        return kDsBlend;
    }
    // the depth and stencil buffers, the flag buffers, SP_FS_MRT_REG[8], the LRZ's copy of the format, dither, sRGB, the
    // layer count, the sample counts in the three blocks, RB_RENDER_CNTL, the LRZ registers
    if ((r >= 0x8872 && r < 0x8878) || (r >= 0x8881 && r < 0x8887) || r == 0x8098 || (r >= 0x8900 && r < 0x891b) || (r >= 0xa996 && r < 0xa99e) || r == 0x8102 ||
        r == 0x880e || r == 0x880f || r == 0xa98a || r == 0x8004 || r == 0xb300 || r == 0xb301 || r == 0xb304 || (r >= 0x80a2 && r < 0x80a5) ||
        (r >= 0x8802 && r < 0x8805) || r == 0x88d5 || r == 0x8801 || r == 0x8100 || r == 0x8898 || (r >= 0x8103 && r < 0x8108)) {
        return kDsPass;
    }
    return kDsProgram;
}

// one group as the walk keeps it
struct Group {
    std::vector<std::pair<uint32_t, uint32_t>> regs;   // what each of its registers holds now, by register
    std::vector<uint32_t> loads, loadsBefore;          // the draw's load packets of the group, whole and in order; the draw before's
    std::vector<uint32_t> wrote;                       // the registers the draw itself wrote (the twin's fragments)
    bool changed = false;                              // a register of it took another value in this draw
    bool invalidates = false, invalidatesBefore = false;   // (the program's) the draw wrote HLSQ_INVALIDATE_CMD...
    uint32_t invalidate = 0, invalidateBefore = 0;         // ...this value
    bool set = false;                                  // a fragment of it has been written...
    uint32_t at = 0, words = 0;                        // ...the last one: its word offset in the fragments, its length

    void write(uint32_t reg, uint32_t value) {
        wrote.push_back(reg);
        auto it = std::lower_bound(regs.begin(), regs.end(), reg, [](const std::pair<uint32_t, uint32_t>& a, uint32_t b) { return a.first < b; });
        if (it != regs.end() && it->first == reg) {
            changed = changed || it->second != value;
            it->second = value;
        } else {
            regs.insert(it, {reg, value});
            changed = true;
        }
    }
};

// registers into a stream as type-4 packets, the runs of consecutive ones together
void writeRegs(std::vector<uint32_t>& to, const std::vector<std::pair<uint32_t, uint32_t>>& regs, const std::vector<uint32_t>* only) {
    for (size_t i = 0; i < regs.size();) {
        if (only != nullptr && std::find(only->begin(), only->end(), regs[i].first) == only->end()) {
            i++;
            continue;
        }
        size_t n = 1;
        while (i + n < regs.size() && regs[i + n].first == regs[i].first + n && n < pm4::kMaxType4Count &&
               (only == nullptr || std::find(only->begin(), only->end(), regs[i + n].first) != only->end())) {
            n++;
        }
        to.push_back(pm4::type4(regs[i].first, uint32_t(n)));
        for (size_t k = 0; k < n; k++) to.push_back(regs[i + k].second);
        i += n;
    }
}

}  // namespace

void DrawStates::place(uint64_t blobsAddress) {
    for (uint32_t at : relocs) {
        const uint64_t a = blobsAddress + list[at];
        list[at] = uint32_t(a);
        list[at + 1] = uint32_t(a >> 32);
    }
    relocs.clear();
}

bool drawStates(const uint32_t* body, size_t words, uint32_t modes, DrawStates* out, uint8_t twin, uint32_t fetch) {
    DrawStates s;
    Group groups[kDsGroups];
    std::vector<uint32_t> fragment, barrier;   // (BARRIER: the draw's barrier packets, as recorded)
    bool copies = false, pending = false;   // a copy went to the prologue; state or a barrier waits for its draw
    bool armed = false;                     // the barrier's group is set and not yet let go
    uint32_t barrierAt = 0, barrierWords = 0;   // the barrier's fragment (one for the pass: they are all alike)
    const uint32_t ordered = A6XX_GRAS_SC_CNTL_CCUSINGLECACHELINESIZE(2) | A6XX_GRAS_SC_CNTL_SINGLE_PRIM_MODE(FLUSH_PER_OVERLAP);
    for (size_t i = 0; i < words;) {
        const pm4::Header h = pm4::decode(body[i]);
        if (h.kind == pm4::Header::kBad || h.count > words - i - 1) return false;
        const uint32_t* p = body + i + 1;
        const size_t next = i + 1 + h.count;
        if (h.kind == pm4::Header::kRegisters) {
            for (uint32_t k = 0; k < h.count; k++) {
                if (h.reg + k == REG_A6XX_HLSQ_INVALIDATE_CMD) {
                    // a command, not a value: it leads the program's fragment and is never one of the registers kept
                    groups[kDsProgram].invalidates = true;
                    groups[kDsProgram].invalidate = p[k];
                } else if (h.reg + k == REG_A6XX_GRAS_SC_CNTL && fetch == kDsFetchOrdered) {
                    groups[kDsRaster].write(h.reg + k, ordered);   // (every draw in raster order, fetching or not)
                } else {
                    groups[groupOf(h.reg + k)].write(h.reg + k, p[k]);
                }
            }
            pending = true;
        } else if (h.opcode == CP_LOAD_STATE6_GEOM || h.opcode == CP_LOAD_STATE6_FRAG || h.opcode == CP_LOAD_STATE6) {
            if (h.count < 3) return false;
            // the two stages' programs with the program; every other load -- constants, UBO tables, textures, samplers, the
            // IBO table -- with the draw's loads
            const uint32_t block = (p[0] & CP_LOAD_STATE6_0_STATE_BLOCK__MASK) >> CP_LOAD_STATE6_0_STATE_BLOCK__SHIFT;
            const uint32_t type = (p[0] & CP_LOAD_STATE6_0_STATE_TYPE__MASK) >> CP_LOAD_STATE6_0_STATE_TYPE__SHIFT;
            const bool program = h.opcode != CP_LOAD_STATE6 && type == ST6_SHADER && (block == SB6_VS_SHADER || block == SB6_FS_SHADER);
            std::vector<uint32_t>& to = groups[program ? kDsProgram : kDsLoads].loads;
            to.insert(to.end(), body + i, body + next);
            pending = true;
        } else if (h.opcode == pm4::kMemcpy) {
            s.prologue.insert(s.prologue.end(), body + i, body + next);
            copies = true;
        } else if (h.opcode == pm4::kWaitMemWrites || h.opcode == pm4::kWaitForMe || h.opcode == pm4::kNop) {
            // (the waits for a copy, which the prologue ends with once; appendDeduped's marker)
            if (h.count != 0 && h.opcode != pm4::kNop) return false;
        } else if (h.opcode == pm4::kWaitForIdle || (h.opcode == CP_EVENT_WRITE && h.count == 1)) {
            // a draw's barrier (a framebuffer fetch's): a group of its own, in the stream itself, or left out (gmem.hpp)
            if (h.opcode == CP_EVENT_WRITE && (p[0] & CP_EVENT_WRITE_0_EVENT__MASK) != CACHE_INVALIDATE) return false;
            if (fetch == kDsFetchInline) s.list.insert(s.list.end(), body + i, body + next);
            if (fetch == kDsFetchGroup) barrier.insert(barrier.end(), body + i, body + next);
            pending = true;
        } else if (h.opcode == CP_DRAW_INDX_OFFSET || h.opcode == CP_DRAW_INDIRECT_MULTI) {
            // the draw: a fragment for each group that is no longer what its last fragment holds, then the groups set
            uint32_t sent[kDsGroups], count = 0;
            // The barrier's group: set by a draw recorded with a barrier, let go by the first draw after it that the
            // rasterizer is not ordering (GRAS_SC_CNTL without a primitive mode: a draw that does not fetch, which
            // ends the run of fetching draws the barrier was for). 1: set, 2: let go.
            uint32_t barrierDoes = 0;
            {
                uint32_t scCntl = 0;
                for (const auto& [r, v] : groups[kDsRaster].regs) {
                    if (r == REG_A6XX_GRAS_SC_CNTL) scCntl = v;
                }
                const bool inOrder = (scCntl & A6XX_GRAS_SC_CNTL_SINGLE_PRIM_MODE__MASK) != 0;
                if (!barrier.empty()) {
                    if (barrierWords != barrier.size() || memcmp(&s.blobs[barrierAt], barrier.data(), 4 * barrier.size()) != 0) {
                        barrierAt = uint32_t(s.blobs.size());
                        barrierWords = uint32_t(barrier.size());
                        s.blobs.insert(s.blobs.end(), barrier.begin(), barrier.end());
                        s.fragments++;
                    }
                    barrierDoes = 1;
                    armed = true;
                    s.barriers++;
                    barrier.clear();
                } else if (armed && (!inOrder || twin == 7)) {   // (twin 7: let go at the very next draw, a fetching one too)
                    barrierDoes = 2;
                    armed = false;
                }
            }
            for (uint32_t g = kDsPass; g < kDsGroups; g++) {
                Group& G = groups[g];
                const bool differs = !G.set || G.changed || G.loads != G.loadsBefore || G.invalidates != G.invalidatesBefore ||
                                     (G.invalidates && G.invalidate != G.invalidateBefore);
                if (differs && (!G.regs.empty() || !G.loads.empty() || G.invalidates)) {
                    fragment.clear();
                    if (G.invalidates) {
                        fragment.push_back(pm4::type4(REG_A6XX_HLSQ_INVALIDATE_CMD, 1));
                        fragment.push_back(G.invalidate);
                    }
                    // (twin 1: only what the draw itself wrote -- right until a tile skips the draw that wrote the rest)
                    writeRegs(fragment, G.regs, twin == 1 ? &G.wrote : nullptr);
                    fragment.insert(fragment.end(), G.loads.begin(), G.loads.end());
                    if (fragment.empty() || fragment.size() > CP_SET_DRAW_STATE__0_COUNT__MASK) return false;
                    G.at = uint32_t(s.blobs.size());
                    G.words = uint32_t(fragment.size());
                    G.set = true;
                    s.blobs.insert(s.blobs.end(), fragment.begin(), fragment.end());
                    s.fragments++;
                    sent[count++] = g;
                }
                G.changed = false;
                G.loadsBefore.swap(G.loads);
                G.loads.clear();
                G.wrote.clear();
                G.invalidatesBefore = G.invalidates;
                G.invalidateBefore = G.invalidate;
                G.invalidates = false;
            }
            // the program's fragment invalidates what the stages had loaded: the draw's loads go with it, whatever they
            // are (twin 2: they do not, and are not DIRTY)
            const bool program = std::find(sent, sent + count, uint32_t(kDsProgram)) != sent + count;
            if (program && twin != 2 && groups[kDsLoads].set && std::find(sent, sent + count, uint32_t(kDsLoads)) == sent + count) {
                sent[count++] = kDsLoads;
                std::sort(sent, sent + count);
            }
            if (count != 0 || barrierDoes != 0) {
                s.list.push_back(pm4::type7(CP_SET_DRAW_STATE, 3 * (count + (barrierDoes != 0 ? 1 : 0))));
                if (barrierDoes == 1) {
                    // (DIRTY: every barrier is the same words at the same place, which would not run twice otherwise; for
                    // the tiles alone: a binning pass runs no fragment stage)
                    s.list.push_back(setDrawState0(barrierWords, CP_SET_DRAW_STATE__0_GMEM | CP_SET_DRAW_STATE__0_DIRTY, kDsBarrier));
                    s.relocs.push_back(uint32_t(s.list.size()));
                    s.list.push_back(4 * barrierAt);
                    s.list.push_back(0);
                    s.entries++;
                } else if (barrierDoes == 2) {
                    s.list.push_back(setDrawState0(0, CP_SET_DRAW_STATE__0_DISABLE, kDsBarrier));
                    s.list.push_back(0);
                    s.list.push_back(0);
                    s.entries++;
                }
                for (uint32_t k = 0; k < count; k++) {
                    const Group& G = groups[sent[k]];
                    // (every group DIRTY, gmem.hpp; twin 8: the loads' alone, twin 2: not even theirs)
                    const uint32_t dirty = (sent[k] == kDsLoads ? twin != 2 : twin != 8) ? CP_SET_DRAW_STATE__0_DIRTY : 0u;
                    s.list.push_back(setDrawState0(G.words, modes | dirty, sent[k]));
                    s.relocs.push_back(uint32_t(s.list.size()));
                    s.list.push_back(4 * G.at);
                    s.list.push_back(0);
                }
                s.entries += count;
            }
            s.list.insert(s.list.end(), body + i, body + next);
            s.draws++;
            pending = false;
        } else {
            return false;   // (a packet of no draw this knows: the pass is drawn from its stream as it is)
        }
        i = next;
    }
    if (pending || s.draws == 0) return false;   // (state after the last draw: nothing would run it)
    if (copies) {
        s.prologue.push_back(pm4::type7(pm4::kWaitMemWrites, 0));
        s.prologue.push_back(pm4::type7(pm4::kWaitForMe, 0));
    }
    *out = std::move(s);
    return true;
}

}  // namespace hakios::metal::render
