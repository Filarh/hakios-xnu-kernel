// hakios-metal-hsrcheck: whether this GPU draws a pass with hidden surface removal by depth (CommandBuffer::setHsr, the
// plug-in's metal_hsr; dev-notes/investigacion/anexos/32-interfaz-telegram/02-hsr-por-profundidad.md) into the very
// picture it draws without it, byte for byte. It runs the driver's core on the phone's GPU from a process of its own
// (the kernel's msm client, as hakios-metal-tilecheck does: no Metal, no CoreAnimation, nothing of the window server
// touched), compiles its few shaders there with the driver's compiler (the fixtures' bitcode, tools/kernels/*.ll), and
// reads its results back through 2D copies into LINEAR textures.
//
// Each scene of the design's section 5 is one pass into a compressed (UBWC) BGRA8 colour 0 -- what version 1 serves:
// the screen -- with a second, linear, colour attachment where the scene has one, both filled first with a random
// picture: drawn without HSR, then with it from the same picture, and colour 0 and 1 (and a visibility result's word)
// compared. The scenes: opaque layers stacked; blending, a framebuffer fetch and a discard (tri_kill_fs) each behind
// and in front of an opaque draw; an opaque draw over part of the rest; a barrier (a write to colour 1) between a
// hidden draw and the opaque one over it; a visibility result; a Clear and then draws; two segments; two triangles of
// two colours over each other in one opaque draw; instances over each other (tri_base_vs: each its own colour); a
// vertex stage with an IBO (a barrier); a hidden draw of a barrier's shaders first in its segment; a pass past a draw
// limit lowered to 3 (CommandBuffer::setHsrTest). Each at 750 x 1624 (the screen) and 99 x 209.
// Then the twins that must SHOW (setHsrTest): 1 the prepass left out, 4 the shadow kept across a segment's hole, 5 the
// prepass tested Less (the overlapping triangles and the instances: two primitives of one draw at one Z) -- a twin
// that draws the same picture says the comparison could not see what it is there to see.
// And the GPU time of the stacked layers' pass at 750 x 1624, without HSR and with it (median of the repetitions).
//
// One line per check, "hsrcheck: ok ..." or "hsrcheck: FAIL ...", times as "hsrcheck: time ...", and a last line with
// the count; the exit status is the number of failures (100: no GPU).
//
//   hakios-metal-hsrcheck [--reps N] [--no-twins] [--no-time]
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "hakios-metal/command.hpp"
#include "hakios-metal/compiler.hpp"
#include "hakios-metal/texture.hpp"
#include "../tests/fixtures/tri_attr_vs_bc.hpp"
#include "../tests/fixtures/tri_base_vs_bc.hpp"
#include "../tests/fixtures/tri_cflat_fs_bc.hpp"
#include "../tests/fixtures/tri_color_fs_bc.hpp"
#include "../tests/fixtures/tri_dpos_vs_bc.hpp"
#include "../tests/fixtures/tri_fb_fs_bc.hpp"
#include "../tests/fixtures/tri_flat_vs_bc.hpp"
#include "../tests/fixtures/tri_kill_fs_bc.hpp"
#include "../tests/fixtures/tri_mrt_fs_bc.hpp"

using namespace hakios::metal;

namespace {

int failures = 0;

void report(bool ok, const std::string& what) {
    std::printf("hsrcheck: %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    std::fflush(stdout);
    failures += !ok;
}

bool completed(const CommandBuffer& cb) { return cb.status() == CommandBuffer::kCompleted; }

std::shared_ptr<Texture> makeTexture(Gpu& gpu, bool compressed, uint32_t w, uint32_t h, std::string* why) {
    TextureDesc d;
    d.format = PixelFormat::kBGRA8Unorm;
    d.width = w;
    d.height = h;
    d.usage = kTextureUsageRead | kTextureUsageRenderTarget;
    d.compressed = compressed;
    return Texture::make(gpu, d, why);
}

// a linear texture's W x H pixels, row after row
std::vector<uint8_t> pixelsOf(const Texture& t) {
    std::vector<uint8_t> out(uint64_t(t.width()) * t.height() * 4);
    const auto* src = static_cast<const uint8_t*>(t.contents());
    for (uint32_t y = 0; y < t.height(); y++) memcpy(out.data() + uint64_t(y) * t.width() * 4, src + y * t.bytesPerRow(), uint64_t(t.width()) * 4);
    return out;
}

void fillPixels(const Texture& t, const std::vector<uint8_t>& px) {
    auto* dst = static_cast<uint8_t*>(t.contents());
    for (uint32_t y = 0; y < t.height(); y++) memcpy(dst + y * t.bytesPerRow(), px.data() + uint64_t(y) * t.width() * 4, uint64_t(t.width()) * 4);
}

// how many pixels differ, and the first that does
std::string differ(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, uint32_t w, uint64_t* count) {
    *count = 0;
    if (a.size() != b.size() || a.empty()) {
        *count = ~0ull;
        return " (nothing to compare)";
    }
    std::string first;
    for (uint64_t p = 0; p < a.size() / 4; p++) {
        if (memcmp(a.data() + 4 * p, b.data() + 4 * p, 4) == 0) continue;
        if (*count == 0) {
            char t[120];
            std::snprintf(t, sizeof t, " (first at %llu,%llu: %02x%02x%02x%02x against %02x%02x%02x%02x)", (unsigned long long)(p % w),
                          (unsigned long long)(p / w), a[4 * p], a[4 * p + 1], a[4 * p + 2], a[4 * p + 3], b[4 * p], b[4 * p + 1], b[4 * p + 2], b[4 * p + 3]);
            first = t;
        }
        (*count)++;
    }
    return first;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v.size() % 2 ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
}

enum Kind : uint8_t {
    kFlat,        // opaque: tri_flat_vs + tri_cflat_fs, blending off, its own colour
    kBlend,       // hidden: the same blended (source over), its own colour at half alpha
    kFetch,       // hidden: tri_fb_fs, a framebuffer fetch
    kKill,        // hidden: tri_kill_fs, a checkerboard discarded
    kMrt,         // barrier: tri_mrt_fs, writes colour 1 too
    kVsIbo,       // barrier: tri_dpos_vs, its positions through an IBO
    kQuery,       // barrier: the blended draw counting its samples into a visibility result
    kOverlap,     // opaque: tri_attr_vs + tri_color_fs, a red triangle under a green one in one draw
    kInstances,   // opaque: tri_base_vs + tri_color_fs, three instances over each other, each its own colour
    kBlendC1,     // barrier: the blended draw with colour 1's write mask left whole, though its FS writes no colour 1
};
struct Step {
    Kind kind;
    double x, y, w, h;   // its viewport, in parts of the target
};
struct Scene {
    const char* name;
    std::vector<Step> steps;
    bool clear = false;   // the pass opens with a Clear of colour 0
    bool c1 = false;      // ...and has colour attachment 1 (every pipeline but tri_mrt's and kBlendC1's masks it off)
};

const Scene kScenes[] = {
    {"opaque layers stacked", {{kFlat, 0, 0, 1, 1}, {kFlat, .1, .1, .8, .8}, {kFlat, 0, 0, 1, 1}, {kFlat, .2, .3, .4, .3}}},
    {"blending behind and in front of an opaque draw", {{kBlend, 0, 0, 1, 1}, {kFlat, .1, .15, .5, .6}, {kBlend, .3, .3, .65, .6}}},
    {"a fetch behind and in front of an opaque draw", {{kFetch, 0, 0, 1, 1}, {kFlat, .1, .15, .5, .6}, {kFetch, .3, .3, .65, .6}}},
    {"a discard behind and in front of an opaque draw", {{kKill, 0, 0, 1, 1}, {kFlat, .1, .15, .5, .6}, {kKill, .3, .3, .65, .6}}},
    {"an opaque draw over part of the rest", {{kBlend, 0, 0, 1, 1}, {kFlat, 0, 0, .5, 1}, {kBlend, .2, 0, .6, 1}}},
    {"a barrier (colour 1) between a hidden draw and the opaque one over it",
     {{kBlend, 0, 0, 1, 1}, {kMrt, .2, .3, .4, .4}, {kFlat, 0, 0, 1, 1}, {kBlend, .05, .05, .3, .45}},
     false,
     true},
    {"a visibility result", {{kBlend, 0, 0, 1, 1}, {kQuery, .1, .15, .4, .6}, {kFlat, 0, 0, 1, 1}, {kBlend, .05, .05, .3, .45}}},
    {"a Clear, then draws", {{kFlat, 0, 0, .6, .6}, {kBlend, .2, .3, .7, .6}, {kFlat, .4, 0, .6, 1}}, true},
    {"two segments",
     {{kFlat, 0, 0, 1, 1}, {kBlend, .05, .05, .4, .6}, {kFlat, .3, 0, .7, 1}, {kMrt, 0, 0, .2, .3}, {kBlend, 0, 0, 1, 1}, {kFlat, .5, .45, .5, .55},
      {kFetch, .1, .15, .8, .8}},
     false,
     true},
    {"two triangles of two colours over each other in one opaque draw", {{kBlend, 0, 0, 1, 1}, {kOverlap, 0, 0, 1, 1}, {kBlend, .2, .3, .4, .45}}},
    {"instances over each other", {{kBlend, 0, 0, 1, 1}, {kInstances, .08, .12, .73, .78}, {kBlend, .2, .3, .4, .45}}},
    {"a vertex stage with an IBO (a barrier)", {{kBlend, 0, 0, 1, 1}, {kVsIbo, 0, 0, 1, 1}, {kFlat, .1, .15, .3, .45}}},
    {"a hidden draw of a barrier's shaders first in its segment", {{kBlendC1, 0, 0, 1, 1}, {kBlend, .05, .05, .6, .6}, {kFlat, 0, 0, 1, 1}}, false, true},
    {"seven draws, past a limit of 3",
     {{kFlat, 0, 0, 1, 1}, {kBlend, .05, .05, .4, .6}, {kFlat, .3, 0, .7, 1}, {kFlat, 0, 0, 1, 1}, {kBlend, 0, 0, 1, 1}, {kFlat, .5, .45, .5, .55},
      {kBlend, .1, .15, .8, .8}}},
};
constexpr size_t kLimitScene = 13;   // (drawn with setHsrTest's limit at 3)

struct Programs {
    Program flat, blend, fetch, kill, mrt, dpos, overlap, instances, blendC1;
};

// what one run of a scene left: colour 0 and 1 copied out, the visibility word, what HSR did, why it failed
struct Run {
    std::vector<uint8_t> c0, c1;
    uint64_t visible = 0;
    CommandBuffer::Counts counts;
    double us = 0;
    std::string why;
};

struct Rig {
    Gpu& gpu;
    CommandQueue& queue;
    const Programs& p;
    std::shared_ptr<Buffer> colours, pos, verts, vis;
};

// SCENE at W x H into fresh targets filled with PICTURE (and colour 1 with PICTURE1), with HSR ON or not, TWIN and
// LIMIT setHsrTest's
Run runScene(Rig& rig, const Scene& s, uint32_t W, uint32_t H, const std::vector<uint8_t>& picture, const std::vector<uint8_t>& picture1, bool on,
             uint8_t twin = 0, uint32_t limit = CommandBuffer::kHsrMaxDraws) {
    Run r;
    auto t = makeTexture(rig.gpu, true, W, H, &r.why);
    auto c1 = t != nullptr ? makeTexture(rig.gpu, false, W, H, &r.why) : nullptr;
    auto in = c1 != nullptr ? makeTexture(rig.gpu, false, W, H, &r.why) : nullptr;
    auto out0 = in != nullptr ? makeTexture(rig.gpu, false, W, H, &r.why) : nullptr;
    auto out1 = out0 != nullptr ? makeTexture(rig.gpu, false, W, H, &r.why) : nullptr;
    if (out1 == nullptr) return r;
    fillPixels(*in, picture);
    fillPixels(*c1, picture1);
    memset(rig.vis->contents(), 0xa5, 64);
    const render::BlitRect all{0, 0, W, H};
    // (the picture into the compressed target: a pass of its own, before the scene's)
    {
        auto cb = rig.queue.commandBuffer();
        if (!cb->blit(*in, 0, all, *t, 0, all, false)) {
            r.why = "the picture in: " + cb->error();
            return r;
        }
        cb->commit();
        cb->waitUntilCompleted();
        if (!completed(*cb)) {
            r.why = "the picture in: " + cb->error();
            return r;
        }
    }
    auto cb = rig.queue.commandBuffer();
    cb->setHsr(on);
    cb->setHsrTest(twin, limit);
    cb->setProfile(true);
    RenderPass pass;
    pass.color0 = t.get();
    if (s.c1) pass.more[0].texture = c1.get();
    for (size_t i = 0; i < s.steps.size(); i++) {
        const Step& st = s.steps[i];
        RenderPass p = pass;
        if (i == 0 && s.clear) {
            p.load = LoadAction::kClear;
            p.clearColor[0] = 0.25, p.clearColor[1] = 0.5, p.clearColor[2] = 0.75, p.clearColor[3] = 1;
        }
        Draw d;
        d.vertexCount = 6;
        d.viewport = Viewport{st.x * W, st.y * H, st.w * W, st.h * H, 0, 1};
        Bindings vsb, fsb;
        fsb.buffers = {Binding{rig.colours.get(), 16 * i}};   // (draw I's own colour)
        const Program* program = &rig.p.flat;
        switch (st.kind) {
        case kFlat: break;
        case kBlend: program = &rig.p.blend; break;
        case kFetch: program = &rig.p.fetch; fsb = Bindings{}; break;
        case kKill: program = &rig.p.kill; break;
        case kMrt: program = &rig.p.mrt; fsb = Bindings{}; break;
        case kVsIbo:
            program = &rig.p.dpos;
            vsb.buffers = {Binding{rig.pos.get(), 0}};
            break;
        case kQuery:
            program = &rig.p.blend;
            d.visibility = rig.vis.get();
            d.visibilityMode = 2;
            d.visibilityReset = true;
            break;
        case kOverlap:
            program = &rig.p.overlap;
            vsb.buffers = {Binding{rig.verts.get(), 0}};
            fsb = Bindings{};
            break;
        case kInstances:
            program = &rig.p.instances;
            d.instanceCount = 3;
            fsb = Bindings{};
            break;
        case kBlendC1: program = &rig.p.blendC1; break;
        }
        Program masked;
        if (s.c1 && st.kind != kMrt && st.kind != kBlendC1) {
            // (colour 1 attached: masked off by every pipeline that does not write it, as CoreAnimation's are)
            masked = *program;
            masked.moreBlend[0].writeMask = 0;
            masked.uid = newProgramUid();
            program = &masked;
        }
        if (!cb->draw(*program, p, d, vsb, fsb)) {
            r.why = std::string("draw ") + std::to_string(i) + ": " + cb->error();
            return r;
        }
    }
    if (!cb->blit(*t, 0, all, *out0, 0, all, false) || !cb->blit(*c1, 0, all, *out1, 0, all, false)) {
        r.why = "the copies out: " + cb->error();
        return r;
    }
    cb->commit();
    cb->waitUntilCompleted();
    if (!completed(*cb)) {
        r.why = "the scene: " + cb->error();
        return r;
    }
    for (const CommandBuffer::PassTime& pt : cb->passTimes()) {
        if (pt.kind == CommandBuffer::kPassDraw) r.us += pt.us;
    }
    r.counts = cb->counts();
    r.c0 = pixelsOf(*out0);
    r.c1 = pixelsOf(*out1);
    memcpy(&r.visible, rig.vis->contents(), 8);
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    int reps = 3;
    bool twins = true, times = true;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = std::max(1, atoi(argv[++i]));
        if (strcmp(argv[i], "--no-twins") == 0) twins = false;
        if (strcmp(argv[i], "--no-time") == 0) times = false;
    }
    std::string why;
    auto transport = openKernelGpu(&why);
    if (transport == nullptr) {
        std::printf("hsrcheck: no GPU: %s\n", why.c_str());
        return 100;
    }
    Gpu gpu(std::move(transport));
    uint64_t chip = 0;
    gpu.param(kParamChipId, &chip);
    std::printf("hsrcheck: GPU chip id 0x%llx, FAULTS %llu\n", (unsigned long long)chip, (unsigned long long)gpu.faults());
    auto queue = CommandQueue::make(gpu, &why);
    if (queue == nullptr) {
        report(false, "a queue: " + why);
        return failures;
    }
    if (!hasCompiler()) {
        report(false, "this build has no compiler: the shaders cannot be compiled here");
        return failures;
    }
    const uint64_t faults0 = gpu.faults();
    // the shaders, compiled here
    auto stage = [&](const uint8_t* bc, size_t size) { return std::shared_ptr<CompiledStage>(compileStage(bc, size, &why)); };
    auto flatVs = stage(kernels::tri_flat_vs_bc, kernels::tri_flat_vs_bc_size), cflatFs = stage(kernels::tri_cflat_fs_bc, kernels::tri_cflat_fs_bc_size);
    auto fbFs = stage(kernels::tri_fb_fs_bc, kernels::tri_fb_fs_bc_size), killFs = stage(kernels::tri_kill_fs_bc, kernels::tri_kill_fs_bc_size);
    auto mrtFs = stage(kernels::tri_mrt_fs_bc, kernels::tri_mrt_fs_bc_size), dposVs = stage(kernels::tri_dpos_vs_bc, kernels::tri_dpos_vs_bc_size);
    auto attrVs = stage(kernels::tri_attr_vs_bc, kernels::tri_attr_vs_bc_size), colorFs = stage(kernels::tri_color_fs_bc, kernels::tri_color_fs_bc_size);
    auto baseVs = stage(kernels::tri_base_vs_bc, kernels::tri_base_vs_bc_size);
    if (!flatVs || !cflatFs || !fbFs || !killFs || !mrtFs || !dposVs || !attrVs || !colorFs || !baseVs) {
        report(false, "the shaders, compiled here: " + why);
        return failures;
    }
    report(killFs->stage.hasKill != 0, "tri_kill_fs compiled here discards (Stage::hasKill)");
    const PixelFormat f = PixelFormat::kBGRA8Unorm;
    std::vector<std::unique_ptr<CompiledProgram>> linked;   // (what the Programs point into, alive as long as they are)
    auto link = [&](const std::shared_ptr<CompiledStage>& vs, const std::shared_ptr<CompiledStage>& fs) {
        linked.push_back(linkProgram(vs, fs, f, &why));
        return linked.back() != nullptr ? linked.back()->program : Program{};
    };
    Programs p;
    p.flat = link(flatVs, cflatFs);
    p.fetch = link(flatVs, fbFs);
    p.kill = link(flatVs, killFs);
    p.mrt = link(flatVs, mrtFs);
    p.dpos = link(dposVs, cflatFs);
    p.overlap = link(attrVs, colorFs);
    p.instances = link(baseVs, colorFs);
    if (p.flat.name == nullptr || p.fetch.name == nullptr || p.kill.name == nullptr || p.mrt.name == nullptr || p.dpos.name == nullptr ||
        p.overlap.name == nullptr || p.instances.name == nullptr) {
        report(false, "the programs, linked here: " + why);
        return failures;
    }
    // (each change to a linked Program a uid of its own: the draws find what they kept of a program by it)
    p.blend = p.flat;
    p.blend.blend = ColorBlend{true, 0, 0, 4, 5, 1, 5, 0xf};   // source over: SrcAlpha, OneMinusSrcAlpha; One, OneMinusSrcAlpha
    p.blend.uid = newProgramUid();
    p.blendC1 = p.blend;
    p.blendC1.uid = newProgramUid();
    p.overlap.vertexDescriptor.attributes[0] = VertexAttribute{VertexFormat::kFloat2, 0, 0};
    p.overlap.vertexDescriptor.attributes[1] = VertexAttribute{VertexFormat::kUChar4Normalized, 8, 0};
    p.overlap.vertexDescriptor.layouts[0] = VertexLayout{12, VertexStep::kPerVertex, 1};
    p.overlap.uid = newProgramUid();
    // the buffers: a colour a draw (every other one at half alpha, for the blended ones to show what is under them), the
    // full quad's six corners for tri_dpos, the two triangles, the visibility word
    Rig rig{gpu, *queue, p, newBuffer(gpu, 16 * 16, &why), nullptr, nullptr, newBuffer(gpu, 64, &why)};
    {
        const float xy[6][2] = {{-1, -1}, {1, -1}, {-1, 1}, {1, -1}, {1, 1}, {-1, 1}};
        rig.pos = newBuffer(gpu, sizeof xy, &why);
        if (rig.pos != nullptr) memcpy(rig.pos->contents(), xy, sizeof xy);
        struct V {
            float x, y;
            uint8_t c[4];
        };
        const V v[6] = {{-1, -1, {255, 0, 0, 255}}, {1, -1, {255, 0, 0, 255}}, {0, 1, {255, 0, 0, 255}},
                        {-1, 1, {0, 255, 0, 255}},  {1, 1, {0, 255, 0, 255}},  {0, -1, {0, 255, 0, 255}}};
        rig.verts = newBuffer(gpu, sizeof v, &why);
        if (rig.verts != nullptr) memcpy(rig.verts->contents(), v, sizeof v);
    }
    if (rig.colours == nullptr || rig.pos == nullptr || rig.verts == nullptr || rig.vis == nullptr) {
        report(false, "the buffers: " + why);
        return failures;
    }
    {
        auto* c = static_cast<float*>(rig.colours->contents());
        for (uint32_t i = 0; i < 16; i++) {
            c[4 * i + 0] = float((i * 37 + 11) % 64) / 63.0f;
            c[4 * i + 1] = float((i * 53 + 23) % 64) / 63.0f;
            c[4 * i + 2] = float((i * 29 + 41) % 64) / 63.0f;
            c[4 * i + 3] = i % 2 ? 0.5f : 1.0f;
        }
    }
    struct Size {
        uint32_t w, h;
    };
    std::vector<double> timeOff, timeOn;
    for (int rep = 0; rep < reps; rep++) {
        for (const Size sz : {Size{750, 1624}, Size{99, 209}}) {
            std::vector<uint8_t> picture(uint64_t(sz.w) * sz.h * 4), picture1(picture.size());
            std::mt19937 rng(uint32_t(sz.w * 131 + sz.h + rep));
            for (auto& b : picture) b = uint8_t(rng());
            for (auto& b : picture1) b = uint8_t(rng());
            for (size_t k = 0; k < sizeof kScenes / sizeof kScenes[0]; k++) {
                const Scene& s = kScenes[k];
                const uint32_t limit = k == kLimitScene ? 3 : CommandBuffer::kHsrMaxDraws;
                const std::string at = "rep " + std::to_string(rep + 1) + " " + std::to_string(sz.w) + "x" + std::to_string(sz.h) + " " + s.name;
                const Run off = runScene(rig, s, sz.w, sz.h, picture, picture1, false);
                const Run on = runScene(rig, s, sz.w, sz.h, picture, picture1, true, 0, limit);
                if (off.c0.empty() || on.c0.empty()) {
                    report(false, at + ": did not complete: " + off.why + " / " + on.why + " (FAULTS " + std::to_string(gpu.faults()) + ")");
                    continue;
                }
                uint64_t n0 = 0, n1 = 0;
                const std::string first0 = differ(off.c0, on.c0, sz.w, &n0), first1 = differ(off.c1, on.c1, sz.w, &n1);
                const bool query = k == 6;
                const bool served = off.counts.hsrPasses == 0 && on.counts.hsrPasses == 1;
                report(n0 == 0 && n1 == 0 && served && (!query || off.visible == on.visible),
                       at + ": " + std::to_string(n0) + " pixels of colour 0 differ" + first0 + ", " + std::to_string(n1) + " of colour 1" + first1 +
                           (query ? ", the visibility word " + std::to_string(off.visible) + " against " + std::to_string(on.visible) : std::string()) +
                           " (HSR passes " + std::to_string(on.counts.hsrPasses) + ": " + std::to_string(on.counts.hsrOpaque) + " opaque, " +
                           std::to_string(on.counts.hsrHidden) + " hidden, " + std::to_string(on.counts.hsrBarriers) + " barrier draws)");
                if (k == 0 && sz.w == 750 && rep > 0) {   // (the first repetition warms the caches and the stage cache)
                    timeOff.push_back(off.us);
                    timeOn.push_back(on.us);
                }
            }
            if (!twins || rep != 0) continue;
            // the twins that must show: a picture the same with one of them would say the comparison is blind there
            struct Twin {
                uint8_t twin;
                size_t scene;
            };
            for (const Twin tw : {Twin{1, 0}, Twin{4, 12}, Twin{5, 9}, Twin{5, 10}}) {
                const Scene& s = kScenes[tw.scene];
                const Run off = runScene(rig, s, sz.w, sz.h, picture, picture1, false);
                const Run on = runScene(rig, s, sz.w, sz.h, picture, picture1, true, tw.twin);
                if (off.c0.empty() || on.c0.empty()) {
                    report(false, "twin " + std::to_string(tw.twin) + " on " + s.name + ": did not complete: " + off.why + " / " + on.why);
                    continue;
                }
                uint64_t n0 = 0, n1 = 0;
                differ(off.c0, on.c0, sz.w, &n0);
                differ(off.c1, on.c1, sz.w, &n1);
                report(n0 + n1 != 0, "twin " + std::to_string(tw.twin) + " on " + std::to_string(sz.w) + "x" + std::to_string(sz.h) + " " + s.name + " shows: " +
                                         std::to_string(n0) + " pixels of colour 0 differ, " + std::to_string(n1) + " of colour 1");
            }
        }
    }
    if (times && !timeOff.empty()) {
        std::printf("hsrcheck: time %s at 750x1624, the pass's GPU time: without HSR median %.1f us (min %.1f max %.1f), with it %.1f us (min %.1f max %.1f), n %zu\n",
                    kScenes[0].name, median(timeOff), *std::min_element(timeOff.begin(), timeOff.end()), *std::max_element(timeOff.begin(), timeOff.end()),
                    median(timeOn), *std::min_element(timeOn.begin(), timeOn.end()), *std::max_element(timeOn.begin(), timeOn.end()), timeOff.size());
    }
    report(gpu.faults() == faults0, "no GPU reset during the run (FAULTS " + std::to_string(faults0) + " -> " + std::to_string(gpu.faults()) + ")");
    std::printf("hsrcheck: %d failure(s)\n", failures);
    return failures;
}
