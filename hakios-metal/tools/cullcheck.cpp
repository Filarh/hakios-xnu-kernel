// hakios-metal-cullcheck: whether this GPU draws a pass with its covered draws left out (CommandBuffer::setCull, the
// plug-in's metal_cull; dev-notes/investigacion/anexos/33-panel-lateral/10-oclusion-exacta.md) into the very picture it
// draws with all of them, byte for byte, and what leaving them out saves. It runs the driver's core on the phone's GPU
// from a process of its own (the kernel's msm client, as hakios-metal-hsrcheck does: no Metal, no CoreAnimation,
// nothing of the window server touched), compiles its few shaders there with the driver's compiler (the fixtures'
// bitcode, tools/kernels/*.ll), and reads its results back through 2D copies into LINEAR textures.
//
// What the host's tests cannot say is said here: that the hardware puts a vertex where the driver's interpreter of the
// vertex stage says, within the bound it keeps (cull.hpp: kUlps for each operation, kMargin of a pixel after the stage),
// and that a pixel whose centre lies inside two triangles' rectangle by more than that is drawn. Each scene is one pass
// into a compressed (UBWC) BGRA8 colour 0 -- the screen's kind -- with a second, linear, colour attachment where the
// scene has one, both filled first with a random picture: drawn with every draw, then with culling from the same
// picture, then with culling and HSR by depth (the phone's setting), and colour 0 and 1 compared. The scenes: the ones
// of tests/cull_test.cpp that say what must be left out and what must not (coverers that blend, discard, fetch, are
// no rectangle, are cut by their scissor; a reader in between; two coverers side by side; edges through the pixels'
// centres), and scenes made at random (rectangles on and off the pixel grid, turned quads, strips, indexed quads,
// scissors, every kind of draw). Each with three vertex stages: positions as NDC attributes (tri_attr_vs), and as
// pixels times a projection in a constant buffer, compiled to mad.f32 as CoreAnimation's is (tri_mvp_vs) and to
// mul.f and add.f (tri_mvpx_vs). Each at 750 x 1624 (the screen: 2 / 750 is no float) and 99 x 209.
// Then the twins that must SHOW (setCullTest): 1 a blending draw covers, 2 a coverer's rectangle taken outwards --
// a twin that draws the same picture says the comparison could not see what it is there to see.
// And the GPU time of a pass of the panel's kind at 750 x 1624 -- four flat layers and two cells under an opaque
// wallpaper, three blended bars in front -- with every draw and with the covered ones left out (median of the
// repetitions after the first).
//
// One line per check, "cullcheck: ok ..." or "cullcheck: FAIL ...", times as "cullcheck: time ...", and a last line
// with the counts; the exit status is the number of failures (100: no GPU).
//
//   hakios-metal-cullcheck [--reps N] [--random N] [--no-twins] [--no-time] [--no-hsr] [--linear]
//
// --linear: colour 0 linear -- for this tool linked against the fake GPU on the host, which runs draws into linear
// targets alone and rasterizes nothing: its rules and faults judge every stream, the pictures it compares are not drawn.
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "hakios-metal/command.hpp"
#include "hakios-metal/compiler.hpp"
#include "hakios-metal/texture.hpp"
#include "../tests/fixtures/tri_attr_vs_bc.hpp"
#include "../tests/fixtures/tri_color_fs_bc.hpp"
#include "../tests/fixtures/tri_fb_fs_bc.hpp"
#include "../tests/fixtures/tri_kill_fs_bc.hpp"
#include "../tests/fixtures/tri_mrt_fs_bc.hpp"
#include "../tests/fixtures/tri_mvp_vs_bc.hpp"
#include "../tests/fixtures/tri_mvpx_vs_bc.hpp"

using namespace hakios::metal;

namespace {

int failures = 0, checks = 0;
bool gLinear = false;   // --linear

void report(bool ok, const std::string& what) {
    std::printf("cullcheck: %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    std::fflush(stdout);
    failures += !ok;
    checks++;
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

// --- the scenes (tests/cull_test.cpp's, in pixels of the target) -----------------------------------------------------------
enum Kind : uint8_t {
    kOpaque,   // tri_color_fs, blending off: its vertices' colour
    kBlend,    // the same blended (source over), at half alpha
    kKill,     // tri_kill_fs: a checkerboard discarded
    kFetch,    // tri_fb_fs: a framebuffer fetch
    kMrt,      // tri_mrt_fs: writes colour 1 too (a scene with colour 1)
    kKinds,
};
struct Step {
    Kind kind = kOpaque;
    std::vector<std::array<double, 2>> at;   // six vertices of a list, or four of a strip
    bool strip = false, indexed = false, scissored = false;
    Scissor scissor{};
};
struct Scene {
    std::string name;
    std::vector<Step> steps;
    bool c1 = false;
};
Step quad(Kind k, std::array<double, 2> a, std::array<double, 2> b, std::array<double, 2> c, std::array<double, 2> d) {
    Step s;
    s.kind = k;
    s.at = {a, b, c, c, d, a};
    return s;
}
Step rect(Kind k, double x0, double y0, double x1, double y1) { return quad(k, {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}); }
Step strip(Kind k, double x0, double y0, double x1, double y1) {
    Step s;
    s.kind = k;
    s.strip = true;
    s.at = {{x0, y0}, {x1, y0}, {x0, y1}, {x1, y1}};
    return s;
}
Step scissored(Step s, uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    s.scissored = true;
    s.scissor = Scissor{x, y, w, h};
    return s;
}

// the scenes that say what is left out and what is not, for a target of W x H: A is a layer, B what covers it
std::vector<Scene> namedScenes(uint32_t W, uint32_t H) {
    const double ax0 = W / 5, ay0 = H / 6, ax1 = W - W / 3, ay1 = H - H / 5;   // (whole pixels)
    const double bx0 = ax0 - 7, by0 = ay0 - 5, bx1 = ax1 + 9, by1 = ay1 + 6;
    const double mx = double(int((ax0 + ax1) / 2));
    std::vector<Scene> s;
    s.push_back({"layers under a wallpaper", {rect(kOpaque, ax0, ay0, ax1, ay1), rect(kOpaque, ax0, ay0, ax1, ay1), rect(kBlend, ax0 + 4, ay0 + 4, ax1 - 4, ay1 - 4),
                                              rect(kOpaque, -12, -12.5, W + 14, H + 16.5)}});
    s.push_back({"a blending and a discarding draw under an opaque one", {rect(kBlend, ax0, ay0, ax1, ay1), rect(kKill, ax0 + 2, ay0 + 2, ax1 - 2, ay1 - 2), rect(kOpaque, bx0, by0, bx1, by1)}});
    s.push_back({"a coverer that blends", {rect(kOpaque, ax0, ay0, ax1, ay1), rect(kBlend, bx0, by0, bx1, by1)}});
    s.push_back({"a coverer that discards", {rect(kOpaque, ax0, ay0, ax1, ay1), rect(kKill, bx0, by0, bx1, by1)}});
    s.push_back({"a coverer that fetches", {rect(kOpaque, ax0, ay0, ax1, ay1), rect(kFetch, bx0, by0, bx1, by1)}});
    s.push_back({"a fetch between", {rect(kOpaque, ax0, ay0, ax1, ay1), rect(kFetch, ax0 + 9, ay0 + 9, ax1 + 3, ay1 + 3), rect(kOpaque, bx0, by0, bx1, by1)}});
    s.push_back({"a fetch after the coverer", {rect(kOpaque, ax0, ay0, ax1, ay1), rect(kOpaque, bx0, by0, bx1, by1), rect(kFetch, 0, 0, W, H)}});
    s.push_back({"a draw that writes colour 1 between", {rect(kOpaque, ax0, ay0, ax1, ay1), rect(kMrt, ax0 + 3, ay0 + 3, mx, ay1 - 3), rect(kOpaque, bx0, by0, bx1, by1)}, true});
    s.push_back({"a coverer that is not a rectangle", {rect(kOpaque, ax0, ay0, ax1, ay1), quad(kOpaque, {bx0, by0}, {bx1 + 5, by0}, {bx1 - 12, by1}, {bx0, by1})}});
    s.push_back({"a coverer turned on its corner", {rect(kOpaque, mx - 6, ay0 + 6, mx + 6, ay0 + 18), quad(kOpaque, {mx, by0}, {bx1, H / 2.0}, {mx, by1}, {bx0, H / 2.0})}});
    s.push_back({"a turned draw under a rectangle", {quad(kBlend, {mx, ay0}, {ax1, H / 2.0}, {mx, ay1}, {ax0, H / 2.0}), rect(kOpaque, ax0, ay0, ax1, ay1)}});
    s.push_back({"a strip covers, and is covered", {strip(kOpaque, ax0, ay0, ax1, ay1), strip(kOpaque, bx0, by0, bx1, by1), rect(kOpaque, 0, 0, W, H)}});
    s.push_back({"a coverer a pixel short", {rect(kOpaque, ax0, ay0, ax1, ay1), rect(kOpaque, ax0 + 1, ay0, ax1, ay1)}});
    s.push_back({"two coverers side by side", {rect(kOpaque, ax0, ay0, ax1, ay1), rect(kOpaque, bx0, by0, mx, by1), rect(kOpaque, mx, by0, bx1, by1)}});
    s.push_back({"two coverers with a column between", {rect(kOpaque, ax0, ay0, ax1, ay1), rect(kOpaque, bx0, by0, mx, by1), rect(kOpaque, mx + 1, by0, bx1, by1)}});
    s.push_back({"edges through the pixels' centres", {rect(kOpaque, ax0, ay0, ax1, ay1), rect(kOpaque, ax0 - 0.5, ay0 - 0.5, ax1 + 0.5, ay1 + 0.5), rect(kOpaque, ax0, ay0, ax1, ay1),
                                                       rect(kOpaque, ax0 - 1.5, ay0 - 1.5, ax1 + 1.5, ay1 + 1.5)}});
    // (the bound's own edge: a coverer whose edges lie an eighth of a pixel and a little more outside the centres of the
    // layer's outermost pixels covers it; the layer's box reaches a pixel further when its own edges pass a centre by as much)
    s.push_back({"a coverer 9/64 of a pixel outside the centres", {rect(kOpaque, ax0, ay0, ax1, ay1), rect(kOpaque, ax0 + 0.359375, ay0 + 0.359375, ax1 - 0.359375, ay1 - 0.359375)}});
    s.push_back({"a coverer 7/64 of a pixel outside the centres", {rect(kOpaque, ax0, ay0, ax1, ay1), rect(kOpaque, ax0 + 0.390625, ay0 + 0.390625, ax1 - 0.390625, ay1 - 0.390625)}});
    s.push_back({"a layer 7/64 of a pixel past the centres", {rect(kBlend, ax0 - 0.609375, ay0 - 0.609375, ax1 + 0.609375, ay1 + 0.609375), rect(kOpaque, ax0, ay0, ax1, ay1)}});
    s.push_back({"a coverer cut by its scissor", {rect(kOpaque, ax0, ay0, ax1, ay1), scissored(rect(kOpaque, bx0, by0, bx1, by1), uint32_t(ax0), uint32_t(ay0), uint32_t(ax1 - ax0) - 1, uint32_t(ay1 - ay0))}});
    s.push_back({"a coverer whose scissor still holds the layer", {rect(kOpaque, ax0, ay0, ax1, ay1), scissored(rect(kOpaque, 0, 0, W, H), uint32_t(ax0), uint32_t(ay0), uint32_t(ax1 - ax0), uint32_t(ay1 - ay0))}});
    {
        Step q = rect(kOpaque, bx0, by0, bx1, by1);
        q.indexed = true;
        Step a = rect(kOpaque, ax0, ay0, ax1, ay1);
        a.indexed = true;
        s.push_back({"indexed quads", {a, q}});
    }
    return s;
}

Scene randomScene(uint32_t seed, uint32_t W, uint32_t H) {
    std::mt19937 rng(seed * 2654435761u + 17);
    Scene s;
    s.name = "random " + std::to_string(seed);
    s.c1 = rng() % 4 == 0;
    auto coord = [&](uint32_t size) {
        const double whole = double(int(rng() % (size + 24)) - 12);
        switch (rng() % 8) {
        case 0: return whole + 0.5;
        case 1: return whole + 0.25;
        case 2: return whole + double(rng() % 256) / 256.0;
        default: return whole;
        }
    };
    const uint32_t n = 4 + rng() % 9;
    std::vector<std::array<double, 4>> rects;
    for (uint32_t i = 0; i < n; i++) {
        Kind kind = kOpaque;
        const uint32_t kr = rng() % 20;
        if (kr >= 10 && kr < 14) kind = kBlend;
        else if (kr == 14 || kr == 15) kind = kKill;
        else if (kr == 16 || kr == 17) kind = kFetch;
        else if (kr == 18 && s.c1) kind = kMrt;
        std::array<double, 4> a;
        if (!rects.empty() && rng() % 3 == 0) {
            a = rects[rng() % rects.size()];
            static const double grow[] = {0, 0, 1, -1, 4, 0.5, -0.5, 0.0625, -0.0625, 0.375, -0.375};
            const double g = grow[rng() % 11];
            a = {a[0] - g, a[1] - g, a[2] + g, a[3] + g};
        } else {
            double x0 = coord(W), x1 = coord(W), y0 = coord(H), y1 = coord(H);
            if (rng() % 3 == 0) x0 = -8, x1 = W + 8, y0 = -8.5, y1 = H + 8.5;
            a = {std::min(x0, x1), std::min(y0, y1), std::max(x0, x1), std::max(y0, y1)};
        }
        if (a[0] >= a[2] || a[1] >= a[3]) a = {10, 10, 30, 30};
        rects.push_back(a);
        Step step;
        const uint32_t shape = rng() % 12;
        if (shape == 0) {
            const double cx = (a[0] + a[2]) / 2, cy = (a[1] + a[3]) / 2;
            step = quad(kind, {cx, a[1]}, {a[2], cy}, {cx, a[3]}, {a[0], cy});
        } else if (shape == 1) {
            step = quad(kind, {a[0], a[1]}, {a[2] + double(rng() % 9) - 4, a[1]}, {a[2], a[3]}, {a[0], a[3] + double(rng() % 5) - 2});
        } else if (shape == 2) {
            step = strip(kind, a[0], a[1], a[2], a[3]);
        } else if (shape == 3) {
            step = quad(kind, {a[0], a[1]}, {a[0], a[3]}, {a[2], a[3]}, {a[2], a[1]});
        } else {
            step = rect(kind, a[0], a[1], a[2], a[3]);
        }
        if (!step.strip && rng() % 4 == 0) step.indexed = true;
        if (rng() % 6 == 0) {
            const uint32_t x = rng() % (W - 8), y = rng() % (H - 8);
            step = scissored(step, x, y, 1 + rng() % (W - x), 1 + rng() % (H - y));
        }
        s.steps.push_back(step);
    }
    return s;
}

// --- a scene through the driver -----------------------------------------------------------------------------------------------
enum Vs : uint8_t { kVsAttr, kVsMvp, kVsMvpx, kVsCount };
const char* const kVsNames[kVsCount] = {"attributes in NDC", "a matrix, mad.f32", "a matrix, mul.f and add.f"};
struct Programs {
    Program p[kVsCount][kKinds];
};
struct Rig {
    Gpu& gpu;
    CommandQueue& queue;
    const Programs& programs;
    std::shared_ptr<Buffer> colours;   // tri_kill_fs's, a colour a draw
};
struct Run {
    std::vector<uint8_t> c0, c1;
    CommandBuffer::CullCounts cull;
    double us = 0;
    std::string why;
};
constexpr uint32_t kSlots = 8;

Run runScene(Rig& rig, const Scene& s, Vs vs, uint32_t W, uint32_t H, const std::vector<uint8_t>& picture, const std::vector<uint8_t>& picture1, uint8_t mode,
             bool hsr = false, uint8_t twin = 0) {
    Run r;
    auto t = makeTexture(rig.gpu, !gLinear, W, H, &r.why);
    auto c1 = t != nullptr ? makeTexture(rig.gpu, false, W, H, &r.why) : nullptr;
    auto in = c1 != nullptr ? makeTexture(rig.gpu, false, W, H, &r.why) : nullptr;
    auto out0 = in != nullptr ? makeTexture(rig.gpu, false, W, H, &r.why) : nullptr;
    auto out1 = out0 != nullptr ? makeTexture(rig.gpu, false, W, H, &r.why) : nullptr;
    if (out1 == nullptr) return r;
    fillPixels(*in, picture);
    fillPixels(*c1, picture1);
    // the buffers: each step's vertices at its slots, its indices, the projection (pixels to NDC, columns)
    const bool matrix = vs != kVsAttr;
    const size_t stride = matrix ? 20 : 12, n = s.steps.size();
    std::shared_ptr<Buffer> vb = newBuffer(rig.gpu, std::max<size_t>(n, 1) * kSlots * stride, &r.why);
    std::shared_ptr<Buffer> ib = vb != nullptr ? newBuffer(rig.gpu, std::max<size_t>(n, 1) * kSlots * 2, &r.why) : nullptr;
    std::shared_ptr<Buffer> mb = ib != nullptr ? newBuffer(rig.gpu, 64, &r.why) : nullptr;
    if (mb == nullptr) return r;
    {
        const float m[16] = {2.0f / float(W), 0, 0, 0, 0, -2.0f / float(H), 0, 0, 0, 0, 0, 0, -1, 1, 0, 1};
        memcpy(mb->contents(), m, 64);
    }
    std::vector<uint32_t> counts(n);
    for (size_t i = 0; i < n; i++) {
        const Step& st = s.steps[i];
        std::vector<std::array<double, 2>> v = st.at;
        std::vector<uint16_t> indices;
        if (st.indexed && !st.strip && st.at.size() == 6) {
            v = {st.at[0], st.at[1], st.at[2], st.at[4]};
            indices = {0, 1, 2, 2, 3, 0};
        }
        counts[i] = uint32_t(indices.empty() ? v.size() : indices.size());
        for (size_t j = 0; j < v.size() && j < kSlots; j++) {
            uint8_t* at = static_cast<uint8_t*>(vb->contents()) + (i * kSlots + j) * stride;
            const uint8_t colour[4] = {uint8_t(40 + i * 37), uint8_t(200 - i * 53), uint8_t(90 + i * 29), uint8_t(st.kind == kBlend ? 128 : 255)};
            if (matrix) {
                const float p[4] = {float(v[j][0]), float(v[j][1]), 0.0f, 1.0f};
                memcpy(at, p, 16);
                memcpy(at + 16, colour, 4);
            } else {
                const float p[2] = {float(v[j][0] * 2.0 / W - 1.0), float(1.0 - v[j][1] * 2.0 / H)};
                memcpy(at, p, 8);
                memcpy(at + 8, colour, 4);
            }
        }
        if (!indices.empty()) memcpy(static_cast<uint8_t*>(ib->contents()) + i * kSlots * 2, indices.data(), indices.size() * 2);
    }
    const render::BlitRect all{0, 0, W, H};
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
    cb->setCull(mode);
    cb->setCullTest(twin);
    if (gLinear) cb->setGmem(false);   // (a linear target would be drawn in tiles, which setCull leaves alone: sysmem, as the compressed one is)
    if (hsr) {
        cb->setHsr(true);
        cb->setHsrDepth(gLinear ? CommandBuffer::kHsrDepth16 : CommandBuffer::kHsrDepth16Ubwc, false);
        cb->setHsrTest(0, CommandBuffer::kHsrMaxDraws, gLinear);
    }
    cb->setProfile(true);
    RenderPass pass;
    pass.color0 = t.get();
    if (s.c1) pass.more[0].texture = c1.get();
    for (size_t i = 0; i < n; i++) {
        const Step& st = s.steps[i];
        Draw d;
        d.viewport = Viewport{0, 0, double(W), double(H), 0, 1};
        d.primitive = st.strip ? Primitive::kTriangleStrip : Primitive::kTriangle;
        if (st.scissored) {
            d.scissor = st.scissor;
            d.scissor.width = std::min(d.scissor.width, W - d.scissor.x);
            d.scissor.height = std::min(d.scissor.height, H - d.scissor.y);
        }
        d.vertexCount = counts[i];
        if (st.indexed && !st.strip && st.at.size() == 6) {
            d.indexBuffer = ib.get();
            d.indexOffset = i * kSlots * 2;
            d.baseVertex = int32_t(i * kSlots);
        } else {
            d.vertexStart = uint32_t(i * kSlots);
        }
        Program p = rig.programs.p[vs][st.kind];
        if (s.c1 && st.kind != kMrt) {
            p.moreBlend[0].writeMask = 0;   // (colour 1 attached: masked off by every pipeline that does not write it)
            p.uid = newProgramUid();
        }
        Bindings vsb, fsb;
        vsb.buffers = {Binding{vb.get(), 0}};
        if (matrix) vsb.buffers.push_back(Binding{mb.get(), 0});
        if (st.kind == kKill) fsb.buffers = {Binding{rig.colours.get(), 16 * (i % 16)}};
        if (!cb->draw(p, pass, d, vsb, fsb)) {
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
    r.cull = cb->cullCounts();
    r.c0 = pixelsOf(*out0);
    r.c1 = pixelsOf(*out1);
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    int reps = 5, random = 40;
    bool twins = true, times = true, hsr = true;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = std::max(2, atoi(argv[++i]));
        if (strcmp(argv[i], "--random") == 0 && i + 1 < argc) random = std::max(0, atoi(argv[++i]));
        if (strcmp(argv[i], "--no-twins") == 0) twins = false;
        if (strcmp(argv[i], "--no-time") == 0) times = false;
        if (strcmp(argv[i], "--no-hsr") == 0) hsr = false;
        if (strcmp(argv[i], "--linear") == 0) gLinear = true;
    }
    std::string why;
    auto transport = openKernelGpu(&why);
    if (transport == nullptr) {
        std::printf("cullcheck: no GPU: %s\n", why.c_str());
        return 100;
    }
    Gpu gpu(std::move(transport));
    uint64_t chip = 0;
    gpu.param(kParamChipId, &chip);
    std::printf("cullcheck: GPU chip id 0x%llx, FAULTS %llu\n", (unsigned long long)chip, (unsigned long long)gpu.faults());
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
    auto stage = [&](const uint8_t* bc, size_t size) { return std::shared_ptr<CompiledStage>(compileStage(bc, size, &why)); };
    const std::shared_ptr<CompiledStage> vss[kVsCount] = {stage(kernels::tri_attr_vs_bc, kernels::tri_attr_vs_bc_size), stage(kernels::tri_mvp_vs_bc, kernels::tri_mvp_vs_bc_size),
                                                          stage(kernels::tri_mvpx_vs_bc, kernels::tri_mvpx_vs_bc_size)};
    auto colorFs = stage(kernels::tri_color_fs_bc, kernels::tri_color_fs_bc_size), fbFs = stage(kernels::tri_fb_fs_bc, kernels::tri_fb_fs_bc_size);
    auto killFs = stage(kernels::tri_kill_fs_bc, kernels::tri_kill_fs_bc_size), mrtFs = stage(kernels::tri_mrt_fs_bc, kernels::tri_mrt_fs_bc_size);
    if (!vss[0] || !vss[1] || !vss[2] || !colorFs || !fbFs || !killFs || !mrtFs) {
        report(false, "the shaders, compiled here: " + why);
        return failures;
    }
    std::vector<std::unique_ptr<CompiledProgram>> linked;
    Programs programs;
    for (uint32_t v = 0; v < kVsCount; v++) {
        // (the interpreter has a model of each vertex stage: without one nothing is left out and nothing is checked)
        auto link = [&](const std::shared_ptr<CompiledStage>& fs) {
            linked.push_back(linkProgram(vss[v], fs, PixelFormat::kBGRA8Unorm, &why));
            Program p = linked.back() != nullptr ? linked.back()->program : Program{};
            p.vertexDescriptor.attributes[0] = VertexAttribute{v == kVsAttr ? VertexFormat::kFloat2 : VertexFormat::kFloat4, 0, 0};
            p.vertexDescriptor.attributes[1] = VertexAttribute{VertexFormat::kUChar4Normalized, v == kVsAttr ? 8u : 16u, 0};
            p.vertexDescriptor.layouts[0] = VertexLayout{v == kVsAttr ? 12u : 20u, VertexStep::kPerVertex, 1};
            p.uid = newProgramUid();
            return p;
        };
        Program* p = programs.p[v];
        p[kOpaque] = link(colorFs);
        p[kBlend] = p[kOpaque];
        p[kBlend].blend = ColorBlend{true, 0, 0, 4, 5, 1, 5, 0xf};   // source over
        p[kBlend].uid = newProgramUid();
        p[kKill] = link(killFs);
        p[kFetch] = link(fbFs);
        p[kMrt] = link(mrtFs);
        for (uint32_t k = 0; k < kKinds; k++) {
            if (p[k].name == nullptr) {
                report(false, std::string("the programs, linked here (") + kVsNames[v] + "): " + why);
                return failures;
            }
        }
        report(cull::modelOf(p[kOpaque])->usable, std::string("the interpreter has a model of the vertex stage: ") + kVsNames[v]);
    }
    Rig rig{gpu, *queue, programs, newBuffer(gpu, 16 * 16, &why)};
    if (rig.colours == nullptr) {
        report(false, "the buffers: " + why);
        return failures;
    }
    {
        auto* c = static_cast<float*>(rig.colours->contents());
        for (uint32_t i = 0; i < 16; i++) {
            c[4 * i + 0] = float((i * 37 + 11) % 64) / 63.0f;
            c[4 * i + 1] = float((i * 53 + 23) % 64) / 63.0f;
            c[4 * i + 2] = float((i * 29 + 41) % 64) / 63.0f;
            c[4 * i + 3] = 1.0f;
        }
    }
    uint64_t leftOut = 0, drawn = 0, scenes = 0;
    const uint32_t sizes[2][2] = {{750, 1624}, {99, 209}};
    for (const auto& size : sizes) {
        const uint32_t W = size[0], H = size[1];
        std::vector<uint8_t> picture(uint64_t(W) * H * 4), picture1(picture.size());
        std::mt19937 rng(W * 31 + H);
        for (auto& b : picture) b = uint8_t(rng());
        for (auto& b : picture1) b = uint8_t(rng());
        std::vector<Scene> all = namedScenes(W, H);
        // (fewer of the large ones: each is three passes over the screen and six copies of it)
        for (int i = 0; i < (W > 400 ? random / 2 : random); i++) all.push_back(randomScene(uint32_t(i), W, H));
        for (uint32_t v = 0; v < kVsCount; v++) {
            for (const Scene& s : all) {
                const Run off = runScene(rig, s, Vs(v), W, H, picture, picture1, CommandBuffer::kCullOff);
                const Run on = runScene(rig, s, Vs(v), W, H, picture, picture1, CommandBuffer::kCullOn);
                char head[220];
                std::snprintf(head, sizeof head, "%s, %s, %ux%u: %u of %zu draws left out (%u looked at, %u covering, %.2f Mpx)", s.name.c_str(), kVsNames[v], W, H,
                              on.cull.culled, s.steps.size(), on.cull.seen, on.cull.coverers, double(on.cull.pixels) / 1e6);
                if (!off.why.empty() || !on.why.empty()) {
                    report(false, std::string(head) + ": " + (off.why.empty() ? on.why : off.why));
                    continue;
                }
                uint64_t d0 = 0, d1 = 0;
                const std::string f0 = differ(off.c0, on.c0, W, &d0), f1 = differ(off.c1, on.c1, W, &d1);
                report(d0 == 0 && d1 == 0 && on.cull.undone == 0,
                       std::string(head) + (d0 == 0 && d1 == 0 ? ": the same picture" : ": colour 0 differs in " + std::to_string(d0) + " pixels" + f0 + ", colour 1 in " +
                                                                                              std::to_string(d1) + f1));
                leftOut += on.cull.culled;
                drawn += s.steps.size();
                scenes++;
                if (hsr) {
                    const Run both = runScene(rig, s, Vs(v), W, H, picture, picture1, CommandBuffer::kCullOn, true);
                    if (!both.why.empty()) {
                        report(false, std::string(head) + ", with HSR: " + both.why);
                        continue;
                    }
                    const std::string g0 = differ(off.c0, both.c0, W, &d0), g1 = differ(off.c1, both.c1, W, &d1);
                    report(d0 == 0 && d1 == 0 && both.cull.culled == on.cull.culled,
                           std::string(head) + ", with HSR too" + (d0 == 0 && d1 == 0 ? ": the same picture" : ": colour 0 differs in " + std::to_string(d0) + " pixels" + g0 +
                                                                                                                    ", colour 1 in " + std::to_string(d1) + g1));
                }
            }
        }
        if (twins && !gLinear) {
            // the twins: among the scenes, at least one whose picture each changes
            for (uint8_t twin : {uint8_t(1), uint8_t(2)}) {
                uint32_t shown = 0, tried = 0;
                for (const Scene& s : all) {
                    if (shown != 0) break;
                    const Run off = runScene(rig, s, kVsMvp, W, H, picture, picture1, CommandBuffer::kCullOff);
                    const Run on = runScene(rig, s, kVsMvp, W, H, picture, picture1, CommandBuffer::kCullOn, false, twin);
                    if (!off.why.empty() || !on.why.empty()) continue;
                    uint64_t d0 = 0, d1 = 0;
                    differ(off.c0, on.c0, W, &d0);
                    differ(off.c1, on.c1, W, &d1);
                    tried++;
                    if (d0 != 0 || d1 != 0) shown++;
                }
                report(shown != 0, std::string("twin ") + std::to_string(twin) + (twin == 1 ? " (a blending draw covers)" : " (a coverer's rectangle taken outwards)") + ", " +
                                       std::to_string(W) + "x" + std::to_string(H) + (shown != 0 ? ": a picture differs, after " : ": no picture differs in ") +
                                       std::to_string(tried) + " scenes");
            }
        }
    }
    if (times && !gLinear) {
        // the panel's kind of pass at the screen's size: four flat layers the size of the screen and two cells, all under
        // an opaque wallpaper that reaches past the screen, then three blended bars in front
        const uint32_t W = 750, H = 1624;
        Scene panel{"the panel's layers", {rect(kOpaque, 0, 0, W, H), rect(kOpaque, 0, 0, W, H), rect(kOpaque, 0, 0, W, H), rect(kOpaque, 0, 0, W, H),
                                           rect(kOpaque, 40, 300, 350, 610), rect(kOpaque, 400, 300, 710, 610), rect(kOpaque, -123, -123.5, 873, 1747.5),
                                           rect(kBlend, 0, 0, W, 130), rect(kBlend, 30, 900, 720, 1500), rect(kBlend, 520, 160, 670, 240)}};
        std::vector<uint8_t> picture(uint64_t(W) * H * 4);
        std::mt19937 rng(5);
        for (auto& b : picture) b = uint8_t(rng());
        for (const auto& [mode, hsrOn, name] : {std::tuple<uint8_t, bool, const char*>{CommandBuffer::kCullOff, false, "every draw"},
                                                {CommandBuffer::kCullOn, false, "the covered draws left out"},
                                                {CommandBuffer::kCullOff, true, "every draw, with HSR"},
                                                {CommandBuffer::kCullOn, true, "the covered draws left out, with HSR"}}) {
            std::vector<double> us;
            Run last;
            for (int i = 0; i < reps; i++) {
                last = runScene(rig, panel, kVsMvp, W, H, picture, picture, mode, hsrOn);
                if (!last.why.empty()) break;
                if (i > 0) us.push_back(last.us);
            }
            if (!last.why.empty()) {
                report(false, std::string("time, ") + name + ": " + last.why);
                continue;
            }
            std::sort(us.begin(), us.end());
            std::printf("cullcheck: time %s, 750x1624, %s: %.0f us (%.0f to %.0f, n=%zu); %u of %zu draws left out, %.2f Mpx\n", panel.name.c_str(), name, median(us),
                        us.empty() ? 0.0 : us.front(), us.empty() ? 0.0 : us.back(), us.size(), last.cull.culled, panel.steps.size(), double(last.cull.pixels) / 1e6);
        }
    }
    report(gpu.faults() == faults0, "no GPU fault in the run (FAULTS " + std::to_string(gpu.faults()) + ")");
    std::printf("cullcheck: %d checks, %d failed; %llu of %llu draws left out in %llu scenes\n", checks, failures, (unsigned long long)leftOut, (unsigned long long)drawn,
                (unsigned long long)scenes);
    return failures;
}
