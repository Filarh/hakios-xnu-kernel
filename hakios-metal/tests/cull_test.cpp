// CommandBuffer::setCull: draws left out when later opaque rectangles of their pass cover them.
//
// The fake GPU rasterizes nothing, so the pictures here are a model's: every scene is drawn twice by a rasterizer of
// this file -- once with all its draws, once with the draws the fake GPU was in fact given (read off its log of the
// stream, by each draw's VFD_INDEX_OFFSET) -- and the two must be the same word for word, colour 0 and colour 1. The
// model takes the vertices from the same bytes the driver's buffers hold, puts them through the viewport as the
// hardware's registers say, snaps them to 1/256 of a pixel and fills with exact integer edge functions, under two
// tie rules (top-left and its opposite: the driver's answers must not rest on either). What a draw does to a pixel it
// covers is its kind's: an opaque one writes its colour, a blending one mixes with what is there, a discarding one
// writes every other pixel, a fetching one writes a function of what it reads. So a draw left out that could be seen
// -- under a coverer that blends, discards, is not a rectangle, is cut by its scissor, or with a reader in between --
// changes the picture, and the mistakes made on purpose (setCullTest's twins) are each shown to change one.
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <optional>
#include <random>
#include <set>
#include <vector>

#include "fixtures/tri_attr_vs_air.hpp"
#include "fixtures/tri_color_fs_air.hpp"
#include "fixtures/tri_fb_fs_air.hpp"
#include "fixtures/tri_mrt_fs_air.hpp"
#include "hakios-metal/a6xx.hpp"
#include "hakios-metal/command.hpp"
#include "hakios-metal/cull.hpp"
#include "hakios-metal/fake_gpu.hpp"
#include "hakios-metal/render.hpp"
#include "hakios-metal/texture.hpp"
#include "test.hpp"

using namespace hakios::metal;

namespace {

struct Rig {
    Rig() {
        auto f = std::make_unique<FakeGpu>();
        fake = f.get();
        gpu = std::make_unique<Gpu>(std::move(f));
        std::string why;
        queue = CommandQueue::make(*gpu, &why);
    }
    FakeGpu* fake;
    std::unique_ptr<Gpu> gpu;
    std::unique_ptr<CommandQueue> queue;
};

std::shared_ptr<Texture> target(Rig& r, uint32_t w, uint32_t h, PixelFormat format = PixelFormat::kBGRA8Unorm) {
    TextureDesc d;
    d.format = format;
    d.width = w;
    d.height = h;
    d.usage = kTextureUsageRenderTarget | kTextureUsageRead;
    std::string why;
    auto t = Texture::make(*r.gpu, d, &why);
    CHECK(t != nullptr);
    return t;
}

std::shared_ptr<Buffer> bufferWith(Rig& r, const void* bytes, uint64_t length) {
    std::string why;
    auto b = newBuffer(*r.gpu, length, &why);
    CHECK(b != nullptr);
    if (b != nullptr) memcpy(b->contents(), bytes, length);
    return b;
}

// --- instructions by hand (cull.cpp's layout, which a fuzz against Mesa's disassembler holds: 10-oclusion-exacta.md) -------
constexpr uint32_t reg(uint32_t n, uint32_t c) { return n * 4 + c; }
constexpr uint32_t cst(uint32_t n, uint32_t c) { return 0x1000 | (n * 4 + c); }
struct Word2 {
    uint32_t lo, hi;
};
constexpr Word2 cat2(uint32_t opc, uint32_t dst, uint32_t s1, uint32_t s2, uint32_t rpt = 0, bool r1 = false, bool r2 = false) {
    return Word2{s1 | s2 << 16, 2u << 29 | opc << 21 | 1u << 20 | (r2 ? 1u << 19 : 0) | (r1 ? 1u << 11 : 0) | rpt << 8 | dst};
}
constexpr Word2 mad(uint32_t dst, uint32_t s1, uint32_t s2reg, uint32_t s3, uint32_t rpt = 0, bool r1 = false, bool r2 = false, bool r3 = false) {
    return Word2{s1 | (r2 ? 1u << 15 : 0) | s3 << 16 | (r3 ? 1u << 29 : 0), 3u << 29 | 7u << 23 | s2reg << 15 | (r1 ? 1u << 11 : 0) | rpt << 8 | dst};
}
constexpr Word2 kEnd{0, 0x03000000};

// position = M * attribute 0, M the float4x4 at [[buffer(1)]] pushed to c0..c3 (columns): what a compiler makes of
// `out.position = m * in.position`, with ir3's repeats
constexpr Word2 kMvp[5] = {
    cat2(3, reg(2, 0), cst(0, 0), reg(0, 0), 3, true),
    mad(reg(2, 0), cst(1, 0), reg(0, 1), reg(2, 0), 3, true, false, true),
    mad(reg(2, 0), cst(2, 0), reg(0, 2), reg(2, 0), 3, true, false, true),
    mad(reg(2, 0), cst(3, 0), reg(0, 3), reg(2, 0), 3, true, false, true),
    kEnd,
};
uint32_t gMvpWords[32], gBadWords[32], gTaintWords[32];
void words(uint32_t* out, const Word2* in, size_t n) {
    memset(out, 0, 32 * 4);
    for (size_t i = 0; i < n; i++) out[2 * i] = in[i].lo, out[2 * i + 1] = in[i].hi;
}

Stage mvpVs() {
    words(gMvpWords, kMvp, 5);
    Stage s = kernels::tri_attr_vs_air;
    s.k.words = gMvpWords;
    s.k.maxReg = 2;
    s.k.uboRanges[0] = KernelUboRange{1, 0, 0, 16};
    s.k.numUboRanges = 1;
    s.k.immediatesOffsetVec4 = 4;
    s.k.name = "mvp_vs";
    s.inputs[0].compmask = 0xf;
    s.outputs[0].regid = uint8_t(reg(2, 0));
    s.positionRegid = reg(2, 0);
    return s;
}

// --- the scenes -------------------------------------------------------------------------------------------------------------
enum Kind : uint8_t {
    kOpaque,    // writes its colour
    kBlend,     // mixes its colour with what is there
    kKill,      // discards every other pixel
    kFetch,     // reads colour 0 and writes a function of it there
    kFetchC1,   // (a scene with colour 1) reads colour 0 and writes colour 1: a barrier that reads
    kMrt,       // (a scene with colour 1) writes colour 0 and colour 1
    kNoModel,   // opaque, its vertex stage with an instruction the interpreter has no model of
    kTainted,   // opaque, its position's x written by an instruction the interpreter does not run
};
struct Step {
    Kind kind = kOpaque;
    std::vector<std::array<double, 2>> at;   // its vertices: six of a list (two triangles), or four of a strip
    bool strip = false, indexed = false;
    bool scissored = false;
    Scissor scissor{};
    uint32_t cullMode = 0, instances = 1;
    double z = 0;                            // (a scene with a matrix) its vertices' z
};
struct Scene {
    std::vector<Step> steps;
    bool c1 = false;
    uint32_t w = 96, h = 64;   // (96: x * 2 / 96 - 1 is no float exactly, so the vertices land a hair off the pixels asked for)
    // none: the vertices are float2 positions in NDC (tri_attr_vs). Set: float4 positions as given, times this matrix
    // (columns) by the mvp stage
    std::optional<std::array<float, 16>> mvp;
};

Step quad(Kind k, std::array<double, 2> a, std::array<double, 2> b, std::array<double, 2> c, std::array<double, 2> d) {
    Step s;
    s.kind = k;
    s.at = {a, b, c, c, d, a};
    return s;
}
// (the order CoreAnimation gives a layer's six vertices)
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
Step indexed(Step s) {
    s.indexed = true;
    return s;
}

constexpr uint32_t kSlots = 8;   // vertices kept for each step: its first is its number times this, which VFD_INDEX_OFFSET then names

// a step's vertices as the buffer holds them (x, y, z, w; a scene without a matrix has z 0 and w 1 from the fetch),
// and its triangles; an indexed step holds each corner once
struct Mesh {
    std::vector<std::array<float, 4>> v;
    std::vector<uint16_t> indices;             // (indexed)
    std::vector<std::array<uint32_t, 3>> tri;  // of v
};
Mesh meshOf(const Scene& scene, const Step& s) {
    Mesh m;
    auto put = [&](const std::array<double, 2>& p) {
        if (scene.mvp) m.v.push_back({float(p[0]), float(p[1]), float(s.z), 1.0f});
        else m.v.push_back({float(p[0] * 2.0 / scene.w - 1.0), float(1.0 - p[1] * 2.0 / scene.h), 0.0f, 1.0f});
    };
    if (s.indexed && !s.strip && s.at.size() == 6) {
        for (uint32_t i : {0u, 1u, 2u, 4u}) put(s.at[i]);
        m.indices = {0, 1, 2, 2, 3, 0};
        m.tri = {{0, 1, 2}, {2, 3, 0}};
    } else {
        for (const auto& p : s.at) put(p);
        if (s.strip) {
            for (uint32_t i = 0; i + 2 < s.at.size(); i++) m.tri.push_back({i, i + 1, i + 2});
        } else {
            for (uint32_t i = 0; i + 2 < s.at.size(); i += 3) m.tri.push_back({i, i + 1, i + 2});
        }
    }
    return m;
}

// --- the model ----------------------------------------------------------------------------------------------------------------
struct Picture {
    std::vector<uint32_t> c0, c1;
    bool operator==(const Picture& o) const { return c0 == o.c0 && c1 == o.c1; }
};

// SCENE's steps ORDER, in that order, each triangle filled where a pixel's centre is inside its snapped vertices;
// TOP_LEFT: an edge's own points belong to the triangle when it is a top or a left edge (else: when it is not)
Picture draw(const Scene& scene, const std::vector<uint32_t>& order, bool topLeft) {
    const int W = int(scene.w), H = int(scene.h);
    Picture p;
    p.c0.assign(size_t(W) * H, 0x11111111u);
    p.c1.assign(size_t(W) * H, 0x22222222u);
    for (uint32_t id : order) {
        const Step& s = scene.steps[id];
        const Mesh m = meshOf(scene, s);
        // the vertices in the window: the stage's arithmetic in long double, the viewport's as GRAS_CL_VPORT has it
        std::vector<std::array<int64_t, 2>> at;
        for (const auto& v : m.v) {
            long double x = v[0], y = v[1], w = v[3];
            if (scene.mvp) {
                const auto& M = *scene.mvp;
                x = (long double)M[0] * v[0] + (long double)M[4] * v[1] + (long double)M[8] * v[2] + (long double)M[12] * v[3];
                y = (long double)M[1] * v[0] + (long double)M[5] * v[1] + (long double)M[9] * v[2] + (long double)M[13] * v[3];
                w = (long double)M[3] * v[0] + (long double)M[7] * v[1] + (long double)M[11] * v[2] + (long double)M[15] * v[3];
            }
            const long double sx = scene.w / 2.0L + scene.w / 2.0L * (x / w), sy = scene.h / 2.0L - scene.h / 2.0L * (y / w);
            at.push_back({int64_t(llroundl(sx * 256)), int64_t(llroundl(sy * 256))});
        }
        int x0 = 0, y0 = 0, x1 = W - 1, y1 = H - 1;
        if (s.scissored) {
            x0 = int(s.scissor.x), y0 = int(s.scissor.y);
            x1 = std::min(x1, int(s.scissor.x + s.scissor.width) - 1), y1 = std::min(y1, int(s.scissor.y + s.scissor.height) - 1);
        }
        const uint32_t colour = (id + 1) * 0x9e3779b1u | 1u;
        for (const auto& t : m.tri) {
            std::array<int64_t, 2> a = at[t[0]], b = at[t[1]], c = at[t[2]];
            const int64_t area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            if (area == 0) continue;
            if (area < 0) std::swap(b, c);
            const std::array<int64_t, 2> e[3][2] = {{a, b}, {b, c}, {c, a}};
            for (int y = y0; y <= y1; y++) {
                for (int x = x0; x <= x1; x++) {
                    const int64_t px = int64_t(x) * 256 + 128, py = int64_t(y) * 256 + 128;
                    bool inside = true;
                    for (int k = 0; k < 3 && inside; k++) {
                        const int64_t dx = e[k][1][0] - e[k][0][0], dy = e[k][1][1] - e[k][0][1];
                        const int64_t f = dx * (py - e[k][0][1]) - dy * (px - e[k][0][0]);
                        const bool top = (dy == 0 && dx > 0) || dy < 0;
                        inside = f > 0 || (f == 0 && top == topLeft);
                    }
                    if (!inside) continue;
                    uint32_t& c0 = p.c0[size_t(y) * W + x];
                    uint32_t& c1 = p.c1[size_t(y) * W + x];
                    switch (s.kind) {
                    case kOpaque:
                    case kNoModel:
                    case kTainted: c0 = colour; break;
                    case kBlend: c0 = c0 * 31u + colour; break;
                    case kKill:
                        if ((x + y) & 1) c0 = colour;
                        break;
                    case kFetch: c0 = (c0 ^ colour) * 0x85ebca6bu; break;
                    case kFetchC1: c1 = (c0 ^ colour) * 0xc2b2ae35u; break;
                    case kMrt:
                        c0 = colour;
                        c1 = colour * 3u;
                        break;
                    }
                }
            }
        }
    }
    return p;
}

// --- a scene through the driver -----------------------------------------------------------------------------------------------
struct Run {
    bool completed = false;
    std::string error;
    std::vector<uint32_t> ran;   // the steps the fake GPU drew, in the order it drew them
    CommandBuffer::CullCounts counts;
    std::vector<CommandBuffer::CullEvent> events;
    CommandBuffer::Counts all;
    std::set<uint32_t> culled() const {
        std::set<uint32_t> s;
        for (const auto& e : events) s.insert(e.draw);
        return s;
    }
};
struct Knobs {
    uint8_t mode = CommandBuffer::kCullOn, twin = 0;
    bool hsr = false;
    bool gmem = false;      // the pass may be drawn in tiles
    bool depth = false;     // the pass has a depth attachment of its own
    // run when every draw is recorded, before the commit: with the scene's vertex buffer and (a scene with a matrix) its matrix's
    std::function<void(Buffer&, Buffer*)> beforeCommit;
    // run before the first draw, with the command buffer and the vertex buffer
    std::function<void(CommandBuffer&, Buffer&)> first;
    bool split = false;     // the draws recorded into a second command buffer, appended to the first before the commit
};

Run run(const Scene& scene, const Knobs& k = Knobs{}) {
    Run out;
    Rig r;
    std::string why;
    auto t = target(r, scene.w, scene.h);
    auto c1 = target(r, scene.w, scene.h);
    std::shared_ptr<Texture> depth;
    if (k.depth) {
        TextureDesc d;
        d.format = PixelFormat::kDepth32Float;
        d.width = scene.w;
        d.height = scene.h;
        d.usage = kTextureUsageRenderTarget;
        depth = Texture::make(*r.gpu, d, &why);
        CHECK(depth != nullptr);
    }
    // the buffers: every step's vertices at its slots, each step's indices, the matrix
    struct V2 {
        float x, y;
        uint8_t c[4];
    };
    struct V4 {
        float p[4];
        uint8_t c[4];
    };
    const size_t n = scene.steps.size();
    const size_t stride = scene.mvp ? sizeof(V4) : sizeof(V2);
    std::vector<uint8_t> vbytes(n * kSlots * stride, 0);
    std::vector<uint16_t> ibytes(n * kSlots, 0);
    for (size_t i = 0; i < n; i++) {
        const Mesh m = meshOf(scene, scene.steps[i]);
        CHECK(m.v.size() <= kSlots && m.indices.size() <= kSlots);
        for (size_t j = 0; j < m.v.size() && j < kSlots; j++) {
            uint8_t* at = vbytes.data() + (i * kSlots + j) * stride;
            const uint8_t colour[4] = {uint8_t(i * 37), uint8_t(i * 91), uint8_t(255 - i), 255};
            memcpy(at, m.v[j].data(), scene.mvp ? 16 : 8);
            memcpy(at + (scene.mvp ? 16 : 8), colour, 4);
        }
        for (size_t j = 0; j < m.indices.size() && j < kSlots; j++) ibytes[i * kSlots + j] = m.indices[j];
    }
    auto vb = bufferWith(r, vbytes.data(), vbytes.size());
    auto ib = bufferWith(r, ibytes.data(), ibytes.size() * 2);
    std::shared_ptr<Buffer> mb;
    if (scene.mvp) mb = bufferWith(r, scene.mvp->data(), 64);
    // the programs: one vertex stage for the scene, a fragment stage a kind
    Stage vs = scene.mvp ? mvpVs() : kernels::tri_attr_vs_air;
    auto made = [&](const Stage& v, const Stage& f, const char* name) {
        Program p{v, f, Linkage{}, name};
        p.vertexDescriptor.attributes[0] = VertexAttribute{scene.mvp ? VertexFormat::kFloat4 : VertexFormat::kFloat2, 0, 0};
        p.vertexDescriptor.attributes[1] = VertexAttribute{VertexFormat::kUChar4Normalized, scene.mvp ? 16u : 8u, 0};
        p.vertexDescriptor.layouts[0] = VertexLayout{uint32_t(stride), VertexStep::kPerVertex, 1};
        p.ssboWrites = 0;
        if (scene.c1) p.moreBlend[0].writeMask = 0;   // (colour 1 masked off by every pipeline but kMrt's and kFetchC1's)
        return p;
    };
    const Program opaque = made(vs, kernels::tri_color_fs_air, "cull_opaque");
    Program blend = opaque;
    blend.blend = ColorBlend{true, 0, 0, 4, 5, 1, 5, 0xf};
    Program kill = opaque;
    kill.fs.hasKill = 1;
    const Program fetch = made(vs, kernels::tri_fb_fs_air, "cull_fetch");
    Program fetchC1 = fetch;
    fetchC1.moreBlend[0].writeMask = 0xf;
    Program mrt = made(vs, kernels::tri_mrt_fs_air, "cull_mrt");
    mrt.moreBlend[0].writeMask = 0xf;
    // (a vertex stage with a texture read first, and one whose position's x a reciprocal square root writes)
    Stage badVs = vs, taintVs = vs;
    {
        const Word2 bad[2] = {Word2{0, 5u << 29}, kEnd};
        const Word2 taint[2] = {Word2{0, 4u << 29 | vs.positionRegid}, kEnd};
        words(gBadWords, bad, 2);
        words(gTaintWords, taint, 2);
        badVs.k.words = gBadWords;
        taintVs.k.words = gTaintWords;
    }
    const Program noModel = made(badVs, kernels::tri_color_fs_air, "cull_no_model");
    const Program tainted = made(taintVs, kernels::tri_color_fs_air, "cull_tainted");

    render::Options o;
    o.gmem = k.gmem;
    auto main = r.queue->commandBuffer();
    auto cb = k.split ? r.queue->commandBuffer() : main;
    for (auto& b : {main, cb}) {
        b->setCull(k.mode);
        b->setCullTest(k.twin);
        if (k.hsr) {
            b->setHsr(true);
            b->setHsrTest(0, CommandBuffer::kHsrMaxDraws, true);
        }
    }
    if (k.first) k.first(*cb, *vb);
    r.fake->logDraws(true, true);
    RenderPass pass;
    pass.color0 = t.get();
    if (scene.c1) pass.more[0].texture = c1.get();
    if (depth != nullptr) pass.depth = depth.get();
    for (size_t i = 0; i < n; i++) {
        const Step& s = scene.steps[i];
        const Mesh m = meshOf(scene, s);
        Draw d;
        d.viewport = Viewport{0, 0, double(scene.w), double(scene.h), 0, 1};
        d.primitive = s.strip ? Primitive::kTriangleStrip : Primitive::kTriangle;
        d.cullMode = s.cullMode;
        d.instanceCount = s.instances;
        if (s.scissored) d.scissor = s.scissor;
        if (!m.indices.empty()) {
            d.indexBuffer = ib.get();
            d.indexOffset = i * kSlots * 2;
            d.indexType = 0;
            d.baseVertex = int32_t(i * kSlots);
            d.vertexCount = uint32_t(m.indices.size());
        } else {
            d.vertexStart = uint32_t(i * kSlots);
            d.vertexCount = uint32_t(m.v.size());
        }
        const Program* p = &opaque;
        switch (s.kind) {
        case kOpaque: break;
        case kBlend: p = &blend; break;
        case kKill: p = &kill; break;
        case kFetch: p = &fetch; break;
        case kFetchC1: p = &fetchC1; break;
        case kMrt: p = &mrt; break;
        case kNoModel: p = &noModel; break;
        case kTainted: p = &tainted; break;
        }
        Bindings vsb;
        vsb.buffers = {Binding{vb.get(), 0}};
        if (mb != nullptr) vsb.buffers.push_back(Binding{mb.get(), 0});
        if (!cb->draw(*p, pass, d, vsb, Bindings{}, o)) {
            out.error = cb->error();
            return out;
        }
    }
    if (k.beforeCommit) k.beforeCommit(*vb, mb.get());
    if (k.split) CHECK(main->append(*cb));
    main->commit();
    main->waitUntilCompleted();
    out.completed = main->status() == CommandBuffer::kCompleted;
    const FakeGpu::Stats stats = r.fake->stats();
    if (!out.completed) out.error = main->error() + " | " + stats.lastFault;
    if (stats.faults != 0) out.completed = false, out.error += " | fault: " + stats.lastFault;
    for (const FakeGpu::Stats::DrawLog& l : stats.drawLog) {
        if (l.binning) continue;
        for (uint64_t w : l.state) {
            if (w == ~0ull) break;
            if (uint32_t(w >> 32) == a6xx::REG_A6XX_VFD_INDEX_OFFSET) out.ran.push_back(uint32_t(w) / kSlots);
        }
    }
    out.counts = main->cullCounts();
    out.events = main->cullEvents();
    out.all = main->counts();
    return out;
}

std::vector<uint32_t> everyStep(const Scene& s) {
    std::vector<uint32_t> all;
    for (uint32_t i = 0; i < s.steps.size(); i++) all.push_back(i);
    return all;
}
std::vector<uint32_t> without(const Scene& s, const std::set<uint32_t>& gone) {
    std::vector<uint32_t> left;
    for (uint32_t i = 0; i < s.steps.size(); i++) {
        if (gone.count(i) == 0) left.push_back(i);
    }
    return left;
}
// whether SCENE without the steps GONE is the picture of all of it, under both tie rules
bool samePicture(const Scene& s, const std::set<uint32_t>& gone) {
    const std::vector<uint32_t> all = everyStep(s), left = without(s, gone);
    return draw(s, all, true) == draw(s, left, true) && draw(s, all, false) == draw(s, left, false);
}

// SCENE drawn with culling: the fake drew exactly the steps not left out, in their order, and the picture is the
// whole scene's; the steps left out
std::set<uint32_t> checked(const Scene& s, const char* name, const Knobs& k = Knobs{}) {
    const Run r = run(s, k);
    CHECK(r.completed);
    if (!r.completed) std::printf("       %s: %s\n", name, r.error.c_str());
    const std::set<uint32_t> gone = r.culled();
    const bool drew = r.ran == without(s, r.counts.undone != 0 ? std::set<uint32_t>{} : gone);
    CHECK(drew);
    if (!drew) std::printf("       %s: the fake drew %zu steps, %zu were left out of %zu\n", name, r.ran.size(), gone.size(), s.steps.size());
    const bool same = samePicture(s, gone);
    CHECK(same);
    if (!same) std::printf("       %s: the picture without the %zu steps left out is another\n", name, gone.size());
    return gone;
}

using Gone = std::set<uint32_t>;

}  // namespace

// --- the arithmetic -----------------------------------------------------------------------------------------------------------

TEST(cull_decodes_the_position_instructions_as_mesa_writes_them) {
    // (CoreAnimation's fixed_vert_lph_spc, compiled for this chip: its first, sixth and last instructions and one of its
    // texture-coordinate ones, as ir3's disassembler lists them)
    CHECK(cull::disassemble(0x00001000, 0x40700008) == "mul.f r2.x, c0.x, r0.x");
    CHECK(cull::disassemble(0x00001002, 0x4070090a) == "(rpt1)mul.f r2.z, (r)c0.z, r0.x");
    CHECK(cull::disassemble(0x00081004, 0x63808008) == "mad.f32 r2.x, c1.x, r0.y, r2.x");
    CHECK(cull::disassemble(0x2008100c, 0x63818b08) == "(rpt3)mad.f32 r2.x, (r)c3.x, r0.w, (r)r2.x");
    CHECK(cull::disassemble(0x30129010, 0x63820900) == "(rpt1)mad.f32 r0.x, (r)c4.x, (r)r1.x, (r)c4.z");
    CHECK(cull::disassemble(0x00000000, 0x03000000) == "end");
    CHECK(cull::disassemble(0x00000000, 0x00000000) == "nop");
    // (tri_attr_vs's two: the z and the w a float2 position is given)
    CHECK(cull::disassemble(0x00000000, 0x204cc002) == "mov r0.z, 0x0");
    CHECK(cull::disassemble(0x3f800000, 0x204cc003) == "mov r0.w, 0x3f800000");
    // the matrix stage of this file
    CHECK(cull::disassemble(kMvp[0].lo, kMvp[0].hi) == "(rpt3)mul.f r2.x, (r)c0.x, r0.x");
    CHECK(cull::disassemble(kMvp[3].lo, kMvp[3].hi) == "(rpt3)mad.f32 r2.x, (r)c3.x, r0.w, (r)r2.x");
    // no model: a branch, a texture read, a load from memory, a mov that converts, one to the address register
    CHECK(cull::disassemble(0x00000004, 0x00800000) == "?");
    CHECK(cull::disassemble(0, 5u << 29) == "?");
    CHECK(cull::disassemble(0, 6u << 29) == "?");
    CHECK(cull::disassemble(0x00000000, 0x2044c002) == "?");   // f32 to u32
    CHECK(cull::disassemble(0x00000000, 0x204cc0f4) == "?");   // to a0.x
    // not run, its destination no longer known: another opcode of categories 2 to 4, a half destination, a saturated one
    CHECK(cull::disassemble(0x00001000, 0x40300008) == "taint r2.x");   // min.f
    CHECK(cull::disassemble(0x00001000, 0x40704008) == "taint r2.x");
    CHECK(cull::disassemble(0x00001000, 0x40700408) == "taint r2.x");
    CHECK(cull::disassemble(0x00000000, 0x80000008) == "taint r2.x");
}

TEST(cull_region_is_what_is_left_of_a_rectangle) {
    cull::Region g;
    g.set(cull::Rect{0, 0, 9, 9});
    CHECK(g.subtract(cull::Rect{20, 20, 30, 30}) && g.n == 1);   // (apart)
    CHECK(g.subtract(cull::Rect{3, 3, 5, 5}) && g.n == 4);
    uint64_t pixels = 0;
    for (uint32_t i = 0; i < g.n; i++) pixels += g.r[i].pixels();
    CHECK_EQ(pixels, 100u - 9u);
    CHECK(g.subtract(cull::Rect{0, 0, 9, 2}) && g.subtract(cull::Rect{0, 6, 9, 9}));
    CHECK(g.subtract(cull::Rect{0, 3, 2, 5}) && !g.empty());
    CHECK(g.subtract(cull::Rect{6, 0, 9, 9}) && g.empty());
    // more pieces than it keeps: refused, nothing changed
    g.set(cull::Rect{0, 0, 99, 99});
    CHECK(g.subtract(cull::Rect{10, 10, 12, 12}) && g.n == 4);
    CHECK(g.subtract(cull::Rect{50, 1, 52, 3}));
    const uint32_t n = g.n;
    bool refused = false;
    for (int i = 0; i < 6 && !refused; i++) refused = !g.subtract(cull::Rect{3 + 15 * i, 30 + 9 * i, 5 + 15 * i, 32 + 9 * i});
    CHECK(refused && g.n <= cull::kMaxRegion && g.n >= n);
    // an empty rectangle is no region
    g.set(cull::Rect{5, 5, 4, 9});
    CHECK(g.empty());
}

// Where the interpreter puts a draw's vertices and how sure it is: exact arithmetic gives the margin alone, inexact
// arithmetic widens it, and what it cannot know is said by name.
TEST(cull_works_out_positions_with_a_bound_on_the_gpus_own) {
    Rig r;
    std::string why;
    struct V4 {
        float p[4];
        uint8_t c[4];
    };
    Program p{mvpVs(), kernels::tri_color_fs_air, Linkage{}, "mvp"};
    p.vertexDescriptor.attributes[0] = VertexAttribute{VertexFormat::kFloat4, 0, 0};
    p.vertexDescriptor.attributes[1] = VertexAttribute{VertexFormat::kUChar4Normalized, 16, 0};
    p.vertexDescriptor.layouts[0] = VertexLayout{sizeof(V4), VertexStep::kPerVertex, 1};
    const V4 quadAt[6] = {{{16, 8, 0, 1}, {}}, {{80, 8, 0, 1}, {}}, {{80, 40, 0, 1}, {}}, {{80, 40, 0, 1}, {}}, {{16, 40, 0, 1}, {}}, {{16, 8, 0, 1}, {}}};
    auto vb = bufferWith(r, quadAt, sizeof quadAt);
    std::vector<std::shared_ptr<Buffer>> matrices;   // (alive as long as the records that name them)
    auto evaluate = [&](const std::array<float, 16>& m, uint32_t W, uint32_t H, cull::Positions* pos, cull::Geometry* g, cull::Snapshots* snaps) {
        auto mb = bufferWith(r, m.data(), 64);
        matrices.push_back(mb);
        Draw d;
        d.vertexCount = 6;
        d.viewport = Viewport{0, 0, double(W), double(H), 0, 1};
        Bindings b;
        b.buffers = {Binding{vb.get(), 0}, Binding{mb.get(), 0}};
        const uint32_t area[4] = {0, 0, W - 1, H - 1};
        const cull::Why c = cull::capture(p, d, b, area, g);
        if (c != cull::kOk) return c;
        const cull::Why e = cull::evaluate(*g, snaps, pos);
        return e;
    };
    cull::Positions pos;
    cull::Geometry g;
    cull::Snapshots snaps;
    // pixels to NDC over a 128 x 64 window: every product and sum a float holds -- the margin alone, and the rectangle
    // the pixels 16..79 x 8..39
    std::array<float, 16> m = {2.0f / 128, 0, 0, 0, 0, -2.0f / 64, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1};
    CHECK(evaluate(m, 128, 64, &pos, &g, &snaps) == cull::kOk);
    CHECK_EQ(pos.count, 6u);
    CHECK(pos.p[0].x == 16 && pos.p[0].y == 8 && pos.p[2].x == 80 && pos.p[2].y == 40 && pos.p[0].u == cull::kMargin && pos.p[0].inside);
    cull::Rect rc;
    CHECK(cull::occluderRect(g, pos, &rc) == cull::kOk);
    CHECK(rc.x0 == 16 && rc.y0 == 8 && rc.x1 == 79 && rc.y1 == 39);
    cull::Rect box = cull::coverBox(g, pos);
    CHECK(box.x0 == 16 && box.y0 == 8 && box.x1 == 79 && box.y1 == 39);
    // the bytes it read: the matrix once and each of the six vertices' positions (16 bytes each)
    CHECK_EQ(snaps.bytes(), size_t(64 + 6 * 16));
    CHECK(snaps.same(0, snaps.mark()));
    // over a 96-wide window 2 / 96 is no float: the bound is wider than the margin, by far less than a pixel, and the
    // rectangle is the same pixels
    m[0] = 2.0f / 96;
    CHECK(evaluate(m, 96, 64, &pos, &g, &snaps) == cull::kOk);
    CHECK(pos.p[1].u > cull::kMargin && pos.p[1].u < cull::kMargin + 1e-3 && std::fabs(pos.p[1].x - 80) < 1e-4);
    CHECK(cull::occluderRect(g, pos, &rc) == cull::kOk && rc.x0 == 16 && rc.x1 == 79);
    // a vertex behind the far plane (z = 2 w), and one a hair inside it by inexact arithmetic: not provably inside
    m[0] = 2.0f / 128;
    m[14] = 2;
    CHECK(evaluate(m, 128, 64, &pos, &g, &snaps) == cull::kOk && !pos.p[0].inside);
    CHECK(cull::occluderRect(g, pos, &rc) == cull::kClip);
    m[14] = 0.5f;
    CHECK(evaluate(m, 128, 64, &pos, &g, &snaps) == cull::kOk && pos.p[0].inside);
    m[14] = 0;
    m[2] = 1e-9f;   // z = 16e-9 or 80e-9 with a bound: inside; with the sign turned, outside
    CHECK(evaluate(m, 128, 64, &pos, &g, &snaps) == cull::kOk && pos.p[0].inside);
    m[2] = -1e-9f;
    CHECK(evaluate(m, 128, 64, &pos, &g, &snaps) == cull::kOk && !pos.p[0].inside);
    m[2] = 0;
    // w: 2 everywhere halves the picture about the centre; 0 or negative is refused
    m[15] = 2;
    CHECK(evaluate(m, 128, 64, &pos, &g, &snaps) == cull::kOk && pos.p[0].x == 64 + (16 - 64) / 2.0);
    m[15] = 0;
    CHECK(evaluate(m, 128, 64, &pos, &g, &snaps) == cull::kW);
    m[15] = -1;
    CHECK(evaluate(m, 128, 64, &pos, &g, &snaps) == cull::kW);
    m[15] = 1;
    // a matrix with a NaN or an infinity in what the position reads: unknown
    m[12] = NAN;
    CHECK(evaluate(m, 128, 64, &pos, &g, &snaps) == cull::kUnknown);
    m[12] = INFINITY;
    CHECK(evaluate(m, 128, 64, &pos, &g, &snaps) == cull::kUnknown);
    m[12] = -1;
    // a quarter turn keeps the rectangle a rectangle; a turn by 30 degrees does not
    const std::array<float, 16> quarter = {0, -2.0f / 64, 0, 0, 2.0f / 128, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1};
    CHECK(evaluate(quarter, 128, 64, &pos, &g, &snaps) == cull::kOk && cull::occluderRect(g, pos, &rc) == cull::kOk);
    const float c30 = std::cos(0.5236f), s30 = std::sin(0.5236f);
    const std::array<float, 16> turned = {c30 / 64, s30 / 64, 0, 0, -s30 / 64, c30 / 64, 0, 0, 0, 0, 0, 0, -0.5f, -0.5f, 0, 1};
    CHECK(evaluate(turned, 128, 64, &pos, &g, &snaps) == cull::kOk && cull::occluderRect(g, pos, &rc) == cull::kNotRect);
    // the vertex buffer too short for the last vertex, and a matrix buffer not bound: no data
    {
        Draw d;
        d.vertexStart = 1;
        d.vertexCount = 6;
        d.viewport = Viewport{0, 0, 128, 64, 0, 1};
        auto mb = bufferWith(r, m.data(), 64);
        Bindings b;
        b.buffers = {Binding{vb.get(), 0}, Binding{mb.get(), 0}};
        const uint32_t area[4] = {0, 0, 127, 63};
        cull::Geometry g2;
        CHECK(cull::capture(p, d, b, area, &g2) == cull::kOk && cull::evaluate(g2, &snaps, &pos) == cull::kNoData);
        b.buffers.pop_back();
        d.vertexStart = 0;
        CHECK(cull::capture(p, d, b, area, &g2) == cull::kOk && cull::evaluate(g2, &snaps, &pos) == cull::kNoData);
        // more vertices than it reads, points, instances, an indirect draw: the draw's shape
        d.vertexCount = cull::kMaxVertices + 1;
        CHECK(cull::capture(p, d, b, area, &g2) == cull::kShape);
        d.vertexCount = 6;
        d.instanceCount = 2;
        CHECK(cull::capture(p, d, b, area, &g2) == cull::kShape);
        d.instanceCount = 1;
        d.primitive = Primitive::kLine;
        CHECK(cull::capture(p, d, b, area, &g2) == cull::kShape);
        d.primitive = Primitive::kTriangle;
        d.indirect = vb.get();
        CHECK(cull::capture(p, d, b, area, &g2) == cull::kShape);
    }
    // a byte of what was read changed: the records no longer hold; a buffer the GPU has written is not read at all
    {
        cull::Snapshots s2;
        m = {2.0f / 128, 0, 0, 0, 0, -2.0f / 64, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1};
        CHECK(evaluate(m, 128, 64, &pos, &g, &s2) == cull::kOk && s2.same(0, s2.mark()));
        static_cast<float*>(vb->contents())[5] += 1;   // (vertex 1's y)
        CHECK(!s2.same(0, s2.mark()));
        static_cast<float*>(vb->contents())[5] -= 1;
        CHECK(s2.same(0, s2.mark()));
        vb->bo().noteGpuWrite();
        CHECK(!s2.same(0, s2.mark()));
        CHECK(evaluate(m, 128, 64, &pos, &g, &s2) == cull::kNoData);
    }
}

// --- the scenes -----------------------------------------------------------------------------------------------------------------

// What must be left out and what must not, scene by scene; each also drawn by the model with and without the draws
// left out (checked()).
TEST(cull_leaves_out_what_later_opaque_rectangles_cover_and_nothing_else) {
    struct Case {
        const char* name;
        Scene scene;
        Gone gone;
    };
    std::vector<Case> cases;
    auto add = [&](const char* name, std::vector<Step> steps, Gone gone, bool c1 = false, uint32_t w = 96) {
        Scene s;
        s.steps = std::move(steps);
        s.c1 = c1;
        s.w = w;
        cases.push_back(Case{name, std::move(s), std::move(gone)});
    };
    // the panel's first pass: flat layers under the wallpaper, which reaches past the target on every side
    add("layers under a wallpaper", {rect(kOpaque, 20, 10, 60, 50), rect(kOpaque, 20, 10, 60, 50), rect(kBlend, 30, 20, 50, 40), rect(kOpaque, -12, -12.5, 110, 80.5)},
        {0, 1, 2});
    add("the same rectangle over itself", {rect(kOpaque, 20, 10, 60, 50), rect(kOpaque, 20, 10, 60, 50)}, {0});
    add("a blending, a discarding and a fetching draw under an opaque one",
        {rect(kBlend, 20, 10, 60, 50), rect(kKill, 22, 12, 58, 48), rect(kOpaque, 10, 5, 70, 55)}, {0, 1});
    // coverers that do not cover
    add("a coverer that blends", {rect(kOpaque, 20, 10, 60, 50), rect(kBlend, 10, 5, 70, 55)}, {});
    add("a coverer that discards", {rect(kOpaque, 20, 10, 60, 50), rect(kKill, 10, 5, 70, 55)}, {});
    add("a coverer that fetches", {rect(kOpaque, 20, 10, 60, 50), rect(kFetch, 10, 5, 70, 55)}, {});
    add("a coverer with a face culled", {rect(kOpaque, 20, 10, 60, 50), [] {
            Step s = rect(kOpaque, 10, 5, 70, 55);
            s.cullMode = 2;
            return s;
        }()}, {});
    add("a coverer of three instances", {rect(kOpaque, 20, 10, 60, 50), [] {
            Step s = rect(kOpaque, 10, 5, 70, 55);
            s.instances = 3;
            return s;
        }()}, {});
    add("a coverer whose vertex stage has no model", {rect(kOpaque, 20, 10, 60, 50), rect(kNoModel, 10, 5, 70, 55)}, {});
    add("a coverer whose position is not followed", {rect(kOpaque, 20, 10, 60, 50), rect(kTainted, 10, 5, 70, 55)}, {});
    add("a covered draw whose vertex stage has no model", {rect(kNoModel, 20, 10, 60, 50), rect(kOpaque, 10, 5, 70, 55)}, {});
    // a reader in between: nothing before it is left out for what comes after it; what comes after it still is
    add("a fetch between", {rect(kOpaque, 20, 10, 60, 50), rect(kFetch, 40, 30, 80, 60), rect(kOpaque, 10, 5, 90, 62)}, {1});
    add("a fetch between, apart from both", {rect(kOpaque, 20, 10, 60, 50), rect(kFetch, 80, 2, 90, 6), rect(kOpaque, 10, 8, 70, 55)}, {});
    add("a fetch after the coverer", {rect(kOpaque, 20, 10, 60, 50), rect(kOpaque, 10, 5, 70, 55), rect(kFetch, 0, 0, 96, 64)}, {0});
    add("a fetch after the coverer, itself covered later",
        {rect(kOpaque, 20, 10, 60, 50), rect(kOpaque, 10, 5, 70, 55), rect(kFetch, 15, 8, 65, 52), rect(kOpaque, 0, 0, 96, 64)}, {0, 2});
    add("a reader that writes colour 1", {rect(kOpaque, 20, 10, 60, 50), rect(kFetchC1, 0, 0, 96, 64), rect(kOpaque, 10, 5, 70, 55)}, {}, true);
    add("a draw that writes colour 1 is not left out, and is no reader",
        {rect(kOpaque, 20, 10, 60, 50), rect(kMrt, 30, 20, 50, 40), rect(kOpaque, 10, 5, 70, 55)}, {0}, true);
    add("colour 1 written by the coverer", {rect(kOpaque, 20, 10, 60, 50), rect(kMrt, 10, 5, 70, 55)}, {}, true);
    // shapes
    add("a coverer that is not a rectangle", {rect(kOpaque, 30, 20, 50, 40), quad(kOpaque, {10, 5}, {75, 5}, {70, 55}, {10, 55})}, {});
    add("a coverer turned on its corner", {rect(kOpaque, 40, 25, 56, 39), quad(kOpaque, {48, -20}, {100, 32}, {48, 84}, {-4, 32})}, {});
    add("a coverer whose triangles are the same half twice",
        {rect(kOpaque, 30, 20, 50, 40), [] {
             Step s;
             s.at = {{10, 5}, {70, 5}, {70, 55}, {70, 55}, {70, 5}, {10, 5}};
             return s;
         }()}, {});
    add("a coverer whose triangles overlap and leave a corner",
        {rect(kOpaque, 30, 20, 50, 40), [] {
             Step s;
             s.at = {{10, 5}, {70, 5}, {70, 55}, {10, 5}, {70, 5}, {10, 55}};
             return s;
         }()}, {});
    add("a coverer of one triangle and a sliver", {rect(kOpaque, 30, 20, 50, 40), [] {
            Step s;
            s.at = {{10, 5}, {70, 5}, {70, 55}, {70, 55}, {10, 55}, {10, 55}};
            return s;
        }()}, {});
    add("a turned draw under a rectangle", {quad(kBlend, {48, 10}, {70, 32}, {48, 54}, {26, 32}), rect(kOpaque, 26, 10, 70, 54)}, {0});
    add("a turned draw whose box sticks out", {quad(kBlend, {48, 10}, {70, 32}, {48, 54}, {26, 32}), rect(kOpaque, 27, 10, 70, 54)}, {});
    add("a strip covers, and is covered", {strip(kOpaque, 20, 10, 60, 50), strip(kOpaque, 10, 5, 70, 55), rect(kOpaque, 0, 0, 96, 64)}, {0, 1});
    add("indexed quads", {indexed(rect(kOpaque, 20, 10, 60, 50)), indexed(rect(kOpaque, 10, 5, 70, 55))}, {0});
    // partial cover, and several coverers
    add("a coverer a pixel short", {rect(kOpaque, 20, 10, 60, 50), rect(kOpaque, 21, 10, 60, 50)}, {});
    add("a coverer a pixel short below", {rect(kOpaque, 20, 10, 60, 50), rect(kOpaque, 20, 10, 60, 49)}, {});
    add("two coverers side by side", {rect(kOpaque, 20, 10, 60, 50), rect(kOpaque, 4, 5, 40, 55), rect(kOpaque, 40, 5, 76, 55)}, {0});
    add("two coverers with a column between", {rect(kOpaque, 20, 10, 60, 50), rect(kOpaque, 4, 5, 40, 55), rect(kOpaque, 41, 5, 77, 55)}, {});
    add("four coverers, a quadrant each",
        {rect(kBlend, 30, 20, 50, 44), rect(kOpaque, 0, 0, 40, 32), rect(kOpaque, 40, 0, 80, 32), rect(kOpaque, 0, 32, 40, 64), rect(kOpaque, 40, 32, 80, 64)}, {0});
    add("coverers before the draw do not count", {rect(kOpaque, 4, 5, 40, 55), rect(kOpaque, 20, 10, 60, 50), rect(kOpaque, 40, 5, 76, 55)}, {});
    // a rectangle under cull::kMinCoverer pixels on a side is not held against the draws before it; one of that size is
    add("a coverer of 31 pixels a side", {rect(kOpaque, 40, 20, 60, 40), rect(kOpaque, 35, 15, 66, 46)}, {});
    add("a coverer of 32 pixels a side", {rect(kOpaque, 40, 20, 60, 40), rect(kOpaque, 34, 15, 66, 47)}, {0});
    add("a covered coverer still covers", {rect(kOpaque, 20, 10, 30, 20), rect(kOpaque, 15, 8, 35, 25), rect(kOpaque, 10, 5, 70, 55)}, {0, 1});
    // edges between pixels and through their centres
    add("edges through the pixels' centres are taken inwards",
        {rect(kOpaque, 20, 10, 60, 50), rect(kOpaque, 19.5, 9.5, 60.5, 50.5), rect(kOpaque, 20, 10, 60, 50), rect(kOpaque, 18.5, 8.5, 61.5, 51.5)}, {0, 1, 2});
    add("a draw with edges through the centres is taken outwards", {rect(kOpaque, 19.5, 9.5, 60.5, 50.5), rect(kOpaque, 20, 10, 60, 50)}, {});
    add("...and covered a pixel further out", {rect(kOpaque, 19.5, 9.5, 60.5, 50.5), rect(kOpaque, 19, 9, 61, 51)}, {0});
    add("a coverer a sixteenth of a pixel short covers the same pixels", {rect(kOpaque, 20, 10, 60, 50), rect(kOpaque, 20.0625, 10.0625, 59.9375, 49.9375)}, {0});
    add("a coverer half a pixel short does not", {rect(kOpaque, 20, 10, 60, 50), rect(kOpaque, 20.5, 10, 60, 50)}, {});
    add("a sliver between two pixel centres draws nothing and is in nobody's way", {rect(kOpaque, 20.6, 10, 21.4, 50), rect(kOpaque, 30, 10, 60, 50)}, {});
    // the scissor
    add("a coverer cut by its scissor", {rect(kOpaque, 20, 10, 60, 50), scissored(rect(kOpaque, 10, 5, 70, 55), 10, 5, 49, 50)}, {});
    add("a coverer whose scissor still holds the draw", {rect(kOpaque, 20, 10, 60, 50), scissored(rect(kOpaque, 0, 0, 96, 64), 20, 10, 40, 40)}, {0});
    add("a draw cut by its scissor to what is covered", {scissored(rect(kBlend, 0, 0, 96, 64), 20, 10, 40, 40), rect(kOpaque, 20, 10, 60, 50)}, {0});
    add("a draw wholly outside the target is left out by the first coverer", {rect(kOpaque, 100, 10, 140, 50), rect(kOpaque, 20, 10, 60, 50), rect(kOpaque, 10, 5, 70, 55)},
        {0, 1});
    for (const Case& c : cases) {
        const Gone gone = checked(c.scene, c.name);
        const bool expected = gone == c.gone;
        CHECK(expected);
        if (!expected) {
            std::printf("       %s: left out", c.name);
            for (uint32_t g : gone) std::printf(" %u", g);
            std::printf(" (meant:");
            for (uint32_t g : c.gone) std::printf(" %u", g);
            std::printf(")\n");
        }
        // verifying: the same answers, every draw drawn
        Knobs verify;
        verify.mode = CommandBuffer::kCullVerify;
        const Run v = run(c.scene, verify);
        CHECK(v.completed && v.culled() == c.gone && v.ran == everyStep(c.scene) && v.counts.culled == c.gone.size());
        // off: nothing looked at
        Knobs off;
        off.mode = CommandBuffer::kCullOff;
        const Run n = run(c.scene, off);
        CHECK(n.completed && n.ran == everyStep(c.scene) && n.counts.seen == 0 && n.counts.culled == 0);
        // with HSR by depth too: the same draws left out, the rest all drawn (its opaque ones from their prepass)
        Knobs hsr;
        hsr.hsr = true;
        const Run h = run(c.scene, hsr);
        CHECK(h.completed);
        if (!h.completed) std::printf("       %s with HSR: %s\n", c.name, h.error.c_str());
        const std::vector<uint32_t> left = without(c.scene, c.gone);
        const bool hsrSame = h.culled() == c.gone && std::set<uint32_t>(h.ran.begin(), h.ran.end()) == std::set<uint32_t>(left.begin(), left.end()) &&
                             h.ran.size() == left.size();
        CHECK(hsrSame);
        if (!hsrSame) std::printf("       %s with HSR: %zu drawn, %zu left out\n", c.name, h.ran.size(), h.culled().size());
    }
}

// The matrix stage (position = M * attribute, as CoreAnimation's): the same answers through the multiplications, and
// none where the clip volume or w forbids one.
TEST(cull_follows_a_matrix_in_the_vertex_stage) {
    auto scene = [](std::array<float, 16> m, std::vector<Step> steps, uint32_t w = 96) {
        Scene s;
        s.steps = std::move(steps);
        s.w = w;
        s.mvp = m;
        return s;
    };
    // pixels to NDC, as a 2D compositor's projection
    auto ortho = [](uint32_t w, uint32_t h) { return std::array<float, 16>{2.0f / w, 0, 0, 0, 0, -2.0f / h, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1}; };
    const std::vector<Step> stack = {rect(kOpaque, 20, 10, 60, 50), rect(kBlend, 30, 20, 50, 40), rect(kOpaque, -12, -12.5, 110, 80.5)};
    CHECK(checked(scene(ortho(96, 64), stack), "ortho 96") == Gone({0, 1}));
    CHECK(checked(scene(ortho(128, 64), stack, 128), "ortho 128") == Gone({0, 1}));
    // z: the coverer at the far plane exactly, inside it, behind it
    {
        std::vector<Step> s = stack;
        std::array<float, 16> m = ortho(96, 64);
        m[10] = 1;
        s[2].z = 1;
        CHECK(checked(scene(m, s), "z = w") == Gone({0, 1}));
        s[2].z = 0.75;
        CHECK(checked(scene(m, s), "z inside") == Gone({0, 1}));
        s[2].z = 1.5;
        CHECK(checked(scene(m, s), "z behind the far plane") == Gone{});
        s[2].z = -0.25;
        CHECK(checked(scene(m, s), "z before the near plane") == Gone{});
        // (a covered draw's z is nobody's business: clipped or not, what is left of it is under the coverer)
        s[2].z = 0.5;
        s[0].z = 7;
        CHECK(checked(scene(m, s), "a covered draw outside the clip volume") == Gone({0, 1}));
    }
    // w = 2 everywhere: the picture halved about the centre, the same cover
    {
        std::array<float, 16> m = ortho(96, 64);
        m[15] = 2;
        CHECK(checked(scene(m, stack), "w = 2") == Gone({0, 1}));
        m[15] = -1;
        CHECK(checked(scene(m, stack), "w = -1") == Gone{});
    }
    // a quarter turn: rectangles stay rectangles
    {
        const std::array<float, 16> m = {0, -2.0f / 64, 0, 0, 2.0f / 96, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1};
        Scene s = scene(m, {rect(kOpaque, 10, 20, 50, 60), rect(kOpaque, 5, 10, 55, 70)});
        CHECK(checked(s, "a quarter turn") == Gone({0}));
    }
    // a small turn: the coverer is no rectangle, a turned draw is still covered by an upright one
    {
        const float c = std::cos(0.1f), sn = std::sin(0.1f);
        const std::array<float, 16> m = {c * 2 / 96, -sn * 2 / 64, 0, 0, -sn * 2 / 96, -c * 2 / 64, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1};
        Scene s = scene(m, {rect(kOpaque, 30, 20, 50, 40), rect(kOpaque, 10, 5, 80, 60)});
        CHECK(checked(s, "a turned coverer") == Gone{});
    }
}

// What the caller may still do between a draw and the commit, and what else is in the pass.
TEST(cull_puts_the_draws_back_when_their_bytes_change_before_the_commit) {
    Scene s;
    s.steps = {rect(kOpaque, 20, 10, 60, 50), rect(kBlend, 30, 20, 50, 40), rect(kOpaque, 10, 5, 70, 55), rect(kOpaque, 80, 5, 90, 15)};
    CHECK(checked(s, "untouched") == Gone({0, 1}));
    struct V2 {
        float x, y;
        uint8_t c[4];
    };
    // the coverer's first vertex moved after its draw was recorded: it no longer covers, and every draw is drawn
    for (uint32_t step : {2u, 0u, 3u}) {
        Knobs k;
        k.beforeCommit = [step](Buffer& vb, Buffer*) { static_cast<V2*>(vb.contents())[step * kSlots].x += 0.5f; };
        const Run r = run(s, k);
        CHECK(r.completed);
        // (step 3 is no part of any answer: its bytes may change)
        const bool undone = step != 3;
        CHECK_EQ(r.counts.undone, undone ? 2u : 0u);
        CHECK(r.ran == (undone ? everyStep(s) : without(s, {0, 1})));
        // the picture the model draws from the bytes as they are at the commit is the whole scene's either way; with
        // the check left out (twin 5) the draws stay left out, and the moved coverer shows what was under it
        Knobs twin = k;
        twin.twin = 5;
        const Run t = run(s, twin);
        CHECK(t.completed && t.counts.undone == 0 && t.ran == without(s, {0, 1}));
    }
    {
        // ...which the model sees: the scene with the coverer where it was moved to, with and without the draws under it
        Scene moved = s;
        moved.steps[2] = quad(kOpaque, {34, 5}, {70, 5}, {70, 55}, {10, 55});
        CHECK(!samePicture(moved, {0, 1}));
    }
    // the same through a second command buffer appended to the first: left out, and put back
    {
        Knobs k;
        k.split = true;
        const Run r = run(s, k);
        CHECK(r.completed && r.ran == without(s, {0, 1}) && r.counts.culled == 2);
        k.beforeCommit = [](Buffer& vb, Buffer*) { static_cast<V2*>(vb.contents())[2 * kSlots].x += 0.5f; };
        const Run u = run(s, k);
        CHECK(u.completed && u.ran == everyStep(s) && u.counts.undone == 2);
    }
    // with HSR: an opaque draw left out of its segment's prepass is put back there
    {
        Knobs k;
        k.hsr = true;
        const Run r = run(s, k);
        CHECK(r.completed && r.ran.size() == 2 && r.counts.culled == 2);
        k.beforeCommit = [](Buffer& vb, Buffer*) { static_cast<V2*>(vb.contents())[2 * kSlots].x += 0.5f; };
        const Run u = run(s, k);
        CHECK(u.completed && u.ran.size() == 4 && u.counts.undone == 2);
    }
    // ...also when its segment has closed since (a barrier that is no reader between it and its coverer)
    {
        Scene two;
        two.c1 = true;
        two.steps = {rect(kOpaque, 20, 10, 60, 50), rect(kMrt, 80, 5, 90, 15), rect(kOpaque, 10, 5, 70, 55)};
        Knobs k;
        k.hsr = true;
        const Run r = run(two, k);
        CHECK(r.completed && r.culled() == Gone({0}) && r.ran.size() == 2 && r.all.hsrPrepasses == 2);
        k.beforeCommit = [](Buffer& vb, Buffer*) { static_cast<V2*>(vb.contents())[2 * kSlots].x += 0.5f; };
        const Run u = run(two, k);
        CHECK(u.completed && u.ran.size() == 3 && u.counts.undone == 1);
    }
    // the matrix changed after the draws: the same
    {
        Scene m = s;
        m.mvp = std::array<float, 16>{2.0f / 96, 0, 0, 0, 0, -2.0f / 64, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1};
        CHECK(checked(m, "matrix untouched") == Gone({0, 1}));
        Knobs k;
        k.beforeCommit = [](Buffer&, Buffer* mb) { static_cast<float*>(mb->contents())[12] = -0.5f; };
        const Run u = run(m, k);
        CHECK(u.completed && u.ran == everyStep(m) && u.counts.undone == 2);
    }
    // a vertex buffer the GPU writes (a copy into it earlier in the command buffer): its bytes say nothing
    {
        Knobs k;
        k.first = [&](CommandBuffer& cb, Buffer& vb) { CHECK(cb.copy(vb, 0, vb, 84, 12)); };   // (step 0's first vertex into a slot no step uses)
        const Run g = run(s, k);
        CHECK(g.completed && g.ran == everyStep(s) && g.counts.culled == 0 && g.counts.why[cull::kNoData] != 0);
    }
    // a pass with a depth attachment of its own, and a pass drawn in tiles: left as they are
    {
        Knobs k;
        k.depth = true;
        const Run d = run(s, k);
        CHECK(d.completed && d.ran == everyStep(s) && d.counts.seen == 0);
        Knobs t;
        t.gmem = true;
        const Run g = run(s, t);
        CHECK(g.completed && g.counts.culled == 0);
    }
}

// Scenes made at random -- rectangles on and off the pixel grid, shared edges, turned quads, strips, indexed quads,
// scissors, every kind of draw -- each drawn with culling and held to its whole picture; and each mistake made on
// purpose shown to change the picture of at least one of them.
namespace {
Scene randomScene(uint32_t seed) {
    std::mt19937 rng(seed * 2654435761u + 17);
    Scene s;
    s.w = (seed & 1) ? 96 : 128;
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
        else if (kr == 16) kind = kFetch;
        else if (kr == 17) kind = s.c1 ? (rng() % 2 ? kFetchC1 : kMrt) : kFetch;
        else if (kr == 18) kind = rng() % 2 ? kNoModel : kTainted;
        // a rectangle: new, or an earlier one's again, the same or a pixel or a fraction larger or smaller
        std::array<double, 4> a;
        if (!rects.empty() && rng() % 3 == 0) {
            a = rects[rng() % rects.size()];
            static const double grow[] = {0, 0, 1, -1, 4, 0.5, -0.5, 0.0625, -0.0625};
            const double g = grow[rng() % 9];
            a = {a[0] - g, a[1] - g, a[2] + g, a[3] + g};
        } else {
            double x0 = coord(s.w), x1 = coord(s.w), y0 = coord(s.h), y1 = coord(s.h);
            if (rng() % 3 == 0) x0 = -8, x1 = s.w + 8, y0 = -8.5, y1 = s.h + 8.5;   // (a backdrop)
            a = {std::min(x0, x1), std::min(y0, y1), std::max(x0, x1), std::max(y0, y1)};
        }
        if (a[0] >= a[2] || a[1] >= a[3]) a = {10, 10, 30, 30};
        rects.push_back(a);
        Step step;
        const uint32_t shape = rng() % 12;
        if (shape == 0) {
            const double cx = (a[0] + a[2]) / 2, cy = (a[1] + a[3]) / 2;   // (on its corner)
            step = quad(kind, {cx, a[1]}, {a[2], cy}, {cx, a[3]}, {a[0], cy});
        } else if (shape == 1) {
            step = quad(kind, {a[0], a[1]}, {a[2] + double(rng() % 9) - 4, a[1]}, {a[2], a[3]}, {a[0], a[3] + double(rng() % 5) - 2});
        } else if (shape == 2) {
            step = strip(kind, a[0], a[1], a[2], a[3]);
        } else if (shape == 3) {
            step = quad(kind, {a[0], a[1]}, {a[0], a[3]}, {a[2], a[3]}, {a[2], a[1]});   // (the other winding)
        } else {
            step = rect(kind, a[0], a[1], a[2], a[3]);
        }
        if (!step.strip && rng() % 4 == 0) step.indexed = true;
        if (rng() % 6 == 0) {
            const uint32_t x = rng() % (s.w - 8), y = rng() % (s.h - 8);
            step = scissored(step, x, y, 1 + rng() % (s.w - x), 1 + rng() % (s.h - y));
        }
        if (rng() % 16 == 0) step.cullMode = 1 + rng() % 2;
        s.steps.push_back(step);
    }
    return s;
}
constexpr uint32_t kRandomScenes = 400;
}  // namespace

TEST(cull_keeps_the_picture_of_scenes_made_at_random) {
    uint32_t leftOut = 0, draws = 0, scenes = 0, undone = 0;
    for (uint32_t seed = 0; seed < kRandomScenes; seed++) {
        const Scene s = randomScene(seed);
        char name[32];
        snprintf(name, sizeof name, "random %u", seed);
        const Gone gone = checked(s, name);
        leftOut += uint32_t(gone.size());
        draws += uint32_t(s.steps.size());
        scenes += gone.empty() ? 0 : 1;
        // with HSR: the same draws left out
        if (seed % 4 == 0) {
            Knobs k;
            k.hsr = true;
            const Run h = run(s, k);
            const std::vector<uint32_t> left = without(s, gone);
            CHECK(h.completed && h.culled() == gone && h.ran.size() == left.size() &&
                  std::set<uint32_t>(h.ran.begin(), h.ran.end()) == std::set<uint32_t>(left.begin(), left.end()));
            undone += h.counts.undone;
        }
    }
    // (the scenes do get their draws left out: a tenth of all draws and more, in a third of the scenes and more)
    CHECK(leftOut * 10 >= draws);
    CHECK(scenes * 3 >= kRandomScenes);
    CHECK_EQ(undone, 0u);
    if (leftOut * 10 < draws || scenes * 3 < kRandomScenes) std::printf("       %u of %u draws left out, in %u of %u scenes\n", leftOut, draws, scenes, kRandomScenes);
}

TEST(cull_twins_each_change_a_picture) {
    // 1 a blending draw covers, 2 a coverer's rectangle taken outwards, 3 a reader ends nothing, 4 any two triangles
    // cover their bounding box: each leaves out a draw that can be seen in some scene, and the model's picture says so
    // (and two scenes made for it, ahead of the random ones: a layer a pixel wider than a coverer whose edges pass through
    // the pixels' centres, which twin 2 takes for covered; a rectangle under a coverer of two overlapping triangles)
    std::vector<Scene> made(2);
    made[0].steps = {rect(kOpaque, 19, 9, 61, 51), rect(kOpaque, 19.5, 9.5, 60.5, 50.5)};
    made[1].steps = {rect(kOpaque, 12, 8, 68, 52), [] {
                         Step s;
                         s.at = {{10, 5}, {70, 5}, {70, 55}, {10, 5}, {70, 5}, {10, 55}};
                         return s;
                     }()};
    for (uint8_t twin = 1; twin <= 4; twin++) {
        uint32_t wrong = 0, more = 0;
        for (uint32_t seed = 0; seed < made.size() + kRandomScenes && wrong == 0; seed++) {
            const Scene s = seed < made.size() ? made[seed] : randomScene(uint32_t(seed - made.size()));
            Knobs k;
            k.twin = twin;
            const Run r = run(s, k);
            CHECK(r.completed);
            const Run right = run(s);
            if (r.culled() != right.culled()) more++;
            if (!samePicture(s, r.culled())) wrong++;
        }
        CHECK(wrong != 0);
        if (wrong == 0) std::printf("       twin %u: no picture changed (%u scenes with other draws left out)\n", twin, more);
    }
}
