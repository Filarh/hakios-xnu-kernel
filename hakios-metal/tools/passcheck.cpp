// hakios-metal-passcheck: whether chains of passes come out the same, byte for byte, with light passes
// (CommandBuffer::setLightPasses; ios-sapphire dev-notes anexos/32-interfaz-telegram §4.3 and §5.2) as without them, and
// as the processor works them out -- and what recording a chain costs the processor each way. It runs the driver's core
// on the phone's GPU from a process of its own (the kernel's msm client, as hakios-metal-probe, -tilecheck and
// -ycbcrcheck do: no Metal, no CoreAnimation, nothing of the window server touched), compiles its few shaders there with
// the driver's compiler (the fixtures' bitcode, tools/kernels/*.ll), and reads results back from LINEAR RGBA8 targets.
//
// Every pass draws with exact values: a copy (tri_ptex: pixel (x, y) samples texel (x, y) of a texture of the target's
// size with a nearest sampler) over a rectangle, a constant colour (tri_cflat, n / 255 in each channel) over another,
// a framebuffer fetch (tri_fb: (r, g, b, a) -> (g, b, r, 1)), a Clear. The processor works each chain out on its own
// copy of the pictures, and each target the chain drew is compared with it, all of its bytes. A read the light passes
// left unordered shows as the older picture (a copy that missed a rectangle, a fetch of the colour before), a write as
// a rectangle drawn over before it was read.
//
//   1. chains, each with light passes off and on, each in one command buffer (the light passes' counts printed beside):
//      ping-pong A -> B -> A ... (8 passes); a target read 1, 2 and 5 passes after it was drawn; Clear + draw, read
//      after; a pass after a 2D copy, and after a dispatch (fill_id into a buffer, checked word for word too); a
//      framebuffer fetch in a light pass; a GMEM pass after a light one and light passes after a GMEM one
//   2. cost: a chain of 40 passes, each a copy of the one before and a rectangle, recorded with light passes off and
//      on -- the processor's time from the first draw to the commit, and commit to completion, the median of N runs
//
// One line per check, "passcheck: ok ..." or "passcheck: FAIL ...", times as "passcheck: time ...", and a last line
// with the count; the exit status is the number of failures (100: no GPU).
//
//   hakios-metal-passcheck [--reps N] [--no-time] [--size WxH]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "hakios-metal/command.hpp"
#include "hakios-metal/compiler.hpp"
#include "hakios-metal/texture.hpp"
#include "../tests/fixtures/fill_id_bc.hpp"
#include "../tests/fixtures/tri_cflat_fs_bc.hpp"
#include "../tests/fixtures/tri_color_vs_bc.hpp"
#include "../tests/fixtures/tri_fb_fs_bc.hpp"
#include "../tests/fixtures/tri_flat_vs_bc.hpp"
#include "../tests/fixtures/tri_ptex_fs_bc.hpp"

using namespace hakios::metal;

namespace {

int failures = 0;

void report(bool ok, const std::string& what) {
    std::printf("passcheck: %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    std::fflush(stdout);
    failures += !ok;
}

double nowUs() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Programs {
    Program copy, flat, fetch;
    std::shared_ptr<CompiledKernel> fill;
};

// --- the pictures, as the processor works them out ----------------------------------------------------------------

struct Rect {
    uint32_t x, y, w, h;
};

struct Picture {
    uint32_t w = 0, h = 0;
    std::vector<uint8_t> px;   // RGBA8, rows of 4 * w
    uint8_t* at(uint32_t x, uint32_t y) { return px.data() + 4 * (uint64_t(y) * w + x); }
    const uint8_t* at(uint32_t x, uint32_t y) const { return px.data() + 4 * (uint64_t(y) * w + x); }
};

// One draw of a pass: a copy of SRC (a texture of the chain; -1 the seed) over R, a colour over R, or a fetch over R
struct Op {
    enum Kind { kCopy, kColor, kFetch } kind;
    int src = -1;
    Rect r{};
    uint8_t color[4] = {};
};

// One step of a chain: a pass into texture TO with LOAD (and CLEAR, n of 255 a channel), its draws; or a 2D copy of
// texture FROM into TO; or a dispatch of fill_id into the chain's buffer; GMEM: the pass in tiles if it can be
struct Step {
    enum Kind { kPass, kBlit, kDispatch } kind = kPass;
    int to = 0, from = 0;
    LoadAction load = LoadAction::kLoad;
    uint8_t clear[4] = {};
    std::vector<Op> ops;
    bool gmem = false;
};

struct Chain {
    std::string name;
    int textures = 2;
    std::vector<Step> steps;
};

const uint32_t kFillWords = 4096, kFillC = 0x00c0ffee;

// the chain worked out on the processor from SEED: every texture's picture at its end (0x5a where nothing drew)
std::vector<Picture> expected(const Chain& c, const Picture& seed) {
    std::vector<Picture> t(c.textures, seed);
    for (Picture& p : t) std::fill(p.px.begin(), p.px.end(), 0x5a);
    for (const Step& s : c.steps) {
        if (s.kind == Step::kBlit) {
            t[s.to].px = t[s.from].px;
            continue;
        }
        if (s.kind != Step::kPass) continue;
        Picture& to = t[s.to];
        if (s.load == LoadAction::kClear) {
            for (uint64_t i = 0; i < uint64_t(to.w) * to.h; i++) memcpy(to.px.data() + 4 * i, s.clear, 4);
        }
        for (const Op& o : s.ops) {
            const Picture* from = o.kind == Op::kCopy ? (o.src < 0 ? &seed : &t[o.src]) : nullptr;
            const Picture before = to;   // (a copy of the target itself, or a fetch, reads it as it was)
            for (uint32_t y = o.r.y; y < o.r.y + o.r.h; y++) {
                for (uint32_t x = o.r.x; x < o.r.x + o.r.w; x++) {
                    uint8_t* d = to.at(x, y);
                    if (o.kind == Op::kColor) memcpy(d, o.color, 4);
                    if (o.kind == Op::kCopy) memcpy(d, (from == &to ? before : *from).at(x, y), 4);
                    if (o.kind == Op::kFetch) {
                        const uint8_t* b = before.at(x, y);
                        const uint8_t v[4] = {b[1], b[2], b[0], 255};
                        memcpy(d, v, 4);
                    }
                }
            }
        }
    }
    return t;
}

// --- the chain on the GPU -------------------------------------------------------------------------------------------

std::shared_ptr<Texture> linearTarget(Gpu& gpu, uint32_t w, uint32_t h, std::string* why) {
    TextureDesc d;
    d.format = PixelFormat::kRGBA8Unorm;
    d.width = w;
    d.height = h;
    d.usage = kTextureUsageRead | kTextureUsageRenderTarget;
    return Texture::make(gpu, d, why);
}

Picture pixelsOf(const Texture& t) {
    Picture p;
    p.w = t.width();
    p.h = t.height();
    p.px.resize(uint64_t(p.w) * p.h * 4);
    const auto* src = static_cast<const uint8_t*>(t.contents());
    for (uint32_t y = 0; y < p.h; y++) memcpy(p.px.data() + uint64_t(y) * p.w * 4, src + y * t.bytesPerRow(), uint64_t(p.w) * 4);
    return p;
}

void fillPixels(const Texture& t, const Picture& p) {
    auto* dst = static_cast<uint8_t*>(t.contents());
    for (uint32_t y = 0; y < p.h; y++) memcpy(dst + y * t.bytesPerRow(), p.px.data() + uint64_t(y) * p.w * 4, uint64_t(p.w) * 4);
}

Draw drawOf(const Rect& r) {
    Draw d;
    d.vertexStart = 0;
    d.vertexCount = 6;
    d.viewport = Viewport{double(r.x), double(r.y), double(r.w), double(r.h), 0, 1};
    return d;
}

struct Result {
    std::vector<Picture> got;
    std::vector<uint32_t> words;   // the dispatch's buffer
    CommandBuffer::LightCounts counts;
    uint32_t gmemPasses = 0, passes = 0;
    std::string why;
};

Result runChain(Gpu& gpu, CommandQueue& queue, const Programs& p, const Chain& c, const Picture& seed, bool light) {
    Result r;
    std::shared_ptr<Texture> seedTex = linearTarget(gpu, seed.w, seed.h, &r.why);
    std::vector<std::shared_ptr<Texture>> t;
    for (int i = 0; i < c.textures && seedTex != nullptr; i++) t.push_back(linearTarget(gpu, seed.w, seed.h, &r.why));
    std::vector<std::shared_ptr<Buffer>> colors;
    std::shared_ptr<Buffer> fillOut = newBuffer(gpu, 4 * kFillWords, &r.why), fillC = newBuffer(gpu, 64, &r.why);
    if (seedTex == nullptr || int(t.size()) != c.textures || t.back() == nullptr || fillOut == nullptr || fillC == nullptr) {
        r.why = "textures: " + r.why;
        return r;
    }
    fillPixels(*seedTex, seed);
    for (auto& x : t) memset(x->contents(), 0x5a, x->buffer()->length());
    memset(fillOut->contents(), 0xa5, 4 * kFillWords);
    static_cast<uint32_t*>(fillC->contents())[0] = kFillC;
    Sampler nearest;
    makeSampler(SamplerDesc{}, &nearest, &r.why);
    auto cb = queue.commandBuffer();
    cb->setLightPasses(light);
    cb->setGmem(true);
    for (const Step& s : c.steps) {
        bool ok = true;
        if (s.kind == Step::kBlit) {
            const render::BlitRect all{0, 0, seed.w, seed.h};
            ok = cb->blit(*t[s.from], 0, all, *t[s.to], 0, all, false);
        } else if (s.kind == Step::kDispatch) {
            const uint32_t g[3] = {kFillWords / 64, 1, 1}, local[3] = {64, 1, 1};
            ok = cb->dispatch(p.fill->kernel, g, local, std::vector<Binding>{Binding{fillOut.get(), 0}, Binding{fillC.get(), 0}});
        } else {
            render::Options o;
            o.gmem = s.gmem;
            for (size_t i = 0; i < s.ops.size() && ok; i++) {
                const Op& op = s.ops[i];
                RenderPass pass;
                pass.color0 = t[s.to].get();
                pass.load = i == 0 ? s.load : LoadAction::kLoad;
                for (int k = 0; k < 4; k++) pass.clearColor[k] = s.clear[k] / 255.0;
                Bindings fsb;
                if (op.kind == Op::kCopy) {
                    fsb.textures = {op.src < 0 ? seedTex.get() : t[op.src].get()};
                    fsb.samplers = {&nearest};
                    ok = cb->draw(p.copy, pass, drawOf(op.r), Bindings{}, fsb, o);
                } else if (op.kind == Op::kColor) {
                    auto b = newBuffer(gpu, 16, &r.why);
                    if (b == nullptr) return r;
                    auto* f = static_cast<float*>(b->contents());
                    for (int k = 0; k < 4; k++) f[k] = op.color[k] / 255.0f;
                    colors.push_back(b);
                    fsb.buffers = {Binding{b.get(), 0}};
                    ok = cb->draw(p.flat, pass, drawOf(op.r), Bindings{}, fsb, o);
                } else {
                    ok = cb->draw(p.fetch, pass, drawOf(op.r), Bindings{}, Bindings{}, o);
                }
            }
        }
        if (!ok) {
            r.why = "recording: " + cb->error();
            return r;
        }
    }
    r.counts = cb->lightCounts();
    r.gmemPasses = cb->counts().gmemPasses;
    r.passes = cb->counts().passes;
    cb->commit();
    cb->waitUntilCompleted();
    if (cb->status() != CommandBuffer::kCompleted) {
        r.why = "the command buffer: " + cb->error();
        return r;
    }
    for (auto& x : t) r.got.push_back(pixelsOf(*x));
    r.words.assign(static_cast<const uint32_t*>(fillOut->contents()), static_cast<const uint32_t*>(fillOut->contents()) + kFillWords);
    return r;
}

// how many pixels differ, and the first that does
std::string differ(const Picture& a, const Picture& b, uint64_t* count) {
    *count = 0;
    std::string first;
    for (uint64_t i = 0; i < a.px.size() / 4; i++) {
        if (memcmp(a.px.data() + 4 * i, b.px.data() + 4 * i, 4) == 0) continue;
        if (*count == 0) {
            char s[140];
            const uint8_t *x = a.px.data() + 4 * i, *y = b.px.data() + 4 * i;
            std::snprintf(s, sizeof s, " (first at %llu,%llu: %02x%02x%02x%02x, wanted %02x%02x%02x%02x)", (unsigned long long)(i % a.w),
                          (unsigned long long)(i / a.w), x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]);
            first = s;
        }
        (*count)++;
    }
    return first;
}

// --- the chains -----------------------------------------------------------------------------------------------------

Op copyOf(int src, Rect r) { Op o{Op::kCopy}; o.src = src, o.r = r; return o; }
Op colorOf(Rect r, int n) {
    // (a colour of its own for each rectangle, every channel n of 255 exactly)
    Op o{Op::kColor};
    o.r = r;
    o.color[0] = uint8_t(37 * n + 11), o.color[1] = uint8_t(91 * n + 29), o.color[2] = uint8_t(53 * n + 71), o.color[3] = uint8_t(255 - 13 * n);
    return o;
}
Op fetchOf(Rect r) { Op o{Op::kFetch}; o.r = r; return o; }

std::vector<Chain> chains(uint32_t W, uint32_t H) {
    const Rect all{0, 0, W, H};
    auto rectOf = [&](int n) { return Rect{(17u * n) % (W - W / 5), (23u * n) % (H - H / 5), W / 5, H / 5}; };
    auto pass = [&](int to, int src, int n) {
        Step s;
        s.to = to;
        s.load = LoadAction::kDontCare;
        s.ops = {copyOf(src, all), colorOf(rectOf(n), n)};
        return s;
    };
    std::vector<Chain> out;
    {
        Chain c{"ping-pong A->B->A, 8 passes", 2, {}};
        for (int k = 0; k < 8; k++) c.steps.push_back(pass(k % 2, k == 0 ? -1 : (k - 1) % 2, k));
        out.push_back(c);
    }
    for (const int d : {1, 2, 5}) {
        Chain c{"each pass reads the target drawn " + std::to_string(d) + " pass(es) before, 6 targets", 6, {}};
        for (int k = 0; k < 6; k++) c.steps.push_back(pass(k, k >= d ? k - d : -1, k));
        out.push_back(c);
    }
    {
        // a copy of A into half of B and a rectangle over A again, A read after: the write after the read
        Chain c{"a pass draws over what the pass before read", 3, {}};
        c.steps.push_back(pass(0, -1, 0));
        Step b;
        b.to = 1;
        b.load = LoadAction::kDontCare;
        b.ops = {copyOf(0, all)};
        c.steps.push_back(b);
        Step a;
        a.to = 0;
        a.ops = {colorOf(Rect{0, 0, W, H / 2}, 9)};
        c.steps.push_back(a);
        c.steps.push_back(pass(2, 1, 3));
        out.push_back(c);
    }
    {
        Chain c{"Clear + draw, read by the next pass", 3, {}};
        for (int k = 0; k < 3; k++) {
            Step s;
            s.to = k;
            s.load = LoadAction::kClear;
            s.clear[0] = uint8_t(40 + 50 * k), s.clear[1] = uint8_t(200 - 30 * k), s.clear[2] = 17, s.clear[3] = 255;
            s.ops = {colorOf(rectOf(k), k)};
            if (k > 0) s.ops.insert(s.ops.begin(), copyOf(k - 1, Rect{0, 0, W / 2, H}));
            c.steps.push_back(s);
        }
        out.push_back(c);
    }
    {
        Chain c{"passes after a 2D copy", 4, {}};
        c.steps.push_back(pass(0, -1, 0));
        c.steps.push_back(pass(1, 0, 1));
        Step b;
        b.kind = Step::kBlit, b.from = 1, b.to = 2;
        c.steps.push_back(b);
        c.steps.push_back(pass(3, 2, 2));
        c.steps.push_back(pass(0, 3, 3));
        out.push_back(c);
    }
    {
        Chain c{"passes after a dispatch", 3, {}};
        c.steps.push_back(pass(0, -1, 0));
        c.steps.push_back(pass(1, 0, 1));
        Step d;
        d.kind = Step::kDispatch;
        c.steps.push_back(d);
        c.steps.push_back(pass(2, 1, 2));
        c.steps.push_back(pass(0, 2, 3));
        out.push_back(c);
    }
    {
        Chain c{"a framebuffer fetch in a light pass, read after", 3, {}};
        c.steps.push_back(pass(0, -1, 0));
        c.steps.push_back(pass(1, 0, 1));
        Step f;
        f.to = 0;
        f.ops = {fetchOf(Rect{W / 4, H / 4, W / 2, H / 2})};
        c.steps.push_back(f);
        c.steps.push_back(pass(2, 0, 2));
        out.push_back(c);
    }
    {
        Chain c{"a GMEM pass after light ones, light ones after it", 5, {}};
        c.steps.push_back(pass(0, -1, 0));
        c.steps.push_back(pass(1, 0, 1));
        Step g = pass(2, 1, 2);
        g.gmem = true;
        c.steps.push_back(g);
        c.steps.push_back(pass(3, 2, 3));
        c.steps.push_back(pass(4, 3, 4));
        c.steps.push_back(pass(0, 4, 5));
        out.push_back(c);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    int reps = 7;
    bool times = true;
    uint32_t W = 333, H = 217;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = std::max(1, atoi(argv[++i]));
        if (strcmp(argv[i], "--no-time") == 0) times = false;
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) std::sscanf(argv[++i], "%ux%u", &W, &H);
    }
    W = std::max(W, 16u), H = std::max(H, 16u);
    std::string why;
    auto transport = openKernelGpu(&why);
    if (transport == nullptr) {
        std::printf("passcheck: no GPU: %s\n", why.c_str());
        return 100;
    }
    Gpu gpu(std::move(transport));
    uint64_t chip = 0;
    gpu.param(kParamChipId, &chip);
    std::printf("passcheck: GPU chip id 0x%llx, FAULTS %llu, targets %ux%u RGBA8 linear\n", (unsigned long long)chip, (unsigned long long)gpu.faults(), W, H);
    auto queue = CommandQueue::make(gpu, &why);
    if (queue == nullptr) {
        report(false, "a queue: " + why);
        return 1;
    }
    if (!hasCompiler()) {
        report(false, "this build has no compiler: the shaders cannot be compiled here");
        return failures;
    }
    const uint64_t faults0 = gpu.faults();
    std::shared_ptr<CompiledStage> cvs(compileStage(kernels::tri_color_vs_bc, kernels::tri_color_vs_bc_size, &why));
    std::shared_ptr<CompiledStage> fvs(compileStage(kernels::tri_flat_vs_bc, kernels::tri_flat_vs_bc_size, &why));
    std::shared_ptr<CompiledStage> pfs(compileStage(kernels::tri_ptex_fs_bc, kernels::tri_ptex_fs_bc_size, &why));
    std::shared_ptr<CompiledStage> ofs(compileStage(kernels::tri_cflat_fs_bc, kernels::tri_cflat_fs_bc_size, &why));
    std::shared_ptr<CompiledStage> bfs(compileStage(kernels::tri_fb_fs_bc, kernels::tri_fb_fs_bc_size, &why));
    std::shared_ptr<CompiledProgram> copy = cvs && pfs ? std::shared_ptr<CompiledProgram>(linkProgram(cvs, pfs, PixelFormat::kRGBA8Unorm, &why)) : nullptr;
    std::shared_ptr<CompiledProgram> flat = fvs && ofs ? std::shared_ptr<CompiledProgram>(linkProgram(fvs, ofs, PixelFormat::kRGBA8Unorm, &why)) : nullptr;
    std::shared_ptr<CompiledProgram> fetch = fvs && bfs ? std::shared_ptr<CompiledProgram>(linkProgram(fvs, bfs, PixelFormat::kRGBA8Unorm, &why)) : nullptr;
    std::shared_ptr<CompiledKernel> fill(compileAir(kernels::fill_id_bc, kernels::fill_id_bc_size, &why));
    if (copy == nullptr || flat == nullptr || fetch == nullptr || fill == nullptr) {
        report(false, "the shaders, compiled here: " + why);
        return failures;
    }
    Programs p{copy->program, flat->program, fetch->program, fill};

    Picture seed;
    seed.w = W, seed.h = H;
    seed.px.resize(uint64_t(W) * H * 4);
    std::mt19937 rng(20261120);
    for (uint8_t& b : seed.px) b = uint8_t(rng());

    // 1. the chains, each with light passes off and on
    for (const Chain& c : chains(W, H)) {
        const std::vector<Picture> want = expected(c, seed);
        bool dispatches = false;
        for (const Step& s : c.steps) dispatches = dispatches || s.kind == Step::kDispatch;
        std::vector<Picture> offGot;
        for (const bool light : {false, true}) {
            const Result r = runChain(gpu, *queue, p, c, seed, light);
            char counts[200];
            std::snprintf(counts, sizeof counts, " [%u passes, %u in GMEM; light %u, full %u, read barriers %u, waits before a write %u]", r.passes,
                          r.gmemPasses, r.counts.light, r.counts.full, r.counts.barriers, r.counts.writeWaits);
            const std::string what = c.name + (light ? ", light passes" : ", light passes off");
            if (!r.why.empty() || r.got.size() != want.size()) {
                report(false, what + ": " + r.why);
                continue;
            }
            uint64_t total = 0;
            std::string firsts;
            for (size_t i = 0; i < want.size(); i++) {
                uint64_t n = 0;
                const std::string f = differ(r.got[i], want[i], &n);
                if (n != 0) firsts += " texture " + std::to_string(i) + ": " + std::to_string(n) + " px" + f + ";";
                total += n;
            }
            if (dispatches) {
                uint32_t wrong = 0;
                for (uint32_t i = 0; i < kFillWords; i++) wrong += r.words[i] != i * 0x9E3779B1u + kFillC;
                report(wrong == 0, what + ": fill_id's buffer, " + std::to_string(wrong) + " of " + std::to_string(kFillWords) + " words wrong");
            }
            report(total == 0, what + ": " + std::to_string(total) + " pixels differ from the processor's" + firsts + counts);
            if (light && !offGot.empty()) {
                uint64_t same = 0;
                for (size_t i = 0; i < offGot.size(); i++) {
                    uint64_t n = 0;
                    differ(r.got[i], offGot[i], &n);
                    same += n;
                }
                report(same == 0, c.name + ": light passes on against off, " + std::to_string(same) + " pixels differ");
            }
            if (!light) offGot = r.got;
        }
    }

    // 2. cost: 40 passes, each a copy of the one before and a rectangle, recorded with light passes off and on
    if (times) {
        Chain c{"40", 2, {}};
        const Rect all{0, 0, W, H};
        for (int k = 0; k < 40; k++) {
            Step s;
            s.to = k % 2;
            s.load = LoadAction::kDontCare;
            s.ops = {copyOf(k == 0 ? -1 : (k - 1) % 2, all), colorOf(Rect{uint32_t(3 * k) % (W / 2), uint32_t(5 * k) % (H / 2), W / 4, H / 4}, k)};
            c.steps.push_back(s);
        }
        auto seedTex = linearTarget(gpu, W, H, &why);
        std::shared_ptr<Texture> t[2] = {linearTarget(gpu, W, H, &why), linearTarget(gpu, W, H, &why)};
        auto colorBuf = newBuffer(gpu, 16 * 40, &why);
        if (seedTex == nullptr || t[1] == nullptr || colorBuf == nullptr) {
            report(false, "cost: textures: " + why);
        } else {
            fillPixels(*seedTex, seed);
            auto* f = static_cast<float*>(colorBuf->contents());
            for (int k = 0; k < 40; k++) {
                for (int i = 0; i < 4; i++) f[4 * k + i] = c.steps[k].ops[1].color[i] / 255.0f;
            }
            Sampler nearest;
            makeSampler(SamplerDesc{}, &nearest, &why);
            const std::vector<Picture> want = expected(c, seed);
            for (const bool light : {false, true}) {
                std::vector<double> record, run, perPass;
                uint64_t wrong = 0;
                CommandBuffer::LightCounts counts;
                size_t words = 0;
                for (int rep = 0; rep <= reps; rep++) {
                    auto cb = queue->commandBuffer();
                    // (the plug-in's command buffer with no metal_* boot-arg: GMEM by the draws, sysmem below 56)
                    cb->setGmem(true);
                    cb->setGmemMinDraws(56);
                    cb->setGmemBinning(true);
                    cb->setGmemTuned(true);
                    cb->setLightPasses(light);
                    bool ok = true;
                    const double t0 = nowUs();
                    for (int k = 0; k < 40 && ok; k++) {
                        RenderPass pass;
                        pass.color0 = t[k % 2].get();
                        pass.load = LoadAction::kDontCare;
                        Bindings fsb;
                        fsb.textures = {k == 0 ? seedTex.get() : t[(k - 1) % 2].get()};
                        fsb.samplers = {&nearest};
                        ok = cb->draw(p.copy, pass, drawOf(all), Bindings{}, fsb);
                        pass.load = LoadAction::kLoad;
                        Bindings cfs;
                        cfs.buffers = {Binding{colorBuf.get(), uint64_t(16 * k)}};
                        ok = ok && cb->draw(p.flat, pass, drawOf(c.steps[k].ops[1].r), Bindings{}, cfs);
                    }
                    const double t1 = nowUs();
                    cb->commit();
                    const double t2 = nowUs();
                    cb->waitUntilCompleted();
                    const double t3 = nowUs();
                    if (!ok || cb->status() != CommandBuffer::kCompleted) {
                        report(false, std::string("cost chain, light passes ") + (light ? "on" : "off") + ": " + cb->error());
                        break;
                    }
                    counts = cb->lightCounts();
                    words = cb->streamWords();
                    if (rep == 0) continue;   // (the first: the programs' states made, the queue's pools filled)
                    record.push_back(t2 - t0);
                    perPass.push_back((t1 - t0) / 40);
                    run.push_back(t3 - t2);
                    for (int i = 0; i < 2; i++) {
                        uint64_t n = 0;
                        differ(pixelsOf(*t[i]), want[i], &n);
                        wrong += n;
                    }
                }
                report(wrong == 0, std::string("cost chain of 40 passes, light passes ") + (light ? "on" : "off") + ": " + std::to_string(wrong) +
                                       " pixels differ from the processor's over " + std::to_string(reps) + " runs");
                std::printf("passcheck: time 40 passes, light passes %s: recorded and committed %.0f us (median of %d; %.1f us a pass, draw calls alone), "
                            "commit to completion %.0f us; %zu words; light %u, full %u, read barriers %u, waits before a write %u\n",
                            light ? "on " : "off", median(record), reps, median(perPass), median(run), words, counts.light, counts.full, counts.barriers,
                            counts.writeWaits);
                std::fflush(stdout);
            }
        }
    }

    report(gpu.faults() == faults0, "no GPU reset during the run (FAULTS " + std::to_string(faults0) + " -> " + std::to_string(gpu.faults()) + ")");
    std::printf("passcheck: %d failure(s)\n", failures);
    return failures;
}
