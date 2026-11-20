// hakios-metal-tilecheck: whether this GPU draws, clears, blends, fetches, samples, copies, loads and stores a colour
// render target laid out TILE6_3 (TextureDesc::tiled) and TILE6_3 with UBWC (TextureDesc::compressed) exactly as it does
// a linear one, and what each of those costs in each layout. It runs the driver's core on the phone's GPU from a
// process of its own (the kernel's msm client, as hakios-metal-probe does: no Metal, no CoreAnimation, nothing of the
// window server touched), compiles its few shaders there with the driver's compiler (the fixtures' bitcode,
// tools/kernels/*.ll), and reads results back from LINEAR textures only: the tiled bytes are never interpreted here.
//
//   1. the UBWC and tiling configuration the GPU holds now: RB/TPL1/SP_NC_MODE_CNTL, UCHE_MODE_CNTL, RBBM_NC_MODE_CNTL
//      (read by the CP itself, CP_REG_TO_MEM), decoded as Linux's a6xx_set_ubwc_config writes them
//   2. correctness, per format (BGRA8, RGBA8) and size (752x1626, 99x209, 270x12): the same sequence of passes into a
//      linear, a tiled and a compressed target -- a Clear, a draw, a blended draw, a framebuffer fetch (sysmem); a load,
//      blended draws and a store (GMEM); a Clear in the tiles and a store (GMEM) -- each stage read out by a 2D copy
//      and by sampling into a linear texture, and the tiled and compressed results compared byte for byte with the
//      linear one; and a random picture copied into the layout and back out, compared with itself
//      The compressed target runs the sequence twice: as the driver runs it by default (its "GMEM" passes in sysmem:
//      render::gmemPlan refuses a compressed attachment) and with CommandBuffer::setGmemCompressed (the plug-in's
//      metal_gmemubwc=1), its GMEM passes then in as many tiles as the linear one's
//   3. costs (GPU time by the CP's always-on counter around each pass, the median of N runs): a Clear of 756x1630; a
//      full-screen draw, a blended one and a fetch at 752x1626; sampling it 1:1 into a linear target and at a quarter
//      into a smaller one; a GMEM load and store of it; 2D copies out of it and into it
//
// One line per check, "tilecheck: ok ..." or "tilecheck: FAIL ...", times as "tilecheck: time ...", and a last line
// with the count; the exit status is the number of failures (100: no GPU).
//
//   hakios-metal-tilecheck [--reps N] [--no-time] [--no-ubwc]
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
#include "hakios-metal/pm4.hpp"
#include "hakios-metal/texture.hpp"
#include "../tests/fixtures/tri_cflat_fs_bc.hpp"
#include "../tests/fixtures/tri_color_fs_bc.hpp"
#include "../tests/fixtures/tri_color_vs_bc.hpp"
#include "../tests/fixtures/tri_fb_fs_bc.hpp"
#include "../tests/fixtures/tri_flat_vs_bc.hpp"
#include "../tests/fixtures/tri_tex_fs_bc.hpp"

using namespace hakios::metal;

namespace {

int failures = 0;

void report(bool ok, const std::string& what) {
    std::printf("tilecheck: %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    std::fflush(stdout);
    failures += !ok;
}

enum Layout { kLinear = 0, kTiled = 1, kUbwc = 2 };
const char* const kLayoutName[3] = {"linear", "tiled", "ubwc"};

std::shared_ptr<Texture> makeTexture(Gpu& gpu, Layout l, PixelFormat f, uint32_t w, uint32_t h, std::string* why) {
    TextureDesc d;
    d.format = f;
    d.width = w;
    d.height = h;
    d.usage = kTextureUsageRead | kTextureUsageRenderTarget;
    d.tiled = l == kTiled;
    d.compressed = l == kUbwc;
    return Texture::make(gpu, d, why);
}

bool completed(const CommandBuffer& cb) { return cb.status() == CommandBuffer::kCompleted; }

double gpuUs(const CommandBuffer& c) {
    double us = 0;
    for (const CommandBuffer::PassTime& pt : c.passTimes()) us += pt.us;
    return us;
}

uint32_t tilesOf(const CommandBuffer& c) {
    uint32_t t = 0;
    for (const CommandBuffer::PassTime& pt : c.passTimes()) t += pt.tiles;
    return t;
}

// a linear texture's W x H pixels, row after row (its rows read once each: write-combined memory)
std::vector<uint8_t> pixelsOf(const Texture& t) {
    const uint32_t bpp = t.info().bytesPerPixel;
    std::vector<uint8_t> out(uint64_t(t.width()) * t.height() * bpp);
    const auto* src = static_cast<const uint8_t*>(t.contents());
    for (uint32_t y = 0; y < t.height(); y++) memcpy(out.data() + uint64_t(y) * t.width() * bpp, src + y * t.bytesPerRow(), uint64_t(t.width()) * bpp);
    return out;
}

void fillPixels(const Texture& t, const std::vector<uint8_t>& px) {
    const uint32_t bpp = t.info().bytesPerPixel;
    auto* dst = static_cast<uint8_t*>(t.contents());
    for (uint32_t y = 0; y < t.height(); y++) memcpy(dst + y * t.bytesPerRow(), px.data() + uint64_t(y) * t.width() * bpp, uint64_t(t.width()) * bpp);
}

// how many pixels differ, and the first that does
std::string differ(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, uint32_t w, uint64_t* count) {
    *count = 0;
    if (a.size() != b.size()) {
        *count = ~0ull;
        return "different sizes";
    }
    std::string first;
    for (uint64_t p = 0; p < a.size() / 4; p++) {
        if (memcmp(a.data() + 4 * p, b.data() + 4 * p, 4) != 0) {
            if (*count == 0) {
                char t[120];
                std::snprintf(t, sizeof t, " (first at %llu,%llu: %02x%02x%02x%02x against %02x%02x%02x%02x)", (unsigned long long)(p % w),
                              (unsigned long long)(p / w), a[4 * p], a[4 * p + 1], a[4 * p + 2], a[4 * p + 3], b[4 * p], b[4 * p + 1], b[4 * p + 2],
                              b[4 * p + 3]);
                first = t;
            }
            (*count)++;
        }
    }
    return first;
}

Draw drawOf(uint32_t count, double x, double y, double w, double h) {
    Draw d;
    d.vertexStart = 0;
    d.vertexCount = count;
    d.viewport = Viewport{x, y, w, h, 0, 1};
    return d;
}

struct Programs {
    Program color, tex, over, fb;
};

// the six registers that say how the GPU lays tiled and compressed images out in memory, read by the command processor
void readRegisters(Gpu& gpu) {
    struct R {
        uint32_t reg;
        const char* name;
    };
    // a6xx.xml.h (Mesa 25.0.7): REG_A6XX_RB_NC_MODE_CNTL 0x8e08, TPL1_NC_MODE_CNTL 0xb604, SP_NC_MODE_CNTL 0xae02,
    // UCHE_MODE_CNTL 0x0e01, RBBM_NC_MODE_CNTL 0x0534, RBBM_PERFCTR_CNTL 0x0500, RB_CCU_CNTL 0x8e07
    const R regs[] = {{0x8e08, "RB_NC_MODE_CNTL"},   {0xb604, "TPL1_NC_MODE_CNTL"}, {0xae02, "SP_NC_MODE_CNTL"}, {0x0e01, "UCHE_MODE_CNTL"},
                      {0x0534, "RBBM_NC_MODE_CNTL"}, {0x0500, "RBBM_PERFCTR_CNTL"}, {0x8e07, "RB_CCU_CNTL"}};
    const uint32_t n = sizeof regs / sizeof regs[0];
    std::string why;
    auto out = newBuffer(gpu, 4096, &why);
    int e = 0;
    auto commands = gpu.newBo(4096, kBoGpuReadOnly | kBoWriteCombined, &e);
    if (out == nullptr || commands == nullptr) {
        report(false, "registers: two buffers: " + why);
        return;
    }
    auto* words = static_cast<uint32_t*>(out->contents());
    for (uint32_t i = 0; i < 64; i++) words[i] = 0xa5a5a5a5u;
    pm4::Stream s;
    s.waitForIdle();
    for (uint32_t i = 0; i < n; i++) {
        const uint64_t at = out->gpuAddress() + 4ull * i;
        const uint32_t w[3] = {regs[i].reg | (1u << 18), uint32_t(at), uint32_t(at >> 32)};   // CP_REG_TO_MEM: REG, CNT 1, 32-bit
        s.packet(a6xx::CP_REG_TO_MEM, w, 3);
    }
    s.waitForIdle();
    memcpy(commands->cpu(), s.words().data(), 4 * s.words().size());
    uint32_t fence = 0;
    int sub = gpu.submit(0, *commands, 0, uint32_t(s.words().size()), {BoUse{&out->bo(), true}}, &fence);
    int w = sub == 0 ? gpu.wait(0, fence, 1000000000) : sub;
    if (w != 0) {
        report(false, "registers: the read did not complete: " + std::string(strerror(w)));
        return;
    }
    uint32_t v[8] = {};
    for (uint32_t i = 0; i < n; i++) v[i] = words[i];
    for (uint32_t i = 0; i < n; i++) std::printf("tilecheck: reg %-18s 0x%08x\n", regs[i].name, v[i]);
    // decoded as a6xx_set_ubwc_config writes them (Linux v6.12 a6xx_gpu.c:568-610; KGSL a6xx_start): hbb = 13 +
    // (hbb_hi << 2 | hbb_lo); mode bit 0 (UBWC 1.0 level-1 bank swizzling); min_acc_len bit 3; level2 off bit 12 (RB, SP),
    // bit 6 (TPL1); UCHE: hbb_lo at 21, min_acc_len at 23
    auto hbbRb = [](uint32_t x) { return 13 + ((((x >> 10) & 1) << 2) | ((x >> 1) & 3)); };
    auto hbbTp = [](uint32_t x) { return 13 + ((((x >> 4) & 1) << 2) | ((x >> 1) & 3)); };
    std::printf("tilecheck: (decoded: RB hbb %u mode %u mal %u l2off %u | TPL1 hbb %u mode %u mal %u l2off %u | SP hbb %u mode %u mal %u l2off %u | "
                "UCHE hbb_lo %u mal %u | RBBM macrotile %u | perf counters %s)\n",
                hbbRb(v[0]), v[0] & 1, (v[0] >> 3) & 1, (v[0] >> 12) & 1, hbbTp(v[1]), v[1] & 1, (v[1] >> 3) & 1, (v[1] >> 6) & 1, hbbRb(v[2]), v[2] & 1,
                (v[2] >> 3) & 1, (v[2] >> 12) & 1, (v[3] >> 21) & 3, (v[3] >> 23) & 1, v[4] & 3, v[5] & 1 ? "on" : "off");
    // the layout of a tiled image depends on these (fd6_tiled_memcpy.cc's header: the bank swizzling by the highest bank
    // bit and the swizzle levels, the macrotile mode); what matters for a picture to survive is that the units that
    // write it (RB) and read it (TP, SP) agree
    const bool agree = hbbRb(v[0]) == hbbTp(v[1]) && hbbRb(v[0]) == hbbRb(v[2]) && (v[0] & 1) == (v[1] & 1) && (v[0] & 1) == (v[2] & 1) &&
                       ((v[0] >> 12) & 1) == ((v[1] >> 6) & 1) && ((v[0] >> 12) & 1) == ((v[2] >> 12) & 1) && ((v[3] >> 21) & 3) == ((v[0] >> 1) & 3);
    report(agree, std::string("the RB, the TP, the SP and the UCHE are told the same bank bit, swizzle and level-2 setting") +
                      (agree ? "" : " -- they are NOT: a tiled image written by one unit reads wrong in another"));
}

// The correctness sequence over one target of layout L (a fresh texture), each stage read out into linear textures:
// OUT[0] copy after the sysmem passes, OUT[1] that sampled, OUT[2] copy after the GMEM load/store pass, OUT[3] copy after
// the GMEM Clear-and-store pass, OUT[4] a random picture copied in and out. Empty vectors where something refused.
struct Run {
    std::vector<uint8_t> out[5];
    std::string why;
    uint32_t gmemTiles = 0;
};

// GMEM_UBWC: the GMEM passes' command buffers with setGmemCompressed (the plug-in's metal_gmemubwc=1); without it a
// pass into a compressed target runs in sysmem (render::gmemPlan refuses it), and its "GMEM" stages are sysmem ones
Run runSequence(Gpu& gpu, CommandQueue& queue, const Programs& p, Layout l, PixelFormat f, uint32_t W, uint32_t H, const std::vector<uint8_t>& random,
                bool gmemUbwc = false) {
    Run r;
    auto t = makeTexture(gpu, l, f, W, H, &r.why);
    std::shared_ptr<Texture> lin[5];
    for (auto& x : lin) {
        if (t != nullptr) x = makeTexture(gpu, kLinear, f, W, H, &r.why);
    }
    auto cbuf = newBuffer(gpu, 16, &r.why);
    if (t == nullptr || lin[4] == nullptr || cbuf == nullptr) return r;
    for (auto& x : lin) memset(x->contents(), 0x5a, x->buffer()->length());
    auto* c = static_cast<float*>(cbuf->contents());
    c[0] = 0.8f, c[1] = 0.4f, c[2] = 0.2f, c[3] = 0.5f;
    Bindings cb0;
    cb0.buffers = {Binding{cbuf.get(), 0}};
    Sampler nearest;
    makeSampler(SamplerDesc{}, &nearest, &r.why);
    Bindings tex;
    tex.textures = {t.get()};
    tex.samplers = {&nearest};
    render::Options sysmem;
    sysmem.gmem = false;
    const render::BlitRect all{0, 0, W, H};
    auto submit = [&](CommandBuffer& cb, const char* what) {
        cb.commit();
        cb.waitUntilCompleted();
        if (!completed(cb)) r.why = std::string(what) + ": " + cb.error();
        return completed(cb);
    };
    // sysmem: a Clear, a draw, a blended draw; then a fetch in a pass of its own
    {
        RenderPass pass;
        pass.color0 = t.get();
        pass.load = LoadAction::kClear;
        pass.clearColor[0] = 0.25, pass.clearColor[1] = 0.5, pass.clearColor[2] = 0.75, pass.clearColor[3] = 0.6;
        auto cb = queue.commandBuffer();
        bool ok = cb->draw(p.color, pass, drawOf(6, W / 10, H / 10, W * 7 / 10, H * 6 / 10), Bindings{}, Bindings{}, sysmem);
        RenderPass more;
        more.color0 = t.get();
        ok = ok && cb->draw(p.over, more, drawOf(6, W / 7, H / 5, W / 2, H / 3), Bindings{}, cb0, sysmem);
        if (!ok) { r.why = "sysmem draws: " + cb->error(); return r; }
        if (!submit(*cb, "sysmem draws")) return r;
        auto cf = queue.commandBuffer();
        if (!cf->draw(p.fb, more, drawOf(6, W / 3, H / 2, W / 2, H / 3), Bindings{}, Bindings{}, sysmem)) { r.why = "fetch: " + cf->error(); return r; }
        if (!submit(*cf, "fetch")) return r;
    }
    // read out: a 2D copy, and sampled 1:1 by a draw into a linear target
    {
        auto cb = queue.commandBuffer();
        if (!cb->blit(*t, 0, all, *lin[0], 0, all, false)) { r.why = "copy out: " + cb->error(); return r; }
        RenderPass sp;
        sp.color0 = lin[1].get();
        if (!cb->draw(p.tex, sp, drawOf(6, 0, 0, W, H), Bindings{}, tex, sysmem)) { r.why = "sample: " + cb->error(); return r; }
        if (!submit(*cb, "read out")) return r;
        r.out[0] = pixelsOf(*lin[0]);
        r.out[1] = pixelsOf(*lin[1]);
    }
    // GMEM: the target loaded, two blended draws across it, stored; then read out
    {
        RenderPass gp;
        gp.color0 = t.get();
        auto cb = queue.commandBuffer();
        cb->setProfile(true);
        cb->setGmemCompressed(gmemUbwc);
        bool ok = cb->draw(p.over, gp, drawOf(6, 0, H / 4, W, H / 8), Bindings{}, cb0) && cb->draw(p.over, gp, drawOf(6, W / 4, 0, W / 8, H), Bindings{}, cb0);
        if (!ok) { r.why = "GMEM draws: " + cb->error(); return r; }
        if (!submit(*cb, "GMEM load and store")) return r;
        r.gmemTiles = tilesOf(*cb);
        auto cc = queue.commandBuffer();
        if (!cc->blit(*t, 0, all, *lin[2], 0, all, false) || !submit(*cc, "copy out after GMEM")) return r;
        r.out[2] = pixelsOf(*lin[2]);
    }
    // GMEM: a Clear in the tiles, a draw, stored (nothing loaded); read out
    {
        RenderPass gp;
        gp.color0 = t.get();
        gp.load = LoadAction::kClear;
        gp.clearColor[0] = 0.1, gp.clearColor[1] = 0.2, gp.clearColor[2] = 0.3, gp.clearColor[3] = 0.4;
        auto cb = queue.commandBuffer();
        cb->setProfile(true);
        cb->setGmemCompressed(gmemUbwc);
        bool ok = cb->draw(p.color, gp, drawOf(6, W / 5, H / 5, W / 2, H / 2), Bindings{}, Bindings{}) &&
                  cb->draw(p.over, gp, drawOf(6, W / 3, H / 3, W / 3, H / 3), Bindings{}, cb0);
        if (!ok) { r.why = "GMEM clear: " + cb->error(); return r; }
        if (!submit(*cb, "GMEM clear and store")) return r;
        r.gmemTiles += tilesOf(*cb);
        auto cc = queue.commandBuffer();
        if (!cc->blit(*t, 0, all, *lin[3], 0, all, false) || !submit(*cc, "copy out after the GMEM clear")) return r;
        r.out[3] = pixelsOf(*lin[3]);
    }
    // a random picture into the layout by the 2D engine and back out
    {
        fillPixels(*lin[4], random);
        auto cb = queue.commandBuffer();
        if (!cb->blit(*lin[4], 0, all, *t, 0, all, false) || !submit(*cb, "copy in")) return r;
        memset(lin[4]->contents(), 0x5a, lin[4]->buffer()->length());
        auto cc = queue.commandBuffer();
        if (!cc->blit(*t, 0, all, *lin[4], 0, all, false) || !submit(*cc, "copy back out")) return r;
        r.out[4] = pixelsOf(*lin[4]);
    }
    return r;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v.size() % 2 ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
}

}  // namespace

int main(int argc, char** argv) {
    int reps = 7;
    bool times = true, ubwc = true;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = std::max(1, atoi(argv[++i]));
        if (strcmp(argv[i], "--no-time") == 0) times = false;
        if (strcmp(argv[i], "--no-ubwc") == 0) ubwc = false;
    }
    std::string why;
    auto transport = openKernelGpu(&why);
    if (transport == nullptr) {
        std::printf("tilecheck: no GPU: %s\n", why.c_str());
        return 100;
    }
    Gpu gpu(std::move(transport));
    uint64_t chip = 0;
    gpu.param(kParamChipId, &chip);
    uint64_t hbb = 0;
    const int hbbErr = gpu.param(kParamHighestBankBit, &hbb);
    std::printf("tilecheck: GPU chip id 0x%llx, FAULTS %llu; the kernel answers HIGHEST_BANK_BIT %s\n", (unsigned long long)chip,
                (unsigned long long)gpu.faults(), hbbErr == 0 ? std::to_string(hbb).c_str() : "(no answer)");
    auto queue = CommandQueue::make(gpu, &why);
    if (queue == nullptr) {
        report(false, "a queue: " + why);
        return 1;
    }
    readRegisters(gpu);
    if (!hasCompiler()) {
        report(false, "this build has no compiler: the shaders cannot be compiled here");
        return failures;
    }
    const uint64_t faults0 = gpu.faults();
    for (const PixelFormat f : {PixelFormat::kBGRA8Unorm, PixelFormat::kRGBA8Unorm}) {
        const char* fname = f == PixelFormat::kBGRA8Unorm ? "BGRA8" : "RGBA8";
        std::string cwhy;
        std::shared_ptr<CompiledStage> cvs = compileStage(kernels::tri_color_vs_bc, kernels::tri_color_vs_bc_size, &cwhy);
        std::shared_ptr<CompiledStage> cfs = compileStage(kernels::tri_color_fs_bc, kernels::tri_color_fs_bc_size, &cwhy);
        std::shared_ptr<CompiledStage> tfs = compileStage(kernels::tri_tex_fs_bc, kernels::tri_tex_fs_bc_size, &cwhy);
        std::shared_ptr<CompiledStage> fvs = compileStage(kernels::tri_flat_vs_bc, kernels::tri_flat_vs_bc_size, &cwhy);
        std::shared_ptr<CompiledStage> ofs = compileStage(kernels::tri_cflat_fs_bc, kernels::tri_cflat_fs_bc_size, &cwhy);
        std::shared_ptr<CompiledStage> bfs = compileStage(kernels::tri_fb_fs_bc, kernels::tri_fb_fs_bc_size, &cwhy);
        auto color = cvs && cfs ? linkProgram(cvs, cfs, f, &cwhy) : nullptr;
        auto texp = cvs && tfs ? linkProgram(cvs, tfs, f, &cwhy) : nullptr;
        auto over = fvs && ofs ? linkProgram(fvs, ofs, f, &cwhy) : nullptr;
        auto fb = fvs && bfs ? linkProgram(fvs, bfs, f, &cwhy) : nullptr;
        if (!color || !texp || !over || !fb) {
            report(false, std::string("the shaders, compiled here for ") + fname + ": " + cwhy);
            continue;
        }
        Programs p{color->program, texp->program, over->program, fb->program};
        p.over.blend = ColorBlend{true, 0, 0, 4, 5, 1, 5, 0xf};   // source over: SrcAlpha, OneMinusSrcAlpha; One, OneMinusSrcAlpha
        struct Size {
            uint32_t w, h;
        };
        for (const Size s : {Size{752, 1626}, Size{99, 209}, Size{270, 12}}) {
            std::vector<uint8_t> random(uint64_t(s.w) * s.h * 4);
            std::mt19937 rng(s.w * 131 + s.h);
            for (auto& b : random) b = uint8_t(rng());
            const Run ref = runSequence(gpu, *queue, p, kLinear, f, s.w, s.h, random);
            const std::string at = std::string(fname) + " " + std::to_string(s.w) + "x" + std::to_string(s.h);
            if (ref.out[4].empty()) {
                report(false, at + ": the linear reference did not complete: " + ref.why);
                continue;
            }
            uint64_t n = 0;
            differ(ref.out[4], random, s.w, &n);
            report(n == 0, at + " linear: a random picture copied in and out unchanged (" + std::to_string(n) + " pixels differ)");
            // (UBWC twice: as the driver runs it by default, its GMEM passes in sysmem since render::gmemPlan refuses a
            // compressed target, and with setGmemCompressed -- metal_gmemubwc=1 -- in as many tiles as the linear one)
            struct Case {
                Layout l;
                bool gmemUbwc;
                const char* name;
            };
            for (const Case cs : {Case{kTiled, false, "tiled"}, Case{kUbwc, false, "ubwc"}, Case{kUbwc, true, "ubwc+gmemubwc"}}) {
                const Layout l = cs.l;
                if (l == kUbwc && !ubwc) continue;
                const Run got = runSequence(gpu, *queue, p, l, f, s.w, s.h, random, cs.gmemUbwc);
                const std::string who = at + " " + cs.name;
                if (got.out[4].empty()) {
                    report(false, who + ": did not complete: " + got.why + " (FAULTS " + std::to_string(gpu.faults()) + ")");
                    continue;
                }
                static const char* const kStage[5] = {"sysmem clear+draw+blend+fetch, copied out", "...sampled 1:1 into a linear target",
                                                      "GMEM load+blend+store, copied out", "GMEM clear+draw+store, copied out",
                                                      "a random picture copied in and out"};
                for (int k = 0; k < 5; k++) {
                    const std::string first = differ(k == 4 ? random : ref.out[k], got.out[k], s.w, &n);
                    report(n == 0, who + ": " + kStage[k] + ": " + std::to_string(n) + " pixels differ from " + (k == 4 ? "the picture" : "linear") + first);
                }
                if (l == kUbwc && !cs.gmemUbwc) {
                    report(got.gmemTiles == 0, who + ": the GMEM passes ran in sysmem as the driver runs a compressed target by default (" +
                                                   std::to_string(got.gmemTiles) + " tiles)");
                } else {
                    report(got.gmemTiles == ref.gmemTiles && (got.gmemTiles > 0 || ref.gmemTiles == 0),
                           who + ": the GMEM passes ran in " + std::to_string(got.gmemTiles) + " tiles (linear " + std::to_string(ref.gmemTiles) + ")");
                }
            }
        }
        // costs, BGRA8 alone (what CoreAnimation draws)
        if (!times || f != PixelFormat::kBGRA8Unorm) continue;
        const uint32_t W = 752, H = 1626;
        render::Options sysmem;
        sysmem.gmem = false;
        auto cbuf = newBuffer(gpu, 16, &why);
        auto* c = static_cast<float*>(cbuf->contents());
        c[0] = 0.8f, c[1] = 0.4f, c[2] = 0.2f, c[3] = 0.5f;
        Bindings cb0;
        cb0.buffers = {Binding{cbuf.get(), 0}};
        Sampler nearest, bilinear;
        makeSampler(SamplerDesc{}, &nearest, &why);
        SamplerDesc bd;
        bd.minFilter = bd.magFilter = 1;
        makeSampler(bd, &bilinear, &why);
        auto out1 = makeTexture(gpu, kLinear, f, W, H, &why), out4 = makeTexture(gpu, kLinear, f, W / 4, H / 4, &why);
        for (const Layout l : {kLinear, kTiled, kUbwc}) {
            if (l == kUbwc && !ubwc) continue;
            auto t = makeTexture(gpu, l, f, W, H, &why);
            auto pool = makeTexture(gpu, l, f, 756, 1630, &why);
            if (t == nullptr || pool == nullptr || out1 == nullptr || out4 == nullptr) {
                report(false, std::string("costs: the textures for ") + kLayoutName[l] + ": " + why);
                continue;
            }
            Bindings tex1, tex4;
            tex1.textures = tex4.textures = {t.get()};
            tex1.samplers = {&nearest};
            tex4.samplers = {&bilinear};
            struct Op {
                const char* name;
                std::function<bool(CommandBuffer&)> record;
            };
            const render::BlitRect all{0, 0, W, H};
            RenderPass onT;
            onT.color0 = t.get();
            const std::vector<Op> ops = {
                {"clear 756x1630 (2D fill)", [&](CommandBuffer& cb) {
                     RenderPass pc;
                     pc.color0 = pool.get();
                     pc.load = LoadAction::kClear;
                     return cb.clear(pc, sysmem);
                 }},
                {"draw 752x1626 (write)", [&](CommandBuffer& cb) { return cb.draw(p.color, onT, drawOf(6, 0, 0, W, H), Bindings{}, Bindings{}, sysmem); }},
                {"blend 752x1626 (read+write)", [&](CommandBuffer& cb) { return cb.draw(p.over, onT, drawOf(6, 0, 0, W, H), Bindings{}, cb0, sysmem); }},
                {"fetch 752x1626 (sysmem)", [&](CommandBuffer& cb) { return cb.draw(p.fb, onT, drawOf(6, 0, 0, W, H), Bindings{}, Bindings{}, sysmem); }},
                {"sample 1:1 into linear 752x1626", [&](CommandBuffer& cb) {
                     RenderPass sp;
                     sp.color0 = out1.get();
                     return cb.draw(p.tex, sp, drawOf(6, 0, 0, W, H), Bindings{}, tex1, sysmem);
                 }},
                {"sample 1/4 bilinear into linear 188x406", [&](CommandBuffer& cb) {
                     RenderPass sp;
                     sp.color0 = out4.get();
                     return cb.draw(p.tex, sp, drawOf(6, 0, 0, W / 4, H / 4), Bindings{}, tex4, sysmem);
                 }},
                {"GMEM load+store 752x1626 (2 small draws)", [&](CommandBuffer& cb) {
                     // (the whole window tiled, whatever the two draws touch: every bin loaded and stored)
                     cb.setGmemOff(CommandBuffer::kGmemWholeArea);
                     cb.setGmemCompressed(true);   // (a compressed target too: metal_gmemubwc=1's path)
                     return cb.draw(p.over, onT, drawOf(6, 0, 0, 16, 16), Bindings{}, cb0) && cb.draw(p.over, onT, drawOf(6, W - 16, H - 16, 16, 16), Bindings{}, cb0);
                 }},
                {"copy out to linear (2D)", [&](CommandBuffer& cb) { return cb.blit(*t, 0, all, *out1, 0, all, false); }},
                {"copy in from linear (2D)", [&](CommandBuffer& cb) { return cb.blit(*out1, 0, all, *t, 0, all, false); }},
            };
            for (const Op& op : ops) {
                std::vector<double> us;
                uint32_t tiles = 0;
                std::string err;
                for (int k = 0; k < reps + 1; k++) {
                    auto cb = queue->commandBuffer();
                    cb->setProfile(true);
                    if (!op.record(*cb)) {
                        err = cb->error();
                        break;
                    }
                    cb->commit();
                    cb->waitUntilCompleted();
                    if (!completed(*cb)) {
                        err = cb->error();
                        break;
                    }
                    tiles = tilesOf(*cb);
                    if (k > 0) us.push_back(gpuUs(*cb));   // (the first run warms the caches and the stage cache)
                }
                if (!err.empty() || us.empty()) {
                    report(false, std::string("costs: ") + kLayoutName[l] + " " + op.name + ": " + err);
                    continue;
                }
                if (strstr(op.name, "GMEM") != nullptr && tiles == 0) {
                    report(false, std::string("costs: ") + kLayoutName[l] + " " + op.name + ": ran in sysmem, not in tiles");
                }
                std::printf("tilecheck: time %-6s %-42s median %8.1f us  min %8.1f  max %8.1f  n %zu%s\n", kLayoutName[l], op.name, median(us),
                            *std::min_element(us.begin(), us.end()), *std::max_element(us.begin(), us.end()), us.size(),
                            tiles ? (" (" + std::to_string(tiles) + " tiles)").c_str() : "");
            }
        }
    }
    report(gpu.faults() == faults0, "no GPU reset during the run (FAULTS " + std::to_string(faults0) + " -> " + std::to_string(gpu.faults()) + ")");
    std::printf("tilecheck: %d failure(s)\n", failures);
    return failures;
}
