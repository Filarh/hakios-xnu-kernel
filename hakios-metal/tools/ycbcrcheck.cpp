// hakios-metal-ycbcrcheck: whether this GPU samples a two-plane '420f'/'420v' IOSurface where it lies -- Metal's private
// format 540 (the planes as they are, (Cr, Y, Cb, 1)) and 500 (RGB, the matrix in the fragment stage that samples it) --
// exactly as the plug-in's copy converted on the processor gives it (HakiosMetalCA.m -hakiosConvertYCbCr, copied here as
// cpuConvert), and what each costs. It runs the driver's core on the phone's GPU from a process of its own (the
// kernel's msm client, as hakios-metal-probe and hakios-metal-tilecheck do: no Metal, no CoreAnimation, nothing of the
// window server touched), compiles its shaders there with the driver's compiler (tools/kernels/*.ll as bitcode: the
// YCbCr conversion is compileStage's, for the A610), and reads results back from LINEAR RGBA8 targets only.
//
//   1. layout: for each size, a surface with the planes IOSurface lays out by itself (its own pitches and offsets: are
//      they the 64 bytes the texture unit needs?) and one with both planes' rows and starts set to 64 bytes
//   2. correctness, per size (64x64, 751x1625, 750x1624 -- the app snapshot's --, 1920x1080), FourCC ('420f', '420v')
//      and picture (ramps of Y, Cb and Cr, random, both planes zero, chroma 0, chroma 255, the probe's 20/60/a0):
//      sampled 1:1 with a nearest sampler by tri_tex into a linear target and compared with the processor's conversion:
//      540 byte for byte (0 bytes differ), 500 within one step of 8 bits in every channel at fp32 (tri_tex) and half
//      (tri_htex: CoreAnimation's _lph functions sample as half), BT.601 and BT.709 (and BT.2020 at 64x64)
//   3. coherence (19 N-05): the planes written by the processor after the draw is encoded and before it is committed:
//      the result is the new picture
//   4. filtering, said and not judged: a picture whose chroma alternates by column, magnified 2x with a linear sampler:
//      whether the texture unit interpolates the chroma (CHROMA_LINEAR has no effect before a6xx_gen3), cosited and
//      with the siting midpoint
//   5. costs (GPU time by the CP's always-on counter, the median of N runs) at 750x1624: 500 sampled 1:1 from the
//      surface; the same picture as an RGBA8 texture, linear and UBWC, sampled 1:1 (the copy's read today), and the
//      UBWC copy's 2D copy-in after a replace; and the processor's conversion of the plug-in, timed in this process
//
// One line per check, "ycbcrcheck: ok ..." or "ycbcrcheck: FAIL ...", "ycbcrcheck: info ..." and "ycbcrcheck: time ...",
// and a last line with the count; the exit status is the number of failures (100: no GPU).
//
//   hakios-metal-ycbcrcheck [--reps N] [--no-time] [--quick]   (--quick: 64x64 and 750x1624 alone)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurfaceRef.h>

#include "hakios-metal/command.hpp"
#include "hakios-metal/compiler.hpp"
#include "hakios-metal/texture.hpp"
#include "../tests/fixtures/tri_color_vs_bc.hpp"
#include "../tests/fixtures/tri_htex_fs_bc.hpp"
#include "../tests/fixtures/tri_tex_fs_bc.hpp"

using namespace hakios::metal;

namespace {

int failures = 0;

void report(bool ok, const std::string& what) {
    std::printf("ycbcrcheck: %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    std::fflush(stdout);
    failures += !ok;
}

void info(const std::string& what) {
    std::printf("ycbcrcheck: info %s\n", what.c_str());
    std::fflush(stdout);
}

constexpr uint32_t k420f = 0x34323066u, k420v = 0x34323076u;

// --- the surfaces -----------------------------------------------------------------------------------------------------

CFNumberRef number(int v) { return CFNumberCreate(nullptr, kCFNumberIntType, &v); }
void put(CFMutableDictionaryRef d, CFStringRef key, int v) {
    CFNumberRef n = number(v);
    CFDictionarySetValue(d, key, n);
    CFRelease(n);
}

// a W x H two-plane surface of FOURCC; ALIGNED: both planes' rows and starts set to multiples of 64 bytes here, else
// what IOSurface lays out by itself from the planes' sizes
IOSurfaceRef makeSurface(uint32_t W, uint32_t H, uint32_t fourcc, bool aligned) {
    const uint32_t cw = (W + 1) / 2, ch = (H + 1) / 2;
    const uint32_t row0 = (W + 63) & ~63u, row1 = (2 * cw + 63) & ~63u;
    const uint32_t size0 = row0 * H, off1 = (size0 + 4095) & ~4095u, size1 = row1 * ch;
    CFMutableDictionaryRef props = CFDictionaryCreateMutable(nullptr, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    put(props, kIOSurfaceWidth, int(W));
    put(props, kIOSurfaceHeight, int(H));
    put(props, kIOSurfacePixelFormat, int(fourcc));
    if (aligned) put(props, kIOSurfaceAllocSize, int(off1 + size1));
    CFMutableArrayRef planes = CFArrayCreateMutable(nullptr, 2, &kCFTypeArrayCallBacks);
    for (int p = 0; p < 2; p++) {
        CFMutableDictionaryRef d = CFDictionaryCreateMutable(nullptr, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        put(d, kIOSurfacePlaneWidth, int(p == 0 ? W : cw));
        put(d, kIOSurfacePlaneHeight, int(p == 0 ? H : ch));
        put(d, kIOSurfacePlaneBytesPerElement, p == 0 ? 1 : 2);
        put(d, kIOSurfacePlaneElementWidth, 1);
        put(d, kIOSurfacePlaneElementHeight, 1);
        if (aligned) {
            put(d, kIOSurfacePlaneBytesPerRow, int(p == 0 ? row0 : row1));
            put(d, kIOSurfacePlaneOffset, int(p == 0 ? 0 : off1));
            put(d, kIOSurfacePlaneSize, int(p == 0 ? size0 : size1));
        }
        CFArrayAppendValue(planes, d);
        CFRelease(d);
    }
    CFDictionarySetValue(props, kIOSurfacePlaneInfo, planes);
    CFRelease(planes);
    IOSurfaceRef s = IOSurfaceCreate(props);
    CFRelease(props);
    if (s != nullptr && IOSurfaceGetPlaneCount(s) != 2) {
        CFRelease(s);
        return nullptr;
    }
    return s;
}

struct Layout {
    uint64_t chromaOffset;
    size_t row0, row1;
    uintptr_t offset0;
};
Layout layoutOf(IOSurfaceRef s) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, nullptr);
    const auto base = uintptr_t(IOSurfaceGetBaseAddress(s));
    Layout l{uint64_t(uintptr_t(IOSurfaceGetBaseAddressOfPlane(s, 1)) - base), IOSurfaceGetBytesPerRowOfPlane(s, 0), IOSurfaceGetBytesPerRowOfPlane(s, 1),
             uintptr_t(IOSurfaceGetBaseAddressOfPlane(s, 0)) - base};
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, nullptr);
    return l;
}

enum Picture { kRamps, kRandom, kZero, kChroma0, kChroma255, kProbe, kStripes, kPictures };
const char* const kPictureName[kPictures] = {"ramps", "random", "zero planes", "chroma 0", "chroma 255", "20/60/a0", "chroma stripes"};

void fill(IOSurfaceRef s, Picture p, uint32_t seed) {
    IOSurfaceLock(s, 0, nullptr);
    const uint32_t W = uint32_t(IOSurfaceGetWidthOfPlane(s, 0)), H = uint32_t(IOSurfaceGetHeightOfPlane(s, 0));
    const uint32_t cw = uint32_t(IOSurfaceGetWidthOfPlane(s, 1)), ch = uint32_t(IOSurfaceGetHeightOfPlane(s, 1));
    auto* y = static_cast<uint8_t*>(IOSurfaceGetBaseAddressOfPlane(s, 0));
    auto* c = static_cast<uint8_t*>(IOSurfaceGetBaseAddressOfPlane(s, 1));
    const size_t r0 = IOSurfaceGetBytesPerRowOfPlane(s, 0), r1 = IOSurfaceGetBytesPerRowOfPlane(s, 1);
    std::mt19937 rng(seed);
    for (uint32_t j = 0; j < H; j++) {
        for (uint32_t i = 0; i < W; i++) {
            uint8_t v = 0;
            switch (p) {
            case kRamps: v = uint8_t((i + 3 * j) & 255); break;
            case kRandom: v = uint8_t(rng()); break;
            case kZero: v = 0; break;
            case kChroma0:
            case kChroma255: v = uint8_t((i * 255) / std::max(W - 1, 1u)); break;
            case kProbe: v = 0x20; break;
            case kStripes: v = 128; break;
            default: break;
            }
            y[j * r0 + i] = v;
        }
    }
    for (uint32_t j = 0; j < ch; j++) {
        for (uint32_t i = 0; i < cw; i++) {
            uint8_t cb = 0, cr = 0;
            switch (p) {
            case kRamps: cb = uint8_t((i * 255) / std::max(cw - 1, 1u)), cr = uint8_t((j * 255) / std::max(ch - 1, 1u)); break;
            case kRandom: cb = uint8_t(rng()), cr = uint8_t(rng()); break;
            case kZero: break;
            case kChroma0: break;
            case kChroma255: cb = cr = 255; break;
            case kProbe: cb = 0x60, cr = 0xa0; break;
            case kStripes: cb = (i & 1) ? 255 : 0, cr = 128; break;
            default: break;
            }
            c[j * r1 + 2 * i] = cb;
            c[j * r1 + 2 * i + 1] = cr;
        }
    }
    IOSurfaceUnlock(s, 0, nullptr);
}

// --- the processor's conversion: HakiosMetalCA.m -hakiosConvertYCbCr, as a function of the surface -----------------------
// MODEL 1 BT.601, 2 BT.709, 3 BT.2020 (the plug-in's _ycbcr); FULL: '420f'; RAW: format 540, the planes as they are.
// W x H RGBA8 pixels, row after row.
uint8_t hm_clamp8(float v) { return v <= 0 ? 0 : v >= 255 ? 255 : (uint8_t)(v + 0.5f); }

std::vector<uint8_t> cpuConvert(IOSurfaceRef s, uint32_t w, uint32_t h, int model, bool full, bool raw) {
    std::vector<uint8_t> rows(size_t(w) * h * 4);
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, nullptr);
    const uint8_t *luma = static_cast<const uint8_t*>(IOSurfaceGetBaseAddressOfPlane(s, 0)), *chroma = static_cast<const uint8_t*>(IOSurfaceGetBaseAddressOfPlane(s, 1));
    const size_t lumaRow = IOSurfaceGetBytesPerRowOfPlane(s, 0), chromaRow = IOSurfaceGetBytesPerRowOfPlane(s, 1);
    const size_t cw = IOSurfaceGetWidthOfPlane(s, 1), ch = IOSurfaceGetHeightOfPlane(s, 1);
    const float kr = model == 2 ? 0.2126f : model == 3 ? 0.2627f : 0.299f, kb = model == 2 ? 0.0722f : model == 3 ? 0.0593f : 0.114f;
    const float kg = 1.0f - kr - kb;
    const float ys = full ? 1.0f : 255.0f / 219.0f, yo = full ? 0.0f : 16.0f, cs = full ? 1.0f : 255.0f / 224.0f;
    for (size_t j = 0; j < h; j++) {
        const uint8_t *yl = luma + j * lumaRow, *cl = chroma + std::min(j / 2, ch - 1) * chromaRow;
        uint8_t* out = rows.data() + j * w * 4;
        for (size_t i = 0; i < w; i++) {
            const size_t c = std::min(i / 2, cw - 1) * 2;
            if (raw) {
                out[4 * i] = cl[c + 1];
                out[4 * i + 1] = yl[i];
                out[4 * i + 2] = cl[c];
                out[4 * i + 3] = 255;
                continue;
            }
            const float y = (yl[i] - yo) * ys, cb = (cl[c] - 128.0f) * cs, cr = (cl[c + 1] - 128.0f) * cs;
            const float r = y + 2.0f * (1.0f - kr) * cr, b = y + 2.0f * (1.0f - kb) * cb;
            const float g = (y - kr * r - kb * b) / kg;
            out[4 * i] = hm_clamp8(r);
            out[4 * i + 1] = hm_clamp8(g);
            out[4 * i + 2] = hm_clamp8(b);
            out[4 * i + 3] = 255;
        }
    }
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, nullptr);
    return rows;
}

// --- the GPU's side ----------------------------------------------------------------------------------------------------

Draw drawOf(uint32_t count, double x, double y, double w, double h) {
    Draw d;
    d.vertexStart = 0;
    d.vertexCount = count;
    d.viewport = Viewport{x, y, w, h, 0, 1};
    return d;
}

std::shared_ptr<Texture> linearTarget(Gpu& gpu, uint32_t w, uint32_t h, std::string* why) {
    TextureDesc d;
    d.format = PixelFormat::kRGBA8Unorm;
    d.width = w;
    d.height = h;
    d.usage = kTextureUsageRead | kTextureUsageRenderTarget;
    return Texture::make(gpu, d, why);
}

std::vector<uint8_t> pixelsOf(const Texture& t) {
    std::vector<uint8_t> out(uint64_t(t.width()) * t.height() * 4);
    const auto* src = static_cast<const uint8_t*>(t.contents());
    for (uint32_t y = 0; y < t.height(); y++) memcpy(out.data() + uint64_t(y) * t.width() * 4, src + y * t.bytesPerRow(), uint64_t(t.width()) * 4);
    return out;
}

// the surface as the plug-in makes it under metal_ycbcrgpu=1 (hakios_metal_texture_new_ycbcr): both planes as one
// R8_G8B8_2PLANE_420_UNORM texture, swizzled to (Cr, Y, Cb, 1)
std::shared_ptr<Texture> ycbcrTexture(Gpu& gpu, IOSurfaceRef s, uint8_t siting, std::string* why) {
    const Layout l = layoutOf(s);
    Texture::SurfacePlane chroma{l.chromaOffset, uint32_t(l.row1)};
    chroma.midpoint = siting;
    TextureDesc d;
    d.format = PixelFormat::kYCbCr8_420_2P;
    d.width = uint32_t(IOSurfaceGetWidth(s));
    d.height = uint32_t(IOSurfaceGetHeight(s));
    d.usage = kTextureUsageRead;
    auto planes = Texture::wrapSurface(gpu, IOSurfaceGetID(s), d, why, &chroma);
    const uint8_t crYCb1[4] = {4, 2, 3, 1};
    return planes != nullptr ? Texture::swizzled(planes, crYCb1, why) : nullptr;
}

struct Programs {
    std::shared_ptr<CompiledStage> vs;
    std::map<std::pair<int, uint64_t>, std::shared_ptr<CompiledProgram>> made;   // by (half, YCbCr word)
    // tri_tex (HALF: tri_htex) compiled with WORD for slot 0, linked for RGBA8
    const Program* get(bool half, uint64_t word, std::string* why) {
        auto key = std::make_pair(half ? 1 : 0, word);
        auto it = made.find(key);
        if (it != made.end()) return it->second ? &it->second->program : nullptr;
        std::shared_ptr<CompiledStage> fs(half ? compileStage(kernels::tri_htex_fs_bc, kernels::tri_htex_fs_bc_size, why, {}, 0, nullptr, word)
                                               : compileStage(kernels::tri_tex_fs_bc, kernels::tri_tex_fs_bc_size, why, {}, 0, nullptr, word));
        std::shared_ptr<CompiledProgram> p = fs != nullptr ? std::shared_ptr<CompiledProgram>(linkProgram(vs, fs, PixelFormat::kRGBA8Unorm, why)) : nullptr;
        made[key] = p;
        return p ? &p->program : nullptr;
    }
};

double gpuUs(const CommandBuffer& c) {
    double us = 0;
    for (const CommandBuffer::PassTime& pt : c.passTimes()) us += pt.us;
    return us;
}

// SRC sampled by P into a fresh linear OUT_W x OUT_H target; BETWEEN runs after the draw is encoded and before the commit
bool sample(Gpu& gpu, CommandQueue& queue, const Program& p, Texture& src, const Sampler& smp, uint32_t outW, uint32_t outH, std::vector<uint8_t>* px,
            std::string* why, const std::function<void()>& between = {}, double* us = nullptr) {
    auto out = linearTarget(gpu, outW, outH, why);
    if (out == nullptr) return false;
    memset(out->contents(), 0x5a, out->buffer()->length());
    RenderPass pass;
    pass.color0 = out.get();
    Bindings fsb;
    fsb.textures = {&src};
    fsb.samplers = {&smp};
    render::Options sysmem;
    sysmem.gmem = false;
    auto cb = queue.commandBuffer();
    cb->setProfile(us != nullptr);
    if (!cb->draw(p, pass, drawOf(6, 0, 0, outW, outH), Bindings{}, fsb, sysmem)) {
        *why = "the draw: " + cb->error();
        return false;
    }
    if (between) between();
    cb->commit();
    cb->waitUntilCompleted();
    if (cb->status() != CommandBuffer::kCompleted) {
        *why = "the command buffer: " + cb->error();
        return false;
    }
    if (us != nullptr) *us = gpuUs(*cb);
    if (px != nullptr) *px = pixelsOf(*out);
    return true;
}

// how far GOT is from WANT, channel by channel: the largest difference, how many channels differ by 1 and by more, and
// the first pixel past TOLERANCE
struct Diff {
    uint32_t worst = 0;
    uint64_t ones = 0, more = 0, bytes = 0;
    std::string first;
};
Diff compare(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want, uint32_t w, uint32_t tolerance) {
    Diff d;
    if (got.size() != want.size()) {
        d.worst = 255;
        d.first = " (different sizes)";
        return d;
    }
    for (size_t k = 0; k < got.size(); k++) {
        const uint32_t e = uint32_t(std::abs(int(got[k]) - int(want[k])));
        d.worst = std::max(d.worst, e);
        d.bytes += e != 0;
        d.ones += e == 1;
        d.more += e > 1;
        if (e > tolerance && d.first.empty()) {
            const size_t p = k / 4;
            char t[160];
            std::snprintf(t, sizeof t, " (first at %zu,%zu: %02x%02x%02x%02x against %02x%02x%02x%02x)", p % w, p / w, got[4 * p], got[4 * p + 1], got[4 * p + 2],
                          got[4 * p + 3], want[4 * p], want[4 * p + 1], want[4 * p + 2], want[4 * p + 3]);
            d.first = t;
        }
    }
    return d;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v.size() % 2 ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
}

const char* modelName(int model) { return model == 1 ? "BT.601" : model == 2 ? "BT.709" : "BT.2020"; }

}  // namespace

int main(int argc, char** argv) {
    int reps = 7;
    bool times = true, quick = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = std::max(1, atoi(argv[++i]));
        if (strcmp(argv[i], "--no-time") == 0) times = false;
        if (strcmp(argv[i], "--quick") == 0) quick = true;
    }
    std::string why;
    auto transport = openKernelGpu(&why);
    if (transport == nullptr) {
        std::printf("ycbcrcheck: no GPU: %s\n", why.c_str());
        return 100;
    }
    Gpu gpu(std::move(transport));
    uint64_t chip = 0;
    gpu.param(kParamChipId, &chip);
    std::printf("ycbcrcheck: GPU chip id 0x%llx, FAULTS %llu\n", (unsigned long long)chip, (unsigned long long)gpu.faults());
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
    Programs progs;
    progs.vs = std::shared_ptr<CompiledStage>(compileStage(kernels::tri_color_vs_bc, kernels::tri_color_vs_bc_size, &why));
    if (progs.vs == nullptr) {
        report(false, "tri_color_vs, compiled here: " + why);
        return failures;
    }
    Sampler nearest, bilinear;
    makeSampler(SamplerDesc{}, &nearest, &why);
    SamplerDesc bd;
    bd.minFilter = bd.magFilter = 1;
    makeSampler(bd, &bilinear, &why);

    struct Size {
        uint32_t w, h;
    };
    std::vector<Size> sizes = {{64, 64}, {751, 1625}, {750, 1624}, {1920, 1080}};
    if (quick) sizes = {{64, 64}, {750, 1624}};

    // 1 and 2: the layouts, and every picture through 540 and 500
    for (const Size sz : sizes) {
        for (const uint32_t fourcc : {k420f, k420v}) {
            const bool full = fourcc == k420f;
            for (const bool aligned : {false, true}) {
                const std::string at = std::string(full ? "'420f' " : "'420v' ") + std::to_string(sz.w) + "x" + std::to_string(sz.h) +
                                       (aligned ? " (64-byte planes)" : " (IOSurface's own layout)");
                IOSurfaceRef s = makeSurface(sz.w, sz.h, fourcc, aligned);
                if (s == nullptr) {
                    if (aligned) report(false, at + ": IOSurfaceCreate gave no two-plane surface");
                    else info(at + ": IOSurfaceCreate gave no two-plane surface without the rows and offsets said");
                    continue;
                }
                const Layout l = layoutOf(s);
                char lay[200];
                std::snprintf(lay, sizeof lay, "plane 0 at +%lu, %zu bytes a row; plane 1 at +%llu, %zu bytes a row; %zu bytes", (unsigned long)l.offset0, l.row0,
                              (unsigned long long)l.chromaOffset, l.row1, IOSurfaceGetAllocSize(s));
                const bool fits = l.row0 % 64 == 0 && l.row1 % 64 == 0 && l.offset0 % 64 == 0 && l.chromaOffset % 64 == 0;
                info(at + ": " + lay + (fits ? ": the texture unit can read it where it lies" : ": NOT 64-byte rows and starts: read through the copy"));
                std::string twhy;
                auto t = ycbcrTexture(gpu, s, 0, &twhy);
                if (!fits) {
                    report(t == nullptr, at + ": the core refuses a surface it cannot read in place" + (t == nullptr ? " (" + twhy + ")" : std::string(" -- it took it")));
                    CFRelease(s);
                    continue;
                }
                if (t == nullptr) {
                    report(false, at + ": the YCbCr texture over it: " + twhy);
                    CFRelease(s);
                    continue;
                }
                for (int pi = 0; pi < kPictures; pi++) {
                    const Picture pic = Picture(pi);
                    if (pic == kStripes) continue;
                    // (the large sizes take three pictures: the costs of the conversion on the processor add up)
                    if (sz.w > 64 && pic != kRamps && pic != kRandom && pic != kZero) continue;
                    fill(s, pic, sz.w * 7 + sz.h + pi);
                    const std::string what = at + ", " + kPictureName[pi];
                    // 540: the planes as they are
                    {
                        std::vector<uint8_t> got;
                        std::string swhy;
                        const Program* p = progs.get(false, 0, &swhy);
                        if (p == nullptr || !sample(gpu, *queue, *p, *t, nearest, sz.w, sz.h, &got, &swhy)) {
                            report(false, what + ", 540: " + swhy + " (FAULTS " + std::to_string(gpu.faults()) + ")");
                        } else {
                            const Diff d = compare(got, cpuConvert(s, sz.w, sz.h, 1, full, true), sz.w, 0);
                            report(d.bytes == 0, what + ", 540: " + std::to_string(d.bytes) + " bytes differ from the planes" + d.first);
                        }
                    }
                    // 500: the stage's matrix, fp32 and half
                    for (int model = 1; model <= 3; model++) {
                        if (model == 3 && sz.w > 64) continue;
                        for (const bool half : {false, true}) {
                            if (half && model != 2 && sz.w > 64) continue;
                            std::vector<uint8_t> got;
                            std::string swhy;
                            const uint64_t word = ycbcrCode(uint32_t(model - 1), !full);
                            const Program* p = progs.get(half, word, &swhy);
                            if (p == nullptr || !sample(gpu, *queue, *p, *t, nearest, sz.w, sz.h, &got, &swhy)) {
                                report(false, what + ", 500 " + modelName(model) + (half ? " half" : " fp32") + ": " + swhy);
                                continue;
                            }
                            const Diff d = compare(got, cpuConvert(s, sz.w, sz.h, model, full, false), sz.w, 1);
                            report(d.worst <= 1, what + ", 500 " + modelName(model) + (half ? " half" : " fp32") + ": at most " + std::to_string(d.worst) +
                                                     " apart; " + std::to_string(d.ones) + " channels 1 apart, " + std::to_string(d.more) + " more" + d.first);
                        }
                    }
                }
                t.reset();
                CFRelease(s);
            }
        }
    }

    // 3: the planes written after the draw is encoded and before the commit (N-05): the draw reads them as they are then
    {
        IOSurfaceRef s = makeSurface(750, 1624, k420f, true);
        std::string twhy;
        auto t = s != nullptr ? ycbcrTexture(gpu, s, 0, &twhy) : nullptr;
        if (t == nullptr) {
            report(false, "coherence: a 750x1624 '420f' surface and its texture: " + twhy);
        } else {
            for (const bool raw : {true, false}) {
                fill(s, kZero, 1);
                std::vector<uint8_t> got;
                std::string swhy;
                const Program* p = progs.get(false, raw ? 0 : ycbcrCode(1, false), &swhy);
                const bool ran = p != nullptr && sample(gpu, *queue, *p, *t, nearest, 750, 1624, &got, &swhy, [&] { fill(s, kRandom, 99); });
                if (!ran) {
                    report(false, std::string("coherence, ") + (raw ? "540" : "500") + ": " + swhy);
                    continue;
                }
                const Diff d = compare(got, cpuConvert(s, 750, 1624, 2, true, raw), 750, raw ? 0 : 1);
                report(raw ? d.bytes == 0 : d.worst <= 1, std::string("coherence, ") + (raw ? "540" : "500 BT.709") +
                                                              ": the planes written after the encode and before the commit are what the draw read (" +
                                                              std::to_string(d.bytes) + " bytes differ, at most " + std::to_string(d.worst) + ")" + d.first);
            }
        }
        t.reset();
        if (s != nullptr) CFRelease(s);
    }

    // 4: the chroma's filtering, said: columns of Cb 0 and 255, magnified 2x by a linear sampler, cosited and midpoint
    {
        IOSurfaceRef s = makeSurface(64, 64, k420f, true);
        if (s != nullptr) {
            fill(s, kStripes, 0);
            for (const uint8_t siting : {uint8_t(0), uint8_t(3)}) {
                std::string twhy;
                auto t = ycbcrTexture(gpu, s, siting, &twhy);
                const Program* p = progs.get(false, 0, &twhy);
                std::vector<uint8_t> got;
                if (t == nullptr || p == nullptr || !sample(gpu, *queue, *p, *t, bilinear, 128, 128, &got, &twhy)) {
                    report(false, "filtering: " + twhy);
                    continue;
                }
                // row 64, Cb (blue) across: only 0 and 255 if the chroma is taken nearest, values between if interpolated
                std::map<int, int> values;
                std::string row;
                for (uint32_t x = 0; x < 128; x++) {
                    values[got[(64 * 128 + x) * 4 + 2]]++;
                    if (x < 12) row += (x ? " " : "") + std::to_string(got[(64 * 128 + x) * 4 + 2]);
                }
                info(std::string("filtering, siting ") + (siting ? "midpoint" : "cosited") + ": Cb along a row magnified 2x by a linear sampler takes " +
                     std::to_string(values.size()) + " values (" + (values.size() > 2 ? "the chroma is interpolated" : "the chroma is taken nearest") +
                     "); first pixels: " + row);
            }
            CFRelease(s);
        }
    }

    // 5: the costs at 750x1624
    if (times) {
        const uint32_t W = 750, H = 1624;
        IOSurfaceRef s = makeSurface(W, H, k420f, true);
        std::string twhy;
        auto t = s != nullptr ? ycbcrTexture(gpu, s, 0, &twhy) : nullptr;
        TextureDesc ld;
        ld.format = PixelFormat::kRGBA8Unorm;
        ld.width = W;
        ld.height = H;
        ld.usage = kTextureUsageRead;
        auto lin = Texture::make(gpu, ld, &twhy);
        ld.compressed = true;
        auto ubwc = Texture::make(gpu, ld, &twhy);
        const Program* p500 = progs.get(false, ycbcrCode(1, false), &twhy);
        const Program* plain = progs.get(false, 0, &twhy);
        if (t == nullptr || lin == nullptr || ubwc == nullptr || p500 == nullptr || plain == nullptr) {
            report(false, "costs: the textures and programs: " + twhy);
        } else {
            fill(s, kRandom, 5);
            const std::vector<uint8_t> rgba = cpuConvert(s, W, H, 2, true, false);
            lin->replace(0, 0, W, H, rgba.data(), uint64_t(W) * 4, &twhy);
            ubwc->replace(0, 0, W, H, rgba.data(), uint64_t(W) * 4, &twhy);
            struct Op {
                const char* name;
                std::function<bool(double*)> run;
            };
            const std::vector<Op> ops = {
                {"500 sampled 1:1 from the surface (NV12, the matrix in the stage)", [&](double* us) { return sample(gpu, *queue, *p500, *t, nearest, W, H, nullptr, &twhy, {}, us); }},
                {"540 sampled 1:1 from the surface", [&](double* us) { return sample(gpu, *queue, *plain, *t, nearest, W, H, nullptr, &twhy, {}, us); }},
                {"RGBA8 linear copy sampled 1:1", [&](double* us) { return sample(gpu, *queue, *plain, *lin, nearest, W, H, nullptr, &twhy, {}, us); }},
                {"RGBA8 UBWC copy sampled 1:1", [&](double* us) { return sample(gpu, *queue, *plain, *ubwc, nearest, W, H, nullptr, &twhy, {}, us); }},
                {"RGBA8 UBWC copy: replace, 2D copy-in and sampled 1:1", [&](double* us) {
                     return ubwc->replace(0, 0, W, H, rgba.data(), uint64_t(W) * 4, &twhy) && sample(gpu, *queue, *plain, *ubwc, nearest, W, H, nullptr, &twhy, {}, us);
                 }},
            };
            for (const Op& op : ops) {
                std::vector<double> us;
                bool ok = true;
                for (int k = 0; k < reps + 1 && ok; k++) {
                    double u = 0;
                    ok = op.run(&u);
                    if (k > 0) us.push_back(u);   // (the first run warms the caches)
                }
                if (!ok) {
                    report(false, std::string("costs: ") + op.name + ": " + twhy);
                    continue;
                }
                std::printf("ycbcrcheck: time %-62s median %8.1f us  min %8.1f  max %8.1f  n %zu (GPU)\n", op.name, median(us), *std::min_element(us.begin(), us.end()),
                            *std::max_element(us.begin(), us.end()), us.size());
            }
            // the processor's side of today's path: the conversion and the replace into the UBWC copy's staging
            std::vector<double> conv, repl;
            for (int k = 0; k < reps; k++) {
                const auto a = std::chrono::steady_clock::now();
                const std::vector<uint8_t> px = cpuConvert(s, W, H, 2, true, false);
                const auto b = std::chrono::steady_clock::now();
                ubwc->replace(0, 0, W, H, px.data(), uint64_t(W) * 4, &twhy);
                const auto c = std::chrono::steady_clock::now();
                conv.push_back(std::chrono::duration<double, std::micro>(b - a).count());
                repl.push_back(std::chrono::duration<double, std::micro>(c - b).count());
            }
            std::printf("ycbcrcheck: time %-62s median %8.1f us  min %8.1f  max %8.1f  n %zu (CPU)\n", "the plug-in's conversion of 750x1624 on this processor",
                        median(conv), *std::min_element(conv.begin(), conv.end()), *std::max_element(conv.begin(), conv.end()), conv.size());
            std::printf("ycbcrcheck: time %-62s median %8.1f us  min %8.1f  max %8.1f  n %zu (CPU)\n", "...and its replace into the UBWC copy's staging", median(repl),
                        *std::min_element(repl.begin(), repl.end()), *std::max_element(repl.begin(), repl.end()), repl.size());
        }
        t.reset();
        if (s != nullptr) CFRelease(s);
    }
    report(gpu.faults() == faults0, "no GPU reset during the run (FAULTS " + std::to_string(faults0) + " -> " + std::to_string(gpu.faults()) + ")");
    std::printf("ycbcrcheck: %d failure(s)\n", failures);
    return failures;
}
