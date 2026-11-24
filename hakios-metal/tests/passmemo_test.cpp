// Render-pass memoization (passmemo.hpp) on the fake GPU: a pass repeated frame after frame, into a target that
// rotates as CoreAnimation's surfaces do, is run, kept, and then copied from what was kept; anything it reads that
// changes -- a byte of a bound buffer (also after encoding, before commit), a texture rewritten by the processor or the
// GPU, the clear colour, a draw's parameters -- runs it again. The fake rasterizes nothing but carries out clears and
// copies, so what a target holds after a frame is what the clear (or the copy standing in for the pass) put there:
// enough to hold a run with memoization against one without, word for word.
#include <cstdio>
#include <cstring>
#include <vector>

#include "hakios-metal/command.hpp"
#include "hakios-metal/fake_gpu.hpp"
#include "hakios-metal/render.hpp"
#include "hakios-metal/texture.hpp"
#include "fixtures/tri_cflat_fs_air.hpp"
#include "fixtures/tri_color_vs_air.hpp"
#include "fixtures/tri_flat_fs_air.hpp"
#include "fixtures/tri_flat_vs_air.hpp"
#include "fixtures/tri_tex_fs_air.hpp"
#include "test.hpp"

using namespace hakios::metal;
namespace kernels = hakios::metal::kernels;

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

std::shared_ptr<Texture> texture(Rig& r, uint32_t w, uint32_t h, uint32_t usage = kTextureUsageRenderTarget | kTextureUsageRead) {
    TextureDesc d;
    d.format = PixelFormat::kRGBA8Unorm;
    d.width = w;
    d.height = h;
    d.usage = usage;
    std::string why;
    auto t = Texture::make(*r.gpu, d, &why);
    CHECK(t != nullptr);
    return t;
}

void fillWords(const Texture& t, uint32_t value) {
    auto* w = static_cast<uint32_t*>(t.contents());
    for (uint64_t i = 0; i < t.buffer()->length() / 4; i++) w[i] = value;
}

std::vector<uint32_t> wordsOf(const Texture& t) {
    const auto* w = static_cast<const uint32_t*>(t.contents());
    return std::vector<uint32_t>(w, w + t.buffer()->length() / 4);
}

uint32_t pixel(const Texture& t, uint32_t x, uint32_t y) {
    const auto* row = static_cast<const uint8_t*>(t.contents()) + uint64_t(y) * t.bytesPerRow();
    uint32_t v;
    memcpy(&v, row + 4 * x, 4);
    return v;
}

// programs with a uid, as linkProgram gives them (memoization keys a program by it)
Program cflat() {
    Program p{kernels::tri_flat_vs_air, kernels::tri_cflat_fs_air, Linkage{}, "tri_cflat"};
    p.uid = 0x1001;
    return p;
}
Program flat() {
    Program p{kernels::tri_flat_vs_air, kernels::tri_flat_fs_air, Linkage{}, "tri_flat"};
    p.uid = 0x1002;
    return p;
}
Program tex() {
    Program p{kernels::tri_color_vs_air, kernels::tri_tex_fs_air, Linkage{}, "tri_tex"};
    p.uid = 0x1003;
    return p;
}

Draw whole(uint32_t vertexCount = 3, uint32_t w = 64, uint32_t h = 64) {
    Draw d;
    d.vertexCount = vertexCount;
    d.viewport = Viewport{0, 0, double(w), double(h), 0, 1};
    return d;
}

Sampler nearest() {
    Sampler s;
    std::string why;
    CHECK(makeSampler(SamplerDesc{}, &s, &why));
    return s;
}

// What one frame did on the fake: its draws run and its copies made
struct Did {
    uint32_t draws = 0, copies = 0;
    bool ok = false;
};

// one frame: a pass into T of DRAWS draws of P (its colour at [[buffer(0)]]), cleared to CLEAR; BEFORE_COMMIT runs
// between the recording and the commit
template <class F>
Did frame(Rig& r, const Texture& t, const Program& p, const Bindings& fsb, uint32_t draws, double clear, F beforeCommit, Draw d = whole()) {
    const FakeGpu::Stats before = r.fake->stats();
    RenderPass pass;
    pass.color0 = &t;
    pass.load = LoadAction::kClear;
    pass.clearColor[0] = clear;
    pass.clearColor[3] = 1;
    auto cb = r.queue->commandBuffer();
    bool ok = true;
    for (uint32_t i = 0; i < draws; i++) {
        ok = cb->draw(p, pass, d, Bindings{}, fsb) && ok;
        pass.load = LoadAction::kLoad;   // (the plug-in's way: the Clear spent, the pass's later draws load what it left)
    }
    beforeCommit();
    cb->commit();
    cb->waitUntilCompleted();
    const FakeGpu::Stats after = r.fake->stats();
    Did did;
    did.ok = ok && cb->status() == CommandBuffer::kCompleted && after.faults == before.faults;
    if (!did.ok) std::printf("       %s | %s\n", cb->error().c_str(), after.lastFault.c_str());
    did.draws = after.draws - before.draws;
    did.copies = after.copies - before.copies;
    return did;
}
Did frame(Rig& r, const Texture& t, const Program& p, const Bindings& fsb, uint32_t draws, double clear = 1.0) {
    return frame(r, t, p, fsb, draws, clear, [] {});
}

}  // namespace

// The premise: the same pass five frames running, into three surfaces in turn. The first is run, the second run and
// kept (a copy into the cache after it), the third to fifth are a copy each and no draw; every surface ends as a run
// without memoization leaves it, word for word.
TEST(a_repeated_cleared_pass_is_run_kept_and_then_copied) {
    std::vector<std::vector<uint32_t>> want;
    for (int memoOn = 0; memoOn < 2; memoOn++) {
        Rig r;
        r.queue->setMemo(memoOn ? memo::kStrict : memo::kOff, 2);
        std::shared_ptr<Texture> t[3] = {texture(r, 64, 64), texture(r, 64, 64), texture(r, 64, 64)};
        for (auto& x : t) fillWords(*x, 0xa5a5a5a5u);
        std::string why;
        auto colour = newBuffer(*r.gpu, 16, &why);
        const float c[4] = {0.25f, 0.5f, 0.75f, 1};
        memcpy(colour->contents(), c, 16);
        Bindings fsb;
        fsb.buffers = {Binding{colour.get(), 0}};
        Did did[5];
        for (int f = 0; f < 5; f++) {
            did[f] = frame(r, *t[f % 3], cflat(), fsb, 4);
            CHECK(did[f].ok);
        }
        if (memoOn == 0) {
            for (auto& x : t) want.push_back(wordsOf(*x));
            for (const Did& d : did) CHECK_EQ(d.draws, 4u);
            CHECK_EQ(r.queue->memoStats().candidates, 0u);
            continue;
        }
        const memo::Stats s = r.queue->memoStats();
        CHECK_EQ(s.candidates, 5u);
        CHECK_EQ(s.hits, 3u);
        CHECK_EQ(s.stores, 1u);
        CHECK_EQ(s.misses, 2u);
        CHECK_EQ(s.skippedDraws, 12u);
        CHECK_EQ(s.refused, 0u);
        CHECK(s.hashedBytes >= 5 * 16u && s.hashedBytes < 5 * 4 * 16u);   // (the one range all four draws bind, hashed once a pass)
        CHECK_EQ(s.cacheBytes, t[0]->buffer()->length());
        CHECK_EQ(did[0].draws, 4u);
        CHECK_EQ(did[0].copies, 0u);   // (seen once: not kept, its store NOPped)
        CHECK_EQ(did[1].draws, 4u);
        CHECK_EQ(did[1].copies, 1u);   // (seen twice: kept)
        for (int f = 2; f < 5; f++) {
            CHECK_EQ(did[f].draws, 0u);
            CHECK_EQ(did[f].copies, 1u);
        }
        for (int i = 0; i < 3; i++) CHECK(wordsOf(*t[i]) == want[size_t(i)]);
    }
}

// The bound bytes are compared as the GPU will read them: at commit. A byte changed between frames, or after the draws
// were recorded and before the commit, runs the pass; the same bytes again are kept and copied again.
TEST(a_changed_byte_of_a_bound_buffer_runs_the_pass_again_even_after_encoding) {
    Rig r;
    r.queue->setMemo(memo::kStrict, 2);
    std::shared_ptr<Texture> t[3] = {texture(r, 64, 64), texture(r, 64, 64), texture(r, 64, 64)};
    std::string why;
    auto colour = newBuffer(*r.gpu, 16, &why);
    float c[4] = {0.25f, 0.5f, 0.75f, 1};
    memcpy(colour->contents(), c, 16);
    Bindings fsb;
    fsb.buffers = {Binding{colour.get(), 0}};
    int f = 0;
    for (; f < 3; f++) frame(r, *t[f % 3], cflat(), fsb, 3);
    CHECK_EQ(r.queue->memoStats().hits, 1u);
    // a byte between frames
    c[1] = 0.5001f;
    memcpy(colour->contents(), c, 16);
    Did d = frame(r, *t[f++ % 3], cflat(), fsb, 3);
    CHECK_EQ(d.draws, 3u);
    CHECK_EQ(r.queue->memoStats().hits, 1u);
    CHECK(r.queue->memoStats().lastMiss == "contents");
    d = frame(r, *t[f++ % 3], cflat(), fsb, 3);   // (kept)
    CHECK_EQ(d.draws, 3u);
    d = frame(r, *t[f++ % 3], cflat(), fsb, 3);
    CHECK_EQ(d.draws, 0u);
    CHECK_EQ(r.queue->memoStats().hits, 2u);
    // a byte after encoding, before commit: Metal lets the program write the buffer until then
    d = frame(r, *t[f++ % 3], cflat(), fsb, 3, 1.0, [&] {
        c[2] = 0.7f;
        memcpy(colour->contents(), c, 16);
    });
    CHECK(d.ok);
    CHECK_EQ(d.draws, 3u);
    CHECK_EQ(r.queue->memoStats().hits, 2u);
    CHECK(r.queue->memoStats().lastMiss == "contents");
    if (r.queue->memoStats().lastMiss != "contents") std::printf("       lastMiss %s\n", r.queue->memoStats().lastMiss.c_str());
}

// A sampled texture is taken as unchanged while its memory's write epoch is: one more for a replaceRegion, and one more
// for any GPU write this process records (here a copy into it, in a command buffer of its own)
TEST(a_texture_rewritten_by_the_processor_or_the_gpu_runs_the_pass_again) {
    Rig r;
    r.queue->setMemo(memo::kStrict, 2);
    std::shared_ptr<Texture> t[3] = {texture(r, 64, 64), texture(r, 64, 64), texture(r, 64, 64)};
    auto image = texture(r, 4, 4, kTextureUsageRead | kTextureUsageRenderTarget);
    auto other = texture(r, 4, 4, kTextureUsageRead | kTextureUsageRenderTarget);
    Sampler s = nearest();
    Bindings fsb;
    fsb.textures = {image.get()};
    fsb.samplers = {&s};
    int f = 0;
    for (; f < 3; f++) CHECK(frame(r, *t[f % 3], tex(), fsb, 2).ok);
    CHECK_EQ(r.queue->memoStats().hits, 1u);
    // the processor, through the driver
    const uint32_t texel = 0x11223344u;
    std::string why;
    CHECK(image->replace(1, 1, 1, 1, &texel, 4, &why));
    Did d = frame(r, *t[f++ % 3], tex(), fsb, 2);
    CHECK_EQ(d.draws, 2u);
    CHECK(r.queue->memoStats().lastMiss == "structure");
    frame(r, *t[f++ % 3], tex(), fsb, 2);
    d = frame(r, *t[f++ % 3], tex(), fsb, 2);
    CHECK_EQ(d.draws, 0u);
    CHECK_EQ(r.queue->memoStats().hits, 2u);
    // the GPU: a copy into it
    {
        auto cb = r.queue->commandBuffer();
        CHECK(cb->blit(*other, 0, render::BlitRect{0, 0, 4, 4}, *image, 0, render::BlitRect{0, 0, 4, 4}, false));
        cb->commit();
        cb->waitUntilCompleted();
    }
    d = frame(r, *t[f++ % 3], tex(), fsb, 2);
    CHECK_EQ(d.draws, 2u);
    CHECK_EQ(r.queue->memoStats().hits, 2u);
}

// A texture whose memory the driver does not see written (one over a buffer's rows, as CoreAnimation's client-storage
// images are): its rows are hashed at commit; a byte written straight into the buffer runs the pass again.
TEST(a_texture_over_a_buffer_is_compared_by_its_rows) {
    Rig r;
    r.queue->setMemo(memo::kStrict, 2);
    std::shared_ptr<Texture> t[3] = {texture(r, 64, 64), texture(r, 64, 64), texture(r, 64, 64)};
    std::string why;
    auto rows = newBuffer(*r.gpu, 16 * 256, &why);
    TextureDesc desc;
    desc.format = PixelFormat::kRGBA8Unorm;
    desc.width = 16;
    desc.height = 16;
    desc.usage = kTextureUsageRead;
    auto image = Texture::overBuffer(rows, 0, 256, desc, &why);
    CHECK(image != nullptr);
    if (image == nullptr) {
        std::printf("       %s\n", why.c_str());
        return;
    }
    Sampler s = nearest();
    Bindings fsb;
    fsb.textures = {image.get()};
    fsb.samplers = {&s};
    int f = 0;
    for (; f < 3; f++) CHECK(frame(r, *t[f % 3], tex(), fsb, 2).ok);
    CHECK_EQ(r.queue->memoStats().hits, 1u);
    static_cast<uint8_t*>(rows->contents())[300] ^= 1;
    Did d = frame(r, *t[f++ % 3], tex(), fsb, 2);
    CHECK_EQ(d.draws, 2u);
    CHECK(r.queue->memoStats().lastMiss == "contents");
    CHECK(r.queue->memoStats().hashedBytes >= 4 * 16 * 256u);
    // and with a version hook (the kernel's write sequence, an IOSurface's seed) answering for it, no rows are read
    uint64_t seq = 7;
    r.queue->setMemoVersion([&](const Texture& x, uint64_t* v) {
        if (&x != image.get()) return false;
        *v = seq;
        return true;
    });
    const uint64_t hashed = r.queue->memoStats().hashedBytes;
    frame(r, *t[f++ % 3], tex(), fsb, 2);
    frame(r, *t[f++ % 3], tex(), fsb, 2);
    d = frame(r, *t[f++ % 3], tex(), fsb, 2);
    CHECK_EQ(d.draws, 0u);
    CHECK(r.queue->memoStats().hashedBytes - hashed < 3 * 4096u);
    seq++;
    d = frame(r, *t[f++ % 3], tex(), fsb, 2);
    CHECK_EQ(d.draws, 2u);
}

// What the pass is made of: the clear colour, a draw's vertex count, the number of draws -- each change runs it again
TEST(a_changed_clear_colour_or_draw_runs_the_pass_again) {
    Rig r;
    r.queue->setMemo(memo::kStrict, 2);
    std::shared_ptr<Texture> t[3] = {texture(r, 64, 64), texture(r, 64, 64), texture(r, 64, 64)};
    Bindings none;
    int f = 0;
    for (; f < 3; f++) frame(r, *t[f % 3], flat(), none, 2);
    CHECK_EQ(r.queue->memoStats().hits, 1u);
    Did d = frame(r, *t[f++ % 3], flat(), none, 2, 0.5);
    CHECK_EQ(d.draws, 2u);
    CHECK(r.queue->memoStats().lastMiss == "structure");
    d = frame(r, *t[f++ % 3], flat(), none, 3, 0.5);
    CHECK_EQ(d.draws, 3u);
    d = frame(r, *t[f++ % 3], flat(), none, 2, 0.5, [] {}, whole(6));
    CHECK_EQ(d.draws, 2u);
    CHECK_EQ(r.queue->memoStats().hits, 1u);
    // and a target of another size is another slot, with a cache of its own
    auto big = texture(r, 128, 64);
    d = frame(r, *big, flat(), none, 2, 0.5, [] {}, whole(3, 128, 64));
    CHECK(d.ok);
    CHECK_EQ(d.draws, 2u);
}

// A pass that loads its target (CoreAnimation's screen pass) is kept only in mode 2, and only when its first draw is
// an opaque rectangle the caller vouched for and every later draw lies inside it; the hit's copy is that rectangle
// alone, the rest of the target left as it was.
TEST(a_load_pass_is_kept_under_a_vouched_opaque_first_draw_and_copied_over_it_alone) {
    for (uint32_t mode : {uint32_t(memo::kStrict), uint32_t(memo::kCover)}) {
        Rig r;
        r.queue->setMemo(mode, 2);
        std::shared_ptr<Texture> t[3] = {texture(r, 64, 64), texture(r, 64, 64), texture(r, 64, 64)};
        Did did[4];
        for (int f = 0; f < 4; f++) {
            const Texture& x = *t[f % 3];
            fillWords(x, 0x01010101u * uint32_t(f + 1));   // (what the surface held: another frame's picture)
            RenderPass pass;
            pass.color0 = &x;
            pass.load = LoadAction::kLoad;
            auto cb = r.queue->commandBuffer();
            const FakeGpu::Stats before = r.fake->stats();
            memo::Hint cover;
            cover.set = cover.rect = true;
            cover.box[0] = 8, cover.box[1] = 16, cover.box[2] = 40, cover.box[3] = 48;
            cb->memoHint(cover);
            CHECK(cb->draw(flat(), pass, whole(6), Bindings{}, Bindings{}));
            Draw inside = whole();
            inside.scissor = Scissor{10, 20, 20, 20};
            memo::Hint small;
            small.set = true;
            small.box[0] = 12, small.box[1] = 22, small.box[2] = 28, small.box[3] = 38;
            cb->memoHint(small);
            CHECK(cb->draw(flat(), pass, inside, Bindings{}, Bindings{}));
            cb->draw(flat(), pass, inside, Bindings{}, Bindings{});
            cb->commit();
            cb->waitUntilCompleted();
            CHECK(cb->status() == CommandBuffer::kCompleted);
            did[f].draws = r.fake->stats().draws - before.draws;
            did[f].copies = r.fake->stats().copies - before.copies;
        }
        const memo::Stats s = r.queue->memoStats();
        if (mode == memo::kStrict) {
            CHECK_EQ(s.hits, 0u);
            CHECK_EQ(s.refused, 4u);
            CHECK(s.lastRefusal.find("metal_passmemo=2") != std::string::npos);
            continue;
        }
        CHECK_EQ(s.hits, 2u);
        CHECK_EQ(s.stores, 1u);
        CHECK_EQ(did[2].draws, 0u);
        CHECK_EQ(did[3].draws, 0u);
        // frame 1 kept what its pass left over the cover (on the fake: what the surface held there, 0x02020202); frame 3
        // (into t[0], which held 0x04040404) has that over the cover and its own words elsewhere
        const Texture& last = *t[0];
        CHECK_EQ(pixel(last, 8, 16), 0x02020202u);
        CHECK_EQ(pixel(last, 39, 47), 0x02020202u);
        CHECK_EQ(pixel(last, 7, 16), 0x04040404u);
        CHECK_EQ(pixel(last, 40, 47), 0x04040404u);
        CHECK_EQ(pixel(last, 8, 48), 0x04040404u);
        CHECK_EQ(pixel(last, 8, 15), 0x04040404u);
    }
}

// ...and refused when a later draw can write outside the cover, or the first draw is not vouched for or blends
TEST(a_load_pass_with_a_draw_outside_its_cover_or_no_cover_is_refused) {
    Rig r;
    r.queue->setMemo(memo::kCover, 2);
    auto t = texture(r, 64, 64);
    RenderPass pass;
    pass.color0 = t.get();
    pass.load = LoadAction::kLoad;
    auto record = [&](bool vouch, bool outside, bool blend) {
        auto cb = r.queue->commandBuffer();
        if (vouch) {
            memo::Hint h;
            h.set = h.rect = true;
            h.box[0] = 8, h.box[1] = 16, h.box[2] = 40, h.box[3] = 48;
            cb->memoHint(h);
        }
        Program first = flat();
        if (blend) {
            first.blend.enabled = true;
            first.blend.dstRgb = 5;   // (OneMinusSourceAlpha)
            first.uid = 0x1010;
        }
        cb->draw(first, pass, whole(6), Bindings{}, Bindings{});
        Draw later = whole();
        if (!outside) later.scissor = Scissor{10, 20, 20, 20};
        cb->draw(flat(), pass, later, Bindings{}, Bindings{});
        cb->commit();
        cb->waitUntilCompleted();
        return r.queue->memoStats().lastRefusal;
    };
    CHECK(record(false, false, false).find("vouched rectangle") != std::string::npos);
    CHECK(record(true, true, false).find("outside the first draw's cover") != std::string::npos);
    CHECK(record(true, false, true).find("blends") != std::string::npos);
    CHECK_EQ(r.queue->memoStats().hits, 0u);
    CHECK_EQ(r.queue->memoStats().refused, 3u);
}

// Refused by name: a program without a uid, a depth attachment; and off, nothing is looked at
TEST(passes_memoization_cannot_judge_are_refused_and_off_is_off) {
    Rig r;
    r.queue->setMemo(memo::kStrict, 1);
    auto t = texture(r, 64, 64);
    {
        Program p = flat();
        p.uid = 0;
        frame(r, *t, p, Bindings{}, 2);
        CHECK(r.queue->memoStats().lastRefusal == "a program without a uid");
    }
    {
        TextureDesc d;
        d.format = PixelFormat::kDepth32Float;
        d.width = d.height = 64;
        d.usage = kTextureUsageRenderTarget;
        std::string why;
        auto depth = Texture::make(*r.gpu, d, &why);
        CHECK(depth != nullptr);
        RenderPass pass;
        pass.color0 = t.get();
        pass.load = LoadAction::kClear;
        pass.depth = depth.get();
        auto cb = r.queue->commandBuffer();
        cb->draw(flat(), pass, whole(), Bindings{}, Bindings{});
        pass.load = LoadAction::kLoad;
        cb->draw(flat(), pass, whole(), Bindings{}, Bindings{});
        cb->commit();
        cb->waitUntilCompleted();
        CHECK(r.queue->memoStats().lastRefusal == "a depth or stencil attachment");
    }
    CHECK_EQ(r.queue->memoStats().refused, 2u);
    r.queue->setMemo(memo::kOff);
    for (int f = 0; f < 4; f++) CHECK_EQ(frame(r, *t, flat(), Bindings{}, 2).draws, 2u);
    CHECK_EQ(r.queue->memoStats().candidates, 0u);
}

// A pass's place in the frame tells passes into the same kind of target apart: two passes a frame into two surfaces
// of one size, each repeated, are two slots, each kept and copied on its own
TEST(two_passes_into_one_kind_of_target_are_kept_apart_by_their_place) {
    Rig r;
    r.queue->setMemo(memo::kStrict, 2);
    auto a = texture(r, 64, 64), b = texture(r, 64, 64);
    for (int f = 0; f < 4; f++) {
        auto cb = r.queue->commandBuffer();
        for (int k = 0; k < 2; k++) {
            RenderPass pass;
            pass.color0 = k == 0 ? a.get() : b.get();
            pass.load = LoadAction::kClear;
            pass.clearColor[0] = k == 0 ? 1.0 : 0.0;
            pass.clearColor[3] = 1;
            for (int i = 0; i < 2; i++) {
                cb->draw(flat(), pass, whole(), Bindings{}, Bindings{});
                pass.load = LoadAction::kLoad;
            }
        }
        cb->commit();
        cb->waitUntilCompleted();
        CHECK(cb->status() == CommandBuffer::kCompleted);
    }
    const memo::Stats s = r.queue->memoStats();
    CHECK_EQ(s.hits, 4u);
    CHECK_EQ(s.stores, 2u);
    CHECK(pixel(*a, 0, 0) != pixel(*b, 0, 0));
}
