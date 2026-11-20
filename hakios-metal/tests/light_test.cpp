// Light passes (CommandBuffer::setLightPasses; ios-sapphire dev-notes anexos/32-interfaz-telegram §4.3 and §5.1): a
// sysmem pass after a sysmem pass of the same command buffer begins without the batch start and ends without the waits,
// and the driver orders what Metal's hazard tracking orders. The fake GPU holds every stream to three rules (B1-B3,
// fake_gpu.hpp); here, for each rule, a chain of passes that needs it -- complete on the fake with the switch on -- and
// its mutation twin (render::Options::lightTwin) with the one packet the rule stands on left out, which the fake
// refuses by the rule's name. Then what the switch changes and what it does not: the stamps, memoryBarrier, the passes
// after a copy or a dispatch.
//
// SPDX-License-Identifier: MIT
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "hakios-metal/command.hpp"
#include "hakios-metal/fake_gpu.hpp"
#include "hakios-metal/render.hpp"
#include "hakios-metal/texture.hpp"
#include "fixtures/tri_cflat_fs_air.hpp"
#include "fixtures/tri_color_vs_air.hpp"
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
        TextureDesc d;
        d.format = PixelFormat::kRGBA8Unorm;
        d.width = d.height = 64;
        d.usage = kTextureUsageRenderTarget | kTextureUsageRead;
        for (auto& t : rt) t = Texture::make(*gpu, d, &why);
        d.width = d.height = 16;
        d.usage = kTextureUsageRead;
        icon = Texture::make(*gpu, d, &why);
        makeSampler(SamplerDesc{}, &smp, &why);
    }
    FakeGpu* fake;
    std::unique_ptr<Gpu> gpu;
    std::unique_ptr<CommandQueue> queue;
    std::shared_ptr<Texture> rt[3], icon;
    Sampler smp;
};

const Program& texProgram() {
    static const Program p{kernels::tri_color_vs_air, kernels::tri_tex_fs_air, Linkage{}, "tri_tex"};
    return p;
}

Draw quad() {
    Draw d;
    d.vertexCount = 6;
    d.viewport = Viewport{0, 0, 64, 64, 0, 1};
    return d;
}

// One step of a chain: a pass into TO (LOAD its action) whose one draw samples FROM
struct Step {
    int to;
    int from;   // -1: the icon, which no pass writes
    LoadAction load = LoadAction::kDontCare;
};

struct Ran {
    bool completed = false;
    uint32_t faults = 0;
    std::string fault;
    CommandBuffer::LightCounts counts;
    uint32_t waits = 0;   // CP_WAIT_FOR_IDLE the command processor ran
    size_t words = 0, passTimes = 0;
};

// the steps in one command buffer, every pass in sysmem, with light passes on or off and the twin's packet left out
Ran run(const std::vector<Step>& steps, bool light, uint8_t twin = 0, bool profile = false, bool tuned = false) {
    Rig r;
    auto cb = r.queue->commandBuffer();
    cb->setGmem(false);
    cb->setLightPasses(light);
    cb->setProfile(profile);
    cb->setGmemTuned(tuned);
    render::Options o;
    o.lightTwin = twin;
    for (const Step& s : steps) {
        RenderPass pass;
        pass.color0 = r.rt[s.to].get();
        pass.load = s.load;
        pass.clearColor[0] = 0.25, pass.clearColor[3] = 1;
        Bindings fsb;
        fsb.textures = {s.from < 0 ? r.icon.get() : r.rt[s.from].get()};
        fsb.samplers = {&r.smp};
        CHECK(cb->draw(texProgram(), pass, quad(), Bindings{}, fsb, o));
    }
    const uint32_t waits0 = r.fake->stats().cpWaits;
    cb->commit();
    cb->waitUntilCompleted();
    Ran out;
    out.completed = cb->status() == CommandBuffer::kCompleted;
    out.faults = r.fake->stats().faults;
    out.fault = r.fake->stats().lastFault;
    out.counts = cb->lightCounts();
    out.waits = r.fake->stats().cpWaits - waits0;
    out.words = cb->streamWords();
    out.passTimes = cb->passTimes().size();
    return out;
}

void expectRule(const Ran& r, const char* rule) {
    CHECK(!r.completed);
    CHECK_EQ(r.faults, 1u);
    CHECK(r.fault.find(rule) != std::string::npos);
    if (r.fault.find(rule) == std::string::npos) std::printf("       wanted %s, the fake said: %s\n", rule, r.fault.c_str());
}

void expectClean(const Ran& r) {
    CHECK(r.completed);
    CHECK_EQ(r.faults, 0u);
    if (r.faults != 0) std::printf("       the fake said: %s\n", r.fault.c_str());
}

// A -> B -> A -> B ...: each pass samples the target the pass before it drew
std::vector<Step> pingPong(int passes) {
    std::vector<Step> s;
    for (int i = 0; i < passes; i++) s.push_back(Step{i % 2, i == 0 ? -1 : (i - 1) % 2});
    return s;
}
}  // namespace

// B1: a read after a write. Every pass of a ping-pong after the first is light and samples the target the pass before
// drew: the read barrier before each one's draw (CACHE_INVALIDATE, CP_WAIT_FOR_IDLE). Its twin leaves the barrier out.
TEST(light_passes_read_what_an_earlier_pass_wrote_after_the_read_barrier) {
    const Ran on = run(pingPong(6), true);
    expectClean(on);
    CHECK_EQ(on.counts.light, 5u);
    CHECK_EQ(on.counts.full, 1u);
    CHECK_EQ(on.counts.barriers, 5u);
    expectRule(run(pingPong(6), true, 1), "(B1)");
    // (and without the switch the same chain, as it always was: every pass full)
    const Ran off = run(pingPong(6), false);
    expectClean(off);
    CHECK_EQ(off.counts.light, 0u);
    CHECK(on.words < off.words);
    CHECK(on.waits < off.waits);
}

// B2: the colour CCU between passes -- flushed by a light tail, invalidated by the light head that follows it. The
// twins: no flush (the invalidate then drops what the pass before drew), no invalidate (the next pass's draw comes to a
// CCU still holding the other pass's lines).
TEST(light_passes_flush_the_colour_cache_and_invalidate_it_between_them) {
    const std::vector<Step> apart = {{0, -1}, {1, -1}, {2, -1}};
    expectClean(run(apart, true));
    expectRule(run(apart, true, 2), "(B2)");
    expectRule(run(apart, true, 3), "(B2)");
}

// B3: a write after a read. A draws, B samples A, then A is drawn again with nothing of B read: the pass that writes A
// waits first for B's reads (a CP_WAIT_FOR_IDLE before its head). Its twin leaves that WFI out. With a Clear the fill
// is the write, before any draw.
TEST(a_light_pass_that_writes_what_was_read_waits_for_the_reads_first) {
    const std::vector<Step> back = {{0, -1}, {1, 0}, {0, -1}};
    const Ran on = run(back, true);
    expectClean(on);
    CHECK_EQ(on.counts.writeWaits, 1u);
    expectRule(run(back, true, 4), "(B3)");
    const std::vector<Step> cleared = {{0, -1}, {1, 0}, {0, -1, LoadAction::kClear}};
    expectClean(run(cleared, true));
    expectRule(run(cleared, true, 4), "(B3)");
    // (a pass whose target nothing read since the last WFI has none: B read A, then a barrier came before C's draw)
    const std::vector<Step> after = {{0, -1}, {1, 0}, {2, 1}, {0, -1}};
    const Ran later = run(after, true);
    expectClean(later);
    CHECK_EQ(later.counts.writeWaits, 0u);
}

// Reads 1, 2 and N passes after the write: one barrier covers every pass closed before it, so a second read of an older
// target finds the caches fresh
TEST(one_read_barrier_covers_every_pass_closed_before_it) {
    const std::vector<Step> s = {{0, -1}, {1, -1}, {2, 0}, {1, 0}, {0, 2}};
    const Ran on = run(s, true);
    expectClean(on);
    // pass 3 reads pass 1's A (a barrier), pass 4 reads A again (none: written before the barrier), pass 5 reads C,
    // written by pass 3 after it (a barrier)
    CHECK_EQ(on.counts.barriers, 2u);
    expectRule(run(s, true, 1), "(B1)");
}

// With a profile each light pass keeps a CP_WAIT_FOR_IDLE before its end stamp, so that the stamps still time it; with
// the queue's tuning alone (the plug-in's default) and no pass of a tuning key, no stamp at all
TEST(a_light_pass_is_stamped_only_when_its_time_is_read) {
    const std::vector<Step> s = pingPong(5);
    const Ran plain = run(s, true), profiled = run(s, true, 0, true), tuned = run(s, true, 0, false, true);
    expectClean(plain);
    expectClean(profiled);
    expectClean(tuned);
    CHECK_EQ(profiled.passTimes, 5u);
    // (the four light passes' own, which then also stand for the WFIs a ping-pong's writes after reads had)
    CHECK_EQ(plain.counts.writeWaits, 3u);
    CHECK_EQ(profiled.counts.writeWaits, 0u);
    CHECK_EQ(profiled.waits, plain.waits + 4 - 3);
    CHECK_EQ(tuned.passTimes, 0u);
    CHECK_EQ(tuned.words, plain.words);
    // (without the switch the tuning stamps every pass, as before)
    CHECK_EQ(run(s, false, 0, false, true).passTimes, 5u);
}

// Metal's memoryBarrier between two draws of one pass: with light passes the read barrier before the next draw; without
// them nothing at all
TEST(a_memory_barrier_is_a_read_barrier_with_light_passes_and_nothing_without) {
    for (const bool light : {false, true}) {
        size_t words[2] = {};
        for (int barrier = 0; barrier < 2; barrier++) {
            Rig r;
            auto cb = r.queue->commandBuffer();
            cb->setGmem(false);
            cb->setLightPasses(light);
            RenderPass pass;
            pass.color0 = r.rt[0].get();
            Bindings fsb;
            fsb.textures = {r.icon.get()};
            fsb.samplers = {&r.smp};
            CHECK(cb->draw(texProgram(), pass, quad(), Bindings{}, fsb));
            if (barrier) cb->memoryBarrier();
            pass.load = LoadAction::kLoad;
            CHECK(cb->draw(texProgram(), pass, quad(), Bindings{}, fsb));
            CHECK_EQ(cb->lightCounts().barriers, light && barrier ? 1u : 0u);
            cb->commit();
            cb->waitUntilCompleted();
            CHECK(cb->status() == CommandBuffer::kCompleted);
            words[barrier] = cb->streamWords();
        }
        if (light) CHECK_EQ(words[1], words[0] + 3);   // (CACHE_INVALIDATE, 2 words, and the WFI, 1)
        else CHECK_EQ(words[1], words[0]);
    }
}

// A copy and a dispatch end the pass before them with the full tail, and the pass after them begins with the full head
TEST(a_pass_after_a_copy_is_a_full_one) {
    Rig r;
    auto cb = r.queue->commandBuffer();
    cb->setGmem(false);
    cb->setLightPasses(true);
    Bindings fsb;
    fsb.textures = {r.icon.get()};
    fsb.samplers = {&r.smp};
    RenderPass a, b;
    a.color0 = r.rt[0].get();
    b.color0 = r.rt[1].get();
    CHECK(cb->draw(texProgram(), a, quad(), Bindings{}, fsb));
    CHECK(cb->draw(texProgram(), b, quad(), Bindings{}, fsb));   // light
    CHECK(cb->blit(*r.rt[1], 0, render::BlitRect{0, 0, 64, 64}, *r.rt[2], 0, render::BlitRect{0, 0, 64, 64}, false));
    fsb.textures = {r.rt[2].get()};
    CHECK(cb->draw(texProgram(), a, quad(), Bindings{}, fsb));   // full: the copy came between
    CHECK_EQ(cb->lightCounts().light, 1u);
    CHECK_EQ(cb->lightCounts().full, 2u);
    cb->commit();
    cb->waitUntilCompleted();
    CHECK(cb->status() == CommandBuffer::kCompleted);
    CHECK_EQ(r.fake->stats().faults, 0u);
}
