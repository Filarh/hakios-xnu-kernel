// What CommandBuffer::draw keeps of a program between draws (render.cpp: ProgramState, PassCache), held to the same
// draw encoded in full. The runner turns CommandBuffer::verifyDraws on, so every draw taken from a kept state anywhere
// in the tests is checked word for word against the long way (a difference ends the buffer in error); the tests here
// are the cases that make a kept state wrong if it is kept wrongly -- a program changed after it was drawn with, a pass
// of another kind, bindings the kept state cannot take -- and the rule the leaving-out of a group stands on.
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <thread>
#include <vector>

#include "hakios-metal/a6xx.hpp"
#include "hakios-metal/command.hpp"
#include "hakios-metal/fake_gpu.hpp"
#include "hakios-metal/hakios_metal.h"
#include "hakios-metal/render.hpp"
#include "hakios-metal/testing.hpp"
#include "hakios-metal/texture.hpp"
#include "fixtures/tri_arr_vs_air.hpp"
#include "fixtures/tri_attr_vs_air.hpp"
#include "fixtures/tri_base_vs_air.hpp"
#include "fixtures/tri_cflat_fs_air.hpp"
#include "fixtures/tri_color_fs_air.hpp"
#include "fixtures/tri_color_vs_air.hpp"
#include "fixtures/tri_dbuf_fs_air.hpp"
#include "fixtures/tri_dpos_vs_air.hpp"
#include "fixtures/tri_fb_fs_air.hpp"
#include "fixtures/tri_flat_fs_air.hpp"
#include "fixtures/tri_flat_vs_air.hpp"
#include "fixtures/tri_mrt_fs_air.hpp"
#include "fixtures/tri_nullfb_fs_air.hpp"
#include "fixtures/tri_oldcb_vs_air.hpp"
#include "fixtures/tri_olddb_vs_air.hpp"
#include "fixtures/tri_pvt_fs_air.hpp"
#include "fixtures/tri_tex39_fs_air.hpp"
#include "fixtures/tri_tex_fs_air.hpp"
#include "fixtures/tri_upos_vs_air.hpp"
#include "fixtures/tri_vtex_vs_air.hpp"
#include "golden.hpp"
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

std::shared_ptr<Texture> texture(Rig& r, uint32_t w, uint32_t h, uint32_t usage, PixelFormat format = PixelFormat::kRGBA8Unorm) {
    TextureDesc d;
    d.format = format;
    d.width = w;
    d.height = h;
    d.usage = usage;
    std::string why;
    auto t = Texture::make(*r.gpu, d, &why);
    CHECK(t != nullptr);
    return t;
}

Draw quad(uint32_t w = 64, uint32_t h = 64) {
    Draw d;
    d.vertexCount = 6;
    d.viewport = Viewport{0, 0, double(w), double(h), 0, 1};
    return d;
}

// the buffer committed and waited for -- and the queue with it, until its thread has the submission's memory back: a
// buffer recorded before that is given new memory, and every address made after it is another (the streams here are
// summed, golden.hpp)
void wait(Rig& r, const std::shared_ptr<CommandBuffer>& cb) {
    cb->commit();
    cb->waitUntilCompleted();
    while (cb->status() == CommandBuffer::kCompleted && r.queue->idleCommandBuffers() == 0) std::this_thread::yield();
}
// ...and held to having completed without the fake finding fault
void finish(Rig& r, const std::shared_ptr<CommandBuffer>& cb) {
    wait(r, cb);
    CHECK(cb->status() == CommandBuffer::kCompleted);
    if (cb->status() != CommandBuffer::kCompleted) std::printf("       %s | %s\n", cb->error().c_str(), r.fake->stats().lastFault.c_str());
}

// a Program as linkProgram leaves one: linked, with a uid
Program linked(const Stage& vs, const Stage& fs, const char* name) {
    Program p{vs, fs, Linkage{}, name};
    std::string why;
    CHECK(render::link(p.vs, p.fs, &p.link, &why));
    p.uid = newProgramUid();
    return p;
}

// the C interface's device, queue and a target, for the tests that go the plug-in's way
struct CRig {
    hakios_metal_device* device = hakios_metal_open_fake();
    hakios_metal_queue* queue = hakios_metal_queue_new(device);
    hakios_metal_texture* target = hakios_metal_texture_new(device, 2, 70, 64, 64, 1 | 4);   // 2D RGBA8Unorm, read and render target
    ~CRig() {
        hakios_metal_texture_free(target);
        hakios_metal_queue_free(queue);
        hakios_metal_close(device);
    }
    hakios_metal_render_pass pass() const {
        hakios_metal_render_pass p{};
        p.color0 = target;
        p.load_action = 1;
        p.store_action = 1;
        return p;
    }
};

// a command buffer of the C interface committed and waited for; its status
hakios_metal_status done(hakios_metal_cb* cb) {
    hakios_metal_cb_commit(cb, nullptr, nullptr);
    hakios_metal_cb_wait(cb);
    const hakios_metal_status s = hakios_metal_cb_status(cb);
    if (s != HAKIOS_METAL_COMPLETED) std::printf("       %s\n", hakios_metal_cb_error(cb));
    hakios_metal_cb_free(cb);
    return s;
}
}  // namespace

// The first draw with a program in a kind of pass is judged and written whole; the draws after it, in the same command
// buffer and in the next ones of the queue, are taken from what that one came to.
TEST(a_programs_state_is_kept_by_the_queue_from_one_command_buffer_to_the_next) {
    Rig r;
    auto t = texture(r, 64, 64, kTextureUsageRenderTarget | kTextureUsageRead);
    RenderPass pass;
    pass.color0 = t.get();
    const Program p = linked(kernels::tri_flat_vs_air, kernels::tri_flat_fs_air, "tri_flat");
    auto first = r.queue->commandBuffer();
    for (int i = 0; i < 4; i++) CHECK(first->draw(p, pass, quad(), Bindings{}, Bindings{}));
    CHECK_EQ(first->counts().draws, 4u);
    CHECK_EQ(first->counts().keptDraws, 3u);   // (all but the first)
    finish(r, first);
    auto second = r.queue->commandBuffer();
    for (int i = 0; i < 4; i++) CHECK(second->draw(p, pass, quad(), Bindings{}, Bindings{}));
    CHECK_EQ(second->counts().keptDraws, 4u);  // (the pass's own first draw too)
    finish(r, second);
    // ...and a pass of another kind (another format) is another state: its first draw the long way again
    auto u = texture(r, 64, 64, kTextureUsageRenderTarget | kTextureUsageRead, PixelFormat::kBGRA8Unorm);
    RenderPass other;
    other.color0 = u.get();
    auto third = r.queue->commandBuffer();
    CHECK(third->draw(p, other, quad(), Bindings{}, Bindings{}));
    CHECK(third->draw(p, other, quad(), Bindings{}, Bindings{}));
    CHECK(third->draw(p, pass, quad(), Bindings{}, Bindings{}));
    CHECK_EQ(third->counts().keptDraws, 2u);
    finish(r, third);
    CHECK_EQ(r.fake->stats().faults, 0u);
}

// A Program made by hand carries no promise that it is the one drawn before: the state kept for it is used only while
// the Program is the same byte for byte. Changed in place between two draws -- its blending, a field of a stage -- the
// second draw is the changed program's (the runner's check against the long way says so, and the register here).
TEST(a_program_made_by_hand_and_changed_in_place_is_drawn_as_it_is_now) {
    Rig r;
    auto t = texture(r, 64, 64, kTextureUsageRenderTarget | kTextureUsageRead);
    RenderPass pass;
    pass.color0 = t.get();
    Program p{kernels::tri_flat_vs_air, kernels::tri_flat_fs_air, Linkage{}, "tri_flat"};
    auto cb = r.queue->commandBuffer();
    CHECK(cb->draw(p, pass, quad(), Bindings{}, Bindings{}));
    CHECK(cb->draw(p, pass, quad(), Bindings{}, Bindings{}));
    CHECK_EQ(cb->counts().keptDraws, 1u);
    p.blend.enabled = true;   // premultiplied source-over
    p.blend.srcRgb = 1, p.blend.dstRgb = 5, p.blend.srcAlpha = 1, p.blend.dstAlpha = 5;
    CHECK(cb->draw(p, pass, quad(), Bindings{}, Bindings{}));
    CHECK_EQ(cb->counts().keptDraws, 1u);      // (not the kept one: the long way, and a state of its own)
    CHECK(cb->draw(p, pass, quad(), Bindings{}, Bindings{}));
    CHECK_EQ(cb->counts().keptDraws, 2u);
    finish(r, cb);
    CHECK_EQ(r.fake->stats().faults, 0u);
    CHECK((r.fake->stats().lastDraw.mrtControl & a6xx::A6XX_RB_MRT_CONTROL_BLEND) != 0);
}

// The C interface's setters change a program after it was linked, and each makes it another program to a draw: what
// was kept for it is not used again.
TEST(a_program_changed_by_the_c_interface_is_another_program_to_a_draw) {
    CRig r;
    hakios_metal_program* p = hakios_metal_program_from_stages(kernels::tri_flat_vs_air, kernels::tri_flat_fs_air, 70);
    CHECK(p != nullptr);
    if (p == nullptr) return;
    hakios_metal_render_pass pass = r.pass();
    hakios_metal_cb* cb = hakios_metal_cb_new(r.queue);
    for (int i = 0; i < 3; i++) CHECK_EQ(hakios_metal_cb_draw(cb, p, &pass, nullptr, nullptr, 3, 0, 6, 1, nullptr, nullptr, 0, nullptr, nullptr, 0), 0);
    hakios_metal_program_set_blend(p, 1, 0, 0, 1, 5, 1, 5, 0xf);
    for (int i = 0; i < 3; i++) CHECK_EQ(hakios_metal_cb_draw(cb, p, &pass, nullptr, nullptr, 3, 0, 6, 1, nullptr, nullptr, 0, nullptr, nullptr, 0), 0);
    hakios_metal_program_set_blend(p, 0, 0, 0, 1, 0, 1, 0, 0x7);   // no blending, alpha not written
    for (int i = 0; i < 3; i++) CHECK_EQ(hakios_metal_cb_draw(cb, p, &pass, nullptr, nullptr, 3, 0, 6, 1, nullptr, nullptr, 0, nullptr, nullptr, 0), 0);
    CHECK(done(cb) == HAKIOS_METAL_COMPLETED);
    hakios_metal_program_free(p);
}

// A draw the kept state cannot take is judged the long way and refused by the sentence it always was: a constant buffer
// not bound, after a draw that had it; a texture of the wrong class; a viewport outside the target. And the draw after,
// with what it needs, is drawn.
TEST(a_draw_its_kept_state_cannot_take_is_refused_by_name_the_long_way) {
    CRig r;
    hakios_metal_program* flat = hakios_metal_program_from_stages(kernels::tri_flat_vs_air, kernels::tri_cflat_fs_air, 70);
    hakios_metal_program* tex = hakios_metal_program_from_stages(kernels::tri_color_vs_air, kernels::tri_tex_fs_air, 70);
    CHECK(flat != nullptr && tex != nullptr);
    if (flat == nullptr || tex == nullptr) return;
    hakios_metal_buffer* uniforms = hakios_metal_buffer_new(r.device, 4096);
    hakios_metal_texture* image = hakios_metal_texture_new(r.device, 2, 70, 16, 16, 1);
    hakios_metal_texture* integers = hakios_metal_texture_new(r.device, 2, 53, 16, 16, 1);   // R32Uint
    hakios_metal_sampler_desc sd{};
    sd.normalized_coordinates = 1;
    sd.max_anisotropy = 1;
    hakios_metal_sampler* sampler = hakios_metal_sampler_new(r.device, &sd);
    CHECK(uniforms != nullptr && image != nullptr && integers != nullptr && sampler != nullptr);
    hakios_metal_render_pass pass = r.pass();
    hakios_metal_cb* cb = hakios_metal_cb_new(r.queue);
    hakios_metal_cb_skip_refused(cb, 1);
    hakios_metal_buffer* bound[1] = {uniforms};
    const uint64_t at8[1] = {8};
    for (int i = 0; i < 2; i++) CHECK_EQ(hakios_metal_cb_draw(cb, flat, &pass, nullptr, nullptr, 3, 0, 6, 1, nullptr, nullptr, 0, bound, at8, 1), 0);
    CHECK_EQ(hakios_metal_cb_draw(cb, flat, &pass, nullptr, nullptr, 3, 0, 6, 1, nullptr, nullptr, 0, nullptr, nullptr, 0), -1);
    CHECK(strstr(hakios_metal_error(), "no buffer bound at index 0 (its constant buffer)") != nullptr);
    CHECK_EQ(hakios_metal_cb_draw(cb, flat, &pass, nullptr, nullptr, 3, 0, 6, 1, nullptr, nullptr, 0, bound, at8, 1), 0);
    hakios_metal_texture* textures[1] = {image};
    hakios_metal_sampler* samplers[1] = {sampler};
    for (int i = 0; i < 2; i++) {
        CHECK_EQ(hakios_metal_cb_draw2(cb, tex, &pass, nullptr, nullptr, 3, 0, 6, 1, nullptr, nullptr, 0, nullptr, nullptr, 0, textures, 1, samplers, 1), 0);
    }
    textures[0] = integers;
    CHECK_EQ(hakios_metal_cb_draw2(cb, tex, &pass, nullptr, nullptr, 3, 0, 6, 1, nullptr, nullptr, 0, nullptr, nullptr, 0, textures, 1, samplers, 1), -1);
    CHECK(strstr(hakios_metal_error(), "reads it as texture2d<float>") != nullptr);
    textures[0] = image;
    const hakios_metal_viewport outside{100, 100, 10, 10, 0, 1};
    CHECK_EQ(hakios_metal_cb_draw2(cb, tex, &pass, &outside, nullptr, 3, 0, 6, 1, nullptr, nullptr, 0, nullptr, nullptr, 0, textures, 1, samplers, 1), -1);
    CHECK(strstr(hakios_metal_error(), "covers no pixel of the 64 x 64 target") != nullptr);
    CHECK_EQ(hakios_metal_cb_draw2(cb, tex, &pass, nullptr, nullptr, 3, 0, 6, 1, nullptr, nullptr, 0, nullptr, nullptr, 0, textures, 1, samplers, 1), 0);
    const char* last = nullptr;
    CHECK_EQ(hakios_metal_cb_skipped(cb, &last), 3u);
    CHECK(done(cb) == HAKIOS_METAL_COMPLETED);
    hakios_metal_sampler_free(sampler);
    hakios_metal_texture_free(integers);
    hakios_metal_texture_free(image);
    hakios_metal_buffer_free(uniforms);
    hakios_metal_program_free(tex);
    hakios_metal_program_free(flat);
}

// Programs changing from draw to draw in one pass, as a compositor's layers change them -- a textured one with a
// vertex buffer, one of ten texture slots with one bound (the null texture in the other it samples), one with constants
// bound where the load cannot read, a framebuffer fetch, two colour outputs -- in sysmem, in tiles, and recorded both
// ways; each draw from its kept state is the draw the long way makes (the runner's check).
TEST(a_compositors_pass_of_changing_programs_is_drawn_from_kept_states_as_in_full) {
    for (int mode = 0; mode < 3; mode++) {
        Rig r;
        auto t = texture(r, 256, 256, kTextureUsageRenderTarget | kTextureUsageRead, PixelFormat::kBGRA8Unorm);
        auto u = texture(r, 256, 256, kTextureUsageRenderTarget | kTextureUsageRead, PixelFormat::kBGRA8Unorm);
        auto icon = texture(r, 16, 16, kTextureUsageRead), other = texture(r, 32, 32, kTextureUsageRead);
        std::string why;
        auto vertices = newBuffer(*r.gpu, 4096, &why), indices = newBuffer(*r.gpu, 4096, &why), uniforms = newBuffer(*r.gpu, 4096, &why);
        Sampler smp;
        CHECK(makeSampler(SamplerDesc{}, &smp, &why));
        RenderPass pass;
        pass.color0 = t.get();
        pass.more[0].texture = u.get();
        pass.more[0].load = LoadAction::kDontCare;
        pass.more[0].store = StoreAction::kDontCare;
        Program attr = linked(kernels::tri_attr_vs_air, kernels::tri_tex_fs_air, "attr+tex");
        attr.vertexDescriptor.attributes[0] = VertexAttribute{VertexFormat::kFloat2, 0, 0};
        attr.vertexDescriptor.attributes[1] = VertexAttribute{VertexFormat::kUChar4Normalized, 8, 0};
        attr.vertexDescriptor.layouts[0] = VertexLayout{12, VertexStep::kPerVertex, 1};
        Program slots = linked(kernels::tri_attr_vs_air, kernels::tri_tex39_fs_air, "attr+tex39");
        slots.vertexDescriptor = attr.vertexDescriptor;
        slots.blend.enabled = true;
        slots.blend.srcRgb = 1, slots.blend.dstRgb = 5, slots.blend.srcAlpha = 1, slots.blend.dstAlpha = 5;
        const Program flat = linked(kernels::tri_flat_vs_air, kernels::tri_cflat_fs_air, "flat+cflat");
        const Program fetch = linked(kernels::tri_flat_vs_air, kernels::tri_fb_fs_air, "flat+fb");
        const Program mrt = linked(kernels::tri_flat_vs_air, kernels::tri_mrt_fs_air, "flat+mrt");
        uint32_t kept = 0, draws = 0;
        for (int frame = 0; frame < 2; frame++) {
            auto cb = r.queue->commandBuffer();
            cb->setGmem(mode != 0);
            cb->setGmemMinDraws(mode == 2 ? 8 : 0);   // (2: recorded both ways, tiles for a pass of 8 or more)
            cb->setGmemBinning(mode == 2);
            for (int i = 0; i < 20; i++) {
                Draw d = quad(256, 256);
                if (i % 3 != 0) d.scissor = Scissor{uint32_t(8 * i), 16, 64, 64};
                Bindings vsb, fsb;
                const int k = i % 7;
                const Program* p = k < 2 ? &attr : k == 2 ? &slots : k == 3 ? &fetch : k == 4 ? &mrt : k == 5 ? &slots : &attr;
                if (p == &attr || p == &slots) {
                    vsb.buffers = {Binding{vertices.get(), uint64_t(12 * i)}};
                    d.indexBuffer = indices.get();
                    d.indexOffset = uint64_t(2 * i);
                }
                if (p == &attr) {
                    fsb.textures = {i % 2 ? icon.get() : other.get()};
                    fsb.samplers = {&smp};
                }
                if (p == &slots) {
                    fsb.textures = {nullptr, nullptr, nullptr, icon.get()};   // (slot 9, which it samples too, left unbound)
                    fsb.samplers.assign(10, &smp);
                }
                CHECK(cb->draw(*p, pass, d, vsb, fsb));
                if (!cb->error().empty()) std::printf("       mode %d draw %d: %s\n", mode, i, cb->error().c_str());
            }
            // (the fake holds a submission's constants to each later stage's constlen: the draws with uniforms last)
            for (int i = 0; i < 4; i++) {
                Bindings fsb;
                fsb.buffers = {Binding{uniforms.get(), uint64_t(8 + 16 * i)}};
                CHECK(cb->draw(flat, pass, quad(256, 256), Bindings{}, fsb));
            }
            kept += cb->counts().keptDraws;
            draws += cb->counts().draws;
            finish(r, cb);
        }
        // every draw of the second frame, and all but each program's first in the first
        CHECK_EQ(draws, 48u);
        CHECK_EQ(kept, 48u - 5u);
        CHECK_EQ(r.fake->stats().faults, 0u);
        if (r.fake->stats().faults != 0) std::printf("       mode %d: %s\n", mode, r.fake->stats().lastFault.c_str());
    }
}

// What leaving a group out stands on: a group a draw does not write again is one whose registers still hold what the
// group last wrote, which is so only if no other group of a draw writes them. Each group encoded alone, for programs of
// every shape here: the registers each writes are its own (HLSQ_INVALIDATE_CMD, a command written every time, apart).
TEST(the_groups_of_a_draw_write_registers_of_their_own) {
    Rig r;
    auto t = texture(r, 64, 64, kTextureUsageRenderTarget | kTextureUsageRead);
    auto u = texture(r, 64, 64, kTextureUsageRenderTarget | kTextureUsageRead);
    auto depth = texture(r, 64, 64, kTextureUsageRenderTarget, PixelFormat::kDepth32Float);
    auto icon = texture(r, 16, 16, kTextureUsageRead);
    std::string why;
    auto vertices = newBuffer(*r.gpu, 4096, &why), uniforms = newBuffer(*r.gpu, 4096, &why), table = newBuffer(*r.gpu, render::kTableBytes, &why);
    auto vsWords = newBuffer(*r.gpu, 4096, &why), fsWords = newBuffer(*r.gpu, 4096, &why);
    Sampler smp;
    CHECK(makeSampler(SamplerDesc{}, &smp, &why));
    struct Case {
        Program program;
        Bindings vsb, fsb;
        bool second, withDepth;
    };
    std::vector<Case> cases;
    cases.push_back(Case{Program{kernels::tri_flat_vs_air, kernels::tri_flat_fs_air, Linkage{}, "tri_flat"}, {}, {}, false, false});
    {
        Case c{Program{kernels::tri_attr_vs_air, kernels::tri_tex_fs_air, Linkage{}, "attr+tex"}, {}, {}, false, true};
        c.program.vertexDescriptor.attributes[0] = VertexAttribute{VertexFormat::kFloat2, 0, 0};
        c.program.vertexDescriptor.attributes[1] = VertexAttribute{VertexFormat::kUChar4Normalized, 8, 0};
        c.program.vertexDescriptor.layouts[0] = VertexLayout{12, VertexStep::kPerVertex, 1};
        c.program.blend.enabled = true;
        c.vsb.buffers = {Binding{vertices.get(), 0}};
        c.fsb.textures = {icon.get()};
        c.fsb.samplers = {&smp};
        cases.push_back(c);
    }
    {
        Case c{Program{kernels::tri_flat_vs_air, kernels::tri_cflat_fs_air, Linkage{}, "flat+cflat"}, {}, {}, false, false};
        c.fsb.buffers = {Binding{uniforms.get(), 8}};
        cases.push_back(c);
    }
    cases.push_back(Case{Program{kernels::tri_flat_vs_air, kernels::tri_mrt_fs_air, Linkage{}, "flat+mrt"}, {}, {}, true, false});
    cases.push_back(Case{Program{kernels::tri_flat_vs_air, kernels::tri_fb_fs_air, Linkage{}, "flat+fb"}, {}, {}, false, false});
    // render::Options' groups of a draw, each a bool of the struct (which holds bools and bytes alone)
    const struct {
        const char* name;
        bool render::Options::* on;
    } groups[] = {{"zsNone", &render::Options::zsNone},
                  {"mrt", &render::Options::mrt},
                  {"msaaOff", &render::Options::msaaOff},
                  {"renderCntl", &render::Options::renderCntl},
                  {"lrzOff", &render::Options::lrzOff},
                  {"programConfig", &render::Options::programConfig},
                  {"programVs", &render::Options::programVs},
                  {"programFs", &render::Options::programFs},
                  {"fsTextures", &render::Options::fsTextures},
                  {"vsTextures", &render::Options::vsTextures},
                  {"vfd", &render::Options::vfd},
                  {"vpc", &render::Options::vpc},
                  {"fsInputs", &render::Options::fsInputs},
                  {"fsOutputs", &render::Options::fsOutputs},
                  {"rast", &render::Options::rast},
                  {"viewportScissor", &render::Options::viewportScissor},
                  {"blendZsa", &render::Options::blendZsa}};
    for (const Case& c : cases) {
        RenderPass pass;
        pass.color0 = t.get();
        if (c.second) pass.more[0].texture = u.get();
        if (c.withDepth) pass.depth = depth.get();
        // what a draw writes with every group off: the depth buffer's registers when the pass has one (the group zsNone's
        // other half, written whatever the switch says), and nothing else
        std::map<uint32_t, const char*> owner;
        for (int g = -1; g < int(sizeof groups / sizeof groups[0]); g++) {
            render::Options o;
            memset(static_cast<void*>(&o), 0, sizeof o);
            if (g >= 0) o.*(groups[g].on) = true;
            pm4::Stream s;
            const bool ok = render::encodeInPass(s, c.program, pass, quad(), c.vsb, c.fsb, vsWords->gpuAddress(), fsWords->gpuAddress(), table->gpuAddress(),
                                                 static_cast<uint8_t*>(table->contents()), &why, o);
            CHECK(ok);
            if (!ok) {
                std::printf("       %s: %s\n", c.program.name, why.c_str());
                break;
            }
            const char* name = g < 0 ? "(the depth buffer)" : groups[g].name;
            std::set<uint32_t> mine;
            const std::vector<uint32_t>& w = s.words();
            for (size_t i = 0; i < w.size();) {
                const pm4::Header h = pm4::decode(w[i]);
                for (uint32_t k = 0; h.kind == pm4::Header::kRegisters && k < h.count; k++) mine.insert(h.reg + k);
                i += 1 + h.count;
            }
            for (uint32_t reg : mine) {
                if (reg == a6xx::REG_A6XX_HLSQ_INVALIDATE_CMD) continue;
                auto had = owner.find(reg);
                // (the depth buffer's registers are in every one of these streams when the pass has depth: theirs)
                if (had != owner.end() && strcmp(had->second, "(the depth buffer)") == 0) continue;
                CHECK(had == owner.end());
                if (had != owner.end()) std::printf("       %s: register %#x is written by %s and by %s\n", c.program.name, reg, had->second, name);
                owner.emplace(reg, name);
            }
        }
    }
}

// --- the same random frames, from kept states and written whole ---------------------------------------------------------------
namespace {
// the test's own numbers, the same on every host
struct Rng {
    uint64_t s;
    uint32_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return uint32_t(s >> 24);
    }
    uint32_t below(uint32_t n) { return next() % n; }
    bool one(uint32_t in) { return below(in) == 0; }
};

// what a recording came to
struct Told {
    std::vector<std::string> said;   // each draw: "" taken, else the sentence it was refused by; and each frame's end
    golden::Sums sums;               // what the fake was given
    uint32_t draws = 0, kept = 0;
    uint64_t faults = 0;
};

struct Chain {
    golden::Sums* into;
    void (*next)(FakeGpu&, uint64_t, uint32_t, void*);
    void* nextContext;
};
void chained(FakeGpu& gpu, uint64_t address, uint32_t words, void* context) {
    Chain& c = *static_cast<Chain*>(context);
    golden::hook(gpu, address, words, c.into);
    if (c.next != nullptr) c.next(gpu, address, words, c.nextContext);
}

// what a program needs bound
enum : uint32_t {
    kVertices = 1, kTexture = 2, kTenSlots = 4, kFsUniforms = 8, kVsUbo = 16, kVsDevice = 32, kFsDevice = 64, kVsTexture = 128, kVsUbos = 256,
    kVsDevices = 512, kManySlots = 1024,
};

// FRAMES frames of DRAWS draws by SEED's numbers: programs of every shape the fixtures have, in passes that change. Half
// the draws are made anew -- their numbers and bindings their own -- and half are the draw before with one thing
// changed (what a compositor's draws are to each other, and what the kept state's memory of the last draw is tried by);
// every fourth is then made wrong in one thing. MODE 0 sysmem, 1 tiles, 2 recorded both ways and binned. ALL_UIDS:
// every program one with a uid, so that a frame after the first is drawn from kept states alone.
Told randomFrames(uint64_t seed, bool keep, int mode, uint32_t frames, uint32_t draws, bool allUids) {
    Told out;
    const bool was = CommandBuffer::keepStates(keep);
    Chain chain{&out.sums, FakeGpu::submitHook, FakeGpu::submitHookContext};
    FakeGpu::submitHookContext = &chain;
    FakeGpu::submitHook = chained;
    {
        Rig r;
        Rng rng{(seed + 1) * 0x9e3779b97f4a7c15ull};
        const uint32_t W = 128, H = 96;
        const uint32_t target = kTextureUsageRenderTarget | kTextureUsageRead;
        auto colorA = texture(r, W, H, target, PixelFormat::kBGRA8Unorm), colorB = texture(r, W, H, target, PixelFormat::kBGRA8Unorm);
        auto second = texture(r, W, H, target, PixelFormat::kBGRA8Unorm);
        auto depth = texture(r, W, H, kTextureUsageRenderTarget, PixelFormat::kDepth32Float);
        auto depthStencil = texture(r, W, H, kTextureUsageRenderTarget, PixelFormat::kDepth32FloatStencil8);
        auto icon = texture(r, 16, 16, kTextureUsageRead), photo = texture(r, 32, 32, kTextureUsageRead);
        auto integers = texture(r, 8, 8, kTextureUsageRead, PixelFormat::kR32Uint);
        auto unread = texture(r, 8, 8, kTextureUsageRenderTarget);
        std::shared_ptr<Texture> volume;
        {
            TextureDesc d;
            d.type = TextureType::k3D;
            d.width = d.height = 4;
            d.depth = 2;
            std::string why;
            volume = Texture::make(*r.gpu, d, &why);
            CHECK(volume != nullptr);
        }
        std::string why;
        auto vertices = newBuffer(*r.gpu, 4096, &why), indices = newBuffer(*r.gpu, 4096, &why), uniforms = newBuffer(*r.gpu, 4096, &why);
        auto otherVertices = newBuffer(*r.gpu, 4096, &why);   // (as long as the first: a fetch from it differs in its address alone)
        auto device = newBuffer(*r.gpu, 4096, &why), otherDevice = newBuffer(*r.gpu, 4096, &why), small = newBuffer(*r.gpu, 64, &why);
        auto numbers = newBuffer(*r.gpu, 4096, &why), results = newBuffer(*r.gpu, 4096, &why);
        {
            // an indirect draw's words, at every offset one is asked at: 6 vertices, 1 instance, from 0
            uint32_t* w = static_cast<uint32_t*>(numbers->contents());
            for (uint32_t i = 0; i < 64; i += 4) w[i] = 6, w[i + 1] = 1, w[i + 2] = 0, w[i + 3] = 0;
        }
        Sampler smpA, smpB;
        SamplerDesc linear;
        linear.minFilter = linear.magFilter = 1;
        CHECK(makeSampler(SamplerDesc{}, &smpA, &why) && makeSampler(linear, &smpB, &why));

        struct Recipe {
            Program program;
            uint32_t needs;
        };
        std::vector<Recipe> recipes;
        auto add = [&](const Stage& vs, const Stage& fs, const char* name, uint32_t needs) {
            Program p{vs, fs, Linkage{}, name};
            if (needs & kVertices) {
                p.vertexDescriptor.attributes[0] = VertexAttribute{VertexFormat::kFloat2, 0, 0};
                p.vertexDescriptor.attributes[1] = VertexAttribute{VertexFormat::kUChar4Normalized, 8, 0};
                p.vertexDescriptor.layouts[0] = VertexLayout{12, VertexStep::kPerVertex, 1};
            }
            // (every other one a program with a uid, as linkProgram makes them; the rest made by hand)
            if (allUids || recipes.size() % 2 == 0) {
                std::string no;
                CHECK(render::link(p.vs, p.fs, &p.link, &no));
                p.uid = newProgramUid();
            }
            recipes.push_back(Recipe{p, needs});
        };
        add(kernels::tri_flat_vs_air, kernels::tri_flat_fs_air, "flat", 0);
        add(kernels::tri_attr_vs_air, kernels::tri_tex_fs_air, "attr+tex", kVertices | kTexture);
        add(kernels::tri_attr_vs_air, kernels::tri_tex39_fs_air, "attr+tex39", kVertices | kTenSlots);
        add(kernels::tri_flat_vs_air, kernels::tri_cflat_fs_air, "flat+cflat", kFsUniforms);
        add(kernels::tri_flat_vs_air, kernels::tri_fb_fs_air, "flat+fb", 0);
        add(kernels::tri_flat_vs_air, kernels::tri_mrt_fs_air, "flat+mrt", 0);
        add(kernels::tri_upos_vs_air, kernels::tri_flat_fs_air, "upos+flat", kVsUbo);
        add(kernels::tri_dpos_vs_air, kernels::tri_flat_fs_air, "dpos+flat", kVsDevice);
        add(kernels::tri_vtex_vs_air, kernels::tri_color_fs_air, "vtex+color", kVsTexture);
        add(kernels::tri_arr_vs_air, kernels::tri_flat_fs_air, "arr+flat", 0);
        add(kernels::tri_base_vs_air, kernels::tri_color_fs_air, "base+color", 0);
        add(kernels::tri_flat_vs_air, kernels::tri_pvt_fs_air, "flat+pvt", 0);
        add(kernels::tri_oldcb_vs_air, kernels::tri_flat_fs_air, "oldcb+flat", kVsUbos);
        add(kernels::tri_olddb_vs_air, kernels::tri_flat_fs_air, "olddb+flat", kVsDevices);
        add(kernels::tri_dpos_vs_air, kernels::tri_dbuf_fs_air, "dpos+dbuf", kVsDevice | kFsDevice);
        add(kernels::tri_color_vs_air, kernels::tri_nullfb_fs_air, "color+nullfb", kManySlots);
        add(kernels::tri_attr_vs_air, kernels::tri_tex_fs_air, "attr+tex blended", kVertices | kTexture);
        recipes.back().program.blend.enabled = true;
        recipes.back().program.blend.srcRgb = 1, recipes.back().program.blend.dstRgb = 5;

        RenderPass passes[4];
        passes[0].color0 = colorA.get();
        passes[1].color0 = colorA.get();
        passes[1].more[0].texture = second.get();
        passes[1].more[0].load = LoadAction::kDontCare;
        passes[1].more[0].store = StoreAction::kDontCare;
        passes[2].color0 = colorB.get();
        passes[2].depth = depth.get();
        passes[3].color0 = colorB.get();
        passes[3].depth = depthStencil.get();
        passes[3].stencil = depthStencil.get();
        const double nan = std::numeric_limits<double>::quiet_NaN();

        for (uint32_t frame = 0; frame < frames; frame++) {
            auto cb = r.queue->commandBuffer();
            cb->setSkipRefused(true);
            cb->setGmem(mode != 0);
            cb->setGmemMinDraws(mode == 2 ? 8 : 0);
            cb->setGmemBinning(mode == 2);
            uint32_t which = rng.below(4), recipe = rng.below(uint32_t(recipes.size()));
            Draw lastDraw;
            Bindings lastVs, lastFs;
            bool haveLast = false;
            for (uint32_t i = 0; i < draws; i++) {
                Draw d;
                Bindings vsb, fsb;
                RenderPass pass;
                if (haveLast && rng.one(2)) {
                    // the draw before, with one thing changed (or nothing, where the thing is not this draw's to change)
                    d = lastDraw;
                    vsb = lastVs;
                    fsb = lastFs;
                    pass = passes[which];
                    const bool indexed = d.indexBuffer != nullptr;
                    StencilFace* const face = rng.one(2) ? &d.stencil.front : &d.stencil.back;
                    switch (rng.below(41)) {
                    case 0: d.viewport.x += 1; break;
                    case 1: d.viewport.y += 1; break;
                    case 2: d.viewport.width += 1; break;
                    case 3: d.viewport.height += 1; break;
                    case 4: d.viewport.znear = d.viewport.znear == 0 ? 0.25 : 0; break;
                    case 5: d.viewport.zfar = d.viewport.zfar == 1 ? 0.75 : 1; break;
                    case 6: d.scissor = d.scissor.width == 0 ? Scissor{1, 1, 8, 8} : Scissor{d.scissor.x ^ 1u, d.scissor.y, d.scissor.width, d.scissor.height}; break;
                    case 7: d.scissor = d.scissor.width == 0 ? Scissor{0, 0, W, H} : Scissor{d.scissor.x, d.scissor.y ^ 1u, d.scissor.width, d.scissor.height}; break;
                    case 8: if (d.scissor.width > 1) d.scissor.width -= 1; break;
                    case 9: if (d.scissor.height > 1) d.scissor.height -= 1; break;
                    case 10: d.cullMode = (d.cullMode + 1) % 3; break;
                    case 11: d.winding ^= 1u; break;
                    case 12: if (!indexed) d.vertexStart ^= 1u; break;
                    case 13: if (indexed) d.baseVertex ^= 1; break;
                    case 14: d.baseInstance ^= 1u; break;
                    case 15: if (indexed) d.indexType ^= 1u, d.indexOffset = 0; break;
                    case 16: if (recipes[recipe].needs & kVertices) d.indexBuffer = indexed ? nullptr : indices.get(), d.indexOffset = 0; break;
                    case 17: if (!vsb.buffers.empty() && (recipes[recipe].needs & kVertices)) vsb.buffers[0].offset ^= 12u; break;
                    case 18: d.blendColor[rng.below(4)] += 0.25f; break;
                    case 19: if (pass.depth != nullptr) d.depth.compare = (d.depth.compare + 1) % 8; break;
                    case 20: if (pass.depth != nullptr) d.depth.write = !d.depth.write; break;
                    case 21: if (pass.stencil != nullptr) d.stencil.enabled = !d.stencil.enabled; break;
                    case 22: if (pass.stencil != nullptr) face->compare = (face->compare + 1) % 8; break;
                    case 23: if (pass.stencil != nullptr) face->stencilFail = (face->stencilFail + 1) % 8; break;
                    case 24: if (pass.stencil != nullptr) face->depthFail = (face->depthFail + 1) % 8; break;
                    case 25: if (pass.stencil != nullptr) face->pass = (face->pass + 1) % 8; break;
                    case 26: if (pass.stencil != nullptr) face->readMask ^= 1u; break;
                    case 27: if (pass.stencil != nullptr) face->writeMask ^= 1u; break;
                    case 28: if (pass.stencil != nullptr) d.stencil.frontReference ^= 1u; break;
                    case 29: if (pass.stencil != nullptr) d.stencil.backReference ^= 1u; break;
                    case 30: for (const Texture*& t : fsb.textures) t = t == icon.get() ? photo.get() : t == photo.get() ? icon.get() : t; break;
                    case 31: for (const Sampler*& m : fsb.samplers) m = m == &smpA ? &smpB : &smpA; break;
                    case 32: if (recipes[recipe].needs & kFsUniforms) fsb.buffers[0].offset ^= 16u; break;
                    case 33: d.instanceCount = d.instanceCount == 1 ? 2 : 1; break;
                    case 34: d.vertexCount = d.vertexCount == 6 ? 3 : 6; break;
                    case 35: d.primitive = Primitive((uint32_t(d.primitive) + 1) % 5); break;
                    case 36: if (indexed) d.indexOffset ^= 4u; break;
                    case 37: for (const Texture*& t : vsb.textures) t = t == icon.get() ? photo.get() : icon.get(); break;
                    case 38: if (recipes[recipe].needs & (kVsUbo | kVsUbos)) vsb.buffers[0].offset ^= 16u; break;
                    case 39:
                        if (recipes[recipe].needs & kVertices) vsb.buffers[0].buffer = vsb.buffers[0].buffer == vertices.get() ? otherVertices.get() : vertices.get();
                        break;
                    default: break;   // (the same draw again)
                    }
                } else {
                    // the pass stays for a while, the program for a draw or a few
                    if (rng.one(12)) which = rng.below(4);
                    if (!rng.one(3)) recipe = rng.below(uint32_t(recipes.size()));
                    pass = passes[which];
                    d.vertexCount = rng.one(4) ? 3 : 6;
                    d.viewport = Viewport{0, 0, double(W), double(H), 0, 1};
                    if (rng.one(3)) d.viewport = Viewport{double(rng.below(64)), double(rng.below(48)), double(8 + rng.below(56)), double(8 + rng.below(40)), 0, 1};
                    if (rng.one(3)) d.scissor = Scissor{rng.below(64), rng.below(48), 1 + rng.below(64), 1 + rng.below(48)};
                    if (rng.one(4)) d.cullMode = rng.below(3);
                    if (rng.one(4)) d.winding = rng.below(2);
                    if (rng.one(6)) d.instanceCount = 1 + rng.below(3);
                    if (rng.one(8)) d.baseInstance = rng.below(4);
                    if (rng.one(6)) d.primitive = Primitive(rng.below(5));
                    if (rng.one(8)) {
                        for (float& c : d.blendColor) c = float(rng.below(5)) / 4;
                    }
                    if (pass.depth != nullptr && rng.one(2)) {
                        d.depth.compare = rng.below(8);
                        d.depth.write = rng.one(2);
                    }
                    if (pass.stencil != nullptr && rng.one(2)) {
                        d.stencil.enabled = true;
                        for (StencilFace* face : {&d.stencil.front, &d.stencil.back}) {
                            face->compare = rng.below(8), face->stencilFail = rng.below(8), face->depthFail = rng.below(8), face->pass = rng.below(8);
                            face->readMask = rng.one(2) ? 0xff : rng.below(256), face->writeMask = rng.one(2) ? 0xff : rng.below(256);
                        }
                        d.stencil.frontReference = rng.below(4), d.stencil.backReference = rng.below(4);
                    }
                    const uint32_t needs = recipes[recipe].needs;
                    if (needs & kVertices) {
                        vsb.buffers = {Binding{rng.one(2) ? vertices.get() : otherVertices.get(), uint64_t(12 * rng.below(8))}};
                        if (rng.one(2)) {
                            d.indexBuffer = indices.get();
                            d.indexType = rng.one(4) ? 1 : 0;
                            d.indexOffset = uint64_t((d.indexType == 1 ? 4 : 2) * rng.below(16));
                            d.baseVertex = int32_t(rng.below(3));
                        } else {
                            d.vertexStart = rng.below(4);
                        }
                    }
                    if (needs & kTexture) {
                        fsb.textures = {rng.one(2) ? icon.get() : photo.get()};
                        fsb.samplers = {rng.one(2) ? &smpA : &smpB};
                    }
                    if (needs & kTenSlots) {
                        fsb.textures = {nullptr, nullptr, nullptr, icon.get()};
                        if (rng.one(2)) {
                            fsb.textures.resize(10, nullptr);
                            fsb.textures[9] = photo.get();
                        }
                        fsb.samplers.assign(2, rng.one(2) ? &smpA : &smpB);
                    }
                    if (needs & kManySlots) {
                        fsb.textures.assign(16, nullptr);
                        for (uint32_t t = 0; t < 16; t++) {
                            if (rng.one(3)) fsb.textures[t] = rng.one(2) ? icon.get() : photo.get();
                        }
                        fsb.samplers = {&smpA};
                    }
                    if (needs & kFsUniforms) fsb.buffers = {Binding{uniforms.get(), uint64_t(rng.one(2) ? 16 * rng.below(8) : 8 + 16 * rng.below(8))}};
                    if (needs & kVsUbo) vsb.buffers = {Binding{uniforms.get(), uint64_t(16 * rng.below(4))}};
                    if (needs & kVsUbos) vsb.buffers = {Binding{uniforms.get(), uint64_t(16 * rng.below(4))}, Binding{vertices.get(), uint64_t(16 * rng.below(4))}};
                    if (needs & kVsDevice) vsb.buffers = {Binding{device.get(), uint64_t(4 * rng.below(4))}};
                    if (needs & kVsDevices) vsb.buffers = {Binding{device.get(), uint64_t(4 * rng.below(4))}, Binding{otherDevice.get(), 0}};
                    if (needs & kFsDevice) fsb.buffers = {vsb.buffers[0]};   // (one table for both stages: the same buffer at the same index)
                    if (needs & kVsTexture) {
                        vsb.textures = {rng.one(2) ? icon.get() : photo.get()};
                        vsb.samplers = {&smpA};
                    }
                }
                lastDraw = d;
                lastVs = vsb;
                lastFs = fsb;
                haveLast = true;
                if (rng.one(24)) pass.load = LoadAction::kClear;
                Recipe& rc = recipes[recipe];
                // (a program changed after it was drawn with: by hand in place, one with a uid given another)
                if (rng.one(16)) {
                    rc.program.blend.enabled = !rc.program.blend.enabled;
                    rc.program.blend.srcRgb = 1, rc.program.blend.dstRgb = 5, rc.program.blend.srcAlpha = 1, rc.program.blend.dstAlpha = 5;
                    if (rc.program.uid != 0) rc.program.uid = newProgramUid();
                }
                // every fourth draw wrong in one thing (or in nothing, where the thing is not this draw's to get wrong)
                if (rng.one(4)) {
                    switch (rng.below(54)) {
                    case 0: d.viewport.width = 0; break;
                    case 1: d.viewport.height = -4; break;
                    case 2: d.viewport.x = 4096; break;
                    case 3: d.viewport.znear = nan; break;
                    case 4: d.viewport.y = -1000; break;
                    case 5: d.scissor = Scissor{100, 0, 64, 16}; break;
                    case 6: d.scissor = Scissor{0, 90, 8, 16}; break;
                    case 7: d.scissor = Scissor{0, 0, 8, 0}; break;
                    case 8: d.vertexCount = 0; break;
                    case 9: d.instanceCount = 0; break;
                    case 10: d.cullMode = 3; break;
                    case 11: d.winding = 2; break;
                    case 12: d.primitive = Primitive(7); break;
                    case 13: d.blendColor[2] = float(nan); break;
                    case 14: d.depth.compare = 9; break;
                    case 15: d.depth.compare = 1, d.depth.write = true; break;
                    case 16: d.stencil.enabled = true; break;
                    case 17: d.stencil.enabled = true, d.stencil.front.compare = 8; break;
                    case 18: d.indexBuffer = indices.get(), d.indexType = 2; break;
                    case 19: d.indexBuffer = indices.get(), d.indexType = 0, d.indexOffset = 3; break;
                    case 20: d.indexBuffer = indices.get(), d.indexOffset = 8192; break;
                    case 21: d.indexBuffer = indices.get(), d.indexType = 1, d.indexOffset = 4088, d.vertexCount = 6; break;
                    case 22: d.indexBuffer = indices.get(), d.baseVertex = -1; break;
                    case 23: vsb.buffers.clear(); break;
                    case 24: if (!vsb.buffers.empty()) vsb.buffers[0].offset = 8192; break;
                    case 25: if (!vsb.buffers.empty()) vsb.buffers[0].offset = 4090; break;
                    case 26: if (!vsb.buffers.empty()) vsb.buffers[0].buffer = small.get(), vsb.buffers[0].offset = 0, d.vertexStart = 40; break;
                    case 27: fsb.buffers.clear(); break;
                    case 28: if (!fsb.buffers.empty()) fsb.buffers[0].offset += 2; break;
                    case 29: if (!fsb.buffers.empty()) fsb.buffers[0].offset = 8192; break;
                    case 30: if (!fsb.buffers.empty()) fsb.buffers[0] = Binding{small.get(), 60}; break;
                    case 31: for (const Texture*& t : fsb.textures) t = t != nullptr ? integers.get() : nullptr; break;
                    case 32: if (!fsb.textures.empty()) fsb.textures.back() = unread.get(); break;
                    case 33: for (const Texture*& t : fsb.textures) t = t != nullptr ? volume.get() : nullptr; break;
                    case 34: fsb.textures.clear(); break;
                    case 35: fsb.samplers.clear(); break;
                    case 36: for (const Texture*& t : fsb.textures) t = t != nullptr ? pass.color0 : nullptr; break;
                    case 37: d.indirect = numbers.get(), d.indirectOffset = uint64_t(16 * rng.below(4)); break;
                    case 38: d.visibility = results.get(), d.visibilityMode = 1 + rng.below(2), d.visibilityOffset = uint64_t(8 * rng.below(8)), d.visibilityReset = rng.one(2); break;
                    case 39: for (const Texture*& t : vsb.textures) t = integers.get(); break;
                    case 40: if (!fsb.buffers.empty()) fsb.buffers[0] = Binding{otherDevice.get(), 0}; break;
                    case 41: if (vsb.buffers.size() > 1) vsb.buffers.pop_back(); break;
                    case 42: if (vsb.buffers.size() > 1) vsb.buffers[0].buffer = nullptr; break;
                    case 43: if (!vsb.buffers.empty()) vsb.buffers[0].offset += 1; break;
                    case 44: vsb.textures.clear(), vsb.samplers.clear(); break;
                    case 45: pass.more[1].texture = unread.get(); break;
                    case 46: d.vertexStart = 4000, d.indexBuffer = nullptr; break;
                    case 47: d.viewport.zfar = std::numeric_limits<double>::infinity(); break;
                    case 48: pass.slice = 1; break;
                    case 49: pass.more[0].slice = 1; break;
                    case 50: pass.more[0].level = 1; break;
                    case 51: pass.depthSlice = 1; break;
                    case 52: d.stencil.enabled = true, d.stencil.back.depthFail = 8; break;
                    case 53: pass.stencilStore = StoreAction::kStore; break;
                    }
                }
                const bool taken = cb->draw(rc.program, pass, d, vsb, fsb);
                out.said.push_back(taken ? std::string() : cb->error().empty() ? cb->lastSkipped() : "the buffer ended: " + cb->error());
                out.draws += taken ? 1 : 0;
            }
            out.kept += cb->counts().keptDraws;
            const CommandBuffer::Counts c = cb->counts();
            wait(r, cb);
            out.said.push_back("frame " + std::to_string(frame) + ": " + std::to_string(c.passes) + " passes, " + std::to_string(c.draws) + " draws, " +
                               std::to_string(c.gmemPasses) + " in tiles, " + std::to_string(c.tiles) + " tiles, status " + std::to_string(int(cb->status())) + " " +
                               cb->error() + " | " + cb->gmemRefusal());
        }
        out.faults = r.fake->stats().faults;
    }
    FakeGpu::submitHook = chain.next;
    FakeGpu::submitHookContext = chain.nextContext;
    CommandBuffer::keepStates(was);
    return out;
}
}  // namespace

// What a draw's kept state decides, it decides as the long way does: the same random frames recorded with every draw
// judged and written whole and with the programs' states kept come to the same draws taken, the same draws refused by
// the same sentences, the same passes, and word for word the same submissions.
TEST(random_frames_are_the_same_from_kept_states_as_written_whole) {
    uint32_t taken = 0, kept = 0, refused = 0;
    for (int mode = 0; mode < 3; mode++) {
        for (uint64_t seed = 0; seed < 64; seed++) {
            const bool allUids = seed % 2 == 1;
            const Told whole = randomFrames(seed * 3 + uint64_t(mode), false, mode, 3, 80, allUids);
            const Told fromKept = randomFrames(seed * 3 + uint64_t(mode), true, mode, 3, 80, allUids);
            CHECK_EQ(whole.kept, 0u);
            CHECK_EQ(whole.said.size(), fromKept.said.size());
            for (size_t i = 0; i < whole.said.size() && i < fromKept.said.size(); i++) {
                if (whole.said[i] == fromKept.said[i]) continue;
                CHECK(whole.said[i] == fromKept.said[i]);
                std::printf("       mode %d seed %llu, draw %zu: written whole \"%s\", from its kept state \"%s\"\n", mode, (unsigned long long)seed, i,
                            whole.said[i].c_str(), fromKept.said[i].c_str());
                break;
            }
            CHECK_EQ(whole.sums.submits, fromKept.sums.submits);
            CHECK_EQ(whole.sums.words, fromKept.sums.words);
            CHECK_EQ(whole.sums.draws, fromKept.sums.draws);
            CHECK_EQ(whole.sums.loads, fromKept.sums.loads);
            CHECK_EQ(whole.sums.stream, fromKept.sums.stream);
            CHECK_EQ(whole.sums.state, fromKept.sums.state);
            CHECK_EQ(whole.sums.named, fromKept.sums.named);
            CHECK_EQ(whole.sums.names, fromKept.sums.names);
            CHECK_EQ(whole.faults, fromKept.faults);
            taken += fromKept.draws;
            kept += fromKept.kept;
            for (const std::string& s : fromKept.said) refused += !s.empty() && s.compare(0, 6, "frame ") != 0 ? 1 : 0;
        }
    }
    std::printf("       %u draws taken, %u of them from a kept state; %u refused\n", taken, kept, refused);
    // (the test is of the kept states: most of the draws taken must be from one)
    CHECK(kept * 2 > taken);
    CHECK(refused * 10 > taken);
}

// A command buffer finds a program's state by the program's uid in a few slots; more programs than slots, drawn in
// turn, push each other out of them, and each is still drawn as itself (the runner's check against the long way).
TEST(more_programs_than_a_command_buffer_has_slots_for_are_each_drawn_as_themselves) {
    Rig r;
    auto t = texture(r, 64, 64, kTextureUsageRenderTarget | kTextureUsageRead);
    RenderPass pass;
    pass.color0 = t.get();
    std::vector<Program> programs;
    for (uint32_t i = 0; i < 200; i++) {
        Program p = linked(kernels::tri_flat_vs_air, i % 2 ? kernels::tri_flat_fs_air : kernels::tri_cflat_fs_air, "many");
        p.blend.enabled = (i / 2) % 2 != 0;
        p.blend.srcRgb = 1 + i % 5, p.blend.dstRgb = 5, p.blend.writeMask = 1 + i % 15;
        programs.push_back(p);
    }
    std::string why;
    auto uniforms = newBuffer(*r.gpu, 4096, &why);
    Bindings fsb;
    fsb.buffers = {Binding{uniforms.get(), 16}};
    uint32_t kept = 0;
    for (int frame = 0; frame < 2; frame++) {
        auto cb = r.queue->commandBuffer();
        for (int round = 0; round < 3; round++) {
            for (const Program& p : programs) CHECK(cb->draw(p, pass, quad(), Bindings{}, fsb));
        }
        kept += cb->counts().keptDraws;
        finish(r, cb);
    }
    // all but each program's first draw
    CHECK_EQ(kept, 2u * 3u * 200u - 200u);
    CHECK_EQ(r.fake->stats().faults, 0u);
}
