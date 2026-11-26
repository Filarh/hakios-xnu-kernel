// Whether a draw's primitives can overlap (overlap.hpp), and what a sysmem draw that reads its target is drawn with for
// it: the proof alone, then through a command buffer and the fake GPU. SPDX-License-Identifier: MIT
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "fixtures/tri_attr_vs_air.hpp"
#include "fixtures/tri_fb_fs_air.hpp"
#include "fixtures/tri_flat_fs_air.hpp"
#include "fixtures/tri_flat_vs_air.hpp"
#include "hakios-metal/a6xx.hpp"
#include "hakios-metal/command.hpp"
#include "hakios-metal/fake_gpu.hpp"
#include "hakios-metal/overlap.hpp"
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

// CoreAnimation's fixed_vert_lph_spc (QuartzCore's default.metallib of 20H392, function constant 3 = 1) as this
// driver's compiler makes it for the Adreno: position = mvp_matrix ([[buffer(2)]], pushed to c0-c3) times attribute 0,
// a float4 in r0; the colour, attribute 3, in r1. With Mesa's listing of each instruction (tools/stagecc.cpp's stage):
constexpr uint32_t kMvpWords[] = {
    0x00000000, 0x204cc008,   // mov.u32u32 r2.x, 0
    0x00000000, 0x20598012,   // mov.u8u8 hr4.z, 0
    0x00001000, 0x4070000a,   // mul.f r2.z, c0.x, r0.x
    0x00001001, 0x4070000f,   // mul.f r3.w, c0.y, r0.x
    0x000a1004, 0x6380800a,   // mad.f32 r2.z, c1.x, r0.y, r2.z
    0x000f1005, 0x6380800f,   // mad.f32 r3.w, c1.y, r0.y, r3.w
    0x000a1008, 0x6381000b,   // mad.f32 r2.w, c2.x, r0.z, r2.z
    0x00000000, 0x204cc012,   // mov.u32u32 r4.z, 0
    0x00000000, 0x204cc013,   // mov.u32u32 r4.w, 0
    0x00001002, 0x40700910,   // (rpt1)mul.f r4.x, (r)c0.z, r0.x
    0x01800024, 0xc14c1113,   // stp.u8 p[r2.x+19], r4.z, 1
    0x000f1009, 0x6381000c,   // mad.f32 r3.x, c2.y, r0.z, r3.w
    0x20101006, 0x63808910,   // (rpt1)mad.f32 r4.x, (r)c1.z, r0.y, (r)r4.x
    0x01800024, 0xc14c1112,   // stp.u8 p[r2.x+18], r4.z, 1
    0x2010100a, 0x6381090d,   // (rpt1)mad.f32 r3.y, (r)c2.z, r0.z, (r)r4.x
    0x200b100c, 0x63818b14,   // (rpt3)mad.f32 r5.x, (r)c3.x, r0.w, (r)r2.w
    0x00000000, 0x03000000,   // end
    0x00000000, 0x00000000,   // nop
};
constexpr uint32_t kMvpEndAt = 32;   // (the word the end begins at)

Stage mvpStage(const uint32_t* words, uint32_t numWords) {
    Stage s = kernels::tri_attr_vs_air;
    s.k.words = words;
    s.k.numWords = numWords;
    s.k.uboRanges[0] = KernelUboRange{2, 0, 0, 16};
    s.k.numUboRanges = 1;
    s.inputs[0] = StageInput{15, 0x00, 0xf, 0, 0, 0, 1, 0};   // attribute 0, r0.xyzw
    s.inputs[1] = StageInput{18, 0x04, 0xf, 0, 0, 0, 1, 0};   // attribute 3, r1.xyzw
    s.numInputs = 2;
    s.positionRegid = 20;                                     // r5.x
    return s;
}

// what CoreAnimation binds: its matrix for a W x H target (pixels to clip space, y down), read off the panel's traces
void mvp(float* m, float w, float h) {
    const float v[16] = {2 / w, 0, 0, 0, 0, -2 / h, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1};
    memcpy(m, v, sizeof v);
}

struct Scene {
    Rig r;
    std::shared_ptr<Buffer> vertices, constants, indices;
    Program program;
    Bindings vs;
    Draw draw;
    render::OverlapWitness w;

    // FLOATS a vertex position (2 or 4), STRIDE bytes a vertex
    Scene(uint32_t floats, uint32_t stride, const uint32_t* words = kMvpWords, uint32_t numWords = sizeof kMvpWords / 4)
        : program{mvpStage(words, numWords), kernels::tri_fb_fs_air, Linkage{}, "mvp"}, floats_(floats), stride_(stride) {
        std::string why;
        vertices = newBuffer(*r.gpu, 65536, &why);
        constants = newBuffer(*r.gpu, 4096, &why);
        indices = newBuffer(*r.gpu, 4096, &why);
        CHECK(vertices != nullptr && constants != nullptr && indices != nullptr);
        memset(vertices->contents(), 0, 65536);
        memset(constants->contents(), 0, 4096);
        mvp(static_cast<float*>(constants->contents()) + 16, 750, 1624);
        program.vertexDescriptor.attributes[0] = VertexAttribute{floats == 2 ? VertexFormat::kFloat2 : VertexFormat::kFloat4, 0, 1};
        program.vertexDescriptor.attributes[3] = VertexAttribute{VertexFormat::kHalf4, 16, 1};
        program.vertexDescriptor.layouts[1] = VertexLayout{stride, VertexStep::kPerVertex, 1};
        vs.buffers = {Binding{}, Binding{vertices.get(), 256}, Binding{constants.get(), 64}};
        draw.viewport = Viewport{0, 0, 750, 1624, 0, 1};
    }
    void vertex(uint32_t i, float x, float y) {
        float v[4] = {x, y, 0, 1};
        memcpy(static_cast<uint8_t*>(vertices->contents()) + 256 + size_t(i) * stride_, v, 4 * floats_);
    }
    void set(std::initializer_list<std::pair<float, float>> xy, Primitive primitive = Primitive::kTriangle) {
        uint32_t i = 0;
        for (const auto& p : xy) vertex(i++, p.first, p.second);
        draw.vertexCount = i;
        draw.primitive = primitive;
    }
    render::Overlap answer() {
        w.clear();
        return render::primitivesOverlap(program, draw, vs, &w);
    }

private:
    uint32_t floats_, stride_;
};

const render::Overlap kNever = render::Overlap::kNever, kMay = render::Overlap::kMay;

}  // namespace

// The two triangles of a quad never overlap; two that do, or that the proof does not cover, may.
TEST(a_quads_two_triangles_are_proved_apart_and_nothing_less_is) {
    Scene s(4, 48);
    // (CoreAnimation's order for a layer's rectangle, as the traces print it)
    s.set({{320, 807}, {409, 807}, {409, 907}, {409, 907}, {320, 907}, {320, 807}});
    CHECK(s.answer() == kNever);
    CHECK_EQ(s.w.spans.size(), size_t(2));   // (the matrix and the six positions)
    CHECK(s.w.holds(0, s.w.spans.size()));
    // ...for the bytes it read: a vertex written after is seen, and so is the matrix
    s.vertex(4, 409, 808);
    CHECK(!s.w.holds(0, s.w.spans.size()));
    s.vertex(4, 320, 907);
    CHECK(s.w.holds(0, s.w.spans.size()));
    static_cast<float*>(s.constants->contents())[16] = 1;
    CHECK(!s.w.holds(0, s.w.spans.size()));
    mvp(static_cast<float*>(s.constants->contents()) + 16, 750, 1624);

    // one triangle, whatever it is: nothing read
    s.draw.vertexCount = 3;
    CHECK(s.answer() == kNever);
    CHECK_EQ(s.w.spans.size(), size_t(0));
    s.draw.vertexCount = 5;
    CHECK(s.answer() == kNever);
    // ...but not two instances of it, nor counts the GPU reads
    s.draw.instanceCount = 2;
    CHECK(s.answer() == kMay);
    s.draw.instanceCount = 1;
    s.draw.indirect = s.indices.get();
    CHECK(s.answer() == kMay);
    s.draw.indirect = nullptr;

    // the second triangle on the first one's side of the shared edge: they overlap
    s.set({{320, 807}, {409, 807}, {409, 907}, {409, 907}, {320, 807}, {409, 812}});
    CHECK(s.answer() == kMay);
    CHECK_EQ(s.w.spans.size(), size_t(0));   // (nothing kept of a draw left in order)
    CHECK_EQ(s.w.bytes.size(), size_t(0));
    // two triangles that share a corner alone, apart as they are: not what is proved
    s.set({{100, 100}, {200, 100}, {150, 150}, {150, 150}, {100, 200}, {200, 200}});
    CHECK(s.answer() == kMay);
    // the same triangle twice
    s.set({{100, 100}, {200, 100}, {150, 150}, {100, 100}, {200, 100}, {150, 150}});
    CHECK(s.answer() == kMay);
    // three triangles, the first two a quad
    s.set({{320, 807}, {409, 807}, {409, 907}, {409, 907}, {320, 907}, {320, 807}, {0, 0}, {10, 0}, {0, 10}});
    CHECK(s.answer() == kMay);
    // a quad one of whose halves is thinner than the rasterizer's grid can keep on its side of the diagonal
    s.set({{100, 100}, {150.001f, 150}, {200, 200}, {200, 200}, {100, 200}, {100, 100}});
    CHECK(s.answer() == kMay);
    // ...and one that is a hair fatter than that margin asks (the diagonal 141 pixels, the corner 0.1 off it)
    s.set({{100, 100}, {150.1f, 150}, {200, 200}, {200, 200}, {100, 200}, {100, 100}});
    CHECK(s.answer() == kNever);
    // a quad as a strip of four
    s.set({{320, 807}, {409, 807}, {320, 907}, {409, 907}}, Primitive::kTriangleStrip);
    CHECK(s.answer() == kNever);
    // ...and a strip of five is three triangles
    s.set({{320, 807}, {409, 807}, {320, 907}, {409, 907}, {320, 1000}}, Primitive::kTriangleStrip);
    CHECK(s.answer() == kMay);
}

TEST(an_indexed_quad_is_proved_from_its_indices_and_a_restart_among_them_is_not) {
    Scene s(4, 48);
    s.set({{320, 807}, {409, 807}, {409, 907}, {320, 907}});
    auto* ix = static_cast<uint16_t*>(s.indices->contents());
    const uint16_t quad[8] = {7, 7, 0, 1, 2, 2, 3, 0};
    memcpy(ix, quad, sizeof quad);
    s.draw.indexBuffer = s.indices.get();
    s.draw.indexOffset = 4;
    s.draw.indexType = 0;
    s.draw.vertexCount = 6;
    CHECK(s.answer() == kNever);
    CHECK_EQ(s.w.spans.size(), size_t(3));   // (the matrix, the indices, the four positions)
    ix[5] = 1;                               // (2, 1, 0 for the second triangle: the first again)
    CHECK(!s.w.holds(0, s.w.spans.size()));
    CHECK(s.answer() == kMay);
    ix[5] = 2;
    ix[6] = 0xffff;                          // (a restart: other primitives than the count says)
    CHECK(s.answer() == kMay);
    ix[6] = 3;
    // a base vertex moves which vertices they are
    s.draw.baseVertex = 1;
    s.vertex(4, 320, 907);
    s.vertex(1, 320, 807);
    s.vertex(2, 409, 807);
    s.vertex(3, 409, 907);
    CHECK(s.answer() == kNever);
    s.draw.baseVertex = -1;
    CHECK(s.answer() == kMay);
    // 32-bit indices
    s.draw.baseVertex = 0;
    s.set({{320, 807}, {409, 807}, {409, 907}, {320, 907}});
    const uint32_t wide[6] = {0, 1, 2, 2, 3, 0};
    memcpy(s.indices->contents(), wide, sizeof wide);
    s.draw.indexOffset = 0;
    s.draw.indexType = 1;
    s.draw.vertexCount = 6;
    CHECK(s.answer() == kNever);
    // indices past the buffer
    s.draw.indexOffset = 4096 - 8;
    CHECK(s.answer() == kMay);
}

// What the GPU would clip, or cannot be placed, is not proved.
TEST(a_quad_that_could_be_clipped_keeps_its_order) {
    Scene s(4, 48);
    const auto quad = [&](float x0, float y0, float x1, float y1) { s.set({{x0, y0}, {x1, y0}, {x1, y1}, {x1, y1}, {x0, y1}, {x0, y0}}); };
    float* m = static_cast<float*>(s.constants->contents()) + 16;
    quad(-200, -300, 900, 2000);   // (past the target, inside the guardband)
    CHECK(s.answer() == kNever);
    quad(320, 807, 40000, 907);    // (past the guardband: window x beyond 32767)
    CHECK(s.answer() == kMay);
    quad(320, 807, 409, 907);
    m[15] = 0;                     // w = 0
    CHECK(s.answer() == kMay);
    m[15] = -1;
    CHECK(s.answer() == kMay);
    m[15] = 1;
    m[14] = -0.5f;                 // z = -0.5 for every vertex: all of it behind the near plane, nothing clipped in part
    CHECK(s.answer() == kNever);
    m[14] = 0;
    m[2] = 0.001f;                 // z grows with x: 0.32 to 0.41, inside 0 to w
    CHECK(s.answer() == kNever);
    m[14] = -0.35f;                // ...and now crosses the near plane inside the quad
    CHECK(s.answer() == kMay);
    m[14] = 0.8f;                  // ...or the far one
    CHECK(s.answer() == kMay);
    m[14] = 0;
    m[2] = 0;
    m[3] = 0.001f;                 // w grows with x: a perspective, every w above zero
    CHECK(s.answer() == kNever);
    m[3] = 0;
    m[12] = NAN;
    CHECK(s.answer() == kMay);
    mvp(m, 750, 1624);
    // a viewport of no size, a vertex that is not a number
    s.draw.viewport.width = 0;
    CHECK(s.answer() == kMay);
    s.draw.viewport.width = 750;
    s.vertex(1, INFINITY, 807);
    CHECK(s.answer() == kMay);
    quad(320, 807, 409, 907);
    // no buffer where the matrix is, or one too short for it; a vertex buffer too short
    Bindings kept = s.vs;
    s.vs.buffers[2].offset = 4096 - 32;
    CHECK(s.answer() == kMay);
    s.vs.buffers[2] = Binding{};
    CHECK(s.answer() == kMay);
    s.vs = kept;
    s.vs.buffers[1].offset = 65536 - 5 * 48;
    CHECK(s.answer() == kMay);
    s.vs = kept;
    // an attribute the hardware converts, a step that is not per vertex
    s.program.vertexDescriptor.attributes[0].format = VertexFormat::kShort4Normalized;
    CHECK(s.answer() == kMay);
    s.program.vertexDescriptor.attributes[0].format = VertexFormat::kFloat4;
    s.program.vertexDescriptor.layouts[1].step = VertexStep::kPerInstance;
    CHECK(s.answer() == kMay);
    s.program.vertexDescriptor.layouts[1].step = VertexStep::kPerVertex;
    CHECK(s.answer() == kNever);
    // two positions, the rest the format's (0, 0, 0, 1); and halfs
    s.program.vertexDescriptor.attributes[0].format = VertexFormat::kFloat2;
    CHECK(s.answer() == kNever);
    s.program.vertexDescriptor.attributes[0].format = VertexFormat::kHalf2;
    const uint16_t h320 = 0x5d00, h409 = 0x5e64, h807 = 0x624e, h907 = 0x6316;   // (320, 409, 807, 907 as halfs)
    const uint16_t hq[6][2] = {{h320, h807}, {h409, h807}, {h409, h907}, {h409, h907}, {h320, h907}, {h320, h807}};
    for (uint32_t i = 0; i < 6; i++) memcpy(static_cast<uint8_t*>(s.vertices->contents()) + 256 + i * 48, hq[i], 4);
    CHECK(s.answer() == kNever);
}

// The vertex stage's code: followed while it is affine in the attributes, and no further.
TEST(a_vertex_stage_the_walk_cannot_follow_keeps_the_order) {
    auto with = [&](std::initializer_list<uint32_t> extra, uint32_t positionRegid = 20) {
        // (EXTRA put where the end was, the end after it)
        std::vector<uint32_t> words(kMvpWords, kMvpWords + kMvpEndAt);
        words.insert(words.end(), extra.begin(), extra.end());
        words.push_back(0);
        words.push_back(0x03000000);
        Scene s(4, 48, words.data(), uint32_t(words.size()));
        s.program.vs.positionRegid = positionRegid;
        s.set({{320, 807}, {409, 807}, {409, 907}, {409, 907}, {320, 907}, {320, 807}});
        return s.answer();
    };
    CHECK(with({}) == kNever);
    CHECK(with({0x00000000, 0x00000000}) == kNever);                 // nop
    CHECK(with({0x00000000, 0x204cc008}) == kNever);                 // mov.u32u32 r2.x, 0: not the position's
    CHECK(with({0x00000014, 0x200cc018}) == kNever);                 // mov.u32u32 r6.x, r5.x: a copy elsewhere
    CHECK(with({0x00000030, 0x200cc014}) == kMay);                   // mov.u32u32 r5.x, r12.x: a register nothing here wrote (a vertex id's, say)
    CHECK(with({0x00010000, 0x40700014}) == kMay);                   // mul.f r5.x, r0.x, r0.y: not affine
    CHECK(with({0x00140014, 0x40100014}) == kNever);                 // add.f r5.x, r5.x, r5.x: twice the x, still affine
    CHECK(with({0x00144014, 0x40100014}) == kMay);                   // add.f r5.x, (neg)r5.x, r5.x: x = 0 for every vertex
    CHECK(with({0x00000000, 0x20598014}) == kMay);                   // mov.u8u8 hr5.x, 0: the half register is the full one's low half
    CHECK(with({0x00000014, 0x40700414}) == kMay);                   // (sat)mul.f r5.x, r5.x, r0.x
    CHECK(with({0x00000014, 0x40600014}) == kMay);                   // a cat2 on half registers writing r5.x
    CHECK(with({0x00000000, 0x80000014}) == kMay);                   // a cat4 (rcp and its like): not followed at all
    CHECK(with({0x00000000, 0xa0000014}) == kMay);                   // a cat5 (a texture sample)
    CHECK(with({0x00000008, 0x00800000}) == kMay);                   // a cat0 that is no nop: a jump
    CHECK(with({0x01800024, 0xc14c1113}) == kNever);                 // stp: writes no register
    CHECK(with({0x00c00001, 0xc0000014}) == kMay);                   // another cat6: a load
    CHECK(with({}, 0x30) == kMay);                                   // the position in a register the code never writes
    CHECK(with({0x00000008, 0x200cc014}) == kMay);                   // mov.u32u32 r5.x, r2.x: x the same constant for every vertex
    // a stage with no end
    Scene s(4, 48, kMvpWords, kMvpEndAt);
    s.set({{320, 807}, {409, 807}, {409, 907}, {409, 907}, {320, 907}, {320, 807}});
    CHECK(s.answer() == kMay);
}

// CoreAnimation's coverage spans: lines along pixel rows, at pixel centres.
TEST(lines_apart_are_proved_apart_and_runs_that_touch_are_not) {
    Scene s(2, 16);
    const Primitive lines = Primitive::kLine;
    // two runs on a row with clear pixels between them, a run on the next row under the first
    s.set({{0.5f, 180.5f}, {3.5f, 180.5f}, {5.5f, 180.5f}, {9.5f, 180.5f}, {0.5f, 181.5f}, {1.5f, 181.5f}}, lines);
    CHECK(s.answer() == kNever);
    CHECK_EQ(s.w.spans.size(), size_t(2));
    // the same out of order
    s.set({{0.5f, 181.5f}, {1.5f, 181.5f}, {5.5f, 180.5f}, {9.5f, 180.5f}, {3.5f, 180.5f}, {0.5f, 180.5f}}, lines);
    CHECK(s.answer() == kNever);
    // the next run begins on the pixel after the one the first ends before: one clear pixel, whichever end a line keeps
    s.set({{0.5f, 180.5f}, {3.5f, 180.5f}, {4.5f, 180.5f}, {9.5f, 180.5f}}, lines);
    CHECK(s.answer() == kNever);
    // runs that meet at a pixel: whose it is hangs on the hardware's rule for a line's ends
    s.set({{0.5f, 180.5f}, {3.5f, 180.5f}, {3.5f, 180.5f}, {9.5f, 180.5f}}, lines);
    CHECK(s.answer() == kMay);
    CHECK_EQ(s.w.bytes.size(), size_t(0));
    // runs over each other, a run twice
    s.set({{0.5f, 180.5f}, {5.5f, 180.5f}, {2.5f, 180.5f}, {9.5f, 180.5f}}, lines);
    CHECK(s.answer() == kMay);
    s.set({{0.5f, 180.5f}, {5.5f, 180.5f}, {0.5f, 180.5f}, {5.5f, 180.5f}}, lines);
    CHECK(s.answer() == kMay);
    // a line across rows, a line on the boundary between two rows
    s.set({{0.5f, 180.5f}, {5.5f, 183.5f}, {20.5f, 190.5f}, {25.5f, 190.5f}}, lines);
    CHECK(s.answer() == kMay);
    s.set({{0.5f, 180.0f}, {5.5f, 180.0f}, {20.5f, 190.5f}, {25.5f, 190.5f}}, lines);
    CHECK(s.answer() == kMay);
    // one line is one primitive; a strip of three points is two lines that share a pixel
    s.set({{0.5f, 180.5f}, {5.5f, 183.5f}}, lines);
    CHECK(s.answer() == kNever);
    CHECK_EQ(s.w.spans.size(), size_t(0));
    s.set({{0.5f, 180.5f}, {5.5f, 180.5f}, {9.5f, 180.5f}}, Primitive::kLineStrip);
    CHECK(s.answer() == kMay);
    // points
    s.set({{0.5f, 180.5f}, {5.5f, 180.5f}}, Primitive::kPoint);
    CHECK(s.answer() == kMay);
    // a text's worth: 768 runs, three a row; then one more than is looked at
    for (uint32_t i = 0; i < 1025; i++) {
        s.vertex(2 * i, 0.5f + 12 * (i % 3), 100.5f + i / 3);
        s.vertex(2 * i + 1, 8.5f + 12 * (i % 3), 100.5f + i / 3);
    }
    s.draw.primitive = lines;
    s.draw.vertexCount = 1536;
    CHECK(s.answer() == kNever);
    CHECK_EQ(s.w.bytes.size(), size_t(64 + 1535 * 16 + 8));
    s.vertex(1001, 12.5f, 100.5f + 500 / 3);   // (run 500 turned back over the run before it)
    CHECK(!s.w.holds(0, s.w.spans.size()));
    CHECK(s.answer() == kMay);
    s.vertex(1001, 8.5f + 12 * (500 % 3), 100.5f + 500 / 3);
    s.draw.vertexCount = 2048;
    CHECK(s.answer() == kNever);
    s.draw.vertexCount = 2050;
    CHECK(s.answer() == kMay);
}

// --- through a command buffer ---------------------------------------------------------------------------------------------

namespace {

std::shared_ptr<Texture> target(Rig& r, bool compressed = false) {
    TextureDesc d;
    d.format = compressed ? PixelFormat::kBGRA8Unorm : PixelFormat::kRGBA8Unorm;
    d.width = 64;
    d.height = 64;
    d.usage = kTextureUsageRenderTarget | kTextureUsageRead;
    d.compressed = compressed;
    std::string why;
    return Texture::make(*r.gpu, d, &why);
}

// tri_attr_vs (position = float4(attribute 0, 0, 1), 12-byte vertices) with tri_fb_fs, which reads colour attachment 0
struct Fetching {
    Rig r;
    std::shared_ptr<Texture> t;
    std::shared_ptr<Buffer> vertices;
    Program program{kernels::tri_attr_vs_air, kernels::tri_fb_fs_air, Linkage{}, "attr_fb"};
    RenderPass pass;
    Bindings vs;
    render::Options sysmem;

    explicit Fetching(bool compressed = false) {
        t = target(r, compressed);
        std::string why;
        vertices = newBuffer(*r.gpu, 4096, &why);
        CHECK(vertices != nullptr);
        memset(vertices->contents(), 0, 4096);
        program.vertexDescriptor.attributes[0] = VertexAttribute{VertexFormat::kFloat2, 0, 0};
        program.vertexDescriptor.attributes[1] = VertexAttribute{VertexFormat::kUChar4Normalized, 8, 0};
        program.vertexDescriptor.layouts[0] = VertexLayout{12, VertexStep::kPerVertex, 1};
        pass.color0 = t.get();
        vs.buffers = {Binding{vertices.get(), 0}};
        sysmem.gmem = false;
        r.fake->logDraws(true, true);
    }
    void set(uint32_t first, std::initializer_list<std::pair<float, float>> xy) {
        for (const auto& p : xy) {
            const float v[2] = {p.first, p.second};
            memcpy(static_cast<uint8_t*>(vertices->contents()) + size_t(first++) * 12, v, 8);
        }
    }
    void quad(uint32_t first = 0) { set(first, {{-0.5f, -0.5f}, {0.5f, -0.5f}, {0.5f, 0.5f}, {0.5f, 0.5f}, {-0.5f, 0.5f}, {-0.5f, -0.5f}}); }
    void folded(uint32_t first = 0) { set(first, {{-0.5f, -0.5f}, {0.5f, -0.5f}, {0.5f, 0.5f}, {0.5f, 0.5f}, {-0.5f, -0.5f}, {0.5f, -0.4f}}); }
    Draw draw(uint32_t count, uint32_t first = 0) const {
        Draw d;
        d.vertexStart = first;
        d.vertexCount = count;
        d.viewport = Viewport{0, 0, 64, 64, 0, 1};
        return d;
    }
    // GRAS_SC_CNTL.SINGLE_PRIM_MODE as each draw ran with it, of those run since the last asking
    size_t seen = 0;
    std::vector<uint32_t> modes() {
        using namespace a6xx;
        std::vector<uint32_t> out;
        const auto stats = r.fake->stats();
        for (; seen < stats.drawLog.size(); seen++) {
            const auto& l = stats.drawLog[seen];
            uint32_t value = ~0u;
            for (size_t i = 0; i < l.state.size() && l.state[i] != ~0ull; i++) {
                if (uint32_t(l.state[i] >> 32) == REG_A6XX_GRAS_SC_CNTL) value = uint32_t(l.state[i]);
            }
            out.push_back((value & A6XX_GRAS_SC_CNTL_SINGLE_PRIM_MODE__MASK) >> A6XX_GRAS_SC_CNTL_SINGLE_PRIM_MODE__SHIFT);
        }
        return out;
    }
    bool finish(const std::shared_ptr<CommandBuffer>& cb) {
        cb->commit();
        cb->waitUntilCompleted();
        const bool ok = cb->status() == CommandBuffer::kCompleted && r.fake->stats().faults == 0;
        if (!ok) std::printf("       %s | %s\n", cb->error().c_str(), r.fake->stats().lastFault.c_str());
        return ok;
    }
};

const uint32_t kOrdered = a6xx::FLUSH_PER_OVERLAP_AND_OVERWRITE, kPlain = a6xx::NO_FLUSH;

}  // namespace

// The request's three: a quad is drawn without the ordering mode, two triangles that overlap with it, three triangles with it.
TEST(a_sysmem_draw_that_reads_its_target_is_ordered_only_when_its_primitives_may_overlap) {
    Fetching f;
    if (f.t == nullptr) return;
    const Program flat{kernels::tri_flat_vs_air, kernels::tri_flat_fs_air, Linkage{}, "tri_flat"};
    f.quad(0);
    f.folded(6);
    f.quad(12);
    f.set(18, {{-0.9f, -0.9f}, {-0.8f, -0.9f}, {-0.9f, -0.8f}});
    auto cb = f.r.queue->commandBuffer();
    CHECK(cb->draw(f.program, f.pass, f.draw(6, 0), f.vs, Bindings{}, f.sysmem));    // a quad
    CHECK(cb->draw(f.program, f.pass, f.draw(6, 6), f.vs, Bindings{}, f.sysmem));    // two triangles over each other
    CHECK(cb->draw(f.program, f.pass, f.draw(9, 12), f.vs, Bindings{}, f.sysmem));   // three triangles
    CHECK(cb->draw(f.program, f.pass, f.draw(3, 0), f.vs, Bindings{}, f.sysmem));    // one
    CHECK(cb->draw(flat, f.pass, f.draw(6), Bindings{}, Bindings{}, f.sysmem));      // a draw that reads nothing
    CHECK(cb->draw(f.program, f.pass, f.draw(6, 0), f.vs, Bindings{}, f.sysmem));    // the quad again
    CHECK_EQ(cb->counts().passes, 1u);
    CHECK_EQ(cb->counts().fetchUnordered, 3u);
    CHECK(f.finish(cb));
    CHECK_EQ(cb->counts().fetchReordered, 0u);
    const std::vector<uint32_t> want = {kPlain, kOrdered, kOrdered, kPlain, kPlain, kPlain};
    CHECK(f.modes() == want);

    // the switches: every such draw ordered as before the proof (nothing looked at), or none (the old A/B)
    for (const uint32_t off : {uint32_t(CommandBuffer::kSysmemFetchOrdered), uint32_t(CommandBuffer::kSysmemFetchUnordered)}) {
        auto b = f.r.queue->commandBuffer();
        b->setGmemOff(off);
        CHECK(b->draw(f.program, f.pass, f.draw(6, 0), f.vs, Bindings{}, f.sysmem));
        CHECK(b->draw(f.program, f.pass, f.draw(6, 6), f.vs, Bindings{}, f.sysmem));
        CHECK(b->draw(f.program, f.pass, f.draw(3, 0), f.vs, Bindings{}, f.sysmem));
        CHECK_EQ(b->counts().fetchUnordered, 0u);
        CHECK(f.finish(b));
        const uint32_t all = off == CommandBuffer::kSysmemFetchOrdered ? kOrdered : kPlain;
        CHECK(f.modes() == (std::vector<uint32_t>{all, all, all}));
    }
}

// Metal lets the vertices be written until the commit: a quad folded over itself after its draw was encoded is drawn in
// order after all, and the draws around it as they were.
TEST(a_draw_whose_vertices_change_before_the_commit_gets_its_order_back) {
    for (const bool kept : {true, false}) {
        Fetching f;
        if (f.t == nullptr) return;
        render::Options o = f.sysmem;
        o.dedupRegs = kept;   // (each draw's state deduplicated against the pass's, or written whole)
        f.quad(0);
        f.quad(6);
        f.quad(12);
        auto cb = f.r.queue->commandBuffer();
        CHECK(cb->draw(f.program, f.pass, f.draw(6, 0), f.vs, Bindings{}, o));
        CHECK(cb->draw(f.program, f.pass, f.draw(6, 6), f.vs, Bindings{}, o));
        CHECK(cb->draw(f.program, f.pass, f.draw(6, 12), f.vs, Bindings{}, o));
        CHECK_EQ(cb->counts().fetchUnordered, 3u);
        f.folded(6);
        f.quad(12);           // (written again with what it held: nothing changed)
        CHECK(f.finish(cb));
        CHECK_EQ(cb->counts().fetchReordered, 1u);
        CHECK(f.modes() == (std::vector<uint32_t>{kPlain, kOrdered, kPlain}));
    }
}

// The same through the two places a draw's words move after they are written: a command buffer appended to another, and
// a pass recorded for tiles and for sysmem at once that goes on in sysmem.
TEST(a_fetch_left_unordered_is_found_again_where_its_words_end_up) {
    {
        Fetching f;
        if (f.t == nullptr) return;
        const Program flat{kernels::tri_flat_vs_air, kernels::tri_flat_fs_air, Linkage{}, "tri_flat"};
        f.quad(0);
        f.quad(6);
        auto first = f.r.queue->commandBuffer(), second = f.r.queue->commandBuffer();
        CHECK(first->draw(flat, f.pass, f.draw(3), Bindings{}, Bindings{}, f.sysmem));
        CHECK(first->draw(f.program, f.pass, f.draw(6, 0), f.vs, Bindings{}, f.sysmem));
        CHECK(second->draw(f.program, f.pass, f.draw(6, 6), f.vs, Bindings{}, f.sysmem));
        CHECK(second->draw(f.program, f.pass, f.draw(6, 0), f.vs, Bindings{}, f.sysmem));
        CHECK(first->append(*second));
        CHECK_EQ(first->counts().fetchUnordered, 3u);
        f.folded(6);
        CHECK(f.finish(first));
        CHECK_EQ(first->counts().fetchReordered, 1u);
        CHECK(f.modes() == (std::vector<uint32_t>{kPlain, kPlain, kOrdered, kPlain}));
    }
    {
        Fetching f;
        if (f.t == nullptr) return;
        std::string why;
        auto vis = newBuffer(*f.r.gpu, 64, &why);
        CHECK(vis != nullptr);
        if (vis == nullptr) return;
        const Program flat{kernels::tri_flat_vs_air, kernels::tri_flat_fs_air, Linkage{}, "tri_flat"};
        f.quad(0);
        f.quad(6);
        auto cb = f.r.queue->commandBuffer();
        cb->setGmemMinDraws(64);   // (both ways while its draws are few)
        const render::Options both;
        CHECK(cb->draw(flat, f.pass, f.draw(3), Bindings{}, Bindings{}, both));
        CHECK(cb->draw(f.program, f.pass, f.draw(6, 0), f.vs, Bindings{}, both));
        CHECK(cb->draw(f.program, f.pass, f.draw(6, 6), f.vs, Bindings{}, both));
        Draw counted = f.draw(3);  // (a visibility result: no tile can replay it, so the pass goes on in sysmem)
        counted.visibility = vis.get();
        counted.visibilityMode = 2;
        CHECK(cb->draw(flat, f.pass, counted, Bindings{}, Bindings{}, both));
        CHECK(cb->draw(f.program, f.pass, f.draw(6, 0), f.vs, Bindings{}, both));
        CHECK_EQ(cb->counts().passes, 1u);
        CHECK_EQ(cb->counts().fetchUnordered, 3u);
        f.folded(6);
        CHECK(f.finish(cb));
        CHECK_EQ(cb->counts().gmemPasses, 0u);
        CHECK_EQ(cb->counts().fetchReordered, 1u);
        // (the draw after the one put back in order writes no GRAS_SC_CNTL of its own -- its state is the pass's --
        // and runs in that mode too: slower, the same picture)
        CHECK(f.modes() == (std::vector<uint32_t>{kPlain, kPlain, kOrdered, kOrdered, kPlain}));
    }
    {
        // ...and one that ends in tiles: the sysmem side and its marks are dropped, the tiles ordered as ever
        Fetching f;
        if (f.t == nullptr) return;
        f.quad(0);
        auto cb = f.r.queue->commandBuffer();
        cb->setGmemMinDraws(2);
        const render::Options both;
        CHECK(cb->draw(f.program, f.pass, f.draw(6, 0), f.vs, Bindings{}, both));
        CHECK(cb->draw(f.program, f.pass, f.draw(6, 0), f.vs, Bindings{}, both));
        f.folded(0);
        CHECK(f.finish(cb));
        CHECK_EQ(cb->counts().fetchReordered, 0u);
        for (const uint32_t m : f.modes()) CHECK_EQ(m, uint32_t(a6xx::FLUSH_PER_OVERLAP));
    }
}

// A compressed target keeps the mode whatever the primitives are: there it also keeps a block's flags and data in step.
// (Counted, not run: the fake GPU's model draws into linear targets alone.)
TEST(a_fetch_from_a_compressed_target_keeps_its_order_unless_asked) {
    Fetching f(true);
    if (f.t == nullptr) {
        std::printf("       (no Mesa in this build: no compressed target to draw)\n");
        return;
    }
    f.quad(0);
    for (const uint32_t off : {0u, uint32_t(CommandBuffer::kSysmemFetchUnorderedUbwc)}) {
        auto cb = f.r.queue->commandBuffer();
        cb->setGmemOff(off);
        CHECK(cb->draw(f.program, f.pass, f.draw(6, 0), f.vs, Bindings{}));   // (GMEM allowed: render::gmemPlan refuses a compressed target)
        CHECK(cb->draw(f.program, f.pass, f.draw(3, 0), f.vs, Bindings{}));
        CHECK_EQ(cb->counts().gmemPasses, 0u);
        CHECK_EQ(cb->counts().fetchUnordered, off == 0 ? 0u : 2u);
    }
}
