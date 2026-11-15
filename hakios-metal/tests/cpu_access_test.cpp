// The processor's accesses to GPU memory it caches, told to the kernel. An IOSurface's memory is mapped cacheable in
// this process; a kernel may clean and invalidate it only around the accesses it is told of, as the msm interface lets
// it. So -replaceRegion: and -getBytes: of a texture over a surface go
// between GEM_CPU_PREP (the GPU done with it, every line cleaned and invalidated) and GEM_CPU_FINI (the next submission
// that names it cleans what was written); a texture of the driver's own write-combined memory, which no cache holds, is
// written and read with nothing told. The fake counts the two requests: what is checked is that they are made, and
// where; that the kernel then maintains the caches is the kernel's.
#include <cstring>
#include <string>
#include <vector>

#include "hakios-metal/command.hpp"
#include "hakios-metal/fake_gpu.hpp"
#include "hakios-metal/texture.hpp"
#include "test.hpp"

using namespace hakios::metal;

namespace {
struct Rig {
    Rig() {
        auto f = std::make_unique<FakeGpu>();
        fake = f.get();
        gpu = std::make_unique<Gpu>(std::move(f));
    }
    FakeGpu* fake;
    std::unique_ptr<Gpu> gpu;
};
}  // namespace

TEST(a_surface_textures_cpu_accesses_are_told_to_the_kernel) {
    Rig r;
    const uint32_t kBGRA = 0x42475241u;
    r.fake->addSurface(5, 32, 16, 128, kBGRA);
    TextureDesc d;
    d.format = PixelFormat::kBGRA8Unorm;
    d.width = 32;
    d.height = 16;
    d.usage = kTextureUsageRead | kTextureUsageRenderTarget;
    std::string why;
    auto t = Texture::wrapSurface(*r.gpu, 5, d, &why);
    CHECK(t != nullptr);
    if (t == nullptr) return;
    CHECK(t->buffer()->bo().surface());
    std::vector<uint8_t> rows(32 * 4 * 16, 0x3c), back(rows.size(), 0);
    CHECK(t->replace(0, 0, 32, 16, rows.data(), 32 * 4, &why));
    CHECK_EQ(r.fake->stats().cpuPreps, 1u);
    CHECK_EQ(r.fake->stats().cpuFinis, 1u);
    CHECK(t->read(0, 0, 32, 16, back.data(), 32 * 4, &why));
    CHECK_EQ(r.fake->stats().cpuPreps, 2u);
    CHECK_EQ(r.fake->stats().cpuFinis, 2u);
    CHECK(back == rows);
    // what it wrote is in the surface's memory
    std::vector<uint8_t> surface = r.fake->surfaceBytes(5);
    CHECK(surface.size() >= 128 * 16 && surface[0] == 0x3c && surface[127] == 0x3c);
    // a refusal (a region past the texture) tells the kernel nothing: no access was made
    CHECK(!t->replace(0, 0, 33, 16, rows.data(), 33 * 4, &why));
    CHECK_EQ(r.fake->stats().cpuPreps, 2u);
    // the twin: a texture of the driver's own memory, write-combined, is written and read with nothing told
    auto own = Texture::make(*r.gpu, d, &why);
    CHECK(own != nullptr);
    if (own == nullptr) return;
    CHECK(!own->buffer()->bo().surface());
    CHECK(own->replace(0, 0, 32, 16, rows.data(), 32 * 4, &why));
    CHECK(own->read(0, 0, 32, 16, back.data(), 32 * 4, &why));
    CHECK_EQ(r.fake->stats().cpuPreps, 2u);
    CHECK_EQ(r.fake->stats().cpuFinis, 2u);
}
