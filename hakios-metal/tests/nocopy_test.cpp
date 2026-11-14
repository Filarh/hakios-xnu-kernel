// -newBufferWithBytesNoCopy: is the caller's memory, not a copy of it (19 §2.9, N-12). On the fake GPU a buffer made by
// kUserMemory is the program's own pages: what the GPU reads there is what the program wrote, whenever it wrote it, and
// what the GPU writes there is in the program's memory. Each check has a twin with the copy this driver used to make
// as the buffer was made, which reads what was there THEN: the check tells the two apart. The fake has no caches: that
// the kernel cleans and invalidates the processor's caches around the GPU's use of the pages (xnu patch 0380) is the
// kernel's, checked there.
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>

#include "hakios-metal/command.hpp"
#include "hakios-metal/fake_gpu.hpp"
#include "hakios-metal/render.hpp"
#include "hakios-metal/texture.hpp"
#include "test.hpp"

using namespace hakios::metal;

namespace {
struct Rig {
    explicit Rig(bool async = false, bool shares = true) {
        auto f = std::make_unique<FakeGpu>(async);
        fake = f.get();
        fake->setSharesMemory(shares);
        gpu = std::make_unique<Gpu>(std::move(f));
        std::string why;
        queue = CommandQueue::make(*gpu, &why);
    }
    ~Rig() { queue.reset(); }
    FakeGpu* fake;
    std::unique_ptr<Gpu> gpu;
    std::unique_ptr<CommandQueue> queue;
};

// whole pages of the program's own, as Metal asks the caller of -newBufferWithBytesNoCopy: for
struct Pages {
    explicit Pages(size_t bytes) : size(bytes) {
        void* p = nullptr;
        if (posix_memalign(&p, 4096, bytes) == 0) at = static_cast<uint8_t*>(p);
        if (at != nullptr) memset(at, 0, bytes);
    }
    ~Pages() { free(at); }
    uint8_t* at = nullptr;
    size_t size;
};

bool all(const uint8_t* p, size_t n, uint8_t v) {
    for (size_t i = 0; i < n; i++) {
        if (p[i] != v) return false;
    }
    return true;
}

bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

void run(const std::shared_ptr<CommandBuffer>& cb) {
    cb->commit();
    cb->waitUntilCompleted();
    CHECK(cb->status() == CommandBuffer::kCompleted);
}
}  // namespace

// The GPU reads what the program wrote after the buffer was made, and the program finds what the GPU wrote. The twin:
// a buffer of the driver's own with the bytes copied in when it was made (what -newBufferWithBytesNoCopy: did) gives the
// GPU the bytes of then.
TEST(a_no_copy_buffer_is_the_programs_own_memory) {
    Rig r;
    Pages mem(3 * 4096);
    CHECK(mem.at != nullptr);
    if (mem.at == nullptr) return;
    const uint64_t allocatedBefore = r.gpu->allocatedBytes();
    std::string why;
    auto nc = newBufferNoCopy(*r.gpu, mem.at, mem.size, nullptr, &why);
    CHECK(nc != nullptr);
    if (nc == nullptr) return;
    CHECK(nc->contents() == mem.at);
    CHECK(nc->programMemory());
    CHECK_EQ(nc->length(), uint64_t(mem.size));
    CHECK(nc->gpuAddress() != 0);
    CHECK_EQ(r.gpu->allocatedBytes(), allocatedBefore);   // the program's memory, not the device's allocation
    // the twin, made the old way at the same moment: a copy of what the memory held then (zeros)
    auto copied = newBuffer(*r.gpu, mem.size, &why);
    memcpy(copied->contents(), mem.at, mem.size);

    memset(mem.at, 0x5a, mem.size);   // written after both were made: another process drawing the image, say
    auto fromShared = newBuffer(*r.gpu, mem.size, &why), fromCopy = newBuffer(*r.gpu, mem.size, &why);
    auto cb = r.queue->commandBuffer();
    CHECK(cb->copy(*nc, 0, *fromShared, 0, mem.size));
    CHECK(cb->copy(*copied, 0, *fromCopy, 0, mem.size));
    run(cb);
    CHECK(all(static_cast<uint8_t*>(fromShared->contents()), mem.size, 0x5a));
    CHECK(all(static_cast<uint8_t*>(fromCopy->contents()), mem.size, 0x00));   // the twin: the bytes of then

    // what the GPU writes into it is in the program's memory, and only there
    auto cb2 = r.queue->commandBuffer();
    CHECK(cb2->fill(*nc, 4096, 4096, 0xc3));
    CHECK(cb2->fill(*copied, 4096, 4096, 0xc3));
    run(cb2);
    CHECK(all(mem.at, 4096, 0x5a));
    CHECK(all(mem.at + 4096, 4096, 0xc3));
    CHECK(all(mem.at + 8192, 4096, 0x5a));
    CHECK(all(static_cast<uint8_t*>(copied->contents()) + 4096, 4096, 0xc3));
    // the submissions named the program's pages, the copy read (1) and the fill written (2)
    CHECK_EQ(r.fake->stats().faults, 0u);
    nc.reset();
}

// Memory off a page is refused before the kernel is asked, by the sentence, and nothing is made.
TEST(a_no_copy_buffer_off_a_page_is_refused_by_name) {
    Rig r;
    Pages mem(2 * 4096);
    const uint32_t live = r.fake->liveBuffers();
    std::string why;
    CHECK(newBufferNoCopy(*r.gpu, mem.at + 64, 4096, nullptr, &why) == nullptr);
    CHECK(has(why, "must start on a page and be whole pages"));
    why.clear();
    CHECK(newBufferNoCopy(*r.gpu, mem.at, 100, nullptr, &why) == nullptr);
    CHECK(has(why, "must start on a page and be whole pages"));
    why.clear();
    CHECK(newBufferNoCopy(*r.gpu, nullptr, 4096, nullptr, &why) == nullptr);
    CHECK(has(why, "must start on a page"));
    CHECK_EQ(r.fake->liveBuffers(), live);
    // the twin: the same memory on its page is taken
    auto ok = newBufferNoCopy(*r.gpu, mem.at, 4096, nullptr, &why);
    CHECK(ok != nullptr);
    CHECK_EQ(r.fake->liveBuffers(), live + 1);
}

// A kernel that cannot share a program's memory with the GPU gets no buffer at all, by name: a copy would be a buffer
// that is not the caller's memory, which is what N-12 was. The twin is the same kernel with the request served.
TEST(without_the_kernels_sharing_a_no_copy_buffer_is_refused_not_copied) {
    Rig old(false, false);
    Pages mem(4096);
    std::string why;
    CHECK(!old.gpu->sharesMemory());
    const uint32_t live = old.fake->liveBuffers();
    CHECK(newBufferNoCopy(*old.gpu, mem.at, 4096, nullptr, &why) == nullptr);
    CHECK(has(why, "needs xnu patch 0380"));
    CHECK_EQ(old.fake->liveBuffers(), live);
    int error = 0;
    CHECK(old.gpu->userBo(mem.at, 4096, &error) == nullptr);
    CHECK_EQ(error, ENOSYS);
    Rig now(false, true);
    CHECK(now.gpu->sharesMemory());
    CHECK(newBufferNoCopy(*now.gpu, mem.at, 4096, nullptr, &why) != nullptr);
}

// Metal's deallocator: once the buffer is gone AND every command buffer that used it has completed -- not when the
// caller lets go of it while the GPU still has it (the GPU is held back here so that it cannot have finished).
TEST(the_deallocator_runs_after_the_last_command_buffer_that_used_it) {
    Rig r(true);
    Pages mem(4096);
    std::atomic<int> released{0};
    std::string why;
    auto nc = newBufferNoCopy(*r.gpu, mem.at, 4096, [&] { released++; }, &why);
    CHECK(nc != nullptr);
    if (nc == nullptr) return;
    auto dst = newBuffer(*r.gpu, 4096, &why);
    memset(mem.at, 0x77, 4096);
    r.fake->pause();
    auto cb = r.queue->commandBuffer();
    CHECK(cb->copy(*nc, 0, *dst, 0, 4096));
    cb->keep(nc);
    nc.reset();                       // the caller's last reference, while the GPU has not run it
    cb->commit();
    CHECK_EQ(released.load(), 0);     // the command buffer holds it
    r.fake->resume();
    cb->waitUntilCompleted();
    CHECK(cb->status() == CommandBuffer::kCompleted);
    CHECK_EQ(released.load(), 1);
    CHECK(all(static_cast<uint8_t*>(dst->contents()), 4096, 0x77));
    // the twin: a buffer the caller lets go of with no command buffer holding it is released at once
    std::atomic<int> again{0};
    auto idle = newBufferNoCopy(*r.gpu, mem.at, 4096, [&] { again++; }, &why);
    idle.reset();
    CHECK_EQ(again.load(), 1);
}

// QuartzCore's client-storage image: a no-copy buffer over the image's pages and a texture over its rows at an offset
// (-newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:). What a copy out of the texture reads is the
// image's memory as it is when the GPU runs, written after the texture was made; the twin's texture over a copy made
// with the buffer reads the bytes of then.
TEST(a_texture_over_a_no_copy_buffer_reads_the_programs_rows) {
    Rig r;
    Pages mem(4096 * 4);
    std::string why;
    auto nc = newBufferNoCopy(*r.gpu, mem.at, mem.size, nullptr, &why);
    auto copied = newBuffer(*r.gpu, mem.size, &why);
    memcpy(copied->contents(), mem.at, mem.size);
    CHECK(nc != nullptr && copied != nullptr);
    if (nc == nullptr || copied == nullptr) return;
    TextureDesc d;
    d.format = PixelFormat::kBGRA8Unorm;
    d.width = 40;
    d.height = 30;
    d.usage = kTextureUsageRead;
    const uint64_t offset = 64, pitch = 192;   // a header before the pixels, rows padded to 64 bytes: CA's Shmem bitmap
    auto image = Texture::overBuffer(nc, offset, pitch, d, &why);
    auto imageOfCopy = Texture::overBuffer(copied, offset, pitch, d, &why);
    CHECK(image != nullptr && imageOfCopy != nullptr);
    if (image == nullptr || imageOfCopy == nullptr) return;
    CHECK(image->overBufferRows());
    CHECK_EQ(image->gpuAddress(), nc->gpuAddress() + offset);
    // the image arrives after the texture was made
    for (uint32_t y = 0; y < d.height; y++) {
        for (uint32_t x = 0; x < d.width; x++) {
            uint8_t* px = mem.at + offset + y * pitch + 4 * x;
            px[0] = uint8_t(x);
            px[1] = uint8_t(y);
            px[2] = 0x80;
            px[3] = 0xff;
        }
    }
    TextureDesc o = d;
    o.usage = kTextureUsageRead | kTextureUsageRenderTarget;
    auto out = Texture::make(*r.gpu, o, &why), outOfCopy = Texture::make(*r.gpu, o, &why);
    CHECK(out != nullptr && outOfCopy != nullptr);
    if (out == nullptr || outOfCopy == nullptr) return;
    auto cb = r.queue->commandBuffer();
    CHECK(cb->blit(*image, 0, render::BlitRect{0, 0, d.width, d.height}, *out, 0, render::BlitRect{0, 0, d.width, d.height}, false));
    CHECK(cb->blit(*imageOfCopy, 0, render::BlitRect{0, 0, d.width, d.height}, *outOfCopy, 0, render::BlitRect{0, 0, d.width, d.height}, false));
    run(cb);
    CHECK_EQ(r.fake->stats().faults, 0u);
    bool same = true, oldBytes = true;
    for (uint32_t y = 0; y < d.height; y++) {
        const uint8_t* a = static_cast<const uint8_t*>(out->contents()) + y * out->bytesPerRow();
        const uint8_t* b = static_cast<const uint8_t*>(outOfCopy->contents()) + y * outOfCopy->bytesPerRow();
        same = same && memcmp(a, mem.at + offset + y * pitch, 4 * d.width) == 0;
        oldBytes = oldBytes && all(b, 4 * d.width, 0);
    }
    CHECK(same);
    CHECK(oldBytes);   // the twin
}
