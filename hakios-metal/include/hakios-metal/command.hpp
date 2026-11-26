// Metal's command model on the GPU: a queue, the command buffers committed to it in order, and the blit work the
// command processor can do on its own. It is what MTLCommandQueue, MTLCommandBuffer and MTLBlitCommandEncoder are
// in the plug-in (objc/), and it is testable without them.
//
//   CommandQueue        a submission queue of the driver's own; its command buffers complete in commit order, and
//                       a thread of its own waits for their fences and runs their completed handlers in that order
//   CommandBuffer       Metal's states (not enqueued, enqueued, committed, scheduled, completed, error); its
//                       packets go into a buffer of the queue's (returned to the queue once the GPU is done with it)
//   BlitEncoder         copy and fill of buffers, by the command processor: CP_MEMCPY and CP_MEM_WRITE (pm4.hpp).
//                       Offsets and lengths are in bytes and must be multiples of 4: the command processor moves
//                       words. Metal on macOS asks the same of copies; a blit that is not aligned is refused with
//                       the reason, and the command buffer ends in the error state, as Metal's does on a bad blit.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <algorithm>
#include <unordered_map>
#include <vector>

#include "hakios-metal/compute.hpp"
#include "hakios-metal/cull.hpp"
#include "hakios-metal/render.hpp"
#include "hakios-metal/gpu.hpp"
#include "hakios-metal/pm4.hpp"

namespace hakios::metal {

class CommandQueue;
class Texture;
namespace render {
// (render.cpp, the draw path: what a queue keeps of the programs drawn with it, what a command buffer has at hand of
// that, and what an open pass's stream last held)
struct StateCache;
struct DrawCache;
struct PassCache;
struct ProgramState;
}  // namespace render

// a buffer a program sees: a GEM object with its length (MTLBuffer's -length may be less than the object's size)
class Buffer {
public:
    Buffer(std::unique_ptr<Bo> bo, uint64_t length) : bo_(std::move(bo)), length_(length) {}
    // over the program's own memory (newBufferNoCopy): RELEASED runs once the Buffer is gone and its Bo closed, which is
    // after the last command buffer that used it has completed (each keeps it until then)
    Buffer(std::unique_ptr<Bo> bo, uint64_t length, std::function<void()> released)
        : bo_(std::move(bo)), length_(length), released_(std::move(released)), programMemory_(true) {}
    ~Buffer() {
        bo_.reset();
        if (released_) released_();
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    uint64_t length() const { return length_; }
    void* contents() const { return bo_->cpu(); }
    uint64_t gpuAddress() const { return bo_->gpuAddress(); }
    const Bo& bo() const { return *bo_; }
    // the program's memory (newBufferNoCopy), cached by the processor; every other buffer is the driver's, write-combined
    bool programMemory() const { return programMemory_; }
    Bo& bo() { return *bo_; }

private:
    std::unique_ptr<Bo> bo_;
    uint64_t length_;
    std::function<void()> released_;
    bool programMemory_ = false;
};

std::shared_ptr<Buffer> newBuffer(Gpu& gpu, uint64_t length, std::string* why);
// -newBufferWithBytesNoCopy:length:options:deallocator:: a Buffer that IS the program's memory [BYTES, BYTES + LENGTH),
// as Metal makes one -- the GPU reads what the program writes there before a command buffer that uses it is committed,
// whenever it wrote it, and the program sees what the GPU wrote there once that command buffer has completed. BYTES and
// LENGTH multiples of 4096, the GPU's page (Metal asks the caller for page-aligned memory). The kernel holds the pages
// while the Buffer lives and keeps the processor's caches and the GPU's view of them agreeing (xnu patch 0380); a kernel
// without that is refused by name, for a copy would be another buffer, not this one. RELEASED (Metal's deallocator):
// see Buffer. nullptr with *why.
std::shared_ptr<Buffer> newBufferNoCopy(Gpu& gpu, void* bytes, uint64_t length, std::function<void()> released, std::string* why);
// the live Buffer whose memory holds GPU address ADDRESS, and *OFFSET its offset in it; nullptr when none does. An
// argument buffer names a buffer member by the address it was encoded with (setBuffer:offset:atIndex:, or the
// gpuAddress an application writes itself), and this is how the draw finds the Buffer to bind and keep.
std::shared_ptr<Buffer> bufferAt(uint64_t address, uint64_t* offset);
// IOSurface SURFACE_ID's memory as a Buffer, *INFO what the kernel says of the surface. One Buffer per surface while it
// lives: a second ask for the same surface gets the same Buffer (the kernel's handle for it is one), so that the
// handle is closed once, by the last owner. The last four wrapped are kept past their last user, a surface wrapped
// again then answered without the kernel (the window server wraps its display's page anew every frame). nullptr with *why.
std::shared_ptr<Buffer> surfaceBuffer(Gpu& gpu, uint32_t surfaceId, SurfaceInfo* info, std::string* why);
// B, which surfaceBuffer gave for a texture that was then refused, not kept for the next wrap
void forgetSurfaceBuffer(const Gpu* gpu, const std::shared_ptr<Buffer>& b);

// A depth of setHsr's keeps its contents from pass to pass (anexo 32/03 P1): its Z levels are handed out in one
// decreasing run over all the passes drawn with it, so what an earlier pass left there is always behind (a larger Z)
// whatever a later pass draws, and is cleared only when it is new or the run has used kHsrMaxDraws levels. NEXT is the
// next draw's place in the run; OWNER the command buffer drawing with it and not yet committed (one at a time, so
// that the run's order is the order the GPU runs the buffers in: CommandBuffer::draw, commit).
struct HsrClock {
    std::mutex lock;
    uint32_t next = 0;
    bool fresh = true;              // never cleared: the first pass clears it
    const void* owner = nullptr;
};
class CommandBuffer : public std::enable_shared_from_this<CommandBuffer> {
public:
    enum Status { kNotEnqueued, kEnqueued, kCommitted, kScheduled, kCompleted, kError };

    Status status() const { return status_.load(); }
    const std::string& error() const { return error_; }

    // run after completion (or error), on the queue's thread, in the order of commit; only before commit
    void addCompletedHandler(std::function<void(CommandBuffer&)> handler);
    void enqueue();
    void commit();
    void waitUntilCompleted();
    // how long the GPU had it: from commit to the fence, in nanoseconds (Metal's GPUStartTime..GPUEndTime, roughly)
    int64_t gpuNanoseconds() const { return completedAt_ - committedAt_; }
    // what -commit cost the caller's thread, in nanoseconds of the transport's clock: the last pass's tail (closePass:
    // in tiles its draw-state groups and the tiles' stream), the copy of the stream into its command BO, and the whole
    // of CommandQueue::submit up to the kernel's answer; and the words handed to the GPU. Zero before the commit.
    struct SubmitCost { int64_t closeNs = 0, copyNs = 0, submitNs = 0; uint32_t words = 0; };
    const SubmitCost& submitCost() const { return submitCost_; }

    // --- the blit encoder's work, recorded into this command buffer ---
    bool copy(const Buffer& from, uint64_t fromOffset, const Buffer& to, uint64_t toOffset, uint64_t length);
    bool fill(const Buffer& buffer, uint64_t offset, uint64_t length, uint8_t value);
    // --- the compute encoder's work (compute.hpp): KERNEL over GROUPS work groups of LOCAL threads each (nullptr:
    // the kernel's own size), BUFFERS bound by index, IMMEDIATES in place of the kernel's constants (nullptr: its own).
    // The program and the descriptors go into buffers of their own that live until this completes.
    bool dispatch(const Kernel& kernel, const uint32_t groups[3], const uint32_t* local, const std::vector<Binding>& buffers,
                  const uint32_t* immediates = nullptr, const compute::Options& options = compute::Options{});
    // the same with textures and samplers (compute.hpp: Bindings); the textures' memory is used and kept as buffers are
    bool dispatch(const Kernel& kernel, const uint32_t groups[3], const uint32_t* local, const Bindings& bindings,
                  const uint32_t* immediates = nullptr, const compute::Options& options = compute::Options{});
    // the group counts three 32-bit words in ARGS at OFFSET (a multiple of 4), read when the GPU runs it
    bool dispatchIndirect(const Kernel& kernel, const Buffer& args, uint64_t offset, const uint32_t* local, const Bindings& bindings,
                          const uint32_t* immediates = nullptr, const compute::Options& options = compute::Options{});
    // --- the render encoder's work (render.hpp): one draw of PROGRAM into PASS's colour target, one complete pass per
    // call (begin, state, draw, end). The two programs' words and the pass's table go into buffers of their own that live
    // until this completes; the target's memory is used for write and kept. False with the reason recorded (render::encode's).
    bool draw(const Program& program, const RenderPass& pass, const Draw& draw, const Bindings& vsBindings, const Bindings& fsBindings,
              const render::Options& options = render::Options{});
    // a pass with no draw whose load action is Clear (render::encodeClear): the target filled with the clear colour, a
    // complete pass of its own. What an encoder that recorded no draw needs at its end; anything else is refused.
    bool clear(const RenderPass& pass, const render::Options& options = render::Options{});
    // -copyFromTexture:...toTexture:... and the scaled copies under it: rectangle S of SRC's mip level SL into rectangle D
    // of DST's level DL by the 2D engine, a pass of its own (render::encodeBlit says what it takes)
    bool blit(const Texture& src, uint32_t srcLevel, render::BlitRect s, const Texture& dst, uint32_t dstLevel, render::BlitRect d, bool linear,
              const render::Options& options = render::Options{});
    // the compressed textures among B's sampled ones made current (Texture::staging); a draw and a dispatch do it first
    bool freshen(const Bindings& b, const render::Options& options = {});
private:
    bool blitInto(const Texture& src, uint32_t srcLevel, render::BlitRect s, const Texture& dst, uint32_t dstLevel, render::BlitRect d, bool linear,
                  const render::Options& options);
public:
    // -generateMipmapsForTexture:: every level past 0 made from the one above it, halved with linear filtering, in order
    bool generateMipmaps(const Texture& texture, const render::Options& options = render::Options{});
    // a blit the caller could not record (a buffer not in the GPU's memory, say): the buffer ends in the error state
    void refuse(const std::string& why) { fail(why); }
    // a buffer it uses, kept alive until it completes (the owner may let go of it before that)
    // (by reference, and once: a draw names the same few buffers and textures as the draw before it, and a copy of the
    // pointer for each -- two atomic counts and a slot of a growing vector -- was most of what keeping them cost. One
    // kept a moment ago is found in RECENT and left alone; one pushed out of it is kept twice, which costs a slot.)
    void keep(const std::shared_ptr<const Buffer>& b) {
        if (b != nullptr && !recently(b.get())) kept_.push_back(b);
    }
    void keep(const std::shared_ptr<Buffer>& b) {
        if (b != nullptr && !recently(b.get())) kept_.push_back(b);
    }
    // anything else it uses (a compiled kernel's words), the same way
    template <class T>
    void keepAlive(const std::shared_ptr<T>& p) {
        if (p != nullptr && !recently(p.get())) keptAny_.push_back(p);
    }
    // OTHER's recorded work after this one's, as if it had been recorded here: its packets, what it names and what it
    // keeps alive -- a parallel render encoder's sub-encoders each record into a command buffer of their own, from any
    // thread in any order, and are appended in the order they were made. OTHER is left empty and is not committed; an
    // error it ended in is this one's. False after commit (of either).
    bool append(CommandBuffer& other);
    // What becomes of a draw, dispatch, copy or clear the encoder refuses (a binding it cannot serve, a shape it does not
    // take): by default the command buffer ends in error with the reason, as every test here expects; with skipRefused
    // that one piece of work is left out, counted and its reason kept, and the rest of the buffer still runs. Apple's
    // driver validates nothing and runs a frame whatever is wrong in it; a game whose every frame has one dispatch this
    // does not serve otherwise shows nothing at all (Resident Evil 4). Running out of memory, or work after commit,
    // still fails the buffer either way.
    void setSkipRefused(bool on) { skipRefused_ = on; }
    uint32_t skipped() const { return skipped_; }
    const std::string& lastSkipped() const { return lastSkipped_; }
    // the encoder's refusal of one piece of work, by the rule above: false either way
    bool refuseWork(const std::string& why);
    // What the buffer holds, counted as it is recorded: passes opened (each a pass start, its flushes and the wait for
    // idle at its tail, which is a fixed cost whatever is drawn), the draws in them, and dispatches, copies and clears
    // (each a pass of its own). What a frame's GPU time goes to is read off these beside the time (metal_fps).
    struct Counts {
        uint32_t passes = 0, draws = 0, dispatches = 0, copies = 0, clears = 0, gmemPasses = 0, tiles = 0, binnedPasses = 0;   // (gmemPasses of the passes drawn in tiles, binnedPasses of those binned)
        // ...of the draws, those encoded from what the queue keeps of their program (render.cpp: ProgramState) rather than
        // judged and written whole: all but the first with each program in each kind of pass, when all is well
        uint32_t keptDraws = 0;
        // setHsr's: the passes drawn with it, their draws by class (each drawn in the prepass, tested, or as a barrier),
        // and the prepasses written (segments that had an opaque draw)
        uint32_t hsrPasses = 0, hsrOpaque = 0, hsrHidden = 0, hsrBarriers = 0, hsrPrepasses = 0;
        uint32_t hsrClears = 0, hsrBusy = 0;   // (HsrClock: the depth cleared as a pass opened; a draw without HSR, the clock another buffer's)
        // ...and of those, the ones whose draw before them in the pass was of the same program, taken the same way
        uint32_t sameProgramDraws = 0;
    };
    const Counts& counts() const { return counts_; }
    // With profiling on, the command processor writes its always-on counter (CP_ALWAYS_ON_COUNTER, 19.2 MHz) at the start
    // of each pass and after its tail -- whose wait for idle makes the end the end -- for the first kMaxProfiled passes:
    // what each costs, beside what it was (a draw pass with its target's size and its draws, a copy, a clear, a
    // dispatch). Read once the buffer has completed. metal_fps names a frame's slowest passes with it.
    enum PassKind : uint32_t { kPassDraw = 0, kPassCopy = 1, kPassClear = 2, kPassDispatch = 3 };
    // (TOUCHED: the attachments a tiled pass's draws write or read, as a mask, and above them what the pass was tiled
    // for -- kPassReads a draw reads its target, kPassBelowBar tiled with fewer draws than the bar for what it holds
    // (render::gmemWorth), kPassNoLoad its tiles load nothing: what the plug-in's passes and timeline lines print as
    // "touched")
    // (WROTE, READ: of a tiled pass, the attachments a draw's fragment stage has an output for, and those a draw
    // reads through framebuffer fetch -- what TOUCHED's low byte is the union of)
    enum : uint16_t { kPassReads = 0x100, kPassBelowBar = 0x200, kPassNoLoad = 0x400 };
    struct PassTime { PassKind kind; uint32_t width, height, draws; double us; uint32_t tiles = 0; uint64_t tuneKey = 0; uint8_t attached = 0; uint16_t touched = 0; uint8_t wrote = 0, read = 0, loaded = 0, stored = 0; std::string sysmemWhy; int64_t samples = -1; uint32_t cullSeen = 0, cullCoverers = 0, culled = 0; uint64_t culledPixels = 0; };   // ATTACHED / TOUCHED: the colour attachments set, and those its draws wrote or read (a GMEM pass's)   // tiles 0: in sysmem; tuneKey: CommandQueue::preferGmem's; SAMPLES: setProfileSamples's count, -1 when not counted
    // (CULL_SEEN, CULL_COVERERS, CULLED, CULLED_PIXELS: setCull's, of a pass's draws -- those it looked at, those that
    // covered something, those left out (or, verifying, that would have been) and the pixels of their boxes)
    static constexpr uint32_t kMaxProfiled = 64;
    void setProfile(bool on) { profile_ = on; }
    // ...and the samples each pass's draws passed: the RB's running count copied out by ZPASS_DONE at the pass's start
    // and after its tail, as turnip's autotune takes it (tu_autotune.cc:642-733), into PassTime::samples. What a pass
    // writes through the RB is its samples times the bytes of what it writes (doubled when it blends): the measured half
    // of the choice between tiles and sysmem (dev-notes/investigacion/13 §10.4). Off by default: three packets at each
    // end of a pass.
    void setProfileSamples(bool on) { profileSamples_ = on; }
    // ...and each draw of a sysmem pass on its own, the first kMaxProfiledDraws: a wait for idle before it and after it,
    // the counter at both (so the draws no longer overlap: what each costs alone), with its program's name. Which
    // draws of a heavy frame are heavy is read off these (metal_fps=draws).
    struct DrawTime { std::string name; double us; uint32_t pass = 0, area[4] = {}; };   // PASS: its pass among passTimes'; AREA render::drawArea's
    static constexpr uint32_t kMaxProfiledDraws = 512;
    void setProfileDraws(bool on) { profileDraws_ = on; }
    std::vector<DrawTime> drawTimes() const;
    // render::Options::gmem for this buffer's draws besides each draw's own (both must allow it): the C API's switch
    void setGmem(bool on) { gmem_ = on; }
    // GMEM's refinements off, for a bisect on the phone: kGmemWholeArea tiles the whole window whatever the draws touch,
    // kGmemEveryAttachment loads and stores every attachment, kGmemFetchBarrier orders a fetch by the barrier,
    // kGmemPlainBody draws a binned pass from its draws' plain stream, every tile running every draw's state (no
    // draw-state groups), kGmemNoLeave has each tile read its draws to the last (no CP_SKIP_IB2_ENABLE_LOCAL),
    // kGmemOldTiles lays the bins as before turnip's layout (one column, 16 KiB shares), kGmemEveryBin keeps a bin
    // for every attachment of the pass whether its draws touch it or not, kGmemInlineBarriers leaves a fetch's
    // barrier in the draws' stream, run by every tile that walks past it, kGmemAllOrdered has no barrier between
    // draws and every draw of a pass that fetches in raster order (gmem.hpp: DrawStateFetch), kGmemByDrawsAlone
    // chooses tiles or sysmem for a pass recorded both ways by its draws against the bar and nothing else
    // (render::gmemWorth's criteria off), kSysmemFetchUnordered leaves the primitives of a sysmem draw that reads its
    // target unordered among themselves (render::Options::fbReadOverlap off).
    // (8, 16 and 32 are the plug-in's own: metal_gmem_off's untuned choice, no binning, null textures.)
    enum : uint32_t {
        kGmemWholeArea = 1, kGmemEveryAttachment = 2, kGmemFetchBarrier = 4, kGmemPlainBody = 64, kGmemNoLeave = 128, kGmemOldTiles = 256,
        kGmemEveryBin = 512, kGmemInlineBarriers = 1024, kGmemAllOrdered = 2048, kGmemByDrawsAlone = 4096, kSysmemFetchUnordered = 8192,
    };
    void setGmemOff(uint32_t bits) { gmemOff_ = bits; }
    // GMEM or sysmem for each pass by CommandQueue::preferGmem (each pass stamped for it) rather than GMEM whenever a
    // pass can have it: what the plug-in asks; off by default, so that a test gets the mode it asks for
    void setGmemTuned(bool on) { tuning_ = on; }
    // GMEM for a pass of at least DRAWS draws, sysmem for one of fewer; 0: not by the draws. The draws are counted as
    // they come, so the pass is recorded both ways until it closes -- and DRAWS is then the bar of a pass that loads
    // and stores all it touches: one that reads its target, or whose tiles need load or store less, is tiled with
    // fewer (render::gmemWorth). SpringBoard's passes of 63 to 161 draws took 13 to
    // 22 ms in GMEM against 17 to 31 in sysmem, its passes of 45 to 49 draws 8 to 10 against 6 to 8, its light ones 5
    // against 0.13.
    void setGmemMinDraws(uint32_t draws) { gmemMinDraws_ = draws; }
    // render::Options::gmemBinning for this buffer's draws besides each draw's own (either asks for it)
    void setGmemBinning(bool on) { gmemBinning_ = on; }
    // every fragment texture the null texture (1x1, always in the texture cache): wrong pictures, but a draw's time then
    // says what its shader costs without its texture reads -- a measurement switch, as setGmemOff is a bisect's
    void setNullFsTextures(bool on) { nullFsTextures_ = on; }
    // A measurement instrument (the plug-in's metal_hsrprobe): TAG names the next draw; cancelDraw(TAG) makes that
    // draw's draw packet a CP_NOP of the same length in every stream it went into, while its pass is still open -- its
    // state words stay, so what later draws inherit is unchanged. False when the draw is not in the open pass. What the
    // GPU then draws is no longer what was asked: for timing what a draw that cannot be seen costs, never in use.
    void setDrawTag(uint32_t tag) { drawTag_ = tag; }
    bool cancelDraw(uint32_t tag);
    uint32_t cancelledDraws() const { return cancelled_; }
    // Whole draws left out when what they draw cannot be seen (the plug-in's metal_cull; dev-notes/investigacion/anexos/
    // 33-panel-lateral/10-oclusion-exacta.md). A draw X of a pass is left out -- its draw packet a CP_NOP, its state in
    // place -- when every pixel it can touch is covered by rectangles that LATER draws of the same pass fill whole and
    // no draw between X and the last of them reads the pass's attachments. It rests on nothing the caller promises:
    //   X          writes colour 0 alone, no memory, no depth from its FS, no visibility result (render.cpp's
    //              hsrClassOf: not a barrier); triangles, one instance, not indirect, at most cull::kMaxVertices
    //   a coverer  besides: blending off, every channel written, no discard, no framebuffer fetch, one sample
    //              (hsrClassOf: opaque), no face culled, two triangles that are the halves of an axis-aligned rectangle
    //   a reader   a draw whose FS fetches the framebuffer, or that samples a texture in the memory of an attachment:
    //              nothing drawn before it is left out for what comes after it
    // and where the vertices land is worked out from the vertex stage's own instructions and the bytes it will fetch
    // (cull.hpp), X's box taken outwards and a coverer's rectangle inwards by more than the hardware can move a vertex.
    // Only in a pass with no depth or stencil attachment of the caller's, drawn in sysmem alone (one in tiles, or
    // recorded both ways, is left as it is). The bytes read are held against the buffers again when the command buffer
    // is committed: if a single one differs, or a buffer has since been written by the GPU, every draw left out is put
    // back. kCullVerify works all of it out and leaves nothing out. Off by default.
    enum : uint8_t { kCullOff = 0, kCullOn = 1, kCullVerify = 2 };
    void setCull(uint8_t mode) { cull_ = mode; }
    // (for the tests) TWIN, a mistake made on purpose that the pictures must show: 1 a blending draw covers, 2 a
    // coverer's rectangle taken outwards, 3 a reader ends nothing, 4 any two triangles cover their bounding box, 5 the
    // bytes not held against the buffers at the commit
    void setCullTest(uint8_t twin) { cullTwin_ = twin; }
    struct CullCounts {
        uint32_t seen = 0;        // draws of passes it serves
        uint32_t known = 0;       // ...whose positions were worked out
        uint32_t coverers = 0;    // ...that were a covering rectangle when something could be covered
        uint32_t culled = 0;      // draws left out (verifying: that would have been)
        uint64_t pixels = 0;      // the pixels of their boxes
        uint32_t undone = 0;      // draws put back at the commit: a byte read had changed
        uint32_t why[cull::kWhyCount] = {};   // draws not looked at further, by reason
    };
    const CullCounts& cullCounts() const { return cullCounts_; }
    // the draws left out, the first kMaxCullEvents: PASS its pass among passTimes' (~0: not profiled), DRAW and BY the
    // draw's and its last coverer's numbers among their pass's draws, BOX the pixels it could touch
    struct CullEvent {
        uint32_t pass, draw, by;
        cull::Rect box;
    };
    static constexpr size_t kMaxCullEvents = 256;
    const std::vector<CullEvent>& cullEvents() const { return cullEvents_; }
    // Hidden surface removal by depth (the plug-in's metal_hsr; dev-notes/investigacion/anexos/32-interfaz-telegram/
    // 02-hsr-por-profundidad.md): what a GPU of Apple's does not shade -- a fragment a later opaque draw of the pass
    // covers -- is not shaded here either, and the picture is the one the draws give in their order. Off by default.
    // Version 1 serves the passes that are sysmem whatever is drawn: colour attachment 0 compressed (the UBWC screen,
    // which render::gmemPlan refuses), no depth or stencil attachment of their own, one sample. Such a pass is drawn
    // with a depth attachment of the queue's (CommandQueue::hsrDepth: setHsrDepth's kind, the window's size, cleared to
    // 1 as the pass opens and never stored), and each of its draws I with Z_I = (65533 - 2 I) / 65535 for its depth
    // through the viewport (znear = zfar = Z_I: the clipping before the viewport is the draw's own). Each draw is one of three
    // classes, from its Program, the pass and the Draw alone (render.cpp: hsrClassOf):
    //   opaque    writes colour 0 alone, blending off and every channel, no discard, no framebuffer fetch, no memory
    //             written or read through IBOs by either stage, no depth, sample mask or stencil from the FS, one
    //             sample, no visibility result: drawn in the prepass of its segment, its draw packet in the pass a NOP
    //   hidden    writes colour 0 alone, no memory written by either stage, no depth from the FS, no visibility
    //             result (it may blend, discard or fetch): drawn in its place, tested LEQUAL against the depth, no write
    //   barrier   anything else, and every draw past kHsrMaxDraws: drawn in its place untested; it ends its segment
    // A segment is the draws between two barriers. Its first draw leaves a hole in the stream (a NOP of 3 words) and the
    // register shadow and the cursors are forgotten there, so that this draw writes its whole state; at the segment's
    // end the hole becomes a CP_INDIRECT_BUFFER to the prepass: the segment's opaque draws in REVERSE order, each with
    // its whole state, tested LEQUAL and writing colour and depth -- the last opaque over a pixel is the first to reach
    // it and the earlier ones fail (larger Z), the primitives of one draw (same Z) land as they come.
    void setHsr(bool on) { hsr_ = on; }
    // What setHsr's depth is (the design's version 2, its section 4): the traffic of the depth in memory is what HSR
    // costs, and with Depth32Float it cost more than it saved (dev-notes/investigacion/03-experimentos-y-datos.md, M52).
    //   kHsrDepth32F    Depth32Float, version 1's
    //   kHsrDepth16     Depth16Unorm: half the bytes cleared, written and read (the default)
    //   kHsrDepth16Ubwc Depth16Unorm compressed (UBWC, as turnip lays a D16 image out on this chip; Mesa's layout, so a
    //                   build without Mesa has none and the pass goes without HSR)
    // LRZ: the depth with a low-resolution Z buffer (render::LrzLayout), cleared with it, each draw testing it and the
    // prepass's writing it (render.cpp: lrzCntl) -- 8 x 8 blocks of hidden fragments rejected before the depth is read.
    // The A610 alone (render::kLrzServed); whether its LRZ is written in a sysmem pass at all no source says the same
    // way (turnip turns it off there, fd6 leaves it on), which hakios-metal-hsrcheck measures.
    enum HsrDepth : uint8_t { kHsrDepth32F = 0, kHsrDepth16 = 1, kHsrDepth16Ubwc = 2 };
    void setHsrDepth(HsrDepth kind, bool lrz) {
        hsrKind_ = kind;
        hsrLrz_ = lrz;
    }
    // EACH: every pass drawn with HSR clears its depth as it opens and numbers its draws from 0 (the version measured in
    // M55, for an A/B); off, the depth's clock carries on from pass to pass (HsrClock) and a pass clears it only when the
    // clock says so -- the default
    void setHsrClearEach(bool each) { hsrClearEach_ = each; }
    static constexpr uint32_t kHsrMaxDraws = 32767;   // (render.cpp's hsrZ: Z_I's 16-bit level, 65533 - 2 I, at least 1 below it)
    // (for the tests) TWIN: a mutation of the stream the tests' rules must catch -- 1 the prepass left out (the hole
    // stays a NOP, the opaque draws still NOPs), 2 a barrier tested as a hidden draw, 3 an opaque draw that writes
    // another attachment taken as opaque, 4 the shadow and the cursors kept at a segment's hole, 5 the prepass tested
    // LESS, 6 an opaque draw whose stages have IBOs taken as opaque, 7 the LRZ buffer not cleared as a pass opens, 8 the
    // LRZ written by every draw that tests it (the hidden ones too); 0 the stream as meant. LIMIT: kHsrMaxDraws, or
    // fewer to reach it in a test. ANY_TARGET: a pass whose colour 0 is not compressed taken too (the suites without
    // Mesa have no compressed texture).
    void setHsrTest(uint8_t twin, uint32_t limit = kHsrMaxDraws, bool anyTarget = false) {
        hsrTwin_ = twin;
        hsrLimit_ = limit < kHsrMaxDraws ? limit : kHsrMaxDraws;
        hsrAnyTarget_ = anyTarget;
    }
    // why the last pass that could not be drawn in GMEM was not (gmemPlan's or gmemDrawOk's reason), empty when none was refused
    const std::string& gmemRefusal() const { return gmemWhy_; }
    // the words recorded so far (what the command processor will read; once it is done, what it was given)
    size_t streamWords() const { return streamWords_ != 0 ? streamWords_ : stream_.words().size(); }
    // (for the tests) every draw taken from a program's kept state checked against the same draw encoded in full: a
    // difference in a word of the stream or a byte of the draw's table ends the buffer in error, saying which
    static void verifyDraws(bool on);
    // a program's kept state not used: every draw judged and written whole, as the first with its program is. The
    // stream is the same either way (the tests record it both ways); a bisect's switch. On by default; what it was is
    // returned.
    static bool keepStates(bool on);
    // a draw of a sysmem pass whose draw before it was of the same program, both from its kept state, without the
    // program's HLSQ_INVALIDATE_CMD and its stages' preloads and immediates (render.cpp: gSkipSameProgram): the stream
    // differs. Off by default (metal_sameprog=1 in the plug-in); what it was is returned.
    static bool skipSameProgramLoads(bool on);
    std::vector<PassTime> passTimes() const;
    // why profiled pass I of a draw was drawn in sysmem ("" for a GMEM pass, or one not profiled); valid while the buffer lives
    const char* passSysmemWhy(size_t i) const { return i < profiled_.size() ? profiled_[i].sysmemWhy.c_str() : ""; }

private:
    friend class CommandQueue;
    explicit CommandBuffer(CommandQueue& queue) : queue_(queue) {}
public:
    ~CommandBuffer();   // (its stamp buffers back to the queue: read until the last reader lets go, which is now)
private:
    bool fail(const std::string& why);
    bool skipRefused_ = false;
    Counts counts_;
    bool profile_ = false;
    bool tuning_ = false;     // setGmemTuned's (cleared by a failed stamp buffer too)
    bool profileDraws_ = false;
    std::vector<DrawTime> drawsProfiled_;
    std::shared_ptr<Buffer> drawStamps_;
    void drawStamp(uint32_t slot);
    bool gmem_ = true;
    uint32_t gmemOff_ = 0;
    uint32_t gmemMinDraws_ = 0;
    bool gmemBinning_ = false;
    bool nullFsTextures_ = false;
    // setDrawTag / cancelDraw: where each tagged draw of the open pass put its draw packet (SIZE_MAX: not in that stream)
    struct DrawMark {
        uint32_t tag;
        pm4::Stream* into;
        size_t at;
        size_t atSys;
    };
    uint32_t drawTag_ = 0, cancelled_ = 0;
    std::vector<DrawMark> drawMarks_;
    // setCull: the bytes its answers were read from, which of them a draw left out rests on, and where each such
    // draw's packet lies -- a word of the stream, or of an HSR prepass's buffer -- with the header it had
    uint8_t cull_ = 0, cullTwin_ = 0;
    CullCounts cullCounts_;
    std::vector<CullEvent> cullEvents_;
    cull::Snapshots cullSnapshots_;
    struct CullCheck {
        uint32_t first, count;
    };
    std::vector<CullCheck> cullChecks_;
    struct CullDone {
        size_t at;        // SIZE_MAX: not in the stream
        uint32_t* ptr;
        uint32_t header;
    };
    std::vector<CullDone> cullDone_;
    // the draw just written into the open pass: AT its draw packet in the stream, PREPASS_AT in the HSR prepass (SIZE_MAX: none)
    void cullDraw(const Program& program, const RenderPass& pass, const Draw& draw, const Bindings& vsBindings, const Bindings& fsBindings, size_t at,
                  size_t prepassAt);
    // at the commit, the pass closed: the bytes read still what the buffers hold, or every draw left out put back
    void cullRevalidate();
    bool hsr_ = false, hsrAnyTarget_ = false;   // setHsr's, setHsrTest's
    uint8_t hsrTwin_ = 0;
    uint32_t hsrLimit_ = kHsrMaxDraws;
    HsrDepth hsrKind_ = kHsrDepth16;   // setHsrDepth's
    bool hsrLrz_ = false;
    bool hsrClearEach_ = false;
    std::shared_ptr<HsrClock> hsrClock_;   // the clock this buffer owns (HsrClock::owner), let go at commit or destruction
    std::vector<std::shared_ptr<HsrClock>> hsrHeld_;   // (an appended buffer's clocks of other depths, held the same way)
    void hsrLetGo();
    // setHsr: a segment of the open pass begun (its hole, the shadow and the cursors forgotten), and ended (the hole made
    // the prepass's CP_INDIRECT_BUFFER when the segment had an opaque draw); false with the buffer failed
    void hsrOpenSegment();
    bool hsrCloseSegment();
    uint32_t gmemClears_ = 0, gmemClearWords_[kMaxColorOutputs][4] = {};   // (draw's, for the pass it opens)
    std::string gmemWhy_;
    std::unordered_map<uint64_t, uint32_t> ordinals_;   // passes opened so far into each kind of target
    std::vector<PassTime> profiled_;   // as recorded; us filled in by passTimes from the stamps
    std::shared_ptr<Buffer> profileStamps_;
    bool profileSamples_ = false;
    std::shared_ptr<Buffer> sampleStamps_;   // setProfileSamples's: the sample count at each pass's start and end
    void sampleStamp(uint32_t slot);
    // the counter into stamp slot SLOT (two a pass: start, end), when profiling and there is room
    void stamp(uint32_t slot);
    void stampStart(PassKind kind, uint32_t width, uint32_t height);
    void stampEnd();
    uint32_t skipped_ = 0;
    std::string lastSkipped_;
    // dispatch and dispatchIndirect: GROUPS, or (GROUPS null) the counts at OFFSET of ARGS
    bool dispatchAny(const Kernel& kernel, const uint32_t* groups, const Buffer* args, uint64_t offset, const uint32_t* local, const Bindings& bindings,
                     const uint32_t* immediates, const compute::Options& options);
    // the end of it, however it ends: the status, the handlers (which see that status), and only then the waiters
    void finish(Status s, bool recycle = true);
    // (once for reading and once for writing: the submission names each buffer once whatever the draws, and the list was
    // as long as the frame's draws times their bindings, gone through again for every one of them at the commit)
    // (STORED: the GPU does write it -- false for a draw's bindings named for writing only because their stage has
    // device buffers, which one that stores to none of them does not write: Bo::noteGpuWrite)
    void use(const Buffer& b, bool write, bool stored = true) {
        const Bo* bo = &b.bo();
        if (write && stored) bo->noteGpuWrite();
        const void* key = reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(bo) | (write ? 1u : 0u));
        if (!recently(key)) uses_.push_back(BoUse{bo, write});
    }
    // whether P was asked about a moment ago (and it is, from now on): a small table by address, one entry a slot
    bool recently(const void* p) {
        const uintptr_t a = reinterpret_cast<uintptr_t>(p);
        const void*& slot = recent_[((a >> 4) ^ (a >> 12)) & (kRecent - 1)];
        if (slot == p) return true;
        slot = p;
        return false;
    }
    static constexpr size_t kRecent = 256;
    const void* recent_[kRecent] = {};
public:
    // What each register written in a pass holds, for appendDeduped: a draw writes some 500 of them and almost all
    // hold their value already, so each is looked up once. By the register itself rather than a hash of it: the
    // registers a pass writes lie in a few blocks of the 18-bit space (RB, GRAS, VPC, PC, VFD, SP, HLSQ), each block of
    // 256 a page of values with the generation each was written in, the pages found through a table of the 1024
    // blocks. A register holds a value when its stamp is the current generation, so forgetting everything (a packet
    // behind the stream's back, a pass let go) is one increment, not a fill of the table -- which an open-addressed
    // table needed, as large as the most registers a pass ever held.
    class RegShadow {
    public:
        // whether REG held VALUE; it holds VALUE from now on
        bool hold(uint32_t reg, uint32_t value) {
            if (reg >= kRegs) return holdFar(reg, value);
            const uint32_t block = reg >> kPageBits;
            uint16_t at = block < index_.size() ? index_[block] : 0;
            if (at == 0) at = newPage(block);
            Page& p = pages_[at - 1u];
            const uint32_t k = reg & (kPage - 1);
            const bool same = p.stamp[k] == gen_ && p.value[k] == value;
            p.stamp[k] = gen_;
            p.value[k] = value;
            return same;
        }
        void clear() {
            far_.clear();
            if (++gen_ != 0) return;
            // (the generations ran out: every stamp back to 0, which no generation is)
            for (Page& p : pages_) std::fill(std::begin(p.stamp), std::end(p.stamp), 0u);
            gen_ = 1;
        }
    private:
        static constexpr uint32_t kRegs = 1u << 18;   // a type-4 header's register field
        static constexpr uint32_t kPageBits = 8, kPage = 1u << kPageBits;
        struct Page {
            uint32_t value[kPage];
            uint32_t stamp[kPage];
        };
        uint16_t newPage(uint32_t block) {
            if (index_.empty()) index_.assign(kRegs >> kPageBits, 0);
            pages_.emplace_back();
            Page& p = pages_.back();
            std::fill(std::begin(p.stamp), std::end(p.stamp), 0u);
            index_[block] = uint16_t(pages_.size());
            return index_[block];
        }
        // (a register past the 18 bits a header has: none is written, but the shadow answers for any number)
        bool holdFar(uint32_t reg, uint32_t value) {
            for (auto& [r, v] : far_) {
                if (r != reg) continue;
                const bool same = v == value;
                v = value;
                return same;
            }
            far_.emplace_back(reg, value);
            return false;
        }
        std::vector<Page> pages_;
        // block -> page + 1, 0: none yet (empty until the first page: a shadow is made for nothing at every draw, the
        // check's copy of it)
        std::vector<uint16_t> index_;
        uint32_t gen_ = 1;
        std::vector<std::pair<uint32_t, uint32_t>> far_;
    };

private:
    // the pass draws are joining (render.cpp: draw), closed by its tail before anything else is recorded or it is submitted
    struct OpenPass {
        RenderPass pass;
        render::Options options;
        uint64_t table;   // the GPU address of the pass's table: a piece (below), kept by the command buffer
        RegShadow regs;   // what each register written in the pass holds (render.cpp: appendDeduped)
        std::vector<uint8_t> staticKey;   // what decided the last draw's program-and-pass state (render.cpp: draw)
        // A pass drawn in GMEM (gmemPlan and every draw gmemDrawOk) is recorded rather than streamed: its draws gather
        // in BODIES, deduplicated among themselves from nothing (BODY_REGS) since each tile replays them after the
        // tile's own register writes, and closePass puts them in the tiles.
        bool gmem = false;
        render::GmemPlan plan;
        pm4::Stream bodies;
        RegShadow bodyRegs;
        // what the draws touch: the render area (X0 Y0 X1 Y1, the union of render::drawArea's) and the attachments they
        // write and read (a framebuffer fetch), as masks -- what closePass tiles, loads and stores
        uint32_t area[4] = {~0u, ~0u, 0, 0};
        uint32_t written = 0, read = 0;
        // Both ways at once (setGmemMinDraws): the draws gathered for tiles in BODIES and, beside them, as a sysmem pass in
        // SYS_STREAM (its head included, deduplicated through REGS), the GMEM head held in HEAD_GMEM; closePass takes the
        // one the pass's draws call for, now that they are known
        bool dual = false;
        pm4::Stream sysStream, headGmem;
        // (recorded both ways until its draws left no doubt that it is tiled -- render::gmemWorth's answer could no
        // longer come out sysmem whatever the draws still to come -- and for tiles alone from then on: draw())
        bool tiledEarly = false;
        uint32_t clears = 0;        // the attachments a GMEM pass clears in its tiles (render::gmemClearWords), and their words
        uint32_t clearWords[kMaxColorOutputs][4] = {};
        uint64_t kind = 0;          // render::tuneKeyOf the pass and its ordinal in the buffer among passes of that kind
        uint32_t ordinal = 0, draws = 0;
        bool lastFetched = false;   // the last draw read its attachments (render::encodeInPass's AFTER_FETCH for the next)
        // what the pass's streams last held of a draw's groups, and what its attachments come to (render.cpp: PassCache)
        std::shared_ptr<render::PassCache> cache;
        // setHsr: the pass is drawn so (its depth attachment the queue's), the draws given a Z so far, and the segment
        // open: where its hole lies in the stream, its opaque draws' words each whole (HSR_PREPASS, in their order),
        // and where each of them begins there
        bool hsr = false, hsrSegment = false;
        uint32_t hsrDraws = 0;
        size_t hsrHole = 0;
        pm4::Stream hsrPrepass;
        std::vector<size_t> hsrStarts;
        RegShadow hsrShadow;   // (an empty shadow, for an opaque draw's whole words taken from its program's kept state)
        // setCull: the draws that later ones may still cover, oldest first -- each with what its positions come from,
        // then, once a coverer has asked, its box and what of it is not covered yet, the records its positions were
        // read into and those of the coverers that took a piece -- and the draws left out whose packet in the open
        // segment's prepass has no address yet
        struct CullDraw {
            uint32_t index = 0;
            cull::Geometry geometry;
            bool evaluated = false;
            cull::Rect box;
            cull::Region left;
            CullCheck evidence{0, 0};
            static constexpr uint32_t kMaxCoverers = 6;
            uint32_t coverers = 0;
            CullCheck coverer[kMaxCoverers];
            size_t at = SIZE_MAX, prepassAt = SIZE_MAX;
            uint32_t* prepassPtr = nullptr;
            uint32_t prepassHeader = 0;
        };
        static constexpr size_t kMaxCullDraws = 48;
        std::vector<CullDraw> cull;
        struct CullFix {
            size_t prepassAt;
            uint32_t done;
        };
        std::vector<CullFix> cullFixes;
    };
    // setCull: X of the open pass is covered, the last piece by the pass's draw BY -- counted, and left out unless verifying
    void cullLeaveOut(OpenPass::CullDraw& x, uint32_t by);
    // NEXT: the pass of the draw that could not join the open one, when it is such a draw that closes it (nullptr for
    // anything else: a copy, a dispatch, the commit). A memoryless attachment the tiles wrote is stored for it alone:
    // when NEXT has the same texture attached with Load, the caller's one pass goes on in a second one here.
    // KEPT: stored whatever follows (a command buffer appended to another: what follows is not known here).
    void closePass(const RenderPass* next = nullptr, bool kept = false);
    void useAttachments(const RenderPass& pass);
    // An open pass let go is the queue's to hand to the next pass, as a new one in everything but what its streams and
    // its register shadows have taken from the heap (render.cpp): a frame is eight passes or so, and each made those
    // again from nothing, a doubling at a time.
    struct PassReturn {
        // (constructors of its own: a member initializer here is no use to unique_ptr while CommandBuffer itself is still
        // being defined, and the iOS SDK's libc++ then finds no default constructor for the deleter)
        PassReturn() : queue(nullptr) {}
        explicit PassReturn(CommandQueue* q) : queue(q) {}
        CommandQueue* queue;
        void operator()(OpenPass* p) const;
    };
    using OpenPassPtr = std::unique_ptr<OpenPass, PassReturn>;
    OpenPassPtr open_;
    // A piece of a larger buffer, for a draw's or a dispatch's table and a program's words. A buffer of its own for each
    // was three requests to the kernel for every draw, which on the phone cost about 200 us a draw (probe 9r) while the
    // GPU spent 3 us on it. A piece is the command buffer's: written once, used for writing, kept until completion.
    struct Piece {
        std::shared_ptr<Buffer> buffer;
        uint64_t offset = 0;
        uint64_t gpuAddress() const { return buffer->gpuAddress() + offset; }
        uint8_t* contents() const { return static_cast<uint8_t*>(buffer->contents()) + offset; }
    };
    // AS_IS: the piece may hold what another command buffer wrote there (a slab the queue had back): for a caller that
    // writes every byte the GPU is then told to read. Otherwise the piece is zeros, as memory new from the kernel is.
    bool piece(uint64_t bytes, Piece* out, std::string* why, bool asIs = false);
    std::shared_ptr<Buffer> slab_;
    uint64_t slabUsed_ = 0;
    bool slabUsedBefore_ = false;   // the slab came back from another command buffer: its pieces are zeroed as they are cut
    std::vector<std::shared_ptr<Buffer>> slabs_;   // every slab this buffer took, given back to the queue when it is done
    // A program's words where they are already in this command buffer, so that every draw with the program does not
    // upload them again. Found by the caller's pointer, but taken only when the words still look the same: the pointer
    // is the caller's to free and reuse. They are compared at every 64th word and the last, not all: the whole compare,
    // once a draw, was 7-10 us of a draw's 50 on the phone, and CoreAnimation's uber shader is 31 000 words. (The copy
    // compared is a CPU one: reading the piece back is uncached memory.)
    struct Uploaded {
        std::vector<uint32_t> words;
        uint64_t gpuAddress;
    };
    // the GPU address of K's words in this buffer, uploading them when they are not there; 0 with *WHY when out of memory
    uint64_t programAddress(const Kernel& k, std::string* why);
    std::unordered_multimap<const uint32_t*, Uploaded> uploaded_;
    // the programs this buffer has drawn with and its scratch for a draw (render.cpp: DrawCache), made at the first draw
    std::shared_ptr<render::DrawCache> drawCache_;
    bool drawCacheBegun_ = false;     // drawCache_ is this buffer's (DrawCache::begin: it may be one the queue handed on)
    std::shared_ptr<Texture> null_;   // the queue's null texture, asked for once a buffer rather than once a draw

    CommandQueue& queue_;
    std::atomic<Status> status_{kNotEnqueued};
    std::string error_;
    pm4::Stream stream_;
    size_t streamWords_ = 0;   // the stream's words when the buffer was done and its stream handed back (finish)
    std::vector<BoUse> uses_;
    std::vector<std::shared_ptr<const Buffer>> kept_;   // the buffers it uses stay alive until it completes
    std::vector<std::shared_ptr<const void>> keptAny_;
    std::vector<std::function<void(CommandBuffer&)>> handlers_;
    std::mutex lock_;
    std::condition_variable done_;
    bool finished_ = false;                             // (under lock_) the handlers have run: waitUntilCompleted returns
    uint32_t fence_ = 0;
    uint64_t faultsAtCommit_ = 0;                       // the GPU's reset count then: one more by completion = given up, not run
    std::unique_ptr<Bo> commands_;
    int64_t committedAt_ = 0, completedAt_ = 0;
    SubmitCost submitCost_;
};

class CommandQueue {
public:
    // a queue of its own with the driver; nullptr with *why when the driver has none to give
    static std::unique_ptr<CommandQueue> make(Gpu& gpu, std::string* why);
    ~CommandQueue();

    std::shared_ptr<CommandBuffer> commandBuffer();
    // the number of command buffers held in quarantine: ones whose wait timed out with the GPU neither finishing nor
    // reset, their memory never reused (for tests and diagnostics)
    size_t quarantinedCount();
    Gpu& gpu() { return gpu_; }
    // the buffers kept for command streams, idle (to see that they are given back and used again)
    size_t idleCommandBuffers();
    // What a draw binds at the slot the front end reserves for Metal's null texture (Kernel: KernelTexture::isNull): a
    // 1x1 RGBA8Unorm texture of zeros, made at the first ask, the same for every draw of the queue. Zero is an
    // assumption, not a reading -- nothing reachable says what Apple's null texture samples as (nir_air.h) -- and this
    // is the one place it lives; a probe may write the texel to see which instructions read it. nullptr with *why.
    std::shared_ptr<Texture> nullTexture(std::string* why);
    // Private memory for a stage that wants BYTES a fiber in the PERWAVE layout: one buffer for every program of the
    // queue, as turnip keeps one a device (tu_setup_pvtmem, tu_shader.cc:2132-2190) -- a buffer a pipeline is 4 MiB for
    // 512 bytes a fiber on this GPU -- grown to the next power of two when a program wants more. A grown-out buffer is
    // kept as long as the queue: a draw recorded against it may still be in flight. The buffer and *OUT its layout;
    // nullptr with *why.
    std::shared_ptr<Buffer> privateMemory(uint32_t bytes, bool perWave, PrivateMemory* out, std::string* why);
    // GMEM or sysmem for a pass into the target at KEY (its memory's address), by what the GPU spent on such passes:
    // each mode's time kept as a running average from the passes' own counter stamps (tuned), the cheaper one taken,
    // the other tried again now and then, sysmem tried first and GMEM next -- turnip's autotune, by time instead of samples.
    // GMEM is no gain for a pass that touches a corner of the screen (its tiles load and store the whole target: 23
    // draws of CoreAnimation took 5 ms so, 0.1 ms in sysmem), nor for one of framebuffer fetches (the texture unit
    // reads GMEM at a quarter of the rate it reads memory: probe "full-screen cost").
    // the visibility streams' memory binned GMEM passes share (render::vscBytes): one is enough, the GPU running one
    // submission at a time and each pass's binning before its tiles; nullptr with *why
    std::shared_ptr<Buffer> vscBuffer(std::string* why);
    // The depth attachment of a pass drawn with CommandBuffer::setHsr: a render target of W x H of KIND
    // (CommandBuffer::setHsrDepth), the queue's, with an LRZ buffer when LRZ asks for one (render::lrzLayout's bytes,
    // kept beside it), one for each size and kind asked for lately (kHsrDepthSizes, the least recently asked for let
    // go). Every pass that uses them clears them as it opens and stores nothing of them, and the GPU runs one submission
    // at a time, so the passes of every command buffer share them; a command buffer keeps the ones it drew with alive.
    // A null depth with *why.
    struct HsrTarget {
        std::shared_ptr<Texture> depth;
        std::shared_ptr<Buffer> lrz;
        std::shared_ptr<HsrClock> clock;
    };
    HsrTarget hsrDepth(uint32_t width, uint32_t height, std::string* why, uint8_t kind = 1, bool lrz = false);
    // (the screen's size and the odd other: CoreAnimation's compressed targets are its screens, all of one size; and
    // as many again for a run that tries the kinds one after another, hakios-metal-hsrcheck's)
    static constexpr size_t kHsrDepthSizes = 4;
    // K's words in a buffer of their own, made the first time a command buffer of this queue draws with them and kept
    // for the next: copying them into each command buffer was most of a draw's first cost (CoreAnimation's uber shader is
    // 124 KB, into write-combined memory). Found by the words' pointer and taken only when every word is the same -- a
    // freed program's pointer may come back with other words, and then its entry goes. nullptr with *why.
    std::shared_ptr<Buffer> programBuffer(const Kernel& k, std::string* why);
    // A slab for draw tables (CommandBuffer::piece): one a completed command buffer gave back, zeroed as a new one comes,
    // or a new one. The kernel's making of a 256 KB buffer, once every 32 draws, was most of a draw's CPU time on the
    // phone (40 us a draw on a fast core, where the same encoding on the host takes 5).
    // (*USED_BEFORE: it is one a command buffer gave back, holding what that one wrote -- not zeroed here, a quarter
    // of a megabyte for every 32 draws: CommandBuffer::piece zeroes the pieces that need it)
    std::shared_ptr<Buffer> takeSlab(uint64_t bytes, std::string* why, bool* usedBefore);
    // (for the tests) a slab that comes back is filled with 0xa5 before it is used again: what a draw leaves unwritten
    // in its table is then nothing a GPU could take for a descriptor -- and not the same frame's tables from the time
    // before, which would hide a draw that wrote none
    static void poisonReused(bool on);
    void giveSlabs(std::vector<std::shared_ptr<Buffer>>* slabs);
    // ...and any other buffer a command buffer makes for itself (a large piece, its counter stamps): BYTES rounded up to
    // a power of two from 4 KB, zeroed; given back the same way, 32 MB of them kept at most
    std::shared_ptr<Buffer> takeBuffer(uint64_t bytes, std::string* why);
    void giveBuffer(std::shared_ptr<Buffer> b);
    // whether GMEM passes may still be binned: no pass's visibility streams have overrun their pitch (render::kVscOverflowAt,
    // read by checkVsc once a buffer with a binned pass has completed)
    bool binningOk() const { return !vscOverflowed_.load(); }
    void checkVsc();
    bool preferGmem(uint64_t key);
    void tuned(uint64_t key, bool gmem, double us);
    // What tells passes into one kind of target apart: the draws the pass at that place in a command buffer (the Nth into
    // this kind of target) had the last time, which preferGmem's key takes in as a class (a frame of CoreAnimation is
    // the same passes in the same order as the one before it, light or heavy). 0 when not seen yet.
    uint32_t lastDraws(uint64_t kind, uint32_t ordinal);
    // ...and the most it had in the times since its peak, the peak let go after 16 times below it: a pass that is heavy
    // only now and then (CoreAnimation's: light frames between the heavy ones of an animation) counts as heavy
    uint32_t peakDraws(uint64_t kind, uint32_t ordinal);
    void noteDraws(uint64_t kind, uint32_t ordinal, uint32_t draws);
    // what the queue keeps of the programs drawn with it, from one command buffer to the next (render.cpp: StateCache;
    // under its own lock)
    render::StateCache& states();
    // What a command buffer that is done hands to the next: its stream and its lists, emptied, with the memory they
    // grew to; and what it had at hand of the programs it drew with (render.cpp: DrawCache), which the next frame
    // draws with again. And the open passes let go (CommandBuffer::PassReturn).
    struct Spare {
        pm4::Stream stream;
        std::vector<BoUse> uses;
        std::vector<std::shared_ptr<const Buffer>> kept;
        std::vector<std::shared_ptr<const void>> keptAny;
        std::shared_ptr<render::DrawCache> drawCache;
    };
    void giveSpare(Spare&& spare);
    bool takeSpare(Spare* spare);
    CommandBuffer::OpenPassPtr takePass();
    void givePass(CommandBuffer::OpenPass* pass);


private:
    friend class CommandBuffer;
    CommandQueue(Gpu& gpu, uint32_t id) : gpu_(gpu), id_(id) {}
    void submit(std::shared_ptr<CommandBuffer> cb);
    void waiter();
    std::unique_ptr<Bo> takeCommandBo(size_t bytes, int* error);

    Gpu& gpu_;
    std::shared_ptr<Texture> null_;   // (under lock_) nullTexture's
    std::shared_ptr<Buffer> vsc_;     // (under lock_) vscBuffer's
    struct HsrKept {
        uint8_t kind;
        HsrTarget t;
    };
    std::vector<HsrKept> hsrDepths_;   // (under lock_) hsrDepth's, the last asked for at the back
    // (under poolLock_, a lock of its own: a command buffer gives its stamps back as it is destroyed, which can be under
    // lock_ -- the queue letting go of the last one in flight -- and lock_ there was the window server hung at boot)
    std::mutex poolLock_;
    std::vector<Spare> spares_;                            // (under poolLock_) giveSpare's, a few
    std::vector<CommandBuffer::OpenPass*> sparePasses_;    // (under poolLock_) givePass's, a few
    std::unordered_map<uint64_t, std::vector<std::shared_ptr<Buffer>>> freeBuffers_;   // by size: giveSlabs', giveBuffer's
    uint64_t freeBytes_ = 0;
    struct Resident { std::vector<uint32_t> words; std::shared_ptr<Buffer> buffer; };
    std::unordered_multimap<const uint32_t*, Resident> resident_;   // (under lock_) programBuffer's
    uint64_t residentBytes_ = 0;
    std::atomic<bool> vscOverflowed_{false};
    std::vector<std::shared_ptr<Buffer>> private_[2];   // (under lock_) privateMemory's, by layout, the last the current one
    uint32_t privatePerFiber_[2] = {0, 0};
    uint32_t id_;
    std::mutex lock_;
    struct Tune { double us[2] = {0, 0}; uint32_t runs[2] = {0, 0}; uint32_t passes = 0; };
    std::unordered_map<uint64_t, Tune> tune_;   // (under lock_) preferGmem's, by target
    std::unordered_map<uint64_t, uint32_t> lastDraws_;   // (under lock_) noteDraws', by kind and ordinal
    struct Peak { uint32_t draws = 0, below = 0; };
    std::unordered_map<uint64_t, Peak> peakDraws_;        // (under lock_) noteDraws' too: peakDraws'
    std::shared_ptr<render::StateCache> states_;   // (made by states() at the first ask, under lock_)
    std::condition_variable changed_;
    std::deque<std::shared_ptr<CommandBuffer>> inFlight_;
    // (under lock_) command buffers whose wait timed out with the GPU neither finishing nor reset: it may still be
    // reading their memory, so they are held here for the queue's life -- never freed, never returned to a pool
    std::vector<std::shared_ptr<CommandBuffer>> quarantined_;
    std::vector<std::unique_ptr<Bo>> idle_;
    bool quit_ = false;
    std::thread thread_;
};

}  // namespace hakios::metal
