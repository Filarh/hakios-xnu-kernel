#include "hakios-metal/command.hpp"
#include "hakios-metal/texture.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <map>
#include <mutex>
#include <cstring>

namespace hakios::metal {

namespace {
constexpr size_t kCommandBo = 64 * 1024;           // one command buffer's packets, at most (grown when needed)
constexpr int64_t kPatienceNs = 5000000000;        // a fence not passed in 5 s: the GPU is hung, the buffer is an error
constexpr uint32_t kFillHead = 64;                 // words written by CP_MEM_WRITE before the copies double them
// The most one CP_MEMCPY moves. On the phone (27 October) 125 000 words in one packet took 4.2 ms, and 4 Mi words (16 MB)
// in one packet hung the GPU ("no answer" from the kernel's driver, and the system after it). Cut into packets of this
// size, each waited for before the next, the same copy is a row of small ones.
constexpr uint32_t kMemcpyMax = 64 * 1024;
}  // namespace

void appendStream(pm4::Stream& to, const pm4::Stream& from);   // render.cpp's: a pass of its own onto the buffer's stream

namespace {
std::mutex gAddressLock;
std::map<uint64_t, std::weak_ptr<Buffer>> gByAddress;   // by gpuAddress, every Buffer newBuffer made and still alive

std::shared_ptr<Buffer> addressed(Buffer* raw) {
    const uint64_t at = raw->gpuAddress();
    std::shared_ptr<Buffer> b(raw, [at](Buffer* x) {
        {
            std::lock_guard<std::mutex> g(gAddressLock);
            auto it = gByAddress.find(at);
            if (it != gByAddress.end() && it->second.expired()) gByAddress.erase(it);
        }
        delete x;
    });
    std::lock_guard<std::mutex> g(gAddressLock);
    gByAddress[at] = b;
    return b;
}
}  // namespace

std::shared_ptr<Buffer> bufferAt(uint64_t address, uint64_t* offset) {
    std::lock_guard<std::mutex> g(gAddressLock);
    auto it = gByAddress.upper_bound(address);
    if (it == gByAddress.begin()) return nullptr;
    --it;
    std::shared_ptr<Buffer> b = it->second.lock();
    if (b == nullptr || address - it->first >= b->length()) return nullptr;
    *offset = address - it->first;
    return b;
}

std::atomic<uint32_t> gBuffersMade{0}, gBuffersReused{0};
std::atomic<uint64_t> gBytesMade{0};

std::shared_ptr<Buffer> newBuffer(Gpu& gpu, uint64_t length, std::string* why) {
    gBuffersMade++;
    gBytesMade += length;
    int error = 0;
    std::unique_ptr<Bo> bo = gpu.newBo(length, kBoWriteCombined, &error);
    if (bo == nullptr) {
        *why = "the driver gave no buffer of " + std::to_string(length) + " bytes: " + strerror(error);
        return nullptr;
    }
    return addressed(new Buffer(std::move(bo), length));
}

std::atomic<uint32_t> gBuffersShared{0};
std::atomic<uint64_t> gBytesShared{0};

std::shared_ptr<Buffer> newBufferNoCopy(Gpu& gpu, void* bytes, uint64_t length, std::function<void()> released, std::string* why) {
    const uintptr_t at = reinterpret_cast<uintptr_t>(bytes);
    if (bytes == nullptr || at % 4096 != 0 || length == 0 || length % 4096 != 0) {
        char text[200];
        snprintf(text, sizeof(text), "a no-copy buffer of %llu bytes at %#llx: the memory must start on a page and be whole pages (multiples of 4096 bytes), "
                 "as Metal asks of the caller", (unsigned long long)length, (unsigned long long)at);
        *why = text;
        return nullptr;
    }
    if (!gpu.sharesMemory()) {
        *why = "a no-copy buffer: the kernel does not take a program's own memory as GPU memory (it needs xnu patch 0380), and a copy would be "
               "another buffer, not the caller's memory";
        return nullptr;
    }
    int error = 0;
    std::unique_ptr<Bo> bo = gpu.userBo(bytes, length, &error);
    if (bo == nullptr) {
        *why = "the kernel would not take " + std::to_string(length) + " bytes of this program's memory as a GPU buffer: " + strerror(error);
        return nullptr;
    }
    gBuffersShared++;
    gBytesShared += length;
    return addressed(new Buffer(std::move(bo), length, std::move(released)));
}

namespace {
// One slab is 32 draw tables. Pieces are cut at 256 bytes, which is more than anything in them wants (128 for a
// program's preload and the border colour, 64 for a texture descriptor), and a piece of a page or more starts on a
// page, so that a table lies in memory as it did in a buffer of its own.
constexpr uint64_t kSlabBytes = 256 * 1024;
constexpr uint64_t kPieceAlign = 256;
constexpr uint64_t kPageAlign = 4096;
}  // namespace

bool CommandBuffer::piece(uint64_t bytes, Piece* out, std::string* why, bool asIs) {
    bytes = (std::max<uint64_t>(bytes, 1) + kPieceAlign - 1) & ~(kPieceAlign - 1);
    const uint64_t align = bytes >= kPageAlign ? kPageAlign : kPieceAlign;
    if (slab_ != nullptr) {
        slabUsed_ = std::min(kSlabBytes, (slabUsed_ + align - 1) & ~(align - 1));
    }
    if (bytes > kSlabBytes / 4) {
        // (a piece this large in a slab would leave most of the slab unused when the next one did not fit)
        out->buffer = queue_.takeBuffer(bytes, why);
        out->offset = 0;
        if (out->buffer != nullptr) slabs_.push_back(out->buffer);
    } else {
        if (slab_ == nullptr || kSlabBytes - slabUsed_ < bytes) {
            slab_ = queue_.takeSlab(kSlabBytes, why, &slabUsedBefore_);
            slabUsed_ = 0;
            if (slab_ != nullptr) slabs_.push_back(slab_);
        }
        out->buffer = slab_;
        out->offset = slabUsed_;
        if (slab_ != nullptr) {
            if (slabUsedBefore_ && !asIs) memset(static_cast<uint8_t*>(slab_->contents()) + slabUsed_, 0, bytes);
            slabUsed_ += bytes;
        }
    }
    if (out->buffer == nullptr) {
        return false;
    }
    if (out->offset == 0) {
        use(*out->buffer, true);
        keep(out->buffer);
    }
    return true;
}

uint64_t CommandBuffer::programAddress(const Kernel& k, std::string* why) {
    const uint32_t n = std::max<uint32_t>(k.numWords, 1);
    auto same = [&](const Uploaded& u) {
        if (u.words.size() != k.numWords) return false;
        for (uint32_t i = 0; i < k.numWords; i += 64) {
            if (u.words[i] != k.words[i]) return false;
        }
        return k.numWords == 0 || u.words.back() == k.words[k.numWords - 1];
    };
    const auto [first, last] = uploaded_.equal_range(k.words);
    for (auto it = first; it != last; ++it) {
        if (same(it->second)) return it->second.gpuAddress;
    }
    (void)n;
    std::shared_ptr<Buffer> b = queue_.programBuffer(k, why);
    if (b == nullptr) return 0;
    use(*b, false);
    keep(b);
    uploaded_.emplace(k.words, Uploaded{std::vector<uint32_t>(k.words, k.words + k.numWords), b->gpuAddress()});
    return b->gpuAddress();
}

namespace {
uint64_t sizeClass(uint64_t bytes) {
    uint64_t c = 4096;
    while (c < bytes) c <<= 1;
    return c;
}
}  // namespace

namespace {
std::atomic<bool> gPoisonReused{false};
}  // namespace

std::shared_ptr<Buffer> CommandQueue::takeBuffer(uint64_t bytes, std::string* why) {
    bool usedBefore = false;
    std::shared_ptr<Buffer> b = takeSlab(bytes, why, &usedBefore);
    if (b != nullptr && usedBefore) memset(b->contents(), 0, b->length());   // (as a new one comes from the kernel)
    return b;
}

std::shared_ptr<Buffer> CommandQueue::takeSlab(uint64_t bytes, std::string* why, bool* usedBefore) {
    const uint64_t size = sizeClass(bytes);
    std::shared_ptr<Buffer> b;
    {
        std::lock_guard<std::mutex> g(poolLock_);
        auto it = freeBuffers_.find(size);
        if (it != freeBuffers_.end() && !it->second.empty()) {
            b = std::move(it->second.back());
            it->second.pop_back();
            freeBytes_ -= size;
        }
    }
    *usedBefore = b != nullptr;
    if (b == nullptr) return newBuffer(gpu_, size, why);
    gBuffersReused++;
    if (gPoisonReused.load(std::memory_order_relaxed)) memset(b->contents(), 0xa5, b->length());
    return b;
}

void CommandQueue::poisonReused(bool on) { gPoisonReused = on; }

void CommandQueue::giveBuffer(std::shared_ptr<Buffer> b) {
    // (only one no one else holds, of a size takeBuffer makes: a piece's buffer kept elsewhere is still someone's)
    if (b == nullptr || b.use_count() != 1 || b->contents() == nullptr || b->length() != sizeClass(b->length())) return;
    std::lock_guard<std::mutex> g(poolLock_);
    if (freeBytes_ + b->length() > (32ull << 20)) return;
    freeBytes_ += b->length();
    freeBuffers_[b->length()].push_back(std::move(b));
}

void CommandQueue::giveSlabs(std::vector<std::shared_ptr<Buffer>>* slabs) {
    for (auto& b : *slabs) giveBuffer(std::move(b));
    slabs->clear();
}

std::shared_ptr<Buffer> CommandQueue::programBuffer(const Kernel& k, std::string* why) {
    std::lock_guard<std::mutex> g(lock_);
    const auto [first, last] = resident_.equal_range(k.words);
    for (auto it = first; it != last;) {
        const Resident& r = it->second;
        if (r.words.size() == k.numWords && std::memcmp(r.words.data(), k.words, 4ull * k.numWords) == 0) return r.buffer;
        residentBytes_ -= 4ull * r.words.size();
        it = resident_.erase(it);   // (the pointer's words have changed: a freed program's)
    }
    // (past 64 MB of programs, the ones kept are let go: a command buffer in flight keeps its own)
    if (residentBytes_ > (64ull << 20)) {
        resident_.clear();
        residentBytes_ = 0;
    }
    std::shared_ptr<Buffer> b = newBuffer(gpu_, 4ull * std::max<uint32_t>(k.numWords, 1), why);
    if (b == nullptr || b->contents() == nullptr) return nullptr;
    memcpy(b->contents(), k.words, 4ull * k.numWords);
    resident_.emplace(k.words, Resident{std::vector<uint32_t>(k.words, k.words + k.numWords), b});
    residentBytes_ += 4ull * k.numWords;
    return b;
}

namespace {
// the live surface buffers, by (GPU, kernel handle): the kernel gives this client one handle per surface, so two
// Buffers over it would close it twice. The deleter takes the lock too, so that a handle the kernel hands out again
// cannot be met by a Buffer still being destroyed.
std::mutex gSurfaceLock;
std::map<std::pair<const Gpu*, uint32_t>, std::weak_ptr<Buffer>> gSurfaceBuffers;
}  // namespace

std::shared_ptr<Buffer> surfaceBuffer(Gpu& gpu, uint32_t surfaceId, SurfaceInfo* info, std::string* why) {
    std::lock_guard<std::mutex> g(gSurfaceLock);
    int error = 0;
    std::unique_ptr<Bo> bo = gpu.surfaceBo(surfaceId, info, &error);
    if (bo == nullptr) {
        *why = "IOSurface " + std::to_string(surfaceId) + ": the driver gave no buffer for it: " +
               (error == ENOENT ? std::string("no such surface") : error == ENOTSUP ? std::string("a surface around a program's own memory is not one the GPU takes") : strerror(error));
        return nullptr;
    }
    const auto key = std::make_pair(static_cast<const Gpu*>(&gpu), bo->handle());
    auto live = gSurfaceBuffers.find(key);
    if (live != gSurfaceBuffers.end()) {
        if (std::shared_ptr<Buffer> b = live->second.lock()) {
            gpu.disown(std::move(bo));   // the same handle as the live Buffer's Bo: closed once, by that Bo
            return b;
        }
    }
    const uint64_t length = bo->size();
    std::shared_ptr<Buffer> b(new Buffer(std::move(bo), length), [key](Buffer* x) {
        std::lock_guard<std::mutex> g(gSurfaceLock);
        auto it = gSurfaceBuffers.find(key);
        if (it != gSurfaceBuffers.end() && it->second.expired()) gSurfaceBuffers.erase(it);
        delete x;
    });
    gSurfaceBuffers[key] = b;
    return b;
}

// --- CommandBuffer ----------------------------------------------------------------------------------------------

bool CommandBuffer::fail(const std::string& why) {
    if (error_.empty()) {
        error_ = why;
    }
    return false;
}

void CommandBuffer::stamp(uint32_t slot) {
    if (profileStamps_ == nullptr) {
        std::string why;
        profileStamps_ = queue_.takeBuffer(16 * kMaxProfiled, &why);
        if (profileStamps_ == nullptr) {
            profile_ = tuning_ = false;
            return;
        }
        memset(profileStamps_->contents(), 0, 16 * kMaxProfiled);
        keep(profileStamps_);
    }
    const uint64_t at = profileStamps_->gpuAddress() + 8ull * slot;
    const uint32_t w[3] = {a6xx::CP_REG_TO_MEM_0_REG(0x0980) | a6xx::CP_REG_TO_MEM_0_CNT(2) | a6xx::CP_REG_TO_MEM_0_64B, uint32_t(at),
                           uint32_t(at >> 32)};
    stream_.packet(a6xx::CP_REG_TO_MEM, w, 3);
    use(*profileStamps_, true);
}

std::shared_ptr<Buffer> CommandQueue::vscBuffer(std::string* why) {
    std::lock_guard<std::mutex> g(lock_);
    if (vsc_ == nullptr) vsc_ = newBuffer(gpu_, render::vscBytes(), why);
    return vsc_;
}

void CommandQueue::checkVsc() {
    std::lock_guard<std::mutex> g(lock_);
    if (vsc_ == nullptr || vsc_->contents() == nullptr) return;
    // (where render::encodeGmem tells the command processor to put it: past the draw streams of the chip's own number
    // of pipes -- 32 on the A635, 16 on the A610, where a fixed 32 read a primitive stream's words instead and an
    // overrun went unseen)
    const uint32_t word = *reinterpret_cast<const volatile uint32_t*>(static_cast<const uint8_t*>(vsc_->contents()) + render::vscOverflowWordAt());
    if (word != 0) vscOverflowed_ = true;
}

bool CommandQueue::preferGmem(uint64_t key) {
    std::lock_guard<std::mutex> g(lock_);
    Tune& t = tune_[key];
    t.passes++;
    // (sysmem first: a kind of pass seen once may not come again, and an untried GMEM pass is the dearer gamble -- 5 ms
    // against 0.13 for a light one)
    if (t.runs[0] == 0) return false;
    if (t.runs[1] == 0) return true;
    const bool better = t.us[1] < t.us[0];
    // (the other mode again now and then, what a pass costs drifting: every 16th pass, or every 64th when the one is
    // dearer by half -- a light pass of CoreAnimation took 5 ms in GMEM against 0.13 in sysmem)
    const double ratio = better ? t.us[0] / t.us[1] : t.us[1] / t.us[0];
    return t.passes % (ratio > 1.5 ? 64 : 16) == 0 ? !better : better;
}

namespace {
uint64_t kindKey(uint64_t kind, uint32_t ordinal) { return kind * 0x100000001b3ull ^ ordinal; }
}  // namespace

uint32_t CommandQueue::lastDraws(uint64_t kind, uint32_t ordinal) {
    std::lock_guard<std::mutex> g(lock_);
    auto it = lastDraws_.find(kindKey(kind, ordinal));
    return it == lastDraws_.end() ? 0 : it->second;
}

void CommandQueue::noteDraws(uint64_t kind, uint32_t ordinal, uint32_t draws) {
    std::lock_guard<std::mutex> g(lock_);
    lastDraws_[kindKey(kind, ordinal)] = draws;
    Peak& p = peakDraws_[kindKey(kind, ordinal)];
    if (draws >= p.draws || ++p.below >= 16) p.draws = draws, p.below = 0;
}

uint32_t CommandQueue::peakDraws(uint64_t kind, uint32_t ordinal) {
    std::lock_guard<std::mutex> g(lock_);
    auto it = peakDraws_.find(kindKey(kind, ordinal));
    return it == peakDraws_.end() ? 0 : it->second.draws;
}

void CommandQueue::tuned(uint64_t key, bool gmem, double us) {
    if (!(us > 0)) return;
    std::lock_guard<std::mutex> g(lock_);
    Tune& t = tune_[key];
    double& avg = t.us[gmem ? 1 : 0];
    avg = t.runs[gmem ? 1 : 0]++ == 0 ? us : 0.75 * avg + 0.25 * us;
}

void CommandBuffer::stampStart(PassKind kind, uint32_t width, uint32_t height) {
    if (!(profile_ || tuning_) || profiled_.size() >= kMaxProfiled) return;
    { PassTime t{}; t.kind = kind, t.width = width, t.height = height; profiled_.push_back(std::move(t)); }
    stamp(uint32_t(2 * (profiled_.size() - 1)));
    if (profileSamples_) sampleStamp(uint32_t(2 * (profiled_.size() - 1)));
}

void CommandBuffer::stampEnd() {
    if (!(profile_ || tuning_) || profiled_.empty()) return;
    stamp(uint32_t(2 * (profiled_.size() - 1) + 1));
    if (profileSamples_) sampleStamp(uint32_t(2 * (profiled_.size() - 1) + 1));
}

// The RB's running sample count into slot SLOT: RB_SAMPLE_COUNT_CONTROL.COPY, the address, ZPASS_DONE (turnip's
// tu_autotune_begin_renderpass / _end_renderpass, A6XX path). The copy lands when the RB has it; the submission's own
// end (the kernel's CP_WAIT_FOR_IDLE and CACHE_FLUSH_TS before the fence) is past it when the buffer completes, which
// is when passTimes reads it.
void CommandBuffer::sampleStamp(uint32_t slot) {
    if (sampleStamps_ == nullptr) {
        std::string why;
        sampleStamps_ = queue_.takeBuffer(16 * kMaxProfiled, &why);
        if (sampleStamps_ == nullptr) {
            profileSamples_ = false;
            return;
        }
        memset(sampleStamps_->contents(), 0, 16 * kMaxProfiled);
        keep(sampleStamps_);
    }
    const uint64_t at = sampleStamps_->gpuAddress() + 8ull * slot;
    const uint32_t control = a6xx::A6XX_RB_SAMPLE_COUNT_CONTROL_COPY, address[2] = {uint32_t(at), uint32_t(at >> 32)};
    stream_.registers(a6xx::REG_A6XX_RB_SAMPLE_COUNT_CONTROL, &control, 1);
    stream_.registers(a6xx::REG_A6XX_RB_SAMPLE_COUNT_ADDR, address, 2);
    const uint32_t event = a6xx::eventWrite0(a6xx::ZPASS_DONE);
    stream_.packet(a6xx::CP_EVENT_WRITE, &event, 1);
    use(*sampleStamps_, true);
}

void CommandBuffer::drawStamp(uint32_t slot) {
    if (drawStamps_ == nullptr) {
        std::string why;
        drawStamps_ = queue_.takeBuffer(16 * kMaxProfiledDraws, &why);
        if (drawStamps_ == nullptr) {
            profileDraws_ = false;
            return;
        }
        memset(drawStamps_->contents(), 0, 16 * kMaxProfiledDraws);
        keep(drawStamps_);
    }
    const uint64_t at = drawStamps_->gpuAddress() + 8ull * slot;
    stream_.waitForIdle();
    const uint32_t w[3] = {a6xx::CP_REG_TO_MEM_0_REG(0x0980) | a6xx::CP_REG_TO_MEM_0_CNT(2) | a6xx::CP_REG_TO_MEM_0_64B, uint32_t(at),
                           uint32_t(at >> 32)};
    stream_.packet(a6xx::CP_REG_TO_MEM, w, 3);
    use(*drawStamps_, true);
}

std::vector<CommandBuffer::DrawTime> CommandBuffer::drawTimes() const {
    std::vector<DrawTime> out = drawsProfiled_;
    if (drawStamps_ == nullptr) return {};
    const auto* t = static_cast<const uint64_t*>(drawStamps_->contents());
    for (size_t i = 0; i < out.size(); i++) {
        out[i].us = t[2 * i + 1] > t[2 * i] ? double(t[2 * i + 1] - t[2 * i]) / 19.2 : 0.0;
    }
    return out;
}

std::vector<CommandBuffer::PassTime> CommandBuffer::passTimes() const {
    std::vector<PassTime> out = profiled_;
    if (profileStamps_ == nullptr) return {};
    const auto* t = static_cast<const uint64_t*>(profileStamps_->contents());
    for (size_t i = 0; i < out.size(); i++) {
        out[i].us = t[2 * i + 1] > t[2 * i] ? double(t[2 * i + 1] - t[2 * i]) / 19.2 : 0.0;
    }
    if (sampleStamps_ != nullptr) {
        const auto* c = static_cast<const uint64_t*>(sampleStamps_->contents());
        for (size_t i = 0; i < out.size(); i++) {
            if (c[2 * i + 1] >= c[2 * i]) out[i].samples = int64_t(c[2 * i + 1] - c[2 * i]);
        }
    }
    return out;
}

bool CommandBuffer::refuseWork(const std::string& why) {
    if (!skipRefused_) {
        return fail(why);
    }
    skipped_++;
    lastSkipped_ = why;
    return false;
}

void CommandBuffer::addCompletedHandler(std::function<void(CommandBuffer&)> handler) {
    if (status_ <= kEnqueued) {
        handlers_.push_back(std::move(handler));
    }
}

void CommandBuffer::enqueue() {
    Status expected = kNotEnqueued;
    status_.compare_exchange_strong(expected, kEnqueued);
}

void CommandBuffer::hsrLetGo() {
    for (auto& k : hsrHeld_) {
        std::lock_guard<std::mutex> g(k->lock);
        if (k->owner == this) k->owner = nullptr;
    }
    hsrHeld_.clear();
    if (hsrClock_ == nullptr) return;
    std::lock_guard<std::mutex> g(hsrClock_->lock);
    if (hsrClock_->owner == this) hsrClock_->owner = nullptr;
    hsrClock_ = nullptr;
}

void CommandBuffer::commit() {
    if (status_ > kEnqueued) {
        return;  // Metal asserts here; committing twice does nothing at all in this driver
    }
    // (the HSR depth's clock to the next buffer only now: what this one drew with it is ahead of it in the queue)
    hsrLetGo();
    queue_.submit(shared_from_this());
}

void CommandBuffer::waitUntilCompleted() {
    std::unique_lock<std::mutex> g(lock_);
    // (a buffer never committed is not waited for: Metal would wait forever, which helps nobody)
    done_.wait(g, [this] { return finished_ || status_ <= kEnqueued; });
}

CommandBuffer::~CommandBuffer() {
    hsrLetGo();   // (a buffer let go without a commit: its run of Z levels never reaches the GPU)
    // (a buffer never completed may be destroyed with the GPU still on it -- an error -- and its stamps are then not
    // given back; one that completed has had its readers)
    if (finished_) {
        queue_.giveBuffer(std::move(profileStamps_));
        queue_.giveBuffer(std::move(drawStamps_));
        queue_.giveBuffer(std::move(sampleStamps_));
    }
}

void CommandBuffer::finish(Status s) {
    status_ = s;
    if (s == kCompleted && counts_.binnedPasses != 0) queue_.checkVsc();
    if (s == kCompleted && tuning_) {
        for (const PassTime& t : passTimes()) {
            if (t.tuneKey != 0) queue_.tuned(t.tuneKey, t.tiles != 0, t.us);
        }
    }
    for (auto& h : handlers_) {
        h(*this);
    }
    handlers_.clear();
    kept_.clear();
    slab_.reset();
    queue_.giveSlabs(&slabs_);   // (done with: the GPU has finished with this buffer, or never had it)
    keptAny_.clear();
    // (and what its stream and lists grew to, and the programs it had at hand, to the queue's next command buffer)
    streamWords_ = stream_.size();
    stream_.clear();
    uses_.clear();
    queue_.giveSpare(CommandQueue::Spare{std::move(stream_), std::move(uses_), std::move(kept_), std::move(keptAny_), std::move(drawCache_)});
    stream_ = pm4::Stream{};
    uses_.clear();
    kept_.clear();
    keptAny_.clear();
    drawCache_.reset();
    {
        std::lock_guard<std::mutex> g(lock_);
        finished_ = true;
    }
    done_.notify_all();
}

bool CommandBuffer::copy(const Buffer& from, uint64_t fromOffset, const Buffer& to, uint64_t toOffset, uint64_t length) {
    closePass();
    if (status_ > kEnqueued) {
        return fail("copy after commit");
    }
    if (fromOffset > from.length() || length > from.length() - fromOffset || toOffset > to.length() ||
        length > to.length() - toOffset) {
        return fail("copy outside a buffer");
    }
    if (length == 0) {
        return true;
    }
    if (fromOffset % 4 || toOffset % 4 || length % 4) {
        // the command processor moves words; a copy of bytes goes through the 2D engine instead (a pass of its own).
        // MapKit's snapshot service copies rows of 2 bytes into its textures.
        std::string why;
        Piece table;
        if (!piece(render::kTableBytes, &table, &why)) {
            return fail(std::string("no memory for the copy's table: ") + why);
        }
        pm4::Stream added;
        render::encodeBytes(added, from.gpuAddress() + fromOffset, to.gpuAddress() + toOffset, length, table.gpuAddress(), table.contents());
        appendStream(stream_, added);
        use(from, false);
        use(to, true);
        return true;
    }
    uint64_t words = length / 4, done = 0;
    while (done < words) {
        uint32_t n = uint32_t(std::min<uint64_t>(words - done, kMemcpyMax));
        stream_.memcpy(to.gpuAddress() + toOffset + 4 * done, from.gpuAddress() + fromOffset + 4 * done, n);
        // what comes after may read what this wrote: the copy lands before the next packet is taken
        stream_.waitMemWrites();
        stream_.waitForMe();
        done += n;
    }
    use(from, false);
    use(to, true);
    return true;
}

bool CommandBuffer::fill(const Buffer& b, uint64_t offset, uint64_t length, uint8_t value) {
    closePass();
    if (status_ > kEnqueued) {
        return fail("fill after commit");
    }
    if (offset % 4 || length % 4) {
        return fail("fill of " + std::to_string(length) + " bytes at " + std::to_string(offset) +
                    ": the command processor writes words, so both must be multiples of 4");
    }
    if (offset > b.length() || length > b.length() - offset) {
        return fail("fill outside the buffer");
    }
    if (length == 0) {
        return true;
    }
    uint32_t word = uint32_t(value) * 0x01010101u;
    uint64_t words = length / 4, base = b.gpuAddress() + offset;
    // the first words written as they are, ...
    uint32_t head = uint32_t(std::min<uint64_t>(words, kFillHead));
    std::vector<uint32_t> pattern(head, word);
    stream_.memWrite(base, pattern.data(), head);
    stream_.waitMemWrites();
    stream_.waitForMe();
    // ... and the rest copied from what is already filled, twice as much each time
    uint64_t have = head;
    while (have < words) {
        uint32_t n = uint32_t(std::min<uint64_t>(std::min<uint64_t>(have, words - have), kMemcpyMax));
        stream_.memcpy(base + 4 * have, base, n);
        stream_.waitMemWrites();
        stream_.waitForMe();
        have += n;
    }
    use(b, true);
    return true;
}

// --- CommandQueue -----------------------------------------------------------------------------------------------

std::unique_ptr<CommandQueue> CommandQueue::make(Gpu& gpu, std::string* why) {
    uint32_t id = 0;
    int e = gpu.newQueue(0, &id);
    if (e != 0) {
        *why = std::string("the driver gave no submission queue: ") + strerror(e);
        return nullptr;
    }
    std::unique_ptr<CommandQueue> q(new CommandQueue(gpu, id));
    q->thread_ = std::thread([p = q.get()] { p->waiter(); });
    return q;
}

CommandQueue::~CommandQueue() {
    {
        std::lock_guard<std::mutex> g(lock_);
        quit_ = true;
    }
    changed_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
    idle_.clear();
    for (CommandBuffer::OpenPass* p : sparePasses_) delete p;
    sparePasses_.clear();
    spares_.clear();
    gpu_.closeQueue(id_);
}

std::shared_ptr<CommandBuffer> CommandQueue::commandBuffer() {
    std::shared_ptr<CommandBuffer> cb(new CommandBuffer(*this));
    Spare spare;
    if (takeSpare(&spare)) {
        cb->stream_ = std::move(spare.stream);
        cb->uses_ = std::move(spare.uses);
        cb->kept_ = std::move(spare.kept);
        cb->keptAny_ = std::move(spare.keptAny);
        cb->drawCache_ = std::move(spare.drawCache);
    }
    return cb;
}

void CommandQueue::giveSpare(Spare&& spare) {
    std::lock_guard<std::mutex> g(poolLock_);
    if (spares_.size() < 4) spares_.push_back(std::move(spare));
}

bool CommandQueue::takeSpare(Spare* spare) {
    std::lock_guard<std::mutex> g(poolLock_);
    if (spares_.empty()) return false;
    *spare = std::move(spares_.back());
    spares_.pop_back();
    return true;
}

std::shared_ptr<Texture> CommandQueue::nullTexture(std::string* why) {
    std::lock_guard<std::mutex> g(lock_);
    if (null_ == nullptr) {
        TextureDesc d;
        d.format = PixelFormat::kRGBA8Unorm;
        std::shared_ptr<Texture> t = Texture::make(gpu_, d, why);
        // (written, not taken as the kernel's fresh zeros: the zero is the claim, so it is put there in so many words)
        const uint32_t zero = 0;
        if (t == nullptr || !t->replace(0, 0, 1, 1, &zero, 4, why)) return nullptr;
        null_ = std::move(t);
    }
    return null_;
}

CommandQueue::HsrTarget CommandQueue::hsrDepth(uint32_t width, uint32_t height, std::string* why, uint8_t kind, bool lrz) {
    std::lock_guard<std::mutex> g(lock_);
    const uint8_t key = uint8_t(kind | (lrz ? 0x80 : 0));
    for (size_t i = 0; i < hsrDepths_.size(); i++) {
        const HsrKept& k = hsrDepths_[i];
        if (k.kind != key || k.t.depth->width() != width || k.t.depth->height() != height) continue;
        HsrKept t = k;
        hsrDepths_.erase(hsrDepths_.begin() + long(i));
        hsrDepths_.push_back(t);
        return t.t;
    }
    TextureDesc d;
    d.format = kind == CommandBuffer::kHsrDepth32F ? PixelFormat::kDepth32Float : PixelFormat::kDepth16Unorm;
    d.compressed = kind == CommandBuffer::kHsrDepth16Ubwc;
    d.width = width;
    d.height = height;
    d.usage = kTextureUsageRenderTarget;
    HsrTarget t;
    t.clock = std::make_shared<HsrClock>();
    t.depth = Texture::make(gpu_, d, why);
    if (t.depth == nullptr) return HsrTarget{};
    if (lrz) {
        // (GRAS_LRZ_BUFFER_BASE is 256-aligned, a6xx.xml:3387: a buffer of its own is page-aligned)
        t.lrz = newBuffer(gpu_, render::lrzLayout(width, height).bytes(), why);
        if (t.lrz == nullptr) return HsrTarget{};
    }
    if (hsrDepths_.size() >= kHsrDepthSizes) hsrDepths_.erase(hsrDepths_.begin());
    hsrDepths_.push_back(HsrKept{key, t});
    return t;
}

namespace {
// the A635's SP cores and fibers an SP (freedreno_devices.py, the 7c+ Gen 3 entry: num_ccu 2, fibers_per_sp 128 * 2 * 16)
constexpr uint32_t kSpCores = 2;
constexpr uint32_t kFibersPerSp = 128 * 2 * 16;
}  // namespace

std::shared_ptr<Buffer> CommandQueue::privateMemory(uint32_t bytes, bool perWave, PrivateMemory* out, std::string* why) {
    std::lock_guard<std::mutex> g(lock_);
    const int l = perWave ? 1 : 0;
    if (privatePerFiber_[l] < bytes) {
        uint32_t perFiber = 512;
        while (perFiber < bytes) perFiber *= 2;
        const uint64_t perSp = (uint64_t(perFiber) * kFibersPerSp + 4095) & ~uint64_t(4095);
        std::shared_ptr<Buffer> b = newBuffer(gpu_, perSp * kSpCores, why);
        if (b == nullptr) return nullptr;
        private_[l].push_back(std::move(b));
        privatePerFiber_[l] = perFiber;
    }
    const uint32_t perFiber = privatePerFiber_[l];
    out->address = private_[l].back()->gpuAddress();
    out->perFiber = perFiber;
    out->perSp = uint32_t((uint64_t(perFiber) * kFibersPerSp + 4095) & ~uint64_t(4095));
    out->perWave = perWave;
    return private_[l].back();
}

size_t CommandQueue::idleCommandBuffers() {
    std::lock_guard<std::mutex> g(lock_);
    return idle_.size();
}

std::unique_ptr<Bo> CommandQueue::takeCommandBo(size_t bytes, int* error) {
    {
        std::lock_guard<std::mutex> g(lock_);
        for (auto it = idle_.begin(); it != idle_.end(); ++it) {
            if ((*it)->size() >= bytes) {
                std::unique_ptr<Bo> bo = std::move(*it);
                idle_.erase(it);
                return bo;
            }
        }
    }
    return gpu_.newBo(std::max(bytes, kCommandBo), kBoGpuReadOnly | kBoWriteCombined, error);
}

void CommandQueue::submit(std::shared_ptr<CommandBuffer> cb) {
    cb->committedAt_ = gpu_.transport().nowNs();
    if (!cb->error_.empty()) {
        cb->finish(CommandBuffer::kError);   // a blit that was refused: nothing of it reaches the GPU
        return;
    }
    cb->closePass();   // the last pass's tail
    // every command buffer ends with the GPU idle: what Metal's completion promises, that the work is done
    cb->stream_.waitMemWrites();
    cb->stream_.waitForIdle();
    const std::vector<uint32_t>& w = cb->stream_.words();
    int error = 0;
    cb->commands_ = takeCommandBo(4 * w.size(), &error);
    if (cb->commands_ == nullptr) {
        cb->error_ = std::string("no memory for the command stream: ") + strerror(error);
        cb->finish(CommandBuffer::kError);
        return;
    }
    // the stream is written in order, as write-combined memory wants it, and read back by nobody
    memcpy(cb->commands_->cpu(), w.data(), 4 * w.size());
    std::unique_lock<std::mutex> g(lock_);  // the fences of one queue are handed out in commit order
    cb->status_ = CommandBuffer::kCommitted;
    cb->faultsAtCommit_ = gpu_.faults();
    int e = gpu_.submit(id_, *cb->commands_, 0, uint32_t(w.size()), cb->uses_, &cb->fence_);
    if (e != 0) {
        cb->error_ = std::string("the driver refused the submission: ") + strerror(e);
        idle_.push_back(std::move(cb->commands_));
        g.unlock();                         // (the handlers may commit more: not under the queue's lock)
        cb->finish(CommandBuffer::kError);
        return;
    }
    cb->status_ = CommandBuffer::kScheduled;
    inFlight_.push_back(std::move(cb));
    changed_.notify_all();
}

void CommandQueue::waiter() {
    std::unique_lock<std::mutex> g(lock_);
    for (;;) {
        changed_.wait(g, [this] { return quit_ || !inFlight_.empty(); });
        if (inFlight_.empty()) {
            return;  // quitting, and nothing left in flight
        }
        std::shared_ptr<CommandBuffer> cb = inFlight_.front();
        g.unlock();
        int e = gpu_.wait(id_, cb->fence_, kPatienceNs);
        cb->completedAt_ = gpu_.transport().nowNs();
        // A fence that passed is not yet a submission that ran. When the kernel resets a hung GPU (gpu_recover=1) it
        // gives up everything the GPU had been given, the hung submission and those behind it in the ring, by setting
        // the fence word to the last submission's number -- so their waits end as if they had run (hakios_gpu_recover:
        // "submissions up to N are given up as done"). FAULTS counts the resets; one more since the commit says so.
        uint64_t faults = gpu_.faults();
        bool givenUp = faults > cb->faultsAtCommit_;
        if (e == 0 && givenUp) {
            cb->error_ = "given up by a reset of the GPU (it hung on this submission or on one before it): not run";
            e = EIO;
        } else if (e == ETIMEDOUT) {
            cb->error_ = givenUp ? "the GPU did not finish it in 5 s: it hung, and the kernel has reset it (reset " +
                                       std::to_string(faults) + " of this run)"
                                 : "the GPU did not finish it in 5 s and is not answering; without gpu_recover=1 the "
                                   "kernel waits for nothing again in this run";
        } else if (e != 0) {
            cb->error_ = std::string("waiting for it failed: ") + strerror(e);
        }
        cb->finish(e == 0 ? CommandBuffer::kCompleted : CommandBuffer::kError);
        g.lock();
        inFlight_.pop_front();
        if (cb->commands_ != nullptr) {
            idle_.push_back(std::move(cb->commands_));
        }
    }
}

}  // namespace hakios::metal
