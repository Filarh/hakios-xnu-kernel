// Render-pass memoization: passmemo.hpp says what and why. Here: what a pass and its draws come to as they are
// recorded (memoOpen, memoDraw), what is kept of a closed one (memoClose, memoStore), and the choice at commit
// (memoResolve), which turns a hit's words into a copy of the kept result.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "hakios-metal/command.hpp"
#include "hakios-metal/passmemo.hpp"
#include "hakios-metal/render.hpp"
#include "hakios-metal/texture.hpp"

namespace hakios::metal {

void appendStream(pm4::Stream& to, const pm4::Stream& from);   // render.cpp's

namespace {

// the level a pass draws into: its size (render::windowOf's rule for colour 0)
void levelSize(const RenderPass& pass, uint32_t* w, uint32_t* h) {
    const TextureLevel L = pass.color0->level(pass.level);
    *w = L.width;
    *h = L.height;
}

// The rasterizer's rule, with a margin for its snapping of vertices to sub-pixels (1/256 of a pixel on this GPU; 1/64
// here): a pixel is drawn when its centre lies inside the primitive.
constexpr double kSnap = 1.0 / 64;

// the pixels a rectangle of vertices surely covers (every one whose centre lies inside it by more than the margin), as
// an inclusive X0 Y0 X1 Y1 clipped to the level; false when none
bool coveredPixels(const float box[4], uint32_t w, uint32_t h, uint32_t out[4]) {
    const double x0 = std::ceil(double(box[0]) - 0.5 + kSnap), y0 = std::ceil(double(box[1]) - 0.5 + kSnap);
    const double x1 = std::ceil(double(box[2]) - 0.5 - kSnap) - 1, y1 = std::ceil(double(box[3]) - 0.5 - kSnap) - 1;
    if (!std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(x1) || !std::isfinite(y1)) return false;
    const double cx0 = std::max(0.0, x0), cy0 = std::max(0.0, y0), cx1 = std::min(double(w) - 1, x1), cy1 = std::min(double(h) - 1, y1);
    if (!(cx0 <= cx1 && cy0 <= cy1)) return false;
    out[0] = uint32_t(cx0), out[1] = uint32_t(cy0), out[2] = uint32_t(cx1), out[3] = uint32_t(cy1);
    return true;
}

// ...and every pixel a primitive inside the box can touch (a centre within the margin of it), clipped to the level
bool boundPixels(const float box[4], uint32_t w, uint32_t h, uint32_t out[4]) {
    const double x0 = std::ceil(double(box[0]) - 0.5 - kSnap), y0 = std::ceil(double(box[1]) - 0.5 - kSnap);
    const double x1 = std::floor(double(box[2]) - 0.5 + kSnap), y1 = std::floor(double(box[3]) - 0.5 + kSnap);
    if (!std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(x1) || !std::isfinite(y1)) return false;
    const double cx0 = std::max(0.0, x0), cy0 = std::max(0.0, y0), cx1 = std::min(double(w) - 1, x1), cy1 = std::min(double(h) - 1, y1);
    if (!(cx0 <= cx1 && cy0 <= cy1)) return false;
    out[0] = uint32_t(cx0), out[1] = uint32_t(cy0), out[2] = uint32_t(cx1), out[3] = uint32_t(cy1);
    return true;
}

bool inside(const uint32_t a[4], const uint32_t b[4]) { return a[0] >= b[0] && a[1] >= b[1] && a[2] <= b[2] && a[3] <= b[3]; }

// whether the draw's fragment stage writes colour attachment 0
bool writesColor0(const Program& p) {
    return p.fs.numColorOutputs > 0 && p.fs.colorRegid[0] != a6xx::kRegidNone && (p.blend.writeMask & 0xf) != 0;
}

// the memory a texture's contents live in, and whether the driver sees every write of it
bool volatileMemory(const Texture& t) {
    return t.surface() != 0 || t.buffer()->bo().surface() || t.overBufferRows() || t.buffer()->programMemory();
}

uint64_t rowsBytes(const Texture& t) { return uint64_t(t.bytesPerRow()) * t.height(); }

// a NOP chain over WORDS words at AT (the command processor skips a NOP's payload, whatever it holds)
void nopOver(uint32_t* at, size_t words) {
    while (words > 0) {
        const uint32_t n = uint32_t(std::min<size_t>(words - 1, pm4::kMaxType7Count));
        at[0] = pm4::type7(pm4::kNop, n);
        at += 1 + n;
        words -= 1 + n;
    }
}

int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

// --- the queue's side ----------------------------------------------------------------------------------------------

void CommandQueue::setMemo(uint32_t mode, uint32_t minDraws, uint64_t hashCap) {
    std::lock_guard<std::mutex> g(memoLock_);
    memoMinDraws_ = std::max(1u, minDraws);
    memoHashCap_ = hashCap;
    memoMode_.store(mode > memo::kCover ? uint32_t(memo::kCover) : mode, std::memory_order_relaxed);
    if (mode == memo::kOff) memo_.clear();
}

void CommandQueue::setMemoVersion(std::function<bool(const Texture&, uint64_t*)> hook) {
    std::lock_guard<std::mutex> g(memoLock_);
    memoVersion_ = std::move(hook);
}

memo::Stats CommandQueue::memoStats() {
    std::lock_guard<std::mutex> g(memoLock_);
    memo::Stats s = memoStats_;
    s.cacheBytes = 0;
    for (const auto& [slot, e] : memo_) {
        if (e.cache != nullptr) s.cacheBytes += e.cache->buffer()->length();
    }
    return s;
}

void CommandQueue::memoForget() {
    std::lock_guard<std::mutex> g(memoLock_);
    memo_.clear();
}

// --- the command buffer's side ----------------------------------------------------------------------------------------

void CommandBuffer::memoOpen(const RenderPass& pass, const render::Options& options) {
    memo::Pass& m = open_->memo;
    m = memo::Pass{};
    const uint32_t mode = queue_.memoMode();
    if (mode == memo::kOff) return;
    m.on = m.ok = true;
    m.begin = stream_.size();
    m.profiled = lastStamped_;
    const Texture* c0 = pass.color0;
    if (c0 == nullptr) return m.refuse("no colour attachment 0");
    if (c0->memoryless()) return m.refuse("a memoryless colour attachment 0");
    if (c0->samples() != 1) return m.refuse("a multisample attachment");
    if (c0->type() != TextureType::k2D || pass.slice != 0) return m.refuse("attachment 0 not a 2D texture");
    if (pass.depth != nullptr || pass.stencil != nullptr) return m.refuse("a depth or stencil attachment");
    if (pass.store != StoreAction::kStore) return m.refuse("attachment 0 not stored");
    if (pass.load == LoadAction::kLoad && mode < memo::kCover) return m.refuse("attachment 0 loaded (metal_passmemo=2 takes a vouched cover)");
    memo::Hash& h = m.structure;
    uint32_t w = 0, hgt = 0;
    levelSize(pass, &w, &hgt);
    h.u64(uint64_t(c0->format()) << 32 | uint32_t(pass.load) << 8 | uint32_t(pass.store));
    h.u64(uint64_t(w) << 32 | hgt);
    h.u64(pass.level);
    if (pass.load == LoadAction::kClear) h.pod(pass.clearColor);
    for (uint32_t i = 0; i < kMaxColorOutputs - 1; i++) {
        const ColorAttachment& a = pass.more[i];
        if (a.texture == nullptr) continue;
        if (!a.texture->memoryless()) return m.refuse("an attachment past 0 that is not memoryless");
        if (a.texture->samples() != 1) return m.refuse("a multisample attachment");
        h.u64(uint64_t(i + 1) << 48 | uint64_t(a.texture->format()) << 16 | uint32_t(a.load) << 8 | uint32_t(a.store));
        if (a.load == LoadAction::kClear) h.pod(a.clearColor);
    }
    h.pod(options);
}

void CommandBuffer::memoDraw(const Program& program, const RenderPass& pass, const Draw& d, const Bindings& vsBindings, const Bindings& fsBindings,
                             const render::Options& options, bool first, const memo::Hint& hint) {
    (void)first;
    if (open_ == nullptr) return;
    memo::Pass& m = open_->memo;
    if (!m.on || !m.ok) return;
    const LoadAction load = open_->pass.load;   // (the pass's as it opened: a draw that joins it says Load)
    memo::Hash& h = m.structure;
    if (program.uid == 0) return m.refuse("a program without a uid");
    if (program.vs.k.numSsbos != 0 || program.fs.k.numSsbos != 0) return m.refuse("a stage with device buffers");
    if (d.indirect != nullptr) return m.refuse("an indirect draw");
    if (d.visibilityMode != 0) return m.refuse("an occlusion query");
    h.u64(0xd7a3ull << 48 | m.draws);
    h.u64(program.uid);
    h.u64(uint64_t(d.primitive) | uint64_t(d.cullMode) << 8 | uint64_t(d.winding) << 16 | uint64_t(d.scissorEmpty) << 24);
    h.u64(uint64_t(d.vertexStart) << 32 | d.vertexCount);
    h.u64(uint64_t(d.instanceCount) << 32 | d.baseInstance);
    h.u64(uint64_t(uint32_t(d.baseVertex)) << 32 | d.indexType);
    h.pod(d.viewport);   // (six doubles, and four words below: no padding)
    h.pod(d.scissor);
    h.pod(d.blendColor);
    // (field by field: a struct's padding is whatever the stack held, and hashed it made one frame's draw another's)
    h.u64(uint64_t(d.depth.compare) << 8 | uint64_t(d.depth.write));
    h.u64(uint64_t(d.stencil.enabled));
    if (d.stencil.enabled) {
        for (const StencilFace* f : {&d.stencil.front, &d.stencil.back}) {
            h.u64(uint64_t(f->compare) << 48 | uint64_t(f->stencilFail) << 32 | uint64_t(f->depthFail) << 16 | f->pass);
            h.u64(uint64_t(f->readMask) << 32 | f->writeMask);
        }
        h.u64(uint64_t(d.stencil.frontReference) << 32 | d.stencil.backReference);
    }
    h.pod(options);
    const uint64_t cap = queue_.memoHashCap_;
    // the bound bytes: their contents at commit, their lengths now
    if (d.indexBuffer != nullptr) {
        const uint64_t bytes = uint64_t(d.vertexCount) * (d.indexType == 1 ? 4 : 2);
        if (d.indexBuffer->bo().surface()) return m.refuse("indices in an IOSurface");
        if (d.indexOffset > d.indexBuffer->length() || bytes > d.indexBuffer->length() - d.indexOffset) return m.refuse("indices past their buffer");
        if (bytes > cap) return m.refuse("more indices than the hash cap");
        h.u64(0x1d0ull << 48 | bytes);
        h.u64(m.rangeIndex(memo::Range{d.indexBuffer, d.indexOffset, bytes}));
    }
    const Texture* targets[kMaxColorOutputs] = {pass.color0};
    for (uint32_t i = 0; i < kMaxColorOutputs - 1; i++) targets[i + 1] = pass.more[i].texture;
    for (int stage = 0; stage < 2; stage++) {
        const Bindings& b = stage == 0 ? vsBindings : fsBindings;
        for (size_t i = 0; i < b.buffers.size(); i++) {
            const Binding& x = b.buffers[i];
            if (x.buffer == nullptr) continue;
            if (x.buffer->bo().surface()) return m.refuse("a buffer in an IOSurface");
            if (x.offset > x.buffer->length()) return m.refuse("a binding past its buffer");
            const uint64_t bytes = x.bytes != 0 ? std::min(x.bytes, x.buffer->length() - x.offset) : x.buffer->length() - x.offset;
            if (bytes > cap) return m.refuse("a binding of more bytes than the hash cap");
            h.u64(uint64_t(stage) << 56 | uint64_t(i) << 48 | bytes);
            h.u64(m.rangeIndex(memo::Range{x.buffer, x.offset, bytes}));
        }
        for (size_t i = 0; i < b.textures.size(); i++) {
            const Texture* t = b.textures[i];
            if (t == nullptr || ((b.nullMask >> i) & 1)) continue;
            for (const Texture* a : targets) {
                if (a != nullptr && a->buffer() == t->buffer()) return m.refuse("a draw samples its own attachment");
            }
            h.u64(uint64_t(stage) << 56 | uint64_t(i) << 48 | 0x7e7ull);
            h.u64(t->resourceId());
            h.u64(t->buffer()->writes());
            if (t->staging() != nullptr) h.u64(t->staging()->buffer()->writes());
            if (volatileMemory(*t)) {
                if (t->contents() == nullptr && !queue_.memoVersion_) return m.refuse("a texture written out of the driver's sight, not readable");
                if (rowsBytes(*t) > cap && !queue_.memoVersion_) return m.refuse("a texture written out of the driver's sight, larger than the hash cap");
                h.u64(m.volatileIndex(t, queue_.memoVersion_ ? 0 : rowsBytes(*t)));
            }
        }
        for (size_t i = 0; i < b.samplers.size(); i++) {
            if (b.samplers[i] == nullptr) continue;
            h.u64(uint64_t(stage) << 56 | uint64_t(i) << 48 | 0x5a3ull);
            h.pod(b.samplers[i]->words);
        }
        h.u64(b.nullMask);
    }
    if (m.budget > 4 * cap) return m.refuse("more bytes to hash than four times the cap");
    // what it can write of attachment 0
    if (writesColor0(program)) {
        uint32_t w = 0, hgt = 0;
        levelSize(pass, &w, &hgt);
        uint32_t a[4];
        if (!render::drawArea(pass, d, a)) {
            // (an empty scissor: nothing written)
        } else {
            uint32_t b[4];
            if (hint.set && boundPixels(hint.box, w, hgt, b)) {
                a[0] = std::max(a[0], b[0]), a[1] = std::max(a[1], b[1]), a[2] = std::min(a[2], b[2]), a[3] = std::min(a[3], b[3]);
            }
            if (a[0] <= a[2] && a[1] <= a[3]) {
                if (load == LoadAction::kLoad) {
                    if (m.draws == 0) {
                        // the cover: the first draw, an opaque rectangle the caller vouched for
                        const bool opaque = !program.blend.enabled && (program.blend.writeMask & 0xf) == 0xf && !program.fs.hasKill && !program.fs.fbRead &&
                                            d.depth.compare == 7 && !d.depth.write && !d.stencil.enabled && d.instanceCount == 1;
                        uint32_t c[4];
                        if (!hint.set || !hint.rect) return m.refuse("a Load pass whose first draw is not a vouched rectangle");
                        if (!opaque) return m.refuse("a Load pass whose first draw blends, discards or reads its target");
                        if (!coveredPixels(hint.box, w, hgt, c)) return m.refuse("a Load pass whose first draw covers nothing");
                        uint32_t s[4];
                        render::drawArea(pass, d, s);
                        c[0] = std::max(c[0], s[0]), c[1] = std::max(c[1], s[1]), c[2] = std::min(c[2], s[2]), c[3] = std::min(c[3], s[3]);
                        if (!(c[0] <= c[2] && c[1] <= c[3])) return m.refuse("a Load pass whose first draw is scissored away");
                        memcpy(m.cover, c, sizeof c);
                        m.covered = true;
                    } else if (!m.covered || !inside(a, m.cover)) {
                        return m.refuse("a draw outside the first draw's cover");
                    }
                }
                m.area[0] = std::min(m.area[0], a[0]), m.area[1] = std::min(m.area[1], a[1]);
                m.area[2] = std::max(m.area[2], a[2]), m.area[3] = std::max(m.area[3], a[3]);
            }
        }
    } else if (load == LoadAction::kLoad && m.draws == 0) {
        return m.refuse("a Load pass whose first draw does not write attachment 0");
    }
    m.draws++;
}

std::unique_ptr<CommandBuffer::MemoPending> CommandBuffer::memoClose(const RenderPass* next, bool kept) {
    memo::Pass& m = open_->memo;
    if (!m.on) return nullptr;
    const RenderPass& pass = open_->pass;
    auto refused = [&](const std::string& why) -> std::unique_ptr<MemoPending> {
        std::lock_guard<std::mutex> g(queue_.memoLock_);
        queue_.memoStats_.refused++;
        queue_.memoStats_.lastRefusal = why;
        if (m.profiled >= 0 && size_t(m.profiled) < profiled_.size()) profiled_[m.profiled].memo = 1;
        return nullptr;
    };
    if (!m.ok) return refused(m.why);
    if (m.draws < queue_.memoMinDraws_) return nullptr;   // (light: its copy would cost about what it does)
    // a memoryless attachment the next pass goes on with: what this one leaves there is not in the cache
    for (uint32_t i = 0; i < kMaxColorOutputs - 1; i++) {
        const Texture* t = pass.more[i].texture;
        if (t == nullptr) continue;
        if (kept) return refused("a memoryless attachment kept for an appended buffer");
        if (next != nullptr && next->more[i].texture == t && next->more[i].load == LoadAction::kLoad) return refused("the next pass loads a memoryless attachment");
    }
    uint32_t w = 0, hgt = 0;
    levelSize(pass, &w, &hgt);
    uint32_t r[4];   // inclusive
    if (pass.load == LoadAction::kClear) {
        r[0] = r[1] = 0, r[2] = w - 1, r[3] = hgt - 1;
    } else if (pass.load == LoadAction::kLoad) {
        if (!m.covered) return refused("a Load pass without a cover");
        memcpy(r, m.cover, sizeof r);
    } else {
        if (!(m.area[0] <= m.area[2] && m.area[1] <= m.area[3])) return nullptr;   // (writes nothing of attachment 0)
        memcpy(r, m.area, sizeof r);
    }
    auto p = std::make_unique<MemoPending>();
    p->begin = m.begin;
    p->end = stream_.size();
    p->rect[0] = r[0], p->rect[1] = r[1], p->rect[2] = r[2] - r[0] + 1, p->rect[3] = r[3] - r[1] + 1;
    p->structure = m.structure;
    p->structure.pod(p->rect);
    p->ranges = std::move(m.ranges);
    p->volatiles = std::move(m.volatiles);
    p->profiled = m.profiled;
    p->draws = m.draws;
    // the slot: the kind of target and the pass's place among this buffer's memo passes into it
    memo::Hash kind;
    kind.u64(uint64_t(pass.color0->format()) << 32 | pass.level);
    kind.u64(uint64_t(w) << 32 | hgt);
    const uint32_t ordinal = memoOrdinals_[kind.a]++;
    p->slot = kind.a ^ (uint64_t(ordinal) * 0x9e3779b97f4a7c15ull);
    // the cache: one texture a slot, made the first time (the target's format and level size, render target and read)
    std::shared_ptr<Texture> cache;
    {
        std::lock_guard<std::mutex> g(queue_.memoLock_);
        queue_.memoStats_.candidates++;
        CommandQueue::MemoEntry& e = queue_.memo_[p->slot];
        if (e.cache == nullptr || e.cache->width() != w || e.cache->height() != hgt || e.cache->format() != pass.color0->format()) {
            TextureDesc desc;
            desc.format = pass.color0->format();
            desc.width = w;
            desc.height = hgt;
            desc.usage = kTextureUsageRenderTarget | kTextureUsageRead;
            std::string why;
            e.cache = Texture::make(queue_.gpu(), desc, &why);
            e.valid = false;
            if (e.cache == nullptr) {
                queue_.memoStats_.refused++;
                queue_.memoStats_.lastRefusal = "no cache texture: " + why;
                return nullptr;
            }
        }
        cache = e.cache;
    }
    keep(cache->buffer());
    use(*cache->buffer(), true);
    // the two copies, made now (the hit's lies unused unless commit takes it; the store's goes on the stream after the pass)
    const render::BlitRect R{p->rect[0], p->rect[1], p->rect[2], p->rect[3]};
    std::string why;
    Piece table;
    if (!piece(render::kTableBytes, &table, &why)) return refused("no table for the copy: " + why);
    pm4::Stream hit;
    if (render::encodeBlit(hit, *cache, 0, R, *pass.color0, pass.level, R, false, table.gpuAddress(), table.contents(), &why, open_->options) &&
        hit.size() <= p->end - p->begin) {
        p->hitWords = hit.words();
        p->hitCache = cache;
    }
    Piece storeTable;
    if (!piece(render::kTableBytes, &storeTable, &why)) return refused("no table for the copy: " + why);
    pm4::Stream store;
    if (render::encodeBlit(store, *pass.color0, pass.level, R, *cache, 0, R, false, storeTable.gpuAddress(), storeTable.contents(), &why, open_->options)) {
        p->storeCache = cache;
        p->storeWords = store.words();   // (memoStore puts it on the stream, after the pass's end stamp)
    } else if (p->hitWords.empty()) {
        return refused("the 2D engine cannot copy this target: " + why);
    }
    return p;
}

void CommandBuffer::memoStore(std::unique_ptr<MemoPending> p) {
    if (p == nullptr) return;
    if (!p->storeWords.empty()) {
        stampStart(kPassCopy, p->rect[2], p->rect[3]);
        p->storeBegin = stream_.size();
        stream_.append(p->storeWords.data(), p->storeWords.size());
        p->storeEnd = stream_.size();
        stampEnd();
        p->storeWords.clear();
        p->storeWords.shrink_to_fit();
    }
    memoPending_.push_back(std::move(p));
}

void CommandBuffer::memoResolve() {
    if (memoPending_.empty()) return;
    std::function<bool(const Texture&, uint64_t*)> version;
    {
        std::lock_guard<std::mutex> g(queue_.memoLock_);
        version = queue_.memoVersion_;
    }
    uint32_t* words = stream_.data();
    for (std::unique_ptr<MemoPending>& p : memoPending_) {
        // the contents, as the GPU will find them
        const int64_t began = nowNs();
        memo::Hash full = p->structure;
        uint64_t hashed = 0;
        bool readable = true;
        for (const memo::Range& r : p->ranges) {
            const uint8_t* c = static_cast<const uint8_t*>(r.buffer->contents());
            if (c == nullptr) {
                readable = false;
                break;
            }
            full.bytes(c + r.offset, size_t(r.length));
            hashed += r.length;
        }
        for (const memo::Volatile& v : p->volatiles) {
            uint64_t n = 0;
            if (version && version(*v.texture, &n)) {
                full.u64(n);
                continue;
            }
            const uint8_t* c = static_cast<const uint8_t*>(v.texture->contents());
            if (c == nullptr) {
                readable = false;
                break;
            }
            const Bo& bo = v.texture->buffer()->bo();
            if (bo.surface()) bo.cpuBegin(false);
            full.bytes(c, size_t(rowsBytes(*v.texture)));
            if (bo.surface()) bo.cpuEnd();
            hashed += rowsBytes(*v.texture);
        }
        const int64_t spent = nowNs() - began;
        std::lock_guard<std::mutex> g(queue_.memoLock_);
        memo::Stats& s = queue_.memoStats_;
        s.hashedBytes += hashed;
        s.hashNs += uint64_t(std::max<int64_t>(spent, 0));
        CommandQueue::MemoEntry& e = queue_.memo_[p->slot];
        const bool sameRect = memcmp(e.rect, p->rect, sizeof e.rect) == 0;
        uint8_t outcome = 1;
        if (readable && e.valid && e.full == full && sameRect && e.cache == p->hitCache && !p->hitWords.empty()) {
            // a hit: the pass's words become the copy, the rest NOPs; the store after it, if any, NOPs
            memcpy(words + p->begin, p->hitWords.data(), 4 * p->hitWords.size());
            nopOver(words + p->begin + p->hitWords.size(), p->end - p->begin - p->hitWords.size());
            if (p->storeEnd > p->storeBegin) nopOver(words + p->storeBegin, p->storeEnd - p->storeBegin);
            s.hits++;
            s.skippedDraws += p->draws;
            outcome = 2;
        } else if (readable && p->storeCache != nullptr && p->storeEnd > p->storeBegin && e.seen && e.lastFull == full && e.cache == p->storeCache) {
            // the same pass as at the last frame, not kept yet: kept now (the store runs after it)
            e.valid = true;
            e.full = full;
            e.structure = p->structure;
            memcpy(e.rect, p->rect, sizeof e.rect);
            s.stores++;
            s.misses++;
            outcome = 3;
        } else {
            if (p->storeEnd > p->storeBegin) nopOver(words + p->storeBegin, p->storeEnd - p->storeBegin);
            s.misses++;
            s.lastMiss = !readable                      ? "contents not readable"
                         : !e.seen                      ? "first time"
                         : e.lastFull != full && e.lastStructure != p->structure ? "structure"
                         : e.lastFull != full           ? "contents"
                                                        : "kept result not there yet";
            // (a store NOPped: the cache keeps what it had, and the entry with it)
        }
        if (getenv("HAKIOS_MEMO_DEBUG") != nullptr) {
            fprintf(stderr, "passmemo: slot %016llx structure %016llx full %016llx draws %u rect %u,%u %ux%u outcome %u\n", (unsigned long long)p->slot,
                    (unsigned long long)p->structure.a, (unsigned long long)full.a, p->draws, p->rect[0], p->rect[1], p->rect[2], p->rect[3], outcome);
        }
        e.seen = true;
        e.lastFull = full;
        e.lastStructure = p->structure;
        if (p->profiled >= 0 && size_t(p->profiled) < profiled_.size()) profiled_[p->profiled].memo = outcome;
    }
    memoPending_.clear();
    memoOrdinals_.clear();
}

}  // namespace hakios::metal
