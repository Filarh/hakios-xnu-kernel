#include "hakios-metal/specialize.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace hakios::metal {

namespace {
// the plug-in's hm_hash_bytes: LENGTH bytes as one number, eight at a time
uint64_t hashBytes(const uint8_t* p, size_t length) {
    uint64_t h = 0x9e3779b97f4a7c15ull ^ length;
    while (length != 0) {
        const size_t n = length < 8 ? length : 8;
        uint64_t w = 0;
        memcpy(&w, p, n);
        h = (h ^ w) * 0xff51afd7ed558ccdull;
        h ^= h >> 32;
        p += n;
        length -= n;
    }
    return h;
}
constexpr unsigned kAliases = 8;
}  // namespace

struct SpecializationTable::State {
    std::mutex lock;
    void* base;
    size_t length, most;
    Hooks hooks;
    CompileQueue* queue;
    CompilePriority priority;
    std::string what;
    bool exact;
    uint64_t id = 0;     // the queue's keys are this table's own: an id no other table has had in this process
    bool gone = false;   // the table is gone: a job still queued compiles nothing
    struct Slot {
        uint64_t hash;
        std::vector<uint8_t> key, mask;   // mask: 0xff where the content's byte decided its program (all, until known)
        bool masked = false;
        void* program = nullptr;
        bool failed = false;
        std::string ticket;
    };
    std::vector<Slot> slots;
    uint64_t aliasHashes[kAliases] = {};
    uint32_t aliasSlots[kAliases] = {};
    unsigned aliasNext = 0;
    Stats stats;
    ~State() {
        for (Slot& s : slots) {
            if (s.program != nullptr && hooks.release) hooks.release(s.program);
        }
    }
    // the slot of a content (its very bytes, or the same bytes where a compiled slot's mask says they decided), -1
    int findLocked(const uint8_t* key, uint64_t hash) {
        for (size_t i = 0; i < slots.size(); i++) {
            if (slots[i].hash == hash && memcmp(slots[i].key.data(), key, length) == 0) return int(i);
        }
        auto same = [&](const Slot& s) {
            for (size_t b = 0; b < length; b++) {
                if (((key[b] ^ s.key[b]) & s.mask[b]) != 0) return false;
            }
            return true;
        };
        for (unsigned a = 0; a < kAliases; a++) {
            const uint32_t i = aliasSlots[a];
            if (aliasHashes[a] == hash && i < slots.size() && slots[i].masked && same(slots[i])) return int(i);
        }
        for (size_t i = 0; i < slots.size(); i++) {
            if (!slots[i].masked || !same(slots[i])) continue;
            aliasHashes[aliasNext % kAliases] = hash;
            aliasSlots[aliasNext % kAliases] = uint32_t(i);
            aliasNext++;
            return int(i);
        }
        return -1;
    }
};

SpecializationTable::SpecializationTable(void* base, size_t length, size_t most, Hooks hooks, CompileQueue& queue, CompilePriority priority,
                                         std::string what, bool exact)
    : state_(std::make_shared<State>()) {
    static std::atomic<uint64_t> tables{1};
    state_->id = tables.fetch_add(1);
    state_->base = base;
    state_->length = length > 256 ? 256 : length;
    state_->most = most;
    state_->hooks = std::move(hooks);
    state_->queue = &queue;
    state_->priority = priority;
    state_->what = std::move(what);
    state_->exact = exact;
}

SpecializationTable::~SpecializationTable() {
    std::lock_guard<std::mutex> g(state_->lock);
    state_->gone = true;
}

size_t SpecializationTable::length() const { return state_->length; }

void* SpecializationTable::programFor(const uint8_t* bytes, bool* started) {
    State& s = *state_;
    if (started != nullptr) *started = false;
    uint8_t key[256];
    memcpy(key, bytes, s.length);
    const uint64_t hash = hashBytes(key, s.length);
    std::string ticket;
    size_t slot = 0;
    {
        std::lock_guard<std::mutex> g(s.lock);
        const int found = s.findLocked(key, hash);
        if (found >= 0) {
            State::Slot& at = s.slots[size_t(found)];
            if (at.program != nullptr) {
                s.stats.hits++;
                return at.program;
            }
            if (!at.failed) s.stats.fallbackDraws++;
            return s.base;
        }
        if (s.slots.size() >= s.most) {
            s.stats.full++;
            return s.base;
        }
        slot = s.slots.size();
        State::Slot fresh;
        fresh.hash = hash;
        fresh.key.assign(key, key + s.length);
        fresh.mask.assign(s.length, 0xff);
        char name[64];
        snprintf(name, sizeof(name), "spec:%llu:%zu", (unsigned long long)s.id, slot);
        fresh.ticket = name;
        ticket = fresh.ticket;
        s.slots.push_back(std::move(fresh));
        s.stats.contents++;
        s.stats.fallbackDraws++;
    }
    std::shared_ptr<State> keep = state_;
    std::vector<uint8_t> content(key, key + s.length);
    s.queue->submit(ticket, s.priority, s.what, [keep, slot, content] {
        State& st = *keep;
        {
            std::lock_guard<std::mutex> g(st.lock);
            if (st.gone) return;
        }
        void* p = st.hooks.compile ? st.hooks.compile(content.data(), content.size()) : nullptr;
        std::vector<uint8_t> mask;
        const bool known = p != nullptr && st.hooks.reads && st.hooks.reads(p, &mask) && mask.size() == st.length;
        std::lock_guard<std::mutex> g(st.lock);
        State::Slot& at = st.slots[slot];
        at.program = p;
        at.failed = p == nullptr;
        if (known && !st.exact) {
            at.mask = std::move(mask);
            at.masked = true;
        }
        if (p != nullptr) st.stats.ready++;
        else st.stats.failed++;
    });
    if (started != nullptr) *started = true;
    return s.base;
}

void* SpecializationTable::programForWaiting(const uint8_t* bytes) {
    State& s = *state_;
    (void)programFor(bytes);
    uint8_t key[256];
    memcpy(key, bytes, s.length);
    const uint64_t hash = hashBytes(key, s.length);
    std::string ticket;
    {
        std::lock_guard<std::mutex> g(s.lock);
        const int found = s.findLocked(key, hash);
        if (found < 0) return s.base;
        ticket = s.slots[size_t(found)].ticket;
    }
    s.queue->runOrWait(ticket);
    return programFor(bytes);
}

SpecializationTable::Stats SpecializationTable::stats() {
    std::lock_guard<std::mutex> g(state_->lock);
    return state_->stats;
}

}  // namespace hakios::metal
