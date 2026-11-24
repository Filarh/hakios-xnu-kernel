#include "hakios-metal/compile_queue.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>

#ifdef __APPLE__
#include <pthread.h>
#include <pthread/qos.h>
#endif

namespace hakios::metal {

namespace {

std::atomic<unsigned> gLanes{1};
std::atomic<bool> gForegroundFirst{false};
thread_local CompileLane tLane = CompileLane::kForeground;
thread_local std::string tActivity;

// one slot a compiler: whether it is held and by what, under one small lock (so that a thread that finds it taken
// reads the name of the thread that has it, never a stale one)
struct Slot {
    std::mutex lock;
    std::condition_variable freed;
    bool busy = false;
    unsigned foregroundWaiting = 0;   // a background thread does not take the slot while a foreground one waits for it
    std::string holder;
};
Slot gSlots[2];

double usSince(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t).count();
}

#ifdef __APPLE__
// the worker's processor share follows what it runs: a job something draws a fallback for is the user's, the warm-up is
// not (on the phone: QOS_CLASS_UTILITY may put it on the small cores, which is where it belongs)
void qosFor(CompilePriority p) {
    const qos_class_t q = p <= CompilePriority::kVisible ? QOS_CLASS_USER_INITIATED : QOS_CLASS_UTILITY;
    pthread_set_qos_class_self_np(q, 0);
}
#else
void qosFor(CompilePriority) {}
#endif

}  // namespace

void setCompileLanes(unsigned lanes) { gLanes.store(lanes >= 2 ? 2 : 1); }
unsigned compileLanes() { return gLanes.load(); }
void setCompileForegroundFirst(bool on) { gForegroundFirst.store(on); }
bool compileForegroundFirst() { return gForegroundFirst.load(); }
CompileLane currentCompileLane() { return tLane; }
unsigned compileSlotOf(CompileLane lane) { return gLanes.load() >= 2 && lane == CompileLane::kBackground ? 1 : 0; }

CompileLaneScope::CompileLaneScope(CompileLane lane) : was_(tLane) { tLane = lane; }
CompileLaneScope::~CompileLaneScope() { tLane = was_; }

void setCompileActivity(const std::string& what) { tActivity = what; }
const std::string& compileActivity() { return tActivity; }

// Foreground first (setCompileForegroundFirst): with one compiler for both lanes, a thread that waits for its result is next as soon as the compilation
// in progress ends -- a background thread compiling stage after stage (the warm-up, specialisations) does not get the
// compiler back while one waits. (A plain mutex let it: a pipeline waited behind the next one too, and the next.)
CompilerHold::CompilerHold(CompileLane lane) : slot_(compileSlotOf(lane)) {
    Slot& s = gSlots[slot_];
    const bool foreground = lane == CompileLane::kForeground;
    std::unique_lock<std::mutex> g(s.lock);
    const bool first = gForegroundFirst.load();
    auto free = [&] { return !s.busy && (foreground || !first || s.foregroundWaiting == 0); };
    if (!free()) {
        behind_ = s.holder.empty() ? std::string("(unnamed)") : s.holder;
        const auto asked = std::chrono::steady_clock::now();
        if (foreground) s.foregroundWaiting++;
        s.freed.wait(g, free);
        if (foreground) s.foregroundWaiting--;
        waitedUs_ = usSince(asked);
    }
    s.busy = true;
    s.holder = tActivity.empty() ? std::string(tLane == CompileLane::kBackground ? "background compilation" : "foreground compilation") : tActivity;
}

CompilerHold::~CompilerHold() {
    Slot& s = gSlots[slot_];
    {
        std::lock_guard<std::mutex> g(s.lock);
        s.busy = false;
        s.holder.clear();
    }
    s.freed.notify_all();
}

CompileQueue::CompileQueue(unsigned workers) {
    for (unsigned i = 0; i < workers; i++) workers_.emplace_back([this] { workerLoop(); });
}

CompileQueue::~CompileQueue() {
    {
        std::lock_guard<std::mutex> g(lock_);
        stopping_ = true;
        queued_.clear();
    }
    wake_.notify_all();
    for (std::thread& t : workers_) t.join();
}

CompileQueue& CompileQueue::shared() {
    // (never destroyed: a worker may be compiling as the process exits, and a static's destructor would join it then)
    static CompileQueue* q = new CompileQueue(1);
    return *q;
}

std::shared_ptr<CompileTicket> CompileQueue::submit(const std::string& key, CompilePriority priority, std::string what, std::function<void()> work,
                                                    bool* fresh) {
    std::shared_ptr<CompileTicket> t;
    if (fresh != nullptr) *fresh = false;
    {
        std::lock_guard<std::mutex> g(lock_);
        auto it = byKey_.find(key);
        if (it != byKey_.end()) {
            if (it->second->state == CompileTicket::kQueued && priority < it->second->priority) it->second->priority = priority;
            return it->second;
        }
        t = std::make_shared<CompileTicket>();
        t->key = key;
        t->what = std::move(what);
        t->priority = priority;
        t->sequence = sequence_++;
        t->work = std::move(work);
        byKey_[key] = t;
        queued_.push_back(t);
        if (fresh != nullptr) *fresh = true;
    }
    wake_.notify_one();
    return t;
}

void CompileQueue::raise(const std::string& key, CompilePriority priority) {
    std::lock_guard<std::mutex> g(lock_);
    auto it = byKey_.find(key);
    if (it != byKey_.end() && it->second->state == CompileTicket::kQueued && priority < it->second->priority) it->second->priority = priority;
}

std::shared_ptr<CompileTicket> CompileQueue::takeBestLocked() {
    if (queued_.empty()) return nullptr;
    auto best = std::min_element(queued_.begin(), queued_.end(), [](const auto& a, const auto& b) {
        return a->priority != b->priority ? a->priority < b->priority : a->sequence < b->sequence;
    });
    std::shared_ptr<CompileTicket> t = *best;
    queued_.erase(best);
    t->state = CompileTicket::kRunning;
    return t;
}

void CompileQueue::runTicket(const std::shared_ptr<CompileTicket>& t, const char* by) {
    const std::string was = tActivity;
    tActivity = t->what;
    if (t->work) t->work();
    tActivity = was;
    std::lock_guard<std::mutex> g(lock_);
    t->state = CompileTicket::kDone;
    t->ranOn = by;
    t->work = nullptr;
    finished_.push_back(t->key);
    if (finished_.size() > kRemembered) {
        // forget the oldest done ticket (its key may have been submitted again since: only a done one is dropped)
        auto it = byKey_.find(finished_.front());
        if (it != byKey_.end() && it->second->state == CompileTicket::kDone) byKey_.erase(it);
        finished_.pop_front();
    }
    done_.notify_all();
}

void CompileQueue::workerLoop() {
    CompileLaneScope lane(CompileLane::kBackground);
    for (;;) {
        std::shared_ptr<CompileTicket> t;
        {
            std::unique_lock<std::mutex> g(lock_);
            wake_.wait(g, [&] { return stopping_ || (!paused_ && !queued_.empty()); });
            if (stopping_) return;
            t = takeBestLocked();
        }
        if (t == nullptr) continue;
        qosFor(t->priority);
        runTicket(t, "worker");
    }
}

double CompileQueue::runOrWait(const std::string& key) {
    std::shared_ptr<CompileTicket> t;
    {
        std::unique_lock<std::mutex> g(lock_);
        auto it = byKey_.find(key);
        if (it == byKey_.end() || it->second->state == CompileTicket::kDone) return 0;
        t = it->second;
        if (t->state == CompileTicket::kRunning) {
            const auto began = std::chrono::steady_clock::now();
            done_.wait(g, [&] { return t->state == CompileTicket::kDone; });
            return usSince(began);
        }
        // queued: it is this thread's to run, now, on its own lane
        queued_.erase(std::remove(queued_.begin(), queued_.end(), t), queued_.end());
        t->state = CompileTicket::kRunning;
        t->priority = CompilePriority::kWaited;
    }
    runTicket(t, "caller");
    return 0;
}

void CompileQueue::drain() {
    for (;;) {
        std::shared_ptr<CompileTicket> t;
        {
            std::unique_lock<std::mutex> g(lock_);
            t = takeBestLocked();
            if (t == nullptr) {
                done_.wait(g, [&] {
                    for (const auto& kv : byKey_) {
                        if (kv.second->state == CompileTicket::kRunning) return false;
                    }
                    return true;
                });
                if (queued_.empty()) return;
                continue;
            }
        }
        runTicket(t, "caller");
    }
}

void CompileQueue::pause() {
    std::lock_guard<std::mutex> g(lock_);
    paused_ = true;
}

void CompileQueue::resume() {
    {
        std::lock_guard<std::mutex> g(lock_);
        paused_ = false;
    }
    wake_.notify_all();
}

size_t CompileQueue::pending() {
    std::lock_guard<std::mutex> g(lock_);
    return queued_.size();
}

std::vector<std::string> CompileQueue::finishedOrder() {
    std::lock_guard<std::mutex> g(lock_);
    return std::vector<std::string>(finished_.begin(), finished_.end());
}

}  // namespace hakios::metal
