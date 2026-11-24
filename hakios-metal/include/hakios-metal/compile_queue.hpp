// Who compiles, when, and who waits for whom.
//
// One compiler, one lock (compiler.cpp's gCompilerLock until this file): a pipeline the window server needed now, whose
// own compilation took 3.8 ms, waited 188.8 ms on the phone for whatever another thread was compiling (the console's
// "made in 193.3 ms (1 of 2 stages from the cache in 0.2 ms, 1 compiled in 3.8 ms after 188.8 ms waiting for the
// compiler)"); the plug-in's background specialisations of CoreAnimation's generic uber shader take 0.4 to 1.1 s each
// on the phone and held that one lock all along. Here:
//
//  - LANES. With two lanes there are two compilers (two ir3_compiler objects, each under a lock of its own): the
//    foreground lane is what a thread that is waiting for its result compiles on (a pipeline made on the render
//    thread); the background lane is what the queue's worker compiles on. Background work never holds the compiler a
//    waiting thread needs. With one lane (the default: setCompileLanes(1)) both names are the one compiler, as before.
//  - THE QUEUE. Work nobody waits for (a specialisation while the pipeline's own program draws; the warm-up's recipes)
//    is submitted by KEY at a priority, run by the worker highest priority first (FIFO within one), each key once:
//    submitting a key already queued raises its priority and returns the same ticket. A thread that comes to need a
//    queued key runs it itself, now, on its own lane (runOrWait: the job leaves the queue); one already running is
//    waited for (it started earlier than a copy would; compiling it twice would end no sooner).
//  - ATTRIBUTION. Each lane knows what holds it (setCompileActivity: "specialise fixed_frag_lph_cpf", "warm-up …",
//    "pipeline …"), so a wait can be said with what it waited behind -- the console's line says it from now on.
//
// Nothing here needs Mesa: the jobs are std::functions, and the tests run them with sleeps for compilations.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace hakios::metal {

enum class CompileLane : uint8_t { kForeground = 0, kBackground = 1 };

// Lower runs first. kWaited: a thread is (or will be) blocked on it. kVisible: something draws with a fallback until it
// is done. kPrefetch: likely to be needed soon. kWarmUp: the warm-up's recipes.
enum class CompilePriority : uint8_t { kWaited = 0, kVisible = 1, kPrefetch = 2, kWarmUp = 3 };

// 1 (default): one compiler for every thread, the old behaviour; 2: a foreground and a background compiler. Set once,
// before the first compilation (the plug-in does it as the device is made); a later change applies to compilations
// that start after it.
void setCompileLanes(unsigned lanes);
unsigned compileLanes();
// With one compiler for both lanes: whether a foreground thread waiting for it is served before any background thread
// (off by default: the old lock, first come by the scheduler's choice). With two lanes it changes nothing.
void setCompileForegroundFirst(bool on);
bool compileForegroundFirst();

// the lane this thread compiles on: foreground unless a CompileLaneScope (the queue's worker has one) says otherwise
CompileLane currentCompileLane();
// the compiler slot a lane maps to: 0 or 1 (always 0 with one lane)
unsigned compileSlotOf(CompileLane lane);
struct CompileLaneScope {
    explicit CompileLaneScope(CompileLane lane);
    ~CompileLaneScope();
    CompileLaneScope(const CompileLaneScope&) = delete;
    CompileLaneScope& operator=(const CompileLaneScope&) = delete;
private:
    CompileLane was_;
};

// What this thread is compiling, for whoever waits behind it (copied; "" clears it). The queue sets a job's WHAT.
void setCompileActivity(const std::string& what);
const std::string& compileActivity();

// A held compiler slot. Taking it measures the wait and, when there was one, what held the slot as the wait began.
class CompilerHold {
public:
    explicit CompilerHold(CompileLane lane);
    ~CompilerHold();
    CompilerHold(const CompilerHold&) = delete;
    CompilerHold& operator=(const CompilerHold&) = delete;
    unsigned slot() const { return slot_; }
    double waitedUs() const { return waitedUs_; }
    const std::string& behind() const { return behind_; }   // "" when it was free
private:
    unsigned slot_;
    double waitedUs_ = 0;
    std::string behind_;
};

// One compilation's ticket in the queue (or out of it, once taken).
struct CompileTicket {
    enum State : uint8_t { kQueued, kRunning, kDone };
    std::string key, what;
    CompilePriority priority;
    uint64_t sequence = 0;
    State state = kQueued;
    std::function<void()> work;
    std::string ranOn;   // "worker" or "caller" (who ran it), for the tests and the console
};

class CompileQueue {
public:
    // WORKERS background threads (each compiles on the background lane); 0: none -- jobs run only when waited for or
    // when drain() is called (the tests' deterministic mode)
    explicit CompileQueue(unsigned workers = 1);
    ~CompileQueue();   // the pending jobs are dropped, the running ones finished, the workers joined
    CompileQueue(const CompileQueue&) = delete;
    CompileQueue& operator=(const CompileQueue&) = delete;

    // the process's queue (one worker), made at first use
    static CompileQueue& shared();

    // KEY at PRIORITY: queued (WHAT says what it is), or -- already queued, running or done and still remembered --
    // that ticket, its priority raised to PRIORITY when that is higher (lower in number). Done tickets are forgotten
    // after a while (kRemembered of them): the work's own result (a file of the cache) is what remembers it.
    // *FRESH (when given): whether WORK was queued now (false: the key's ticket was there, and WORK is dropped)
    std::shared_ptr<CompileTicket> submit(const std::string& key, CompilePriority priority, std::string what, std::function<void()> work,
                                          bool* fresh = nullptr);
    // raise a queued key's priority (nothing if it is not queued)
    void raise(const std::string& key, CompilePriority priority);
    // The calling thread needs KEY done: queued, it is taken and run here, now (on the caller's lane); running, waited
    // for; done or unknown, nothing. Returns the microseconds spent waiting for another thread's run of it.
    double runOrWait(const std::string& key);
    // run every queued job on the calling thread, highest priority first, until none is left (and wait for running ones)
    void drain();
    // stop the workers taking jobs (those running finish); resume() starts them again. For the tests' orderings.
    void pause();
    void resume();
    size_t pending();
    // the keys in the order they were finished (the last kRemembered), for the tests
    std::vector<std::string> finishedOrder();

private:
    void workerLoop();
    std::shared_ptr<CompileTicket> takeBestLocked();
    void runTicket(const std::shared_ptr<CompileTicket>& t, const char* by);

    static constexpr size_t kRemembered = 4096;
    std::mutex lock_;
    std::condition_variable wake_, done_;
    std::map<std::string, std::shared_ptr<CompileTicket>> byKey_;
    std::vector<std::shared_ptr<CompileTicket>> queued_;
    std::deque<std::string> finished_;
    uint64_t sequence_ = 0;
    bool paused_ = false, stopping_ = false;
    std::vector<std::thread> workers_;
};

}  // namespace hakios::metal
