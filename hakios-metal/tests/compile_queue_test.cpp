// Compiling without stalling (compile_queue.hpp, specialize.hpp, shader_cache.hpp): the queue's order, the lanes, an
// uber shader's new content drawn at once with its generic program and then with its own, and one cache directory
// shared by processes. The compilations here are sleeps and fixtures (a test build has no compiler): what is under
// test is who waits for whom, what is drawn meanwhile, and what one process's files are to another.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "hakios-metal/command.hpp"
#include "hakios-metal/compile_queue.hpp"
#include "hakios-metal/compiler.hpp"
#include "hakios-metal/fake_gpu.hpp"
#include "hakios-metal/render.hpp"
#include "hakios-metal/shader_cache.hpp"
#include "hakios-metal/specialize.hpp"
#include "hakios-metal/texture.hpp"
#include "fixtures/tri_cflat_fs_air.hpp"
#include "fixtures/tri_flat_fs_air.hpp"
#include "fixtures/tri_flat_vs_air.hpp"
#include "test.hpp"

using namespace hakios::metal;

namespace {

double msSince(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

void sleepMs(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

// until COND or two seconds
template <typename F>
bool eventually(F cond) {
    const auto began = std::chrono::steady_clock::now();
    while (!cond()) {
        if (msSince(began) > 2000) return false;
        sleepMs(1);
    }
    return true;
}

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

Program linked(const Stage& vs, const Stage& fs, const char* name) {
    Program p{vs, fs, Linkage{}, name};
    std::string why;
    CHECK(render::link(p.vs, p.fs, &p.link, &why));
    p.uid = newProgramUid();
    return p;
}

// What a fragment stage writes to r0.x..w, run on the processor: enough of ir3 for the two fixtures compared here and
// no more -- cat1 mov.u32u32 of an immediate (bit 54 of the instruction) or of a constant register (bit 53), then end.
// Anything else is refused (false), so that the comparison below cannot pass by not looking. CONSTS: the constant file
// as the fake GPU saw it loaded for the draw (vec4 units, words).
bool runMovShader(const uint32_t* words, uint32_t count, const std::vector<FakeGpu::ConstLoad>& loads, uint32_t out[4]) {
    std::vector<uint32_t> consts(256 * 4, 0xdeadbeef);
    for (const FakeGpu::ConstLoad& l : loads) {
        for (size_t i = 0; i < l.words.size() && l.dstOffVec4 * 4 + i < consts.size(); i++) consts[l.dstOffVec4 * 4 + i] = l.words[i];
    }
    bool wrote[4] = {};
    for (uint32_t i = 0; i + 1 < count; i += 2) {
        const uint32_t lo = words[i], hi = words[i + 1];
        if (hi == 0x03000000 && lo == 0) {   // end
            return wrote[0] && wrote[1] && wrote[2] && wrote[3];
        }
        if ((hi & ~0x006000ffu) != 0x200cc000u) return false;   // not a cat1 mov.u32u32 to a full register
        const uint32_t dst = hi & 0xff;
        if (dst > 3) return false;
        if (hi & (1u << 22)) out[dst] = lo;                                         // immediate
        else if (hi & (1u << 21)) out[dst] = lo < consts.size() ? consts[lo] : 0;   // constant register c<lo/4>.<lo%4>
        else return false;
        wrote[dst] = true;
    }
    return false;
}

}  // namespace

// The worker takes the highest priority first, the oldest first within one; a key submitted again is the same job, its
// priority raised.
TEST(the_compile_queue_runs_the_highest_priority_first) {
    CompileQueue q(1);
    q.pause();
    auto job = [](const char*) { return [] {}; };
    q.submit("warm-a", CompilePriority::kWarmUp, "warm-up a", job("a"));
    q.submit("warm-b", CompilePriority::kWarmUp, "warm-up b", job("b"));
    q.submit("soon", CompilePriority::kPrefetch, "prefetch", job("p"));
    q.submit("seen-1", CompilePriority::kVisible, "specialise 1", job("1"));
    q.submit("warm-c", CompilePriority::kWarmUp, "warm-up c", job("c"));
    q.submit("seen-2", CompilePriority::kVisible, "specialise 2", job("2"));
    bool fresh = true;
    q.submit("warm-c", CompilePriority::kVisible, "warm-up c, needed", job("c"), &fresh);   // now drawn with a fallback
    CHECK(!fresh);
    CHECK_EQ(q.pending(), 6u);
    q.resume();
    CHECK(eventually([&] { return q.finishedOrder().size() == 6; }));
    const std::vector<std::string> want = {"seen-1", "warm-c", "seen-2", "soon", "warm-a", "warm-b"};
    CHECK(q.finishedOrder() == want);
}

// A job something comes to need while it is still queued runs on the thread that needs it, at once; one already running
// is waited for, not run twice.
TEST(a_queued_job_that_is_waited_for_runs_on_the_waiting_thread) {
    {
        CompileQueue q(0);   // (no worker: nothing runs unless asked)
        std::atomic<int> runs{0};
        auto t = q.submit("k", CompilePriority::kWarmUp, "warm-up", [&] { runs++; });
        CHECK(q.runOrWait("k") == 0);
        CHECK_EQ(runs.load(), 1);
        CHECK(t->state == CompileTicket::kDone);
        CHECK(t->ranOn == "caller");
        q.runOrWait("k");
        CHECK_EQ(runs.load(), 1);
    }
    {
        CompileQueue q(1);
        std::atomic<bool> started{false};
        std::atomic<int> runs{0};
        auto t = q.submit("slow", CompilePriority::kVisible, "specialise", [&] {
            started = true;
            runs++;
            sleepMs(150);
        });
        CHECK(eventually([&] { return started.load(); }));
        const double waited = q.runOrWait("slow") / 1000.0;
        CHECK(waited > 50);
        CHECK_EQ(runs.load(), 1);
        CHECK(t->ranOn == "worker");
    }
}

// The phone's 188.8 ms: a pipeline whose own stage compiles in 3.8 ms waited for the one compiler while the background
// compiled an uber shader's specialisation. With one lane it waits, and says behind what; with two it does not wait.
TEST(the_foreground_compiler_is_never_held_by_background_work) {
    for (unsigned lanes : {1u, 2u}) {
        setCompileLanes(lanes);
        CompileQueue q(1);
        std::atomic<bool> holding{false};
        q.submit("uber", CompilePriority::kVisible, "specialise fixed_frag_lph_cpf", [&] {
            CompilerHold hold(currentCompileLane());   // (the worker's lane: background)
            holding = true;
            sleepMs(300);
        });
        CHECK(eventually([&] { return holding.load(); }));
        setCompileActivity("pipeline fixed_vert_lph_spc+fixed_frag_lph_cph");
        const auto asked = std::chrono::steady_clock::now();
        double waited = 0;
        std::string behind;
        {
            CompilerHold hold(CompileLane::kForeground);
            waited = hold.waitedUs() / 1000.0;
            behind = hold.behind();
            sleepMs(4);   // its own compilation
        }
        const double made = msSince(asked);
        setCompileActivity("");
        if (lanes == 1) {
            CHECK(waited > 150);
            CHECK(behind == "specialise fixed_frag_lph_cpf");
        } else {
            CHECK(waited < 20);
            CHECK(behind.empty());
            CHECK(made < 50);
        }
        q.drain();
    }
    setCompileLanes(1);
}

// A new content of an uber shader's deciding buffer is drawn at once with the generic program (the render thread does
// not wait for the compiler, however long the compilation is), and with its own program once that is ready; a content
// that differs only in bytes the compilation did not read is that program without another compilation.
TEST(an_uber_shaders_new_content_is_drawn_at_once_and_then_with_its_own_program) {
    setCompileLanes(2);
    CompileQueue q(1);
    int generic = 0, special = 0;
    std::atomic<int> compiles{0};
    SpecializationTable::Hooks hooks;
    hooks.compile = [&](const uint8_t* bytes, size_t n) -> void* {
        compiles++;
        CHECK_EQ(n, 8u);
        CHECK_EQ(bytes[0], 7);
        sleepMs(250);   // the phone: 0.4 to 1.1 s
        return &special;
    };
    hooks.reads = [](void*, std::vector<uint8_t>* mask) {
        *mask = {0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0};   // the last four bytes decided nothing (CoreAnimation's padding)
        return true;
    };
    hooks.release = [](void*) {};
    SpecializationTable table(&generic, 8, 128, hooks, q, CompilePriority::kVisible, "specialise uber");
    const uint8_t content[8] = {7, 1, 2, 3, 0xaa, 0xbb, 0xcc, 0xdd};
    const uint8_t padding[8] = {7, 1, 2, 3, 0x11, 0x22, 0x33, 0x44};
    bool started = false;
    const auto asked = std::chrono::steady_clock::now();
    CHECK(table.programFor(content, &started) == &generic);
    CHECK(msSince(asked) < 20);
    CHECK(started);
    // frames meanwhile: the generic program, no wait, no second compilation
    for (int i = 0; i < 20; i++) {
        const auto t = std::chrono::steady_clock::now();
        void* p = table.programFor(content, &started);
        CHECK(msSince(t) < 5);
        CHECK(!started);
        if (p == &special) break;
        sleepMs(5);
    }
    CHECK(eventually([&] { return table.programFor(content) == &special; }));
    CHECK(table.programFor(padding, &started) == &special);
    CHECK(!started);
    CHECK_EQ(compiles.load(), 1);
    const SpecializationTable::Stats st = table.stats();
    CHECK_EQ(st.contents, 1u);
    CHECK_EQ(st.ready, 1u);
    CHECK(st.fallbackDraws >= 2);
    CHECK(st.hits >= 2);
    setCompileLanes(1);
}

// The swap and what each draw runs: a generic program that reads its colour from a constant buffer (tri_cflat_fs) and
// the program specialised for that buffer's bytes, the colour in its instructions (tri_flat_fs). A command buffer
// encoded with the generic one before the swap and committed after it still runs the generic one; the next runs the
// specialised one; and the colour each writes -- computed from the instructions and the constants the fake GPU saw
// loaded for each draw -- is the same. (The fake GPU rasterizes nothing: the comparison is of what each fragment stage
// computes from what it is given, which for these two is the whole of what they do.)
TEST(the_generic_and_the_specialised_program_write_the_same_colour) {
    Rig r;
    setCompileLanes(2);
    CompileQueue q(1);
    const Program generic = linked(kernels::tri_flat_vs_air, kernels::tri_cflat_fs_air, "flat+cflat (generic)");
    const Program special = linked(kernels::tri_flat_vs_air, kernels::tri_flat_fs_air, "flat+flat (specialised)");
    SpecializationTable::Hooks hooks;
    hooks.compile = [&](const uint8_t*, size_t) -> void* {
        sleepMs(100);
        return const_cast<Program*>(&special);
    };
    hooks.reads = [](void*, std::vector<uint8_t>*) { return false; };
    hooks.release = [](void*) {};
    SpecializationTable table(const_cast<Program*>(&generic), 16, 8, hooks, q, CompilePriority::kVisible, "specialise cflat");
    // the deciding buffer: the colour tri_flat_fs has in its instructions (0.2509804, 0.5019608, 0.5019608 rounded down, 1)
    const uint32_t colour[4] = {0x3e008081, 0x3e808081, 0x3f008081, 0x3f800000};
    std::string why;
    auto uniforms = newBuffer(*r.gpu, 4096, &why);
    memcpy(uniforms->contents(), colour, sizeof(colour));
    TextureDesc d;
    d.width = 64;
    d.height = 64;
    d.usage = kTextureUsageRenderTarget | kTextureUsageRead;
    auto target = Texture::make(*r.gpu, d, &why);
    CHECK(uniforms != nullptr && target != nullptr);
    if (uniforms == nullptr || target == nullptr) return;
    RenderPass pass;
    pass.color0 = target.get();
    Draw draw;
    draw.vertexCount = 6;
    draw.viewport = Viewport{0, 0, 64, 64, 0, 1};
    Bindings vsb, fsb;
    fsb.buffers = {Binding{uniforms.get(), 0}};

    auto drawWith = [&](const std::shared_ptr<CommandBuffer>& cb, const Program* p) { CHECK(cb->draw(*p, pass, draw, vsb, fsb)); };
    auto colourOf = [&](const Program* p, uint32_t out[4]) {
        const FakeGpu::Draw& last = r.fake->stats().lastDraw;
        return runMovShader(p->fs.k.words, p->fs.k.numWords, last.fsConsts, out);
    };
    auto commit = [&](const std::shared_ptr<CommandBuffer>& cb) {
        cb->commit();
        cb->waitUntilCompleted();
        CHECK(cb->status() == CommandBuffer::kCompleted);
    };

    const auto* first = static_cast<const Program*>(table.programFor(reinterpret_cast<const uint8_t*>(colour)));
    CHECK(first == &generic);
    auto before = r.queue->commandBuffer();
    drawWith(before, first);   // encoded with the generic program...
    CHECK(eventually([&] { return table.programFor(reinterpret_cast<const uint8_t*>(colour)) == &special; }));
    const auto* second = static_cast<const Program*>(table.programFor(reinterpret_cast<const uint8_t*>(colour)));
    auto after = r.queue->commandBuffer();
    drawWith(after, second);
    commit(before);   // ...committed after the swap: it runs what it was encoded with
    const FakeGpu::Draw a = r.fake->stats().lastDraw;
    uint32_t got[4] = {}, want[4] = {};
    CHECK(colourOf(&generic, got));
    commit(after);
    const FakeGpu::Draw b = r.fake->stats().lastDraw;
    CHECK(colourOf(&special, want));
    CHECK(a.fsConstlen > 0);         // the generic one reads constants...
    CHECK(b.fsConsts.empty());       // ...the specialised one has none to read
    CHECK(a.fsProgram != b.fsProgram);
    for (int i = 0; i < 4; i++) CHECK_EQ(got[i], colour[i]);
    for (int i = 0; i < 4; i++) CHECK_EQ(got[i], want[i]);
    setCompileLanes(1);
}

namespace {
std::string tempDir() {
    char path[] = "/tmp/hm-cache-test-XXXXXX";
    const char* d = mkdtemp(path);
    return d != nullptr ? std::string(d) : std::string();
}
CompiledStage fixtureStage() {
    CompiledStage s;
    s.stage = kernels::tri_flat_fs_air;
    s.words.assign(s.stage.k.words, s.stage.k.words + s.stage.k.numWords);
    s.name = "tri_flat_fs";
    s.stage.k.words = s.words.data();
    s.stage.k.name = s.name.c_str();
    return s;
}
// Stage's bytes with its pointers out of them, for comparing two stages field for field
std::string stageFields(const Stage& st) {
    Stage f = st;
    f.k.words = nullptr;
    f.k.immediates = nullptr;
    f.k.name = nullptr;
    f.k.source = nullptr;
    return std::string(reinterpret_cast<const char*>(&f), sizeof(f));
}
// run FN in a child process; its exit status (0: it said yes)
template <typename F>
int inChild(F fn) {
    const pid_t pid = fork();
    if (pid == 0) _exit(fn() ? 0 : 1);
    int status = -1;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 99;
}
size_t filesUnder(const std::string& dir, const char* containing) {
    size_t n = 0;
    DIR* d = opendir(dir.c_str());
    if (d == nullptr) return 0;
    while (struct dirent* e = readdir(d)) {
        const std::string name = e->d_name;
        if (name == "." || name == "..") continue;
        struct stat st;
        const std::string p = dir + "/" + name;
        if (stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) n += filesUnder(p, containing);
        else if (strstr(name.c_str(), containing) != nullptr) n++;
    }
    closedir(d);
    return n;
}
}  // namespace

// What one process compiled is what another reads, from the shared layout's sealed file: field for field the stage, and
// a program that draws. Eight processes filing the same stage at once leave one whole file and no temporary; a damaged
// file is not loaded (and is gone); the trim drops an old compiler's generation first.
TEST(a_stage_one_process_filed_is_read_by_another) {
    const std::string dir = tempDir();
    CHECK(!dir.empty());
    if (dir.empty()) return;
    setenv("HAKIOS_METAL_CACHE_V2", "1", 1);
    const std::string path = dir + cacheRelativePath("0123456789abcdef0123456789abcdef", ".stage");
    CHECK(path.find("/v2/" + cacheGeneration() + "/01/0123456789abcdef0123456789abcdef.stage") != std::string::npos);
    const CompiledStage made = fixtureStage();

    CHECK_EQ(inChild([&] { return cacheWriteFile(path, stageBytes(made), true); }), 0);   // process A compiles and files it
    CHECK_EQ(inChild([&] {                                                               // process B finds it
        std::string payload;
        if (!cacheReadFile(path, &payload)) return false;
        std::unique_ptr<CompiledStage> s = stageFromBytes(payload);
        return s != nullptr && s->words == made.words && stageFields(s->stage) == stageFields(made.stage) && s->name == "tri_flat_fs";
    }), 0);
    {
        std::string payload;
        CHECK(cacheReadFile(path, &payload));
        std::unique_ptr<CompiledStage> s = stageFromBytes(payload);
        CHECK(s != nullptr);
        if (s != nullptr) {
            Rig r;
            const Program p = linked(kernels::tri_flat_vs_air, s->stage, "flat+flat (from the cache)");
            std::string why;
            TextureDesc d;
            d.width = 64;
            d.height = 64;
            d.usage = kTextureUsageRenderTarget | kTextureUsageRead;
            auto target = Texture::make(*r.gpu, d, &why);
            RenderPass pass;
            pass.color0 = target.get();
            Draw draw;
            draw.vertexCount = 6;
            draw.viewport = Viewport{0, 0, 64, 64, 0, 1};
            auto cb = r.queue->commandBuffer();
            CHECK(cb->draw(p, pass, draw, Bindings{}, Bindings{}));
            cb->commit();
            cb->waitUntilCompleted();
            CHECK(cb->status() == CommandBuffer::kCompleted);
        }
    }

    // eight writers of one key at once
    const std::string racy = dir + cacheRelativePath("fedcba9876543210fedcba9876543210", ".stage");
    std::vector<pid_t> kids;
    for (int i = 0; i < 8; i++) {
        const pid_t pid = fork();
        if (pid == 0) _exit(cacheWriteFile(racy, stageBytes(made), true) ? 0 : 1);
        kids.push_back(pid);
    }
    int ok = 0;
    for (pid_t k : kids) {
        int status = -1;
        waitpid(k, &status, 0);
        ok += WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }
    CHECK_EQ(ok, 8);
    {
        std::string payload;
        CHECK(cacheReadFile(racy, &payload));
        std::unique_ptr<CompiledStage> s = stageFromBytes(payload);
        CHECK(s != nullptr && s->words == made.words);
    }
    CHECK_EQ(filesUnder(dir, ".tmp"), 0u);

    // a damaged file: not loaded, and removed (a miss: compiled again)
    {
        FILE* f = fopen(racy.c_str(), "r+b");
        CHECK(f != nullptr);
        if (f != nullptr) {
            fseek(f, 80, SEEK_SET);
            fputc(0x5a, f);
            fclose(f);
        }
        std::string payload;
        CHECK(!cacheReadFile(racy, &payload));
        struct stat st;
        CHECK(stat(racy.c_str(), &st) != 0);
    }

    // the trim: another generation's files go first, then this one's least recently used
    {
        const std::string old = dir + "/v2/0ldc0mp1ler/aa/aaaa.stage";
        CHECK(cacheWriteFile(old, std::string(200000, 'x'), true));
        const CacheTrimReport t = trimCache(dir, cacheGeneration(), 100000);
        CHECK_EQ(t.generationsRemoved, 1u);
        struct stat st;
        CHECK(stat(old.c_str(), &st) != 0);
        CHECK(stat(path.c_str(), &st) == 0);   // this generation's, under the bound, stays
        const CacheTrimReport none = trimCache(dir, cacheGeneration(), 0);
        CHECK(none.bytesAfter == 0);
        CHECK(stat(path.c_str(), &st) != 0);
    }
    unsetenv("HAKIOS_METAL_CACHE_V2");
    // (and in the old layout: one directory, unsealed, read as it always was)
    const std::string flat = dir + cacheRelativePath("00112233445566778899aabbccddeeff", ".stage");
    CHECK(flat == dir + "/00112233445566778899aabbccddeeff.stage");
    CHECK(cacheWriteFile(flat, stageBytes(made), false));
    std::string payload;
    CHECK(cacheReadFile(flat, &payload));
    CHECK(payload == stageBytes(made));
    std::string cmd = "rm -rf '" + dir + "'";
    CHECK(system(cmd.c_str()) == 0);
}

namespace {
std::vector<uint8_t> kernelBitcode(const char* name) {
    std::vector<uint8_t> bytes;
    FILE* f = fopen((std::string("out/host/kernels/") + name + ".bc").c_str(), "rb");
    if (f == nullptr) return bytes;
    uint8_t buffer[4096];
    for (size_t n; (n = fread(buffer, 1, sizeof buffer, f)) > 0;) bytes.insert(bytes.end(), buffer, buffer + n);
    fclose(f);
    return bytes;
}
}  // namespace

// With the compiler in (make test MESA=1): the two lanes' compilers at work at once -- a pipeline's stages compiled on
// the foreground while the background compiles a long kernel again and again -- give the very words one compiler gives
// alone, and the foreground does not wait; with one lane it waits, behind the background's job by name.
TEST(two_compilers_at_once_compile_what_one_compiles_alone) {
    if (!hasCompiler()) {
        std::printf("       (no compiler in this build: the two lanes did not run)\n");
        return;
    }
    const char* const stages[] = {"tri_flat_vs", "tri_color_fs", "tri_cflat_fs", "tri_tex_fs", "tri_fc_fs"};
    std::vector<std::vector<uint8_t>> bitcode;
    for (const char* s : stages) bitcode.push_back(kernelBitcode(s));
    const std::vector<uint8_t> chain = kernelBitcode("long_chain");
    CHECK(!chain.empty());
    for (const auto& b : bitcode) CHECK(!b.empty());
    if (chain.empty()) return;
    const char* cache = getenv("HAKIOS_METAL_CACHE");
    const std::string keptCache = cache != nullptr ? cache : "";
    unsetenv("HAKIOS_METAL_CACHE");   // (every compileStage a compilation: no file to find)
    std::string why;
    std::vector<std::vector<uint32_t>> alone;
    for (const auto& b : bitcode) {
        auto s = compileStage(b.data(), b.size(), &why);
        CHECK(s != nullptr);
        alone.push_back(s != nullptr ? s->words : std::vector<uint32_t>{});
    }
    auto k = compileAir(chain.data(), chain.size(), &why);
    CHECK(k != nullptr);
    const std::vector<uint32_t> chainAlone = k != nullptr ? k->words : std::vector<uint32_t>{};
    (void)takeCompileTimes();
    for (unsigned lanes : {2u, 1u}) {
        setCompileLanes(lanes);
        setCompileForegroundFirst(lanes == 1);
        CompileQueue q(1);
        std::atomic<bool> stop{false}, began{false};
        std::atomic<int> background{0}, differed{0};
        q.submit("long_chain", CompilePriority::kWarmUp, "warm-up long_chain", [&] {
            while (!stop) {
                std::string w;
                auto c = compileAir(chain.data(), chain.size(), &w);
                if (c == nullptr || c->words != chainAlone) differed++;
                background++;
                began = true;
            }
        });
        CHECK(eventually([&] { return began.load(); }));
        double waited = 0;
        std::string behind;
        for (int round = 0; round < 6; round++) {
            for (size_t i = 0; i < bitcode.size(); i++) {
                auto s = compileStage(bitcode[i].data(), bitcode[i].size(), &why);
                CHECK(s != nullptr && s->words == alone[i]);
                const CompileTimes t = takeCompileTimes();
                waited += t.waitedUs / 1000.0;
                if (!t.behind.empty()) behind = t.behind;
            }
        }
        stop = true;
        q.drain();
        CHECK_EQ(differed.load(), 0);
        CHECK(background.load() >= 1);
        std::printf("       %u lane(s): the foreground waited %.1f ms in all for 30 stages; %d background compilations meanwhile%s%s\n", lanes, waited,
                    background.load(), behind.empty() ? "" : "; behind ", behind.c_str());
        if (lanes == 2) {
            CHECK(waited < 5);
            CHECK(behind.empty());
        } else {
            CHECK(waited > 0);
            CHECK(behind == "warm-up long_chain");
        }
    }
    setCompileLanes(1);
    setCompileForegroundFirst(false);
    if (!keptCache.empty()) setenv("HAKIOS_METAL_CACHE", keptCache.c_str(), 1);
}
