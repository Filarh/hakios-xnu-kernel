// passcost: what a pass costs beside its draws -- the words, registers, waits and events the command processor is
// given for it, and the processor time of opening and closing it -- on the fake GPU, with the chip's constants.
// (A host tool, for dev-notes/investigacion/anexos/32-interfaz-telegram: not part of the driver.)
//
//   passcost chain N [--clear] [--runs R]      N passes of one textured draw each, each sampling the one before (a blur chain)
//   passcost same N [--runs R]                 the same N draws into one target: one pass (what the draws alone cost)
//   passcost frame CSV CB [--runs R]           the passes of command buffer CB of a frame trace's pasadas-*.csv
//                                              (trace-frame.py / clasifica.py): each pass its target's size, load action
//                                              and number of draws, as the phone's window server recorded them
//   --light                                    CommandBuffer::setLightPasses(true); --keep: and setLightKeepAcrossFill(true)
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "hakios-metal/a6xx.hpp"
#include "hakios-metal/command.hpp"
#include "hakios-metal/fake_gpu.hpp"
#include "hakios-metal/render.hpp"
#include "hakios-metal/testing.hpp"
#include "hakios-metal/texture.hpp"
#include "fixtures/tri_color_vs_air.hpp"
#include "fixtures/tri_flat_vs_air.hpp"
#include "fixtures/tri_cflat_fs_air.hpp"
#include "fixtures/tri_tex_fs_air.hpp"

using namespace hakios::metal;
namespace kernels = hakios::metal::kernels;

namespace {

uint64_t nowNs() { return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()); }

struct Hist {
    std::map<std::string, uint64_t> n;
    uint64_t words = 0, regPackets = 0, regs = 0, events = 0;
    std::map<uint32_t, uint64_t> event;   // by vgt_event_type
    void reset() { *this = Hist{}; }
};
Hist gHist;
bool gLight = false, gKeep = false, gUntuned = false, gSysmem = false;
uint32_t gLightPasses = 0, gFullPasses = 0, gBarriers = 0, gWriteWaits = 0;

const char* opcodeName(uint32_t op) {
    using namespace a6xx;
    switch (op) {
    case pm4::kNop: return "CP_NOP";
    case pm4::kWaitMemWrites: return "CP_WAIT_MEM_WRITES";
    case pm4::kWaitForMe: return "CP_WAIT_FOR_ME";
    case pm4::kWaitForIdle: return "CP_WAIT_FOR_IDLE";
    case pm4::kMemWrite: return "CP_MEM_WRITE";
    case pm4::kIndirectBuffer: return "CP_INDIRECT_BUFFER";
    case pm4::kMemcpy: return "CP_MEMCPY";
    case CP_WAIT_MEM_GTE: return "CP_WAIT_MEM_GTE";
    case CP_SKIP_IB2_ENABLE_GLOBAL: return "CP_SKIP_IB2_ENABLE_GLOBAL";
    case CP_BLIT: return "CP_BLIT";
    case CP_LOAD_STATE6_GEOM: return "CP_LOAD_STATE6_GEOM";
    case CP_LOAD_STATE6_FRAG: return "CP_LOAD_STATE6_FRAG";
    case CP_DRAW_INDX_OFFSET: return "CP_DRAW_INDX_OFFSET";
    case CP_WAIT_REG_MEM: return "CP_WAIT_REG_MEM";
    case CP_REG_TO_MEM: return "CP_REG_TO_MEM";
    case CP_SET_DRAW_STATE: return "CP_SET_DRAW_STATE";
    case CP_EVENT_WRITE: return "CP_EVENT_WRITE";
    case CP_SET_MODE: return "CP_SET_MODE";
    case CP_SET_VISIBILITY_OVERRIDE: return "CP_SET_VISIBILITY_OVERRIDE";
    case CP_SET_MARKER: return "CP_SET_MARKER";
    case CP_SET_BIN_DATA5_OFFSET: return "CP_SET_BIN_DATA5_OFFSET";
    case CP_COND_WRITE5: return "CP_COND_WRITE5";
    default: return "type-7 other";
    }
}
const char* eventName(uint32_t e) {
    using namespace a6xx;
    switch (e) {
    case CACHE_FLUSH_TS: return "CACHE_FLUSH_TS";
    case PC_CCU_INVALIDATE_DEPTH: return "PC_CCU_INVALIDATE_DEPTH";
    case PC_CCU_INVALIDATE_COLOR: return "PC_CCU_INVALIDATE_COLOR";
    case PC_CCU_RESOLVE_TS: return "PC_CCU_RESOLVE_TS";
    case PC_CCU_FLUSH_DEPTH_TS: return "PC_CCU_FLUSH_DEPTH_TS";
    case PC_CCU_FLUSH_COLOR_TS: return "PC_CCU_FLUSH_COLOR_TS";
    case BLIT: return "BLIT";
    case LRZ_FLUSH: return "LRZ_FLUSH";
    case CACHE_INVALIDATE: return "CACHE_INVALIDATE";
    case ZPASS_DONE: return "ZPASS_DONE";
    case UNK_2C: return "UNK_2C";
    case UNK_2D: return "UNK_2D";
    default: return "event other";
    }
}

void walk(FakeGpu& gpu, uint64_t address, uint32_t count, int depth) {
    const uint8_t* bytes = gpu.peek(address, 4ull * count);
    if (bytes == nullptr) return;
    const uint32_t* w = reinterpret_cast<const uint32_t*>(bytes);
    gHist.words += count;
    for (uint32_t i = 0; i < count;) {
        const pm4::Header h = pm4::decode(w[i]);
        if (h.kind == pm4::Header::kBad) return;
        if (h.kind == pm4::Header::kRegisters) {
            gHist.regPackets++;
            gHist.regs += h.count;
        } else {
            gHist.n[opcodeName(h.opcode)]++;
            if (h.opcode == a6xx::CP_EVENT_WRITE && h.count >= 1) {
                gHist.events++;
                gHist.event[w[i + 1] & 0xff]++;
            }
            if (h.opcode == pm4::kIndirectBuffer && h.count == 3 && depth < 3) walk(gpu, uint64_t(w[i + 1]) | uint64_t(w[i + 2]) << 32, w[i + 3], depth + 1);
        }
        i += 1 + h.count;
    }
}
void onSubmit(FakeGpu& gpu, uint64_t address, uint32_t words, void*) { walk(gpu, address, words, 0); }

struct Rig {
    Rig() {
        auto t = std::make_unique<FakeGpu>();
        fake = t.get();
        gpu = std::make_unique<Gpu>(std::move(t));
        std::string why;
        queue = CommandQueue::make(*gpu, &why);
    }
    FakeGpu* fake;
    std::unique_ptr<Gpu> gpu;
    std::unique_ptr<CommandQueue> queue;
};

std::shared_ptr<Texture> texture(Rig& r, uint32_t w, uint32_t h, uint32_t usage) {
    TextureDesc d;
    d.format = PixelFormat::kBGRA8Unorm;
    d.width = w;
    d.height = h;
    d.usage = usage;
    std::string why;
    auto t = Texture::make(*r.gpu, d, &why);
    if (t == nullptr) std::printf("texture %ux%u: %s\n", w, h, why.c_str());
    return t;
}

// the command buffer as the CoreAnimation plug-in sets one up with no metal_* boot-arg (drawbench's pluginBuffer)
std::shared_ptr<CommandBuffer> pluginBuffer(Rig& r) {
    auto cb = r.queue->commandBuffer();
    cb->setSkipRefused(true);
    cb->setProfile(false);
    cb->setGmem(!gSysmem);
    cb->setGmemMinDraws(gSysmem ? 0 : 56);
    cb->setGmemBinning(true);
    cb->setNullFsTextures(false);
    cb->setGmemOff(0);
    cb->setGmemTuned(!gUntuned);
    cb->setProfileDraws(false);
    cb->setLightPasses(gLight);
    cb->setLightKeepAcrossFill(gKeep);
    return cb;
}

struct PassSpec {
    uint32_t w = 0, h = 0, draws = 1;
    LoadAction load = LoadAction::kDontCare;
};

struct Totals {
    std::vector<double> firstUs, joinUs, commitUs, frameUs;
    Hist hist;
    uint32_t passes = 0, draws = 0, refused = 0;
    std::string error;
};

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// One frame: the passes in order, each pass's first draw sampling the target of the pass before it (when it is another
// texture), the rest an icon. Timed: each draw call, the pass's first apart from those that join it.
Totals runFrames(const std::vector<PassSpec>& specs, uint32_t runs) {
    Rig r;
    Totals t;
    std::map<uint64_t, std::shared_ptr<Texture>> targets;
    auto target = [&](uint32_t w, uint32_t h, uint32_t nth) {
        // (two textures a size, taken in turn: a chain's pass never samples the texture it draws into)
        const uint64_t key = (uint64_t(w) << 40) | (uint64_t(h) << 16) | (nth & 1);
        auto it = targets.find(key);
        if (it != targets.end()) return it->second;
        return targets[key] = texture(r, w, h, kTextureUsageRenderTarget | kTextureUsageRead);
    };
    std::vector<std::shared_ptr<Texture>> icons;
    for (uint32_t i = 0; i < 24; i++) icons.push_back(texture(r, 96, 96, kTextureUsageRead));
    Sampler smp;
    std::string why;
    makeSampler(SamplerDesc{}, &smp, &why);
    Program tex{kernels::tri_color_vs_air, kernels::tri_tex_fs_air, Linkage{}, "tri_tex"};
    tex.blend.enabled = true;
    tex.blend.srcRgb = 1, tex.blend.dstRgb = 5, tex.blend.srcAlpha = 1, tex.blend.dstAlpha = 5;
    for (uint32_t frame = 0; frame <= runs; frame++) {
        gHist.reset();
        double firstNs = 0, joinNs = 0;
        uint32_t firsts = 0, joins = 0;
        auto cb = pluginBuffer(r);
        const Texture* before = nullptr;
        std::map<uint64_t, uint32_t> nthOf;
        const uint64_t f0 = nowNs();
        for (size_t p = 0; p < specs.size(); p++) {
            const PassSpec& s = specs[p];
            const uint64_t sizeKey = (uint64_t(s.w) << 32) | s.h;
            std::shared_ptr<Texture> to = target(s.w, s.h, nthOf[sizeKey]);
            if (to.get() == before) to = target(s.w, s.h, ++nthOf[sizeKey]);
            RenderPass pass;
            pass.color0 = to.get();
            pass.load = s.load;
            for (uint32_t i = 0; i < s.draws; i++) {
                Draw d;
                d.vertexCount = 6;
                const bool all = i == 0;
                const double w = all ? double(s.w) : std::min(96.0, double(s.w)), h = all ? double(s.h) : std::min(96.0, double(s.h));
                const double x = all ? 0 : double((i * 37) % std::max(1u, s.w - uint32_t(w) + 1)), y = all ? 0 : double((i * 53) % std::max(1u, s.h - uint32_t(h) + 1));
                d.viewport = Viewport{x, y, w, h, 0, 1};
                RenderPass pp = pass;
                if (i != 0) pp.load = LoadAction::kLoad;
                Bindings fsb;
                fsb.textures = {i == 0 && before != nullptr ? before : icons[(p + i) % icons.size()].get()};
                fsb.samplers = {&smp};
                const uint64_t t0 = nowNs();
                const bool ok = cb->draw(tex, pp, d, Bindings{}, fsb);
                const uint64_t t1 = nowNs();
                if (i == 0) firstNs += double(t1 - t0), firsts++;
                else joinNs += double(t1 - t0), joins++;
                if (!ok) t.refused++, t.error = cb->lastSkipped() + " | " + cb->error();
            }
            before = to.get();
        }
        const uint64_t c0 = nowNs();
        cb->commit();
        const uint64_t c1 = nowNs();
        cb->waitUntilCompleted();
        if (cb->status() != CommandBuffer::kCompleted) t.error = "a command buffer ended in error: " + cb->error() + " | " + r.fake->stats().lastFault;
        if (r.fake->stats().faults != 0 && t.error.empty()) t.error = "the fake GPU: " + r.fake->stats().lastFault;
        gLightPasses = cb->lightCounts().light, gFullPasses = cb->lightCounts().full, gBarriers = cb->lightCounts().barriers;
        gWriteWaits = cb->lightCounts().writeWaits;
        if (frame != 0) {
            t.firstUs.push_back(firsts ? firstNs / firsts / 1e3 : 0);
            t.joinUs.push_back(joins ? joinNs / joins / 1e3 : 0);
            t.commitUs.push_back(double(c1 - c0) / 1e3);
            t.frameUs.push_back(double(c1 - f0) / 1e3);
            t.hist = gHist;
            t.passes = firsts;
            t.draws = firsts + joins;
        }
        cb.reset();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return t;
}

void print(const char* name, const Totals& t) {
    std::printf("%s: %u passes, %u draws%s%s%s\n", name, t.passes, t.draws, gLight ? " (light passes)" : "", gKeep ? " (a fill keeps the shadow)" : "",
                gUntuned ? " (no stamps)" : "");
    if (gLight) std::printf("   passes with the light head %u, with the full head %u; read barriers %u, WFIs before a write %u\n", gLightPasses, gFullPasses, gBarriers, gWriteWaits);
    if (!t.error.empty() || t.refused != 0) std::printf("   !! %u refused; %s\n", t.refused, t.error.c_str());
    std::printf("   processor: a pass's first draw %.2f us (median of %zu frames), a draw that joins %.2f us, commit %.1f us, the frame %.1f us\n", median(t.firstUs),
                t.firstUs.size(), median(t.joinUs), median(t.commitUs), median(t.frameUs));
    std::printf("   stream: %llu words, %llu register packets writing %llu registers\n", (unsigned long long)t.hist.words, (unsigned long long)t.hist.regPackets,
                (unsigned long long)t.hist.regs);
    for (const auto& [k, v] : t.hist.n) std::printf("      %-28s %llu\n", k.c_str(), (unsigned long long)v);
    for (const auto& [k, v] : t.hist.event) std::printf("      event %-22s %llu\n", eventName(k), (unsigned long long)v);
}

std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, c)) out.push_back(item);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "";
    uint32_t runs = 5, n = 0, cbWanted = 0;
    bool clear = false, csv = false;
    std::string file;
    int at = 2;
    if (mode == "chain" || mode == "same") n = argc > at ? uint32_t(atoi(argv[at++])) : 0;
    if (mode == "frame") {
        file = argc > at ? argv[at++] : "";
        cbWanted = argc > at ? uint32_t(atoi(argv[at++])) : 0;
    }
    for (int i = at; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--clear") clear = true;
        else if (a == "--light") gLight = true;
        else if (a == "--keep") gKeep = true;
        else if (a == "--untuned") gUntuned = true;
        else if (a == "--verify") CommandBuffer::verifyDraws(true);   // every kept-state draw against the same draw written whole
        else if (a == "--sysmem") gSysmem = true;   // every pass in sysmem, as the phone runs them while its screen is compressed
        else if (a == "--csv") csv = true;
        else if (a == "--runs" && i + 1 < argc) runs = uint32_t(std::max(1, atoi(argv[++i])));
    }
    if ((mode != "chain" && mode != "same" && mode != "frame") || (mode != "frame" && n == 0) || (mode == "frame" && (file.empty() || cbWanted == 0))) {
        std::printf("passcost chain N [--clear] | same N | frame PASADAS.csv CB   [--runs R] [--light] [--csv]\n");
        return 2;
    }
    FakeGpu::submitHook = onSubmit;
    std::vector<PassSpec> specs;
    if (mode == "chain") {
        // (the sizes of the blur chains in the Telegram trace: 273x216, 238x218, 64x64)
        static const uint32_t sizes[3][2] = {{273, 216}, {238, 218}, {64, 64}};
        for (uint32_t i = 0; i < n; i++) {
            PassSpec s;
            s.w = sizes[i % 3][0], s.h = sizes[i % 3][1];
            s.load = clear ? LoadAction::kClear : LoadAction::kDontCare;
            specs.push_back(s);
        }
    } else if (mode == "same") {
        PassSpec s;
        s.w = 273, s.h = 216, s.draws = n;
        specs.push_back(s);
    } else {
        std::ifstream in(file);
        std::string line;
        std::getline(in, line);
        const std::vector<std::string> head = split(line, ',');
        auto col = [&](const char* name) {
            for (size_t i = 0; i < head.size(); i++) {
                if (head[i] == name) return int(i);
            }
            return -1;
        };
        const int cCb = col("cb"), cRt = col("rt"), cLoad = col("load"), cDraws = col("draws");
        while (std::getline(in, line)) {
            const std::vector<std::string> f = split(line, ',');
            if (int(f.size()) <= std::max(std::max(cCb, cRt), std::max(cLoad, cDraws)) || uint32_t(atoi(f[cCb].c_str())) != cbWanted) continue;
            PassSpec s;
            if (std::sscanf(f[cRt].c_str(), "%ux%u", &s.w, &s.h) != 2 || s.w == 0 || s.h == 0) continue;
            s.draws = uint32_t(std::max(1, atoi(f[cDraws].c_str())));
            s.load = f[cLoad] == "clear" ? LoadAction::kClear : f[cLoad] == "load" ? LoadAction::kLoad : LoadAction::kDontCare;
            specs.push_back(s);
        }
        if (specs.empty()) {
            std::printf("no passes of command buffer %u in %s\n", cbWanted, file.c_str());
            return 1;
        }
    }
    const Totals t = runFrames(specs, runs);
    if (csv) {
        uint64_t wfi = 0, waitReg = 0, waitGte = 0;
        for (const auto& [k, v] : t.hist.n) {
            if (k == "CP_WAIT_FOR_IDLE") wfi = v;
            if (k == "CP_WAIT_REG_MEM") waitReg = v;
            if (k == "CP_WAIT_MEM_GTE") waitGte = v;
        }
        std::printf("%s,%d,%u,%u,%.2f,%.2f,%.1f,%.1f,%llu,%llu,%llu,%llu,%llu,%llu\n", mode.c_str(), gLight ? 1 : 0, t.passes, t.draws, median(t.firstUs), median(t.joinUs),
                    median(t.commitUs), median(t.frameUs), (unsigned long long)t.hist.words, (unsigned long long)t.hist.regs, (unsigned long long)wfi,
                    (unsigned long long)waitReg, (unsigned long long)waitGte, (unsigned long long)t.hist.events);
        return t.error.empty() ? 0 : 1;
    }
    print(mode.c_str(), t);
    return t.error.empty() ? 0 : 1;
}
