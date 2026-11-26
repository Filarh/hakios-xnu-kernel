// cull.hpp's arithmetic: a vertex stage's position run on this processor with a bound on the GPU's own result, and the
// rectangles that follow from it.
#include "hakios-metal/cull.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <string>

#include "hakios-metal/command.hpp"

namespace hakios::metal::cull {

const char* whyName(Why w) {
    static const char* const names[kWhyCount] = {"ok", "no_model", "shape", "no_data", "unknown", "w", "range", "not_rect", "clip", "small"};
    return w < kWhyCount ? names[w] : "?";
}

// --- the instructions ----------------------------------------------------------------------------------------------------
// ir3's encoding, 64 bits an instruction, the category in the top three (Mesa's src/freedreno/isa/ir3.xml and
// ir3-cat1.xml to ir3-cat3.xml; instr-a3xx.h's instr_cat1_t, instr_cat2_t, instr_cat3_t name the same bits):
//   every category 1 to 4   hi 0-7 DST, 8-9 REPEAT
//   cat 0   hi 23-26 OPC (0 nop, 6 end); a nop and an end have every other bit but JP (27) and SY (28) clear
//   cat 1   lo the source (a register in 0-7, a constant in 0-10, or 32 immediate bits); hi 11 SRC_R, 14-16 DST_TYPE,
//           17 DST_REL, 18-20 SRC_TYPE, 21 SRC_C, 22 SRC_IM, 23-24 ROUND, 25-26 OPC (0 mov)
//   cat 2   lo 0-15 SRC1, 16-31 SRC2; hi 10 SAT, 11 SRC1_R, 14 DST_HALF (the destination's width the other one), 19
//           SRC2_R, 20 FULL, 21-26 OPC (0 add.f, 3 mul.f)
//   cat 3   lo 0-12 SRC1, 13 SRC2_C, 14 SRC1_NEG, 15 SRC2_R, 16-28 SRC3, 29 SRC3_R, 30 SRC2_NEG, 31 SRC3_NEG; hi 10 SAT,
//           11 SRC1_R, 14 DST_HALF, 15-22 SRC2 (a register), 23-26 OPC (7 mad.f32: SRC1 * SRC2 + SRC3)
// A cat 2 source (16 bits): a register when bits 8-13 are clear (0-7), a constant when bits 11-13 are 010 (0-10: the
// const file's dword), 14 NEG, 15 ABS. A cat 3 source 1 or 3 (13 bits): a register when bits 8-12 are clear, a constant
// when bits 11-12 are 10. A register is n * 4 + component; with REPEAT the instruction runs REPEAT + 1 times, the
// destination and every source marked R one component further each time. With no REPEAT the R bits are (nop) counts.
namespace {

constexpr uint32_t kRegs = 192;   // r0.x to r47.w: the registers a stage's own values live in (above them: shared, a0, p0, none)
constexpr uint32_t kTypeF32 = 1, kTypeU32 = 3, kTypeS32 = 5;

struct Operand {
    enum Kind : uint8_t { kNone, kReg, kConst, kImmed } kind = kNone;
    uint32_t n = 0;         // the register, the const file's dword, or the immediate's bits
    bool neg = false, abs = false, r = false;
};
struct Instr {
    enum Op : uint8_t {
        kBad,     // the stage has no model
        kNop,
        kEnd,
        kMov,     // dst = a
        kAdd,     // dst = a + b
        kMul,     // dst = a * b
        kMad,     // dst = a * b + c
        kTaint,   // dst is no longer known
    } op = kBad;
    uint32_t dst = 0, repeat = 0;
    Operand a, b, c;
};

bool operand2(uint32_t s, Operand* o) {
    o->neg = (s >> 14) & 1;
    o->abs = (s >> 15) & 1;
    if ((s & 0x3f00) == 0) {
        o->kind = Operand::kReg;
        o->n = s & 0xff;
        return true;
    }
    if ((s & 0x3800) == 0x1000) {
        o->kind = Operand::kConst;
        o->n = s & 0x7ff;
        return true;
    }
    return false;   // an immediate, or relative to a0
}
bool operand3(uint32_t s, Operand* o) {
    if ((s & 0x1f00) == 0) {
        o->kind = Operand::kReg;
        o->n = s & 0xff;
        return true;
    }
    if ((s & 0x1800) == 0x1000) {
        o->kind = Operand::kConst;
        o->n = s & 0x7ff;
        return true;
    }
    return false;
}

Instr decode(uint32_t lo, uint32_t hi) {
    Instr i;
    const uint32_t cat = hi >> 29;
    i.dst = hi & 0xff;
    i.repeat = (hi >> 8) & 3;
    const bool known = i.dst < kRegs && i.dst + i.repeat < kRegs;   // (a destination among the stage's own registers)
    switch (cat) {
    case 0: {
        const uint32_t rest = hi & ~((1u << 27) | (1u << 28));
        if (lo == 0 && rest == 0) i.op = Instr::kNop;
        else if (lo == 0 && rest == (6u << 23)) i.op = Instr::kEnd;
        return i;
    }
    case 1: {
        const uint32_t dstType = (hi >> 14) & 7, srcType = (hi >> 18) & 7;
        const bool srcC = (hi >> 21) & 1, srcIm = (hi >> 22) & 1;
        // (a mov of 32 bits as they are; a conversion, a relative destination, an address or predicate register: no model)
        if (((hi >> 25) & 3) != 0 || ((hi >> 17) & 1) != 0 || ((hi >> 23) & 3) != 0 || ((hi >> 10) & 1) != 0 || !known) return i;
        if (dstType != srcType || (srcType != kTypeF32 && srcType != kTypeU32 && srcType != kTypeS32)) return i;
        if (srcC && srcIm) return i;
        i.a.r = (hi >> 11) & 1;
        if (srcIm) {
            i.a.kind = Operand::kImmed;
            i.a.n = lo;
        } else if (srcC) {
            if ((lo & ~0x7ffu) != 0) return i;
            i.a.kind = Operand::kConst;
            i.a.n = lo;
        } else {
            if ((lo & ~0xffu) != 0) return i;
            i.a.kind = Operand::kReg;
            i.a.n = lo;
        }
        i.op = Instr::kMov;
        return i;
    }
    case 2: {
        // (a destination outside the stage's registers is a predicate's, p0: nothing of the registers changes)
        if (!known) {
            i.op = i.dst >= kRegs ? Instr::kNop : Instr::kBad;
            return i;
        }
        const uint32_t opc = (hi >> 21) & 0x3f;
        const bool plain = ((hi >> 20) & 1) != 0 && ((hi >> 14) & 1) == 0 && ((hi >> 10) & 1) == 0;   // full, not half, not saturated
        i.op = Instr::kTaint;
        if ((opc == 0 || opc == 3) && plain && operand2(lo & 0xffff, &i.a) && operand2(lo >> 16, &i.b)) {
            i.a.r = (hi >> 11) & 1;
            i.b.r = (hi >> 19) & 1;
            i.op = opc == 0 ? Instr::kAdd : Instr::kMul;
        }
        return i;
    }
    case 3: {
        if (!known) return i;
        const uint32_t opc = (hi >> 23) & 0xf;
        const bool plain = ((hi >> 14) & 1) == 0 && ((hi >> 10) & 1) == 0;
        i.op = Instr::kTaint;
        if (opc == 7 && plain && ((lo >> 13) & 1) == 0 && operand3(lo & 0x1fff, &i.a) && operand3((lo >> 16) & 0x1fff, &i.c)) {
            i.a.neg = (lo >> 14) & 1;
            i.a.r = (hi >> 11) & 1;
            i.b.kind = Operand::kReg;
            i.b.n = (hi >> 15) & 0xff;
            i.b.neg = (lo >> 30) & 1;
            i.b.r = (lo >> 15) & 1;
            i.c.neg = (lo >> 31) & 1;
            i.c.r = (lo >> 29) & 1;
            i.op = Instr::kMad;
        }
        return i;
    }
    case 4:
        if (known) i.op = Instr::kTaint;
        return i;
    default:
        return i;   // textures, memory, barriers: no model
    }
}

// (as Mesa's disassembler writes an operand, so that the two can be held against each other word for word)
std::string operandText(const Operand& o, bool repeated) {
    static const char comp[] = "xyzw";
    char t[48];
    const bool r = o.r && repeated;
    switch (o.kind) {
    case Operand::kReg: snprintf(t, sizeof t, "%s%s%sr%u.%c", o.neg ? "(neg)" : "", o.abs ? "(abs)" : "", r ? "(r)" : "", o.n / 4, comp[o.n % 4]); break;
    case Operand::kConst: snprintf(t, sizeof t, "%s%s%sc%u.%c", o.neg ? "(neg)" : "", o.abs ? "(abs)" : "", r ? "(r)" : "", o.n / 4, comp[o.n % 4]); break;
    case Operand::kImmed: snprintf(t, sizeof t, "0x%x", o.n); break;
    default: return "?";
    }
    return t;
}

// --- a value and how far the GPU's may lie from it -----------------------------------------------------------------------
struct Val {
    double v = 0, e = 0;
    bool known = false;
};
const Val kUnknownVal{};

double ulp32(double x) {
    const float f = float(std::fabs(x));
    if (!(f < FLT_MAX)) return INFINITY;
    return double(std::nextafterf(f, INFINITY)) - double(f);
}
bool isF32(double x) { return double(float(x)) == x; }
Val checked(double v, double e) {
    // (below the smallest normal float the GPU may give 0: flush to zero is allowed)
    if (v != 0 && std::fabs(v) < double(FLT_MIN)) e += double(FLT_MIN);
    if (!std::isfinite(v) || !std::isfinite(e) || std::fabs(v) + e >= double(FLT_MAX)) return kUnknownVal;
    return Val{v, e, true};
}
// a * b + c, where the product may be rounded on its own (not fused) or not (fused); PRODUCT false: a + c (b unread)
Val madOf(const Val& a, const Val& b, const Val& c, bool product) {
    if (!a.known || !c.known || (product && !b.known)) return kUnknownVal;
    double pv = a.v, pe = a.e;
    bool pExact = a.e == 0;   // (the product as a real number is PV exactly, and a float holds it)
    if (product) {
        if ((a.v == 0 && a.e == 0) || (b.v == 0 && b.e == 0)) {
            pv = 0, pe = 0, pExact = true;   // (0 times a finite number: 0 whatever the rounding)
        } else {
            pv = a.v * b.v;   // (two floats' product is exact in a double; of values with a bound, the double's rounding is in it below)
            pe = std::fabs(a.v) * b.e + std::fabs(b.v) * a.e + a.e * b.e;
            pExact = a.e == 0 && b.e == 0 && isF32(pv) && (pv == 0 || std::fabs(pv) >= double(FLT_MIN));
        }
    }
    if (pExact && pv == 0) return c;                           // (0 + c is c)
    if (pExact && c.e == 0 && c.v == 0) return checked(pv, 0);   // (p + 0 is p)
    const double s = pv + c.v;
    const double bb = s - pv, twoSum = (pv - (s - bb)) + (c.v - bb);   // (what the double's sum lost: 0 when it is exact)
    if (pExact && c.e == 0 && twoSum == 0 && isF32(s) && (s == 0 || std::fabs(s) >= double(FLT_MIN))) return checked(s, 0);
    const double e = pe + c.e + kUlps * ((product ? ulp32(std::fabs(pv) + pe) : 0.0) + ulp32(std::fabs(s) + pe + c.e)) +
                     (std::fabs(pv) + std::fabs(s)) * 0x1p-50;
    return checked(s, e);
}
Val mulOf(const Val& a, const Val& b) { return madOf(a, b, Val{0, 0, true}, true); }

Val flagged(Val v, const Operand& o) {
    if (o.abs) v.v = std::fabs(v.v);
    if (o.neg) v.v = -v.v;
    return v;
}

// one draw's constants, read as they are asked for
struct Consts {
    const Geometry* g;
    Snapshots* snaps;
    bool failed = false;   // a range could not be read: kNoData
    struct Loaded {
        uint8_t state = 0;   // 0 not asked for yet, 1 read, 2 not readable
        float f[kMaxRangeDwords];
    } range[kMaxUboRanges];
    Val at(uint32_t dword) {
        const VsModel& m = *g->vs;
        if (dword >= m.immediatesAt && dword - m.immediatesAt < m.immediates.size()) {
            float f;
            memcpy(&f, &m.immediates[dword - m.immediatesAt], 4);
            return std::isfinite(f) ? Val{double(f), 0, true} : kUnknownVal;
        }
        for (size_t i = 0; i < m.ranges.size() && i < kMaxUboRanges; i++) {
            const VsModel::Range& r = m.ranges[i];
            if (dword < 4 * r.offsetVec4 || dword - 4 * r.offsetVec4 >= r.sizeDwords) continue;
            Loaded& l = range[i];
            if (l.state == 0) {
                const Source& s = g->src[r.source];
                // (the command processor loads the range from the buffer when it runs the draw, straight or through a
                // copy to the draw's table: render.cpp's pushedRanges. One that runs past the buffer's pages is read
                // by the driver as the draw is encoded instead, and is not followed here.)
                l.state = s.buffer != nullptr && s.offset % 4 == 0 && r.sizeDwords <= kMaxRangeDwords &&
                                  snaps->take(*s.buffer, s.offset + 4ull * r.startDword, 4 * r.sizeDwords, l.f)
                              ? 1
                              : 2;
                if (l.state == 2) failed = true;
            }
            if (l.state != 1) return kUnknownVal;
            const float f = l.f[dword - 4 * r.offsetVec4];
            return std::isfinite(f) ? Val{double(f), 0, true} : kUnknownVal;
        }
        return kUnknownVal;   // driver params, a module's constant data, or nothing
    }
};

Val read(const Operand& o, uint32_t step, const Val* regs, Consts* consts) {
    const uint32_t n = o.n + (o.r ? step : 0);
    switch (o.kind) {
    case Operand::kReg: return n < kRegs ? flagged(regs[n], o) : kUnknownVal;
    case Operand::kConst: return flagged(consts->at(n), o);
    case Operand::kImmed: {
        float f;
        memcpy(&f, &o.n, 4);
        return std::isfinite(f) ? Val{double(f), 0, true} : kUnknownVal;
    }
    default: return kUnknownVal;
    }
}

// the stage run for one vertex whose inputs are in REGS; false: an instruction with no model (modelOf has none such)
bool run(const VsModel& m, Val* regs, Consts* consts) {
    for (size_t at = 0; at + 1 < m.words.size(); at += 2) {
        const Instr i = decode(m.words[at], m.words[at + 1]);
        switch (i.op) {
        case Instr::kBad: return false;
        case Instr::kEnd: return true;
        case Instr::kNop: break;
        default:
            for (uint32_t k = 0; k <= i.repeat; k++) {
                Val v;
                if (i.op == Instr::kMov) v = read(i.a, k, regs, consts);
                else if (i.op == Instr::kAdd) v = madOf(read(i.a, k, regs, consts), kUnknownVal, read(i.b, k, regs, consts), false);
                else if (i.op == Instr::kMul) v = mulOf(read(i.a, k, regs, consts), read(i.b, k, regs, consts));
                else if (i.op == Instr::kMad) v = madOf(read(i.a, k, regs, consts), read(i.b, k, regs, consts), read(i.c, k, regs, consts), true);
                regs[i.dst + k] = v;   // (kTaint: unknown)
            }
        }
    }
    return false;   // (no end)
}

constexpr uint32_t kVertAttribGeneric0 = 15;   // (render.cpp's: a [[stage_in]] attribute's slot)
constexpr uint32_t kRegidNone = 0xfc;

std::shared_ptr<const VsModel> build(const Program& program) {
    auto m = std::make_shared<VsModel>();
    const Stage& vs = program.vs;
    const Kernel& k = vs.k;
    // (merged registers, as ir3 has them on this generation: a half register is the full one of the same number, so a
    // write to one makes that register unknown and no other)
    if (vs.positionRegid == kRegidNone || vs.positionRegid + 3 >= kRegs || k.words == nullptr || k.maxReg < 0 || k.maxReg >= int32_t(kRegs / 4) ||
        !k.mergedRegs) {
        return m;
    }
    bool ended = false;
    for (uint32_t at = 0; at + 1 < k.numWords && at < 512 && !ended; at += 2) {
        const Instr i = decode(k.words[at], k.words[at + 1]);
        if (i.op == Instr::kBad) return m;
        m->words.push_back(k.words[at]);
        m->words.push_back(k.words[at + 1]);
        ended = i.op == Instr::kEnd;
    }
    if (!ended) return m;
    m->registers = std::max(uint32_t(k.maxReg + 1) * 4, vs.positionRegid + 4);
    m->positionRegid = vs.positionRegid;
    auto sourceOf = [&](uint32_t bufferIndex) -> uint8_t {
        for (uint32_t s = 0; s < m->sources; s++) {
            if (m->sourceIndex[s] == bufferIndex) return uint8_t(s);
        }
        if (m->sources == kMaxSources || bufferIndex > 0xff) return 0xff;
        m->sourceIndex[m->sources] = uint8_t(bufferIndex);
        return uint8_t(m->sources++);
    };
    // the constants first: a position cannot be had without them, an attribute it does not read can be left unknown
    for (uint32_t i = 0; i < k.numUboRanges && i < kMaxUboRanges; i++) {
        const KernelUboRange& r = k.uboRanges[i];
        VsModel::Range range{0xff, r.offsetVec4, r.startDword, r.sizeDwords};
        // (a module's constant data, loaded from the program's own words: not followed -- 0 dwords, never found)
        if (k.constantDataUbo != 0 && r.block == k.constantDataUbo - 1) range.sizeDwords = 0;
        else range.source = sourceOf(r.block);
        if (range.source == 0xff) range.sizeDwords = 0;
        m->ranges.push_back(range);
    }
    for (uint32_t i = 0; i < vs.numInputs && i < kMaxStageInputs; i++) {
        const StageInput& in = vs.inputs[i];
        if (in.regid == kRegidNone || in.compmask == 0) continue;
        VsModel::Input input{in.regid, in.compmask, 0, 0xff, 0, 0};
        if (!in.sysval && !in.half && in.slot >= kVertAttribGeneric0 && in.slot - kVertAttribGeneric0 < kMaxVertexAttributes) {
            const VertexAttribute& a = program.vertexDescriptor.attributes[in.slot - kVertAttribGeneric0];
            const uint32_t f = uint32_t(a.format), first = uint32_t(VertexFormat::kFloat);
            if (f >= first && f <= uint32_t(VertexFormat::kFloat4) && a.bufferIndex < kMaxVertexBuffers) {
                const VertexLayout& l = program.vertexDescriptor.layouts[a.bufferIndex];
                input.source = l.step == VertexStep::kPerVertex && l.stride != 0 ? sourceOf(a.bufferIndex) : 0xff;
                if (input.source != 0xff) {
                    input.comps = uint8_t(f - first + 1);
                    input.offset = a.offset;
                    input.stride = l.stride;
                }
            }
        }
        m->inputs.push_back(input);
    }
    if (k.immediates != nullptr && k.numImmediates != 0) {
        m->immediates.assign(k.immediates, k.immediates + k.numImmediates);
        m->immediatesAt = 4 * k.immediatesOffsetVec4;
    }
    m->usable = true;
    return m;
}

}  // namespace

std::string disassemble(uint32_t lo, uint32_t hi) {
    const Instr i = decode(lo, hi);
    static const char comp[] = "xyzw";
    std::string out;
    if (i.repeat != 0 && i.op != Instr::kNop && i.op != Instr::kEnd && i.op != Instr::kBad) out = "(rpt" + std::to_string(i.repeat) + ")";
    char dst[24];
    snprintf(dst, sizeof dst, "r%u.%c", i.dst / 4, comp[i.dst % 4]);
    switch (i.op) {
    case Instr::kNop: return "nop";
    case Instr::kEnd: return "end";
    case Instr::kMov: return out + "mov " + dst + ", " + operandText(i.a, i.repeat != 0);
    case Instr::kAdd: return out + "add.f " + dst + ", " + operandText(i.a, i.repeat != 0) + ", " + operandText(i.b, i.repeat != 0);
    case Instr::kMul: return out + "mul.f " + dst + ", " + operandText(i.a, i.repeat != 0) + ", " + operandText(i.b, i.repeat != 0);
    case Instr::kMad:
        return out + "mad.f32 " + dst + ", " + operandText(i.a, i.repeat != 0) + ", " + operandText(i.b, i.repeat != 0) + ", " + operandText(i.c, i.repeat != 0);
    case Instr::kTaint: return out + "taint " + dst;
    default: return "?";
    }
}

std::shared_ptr<const VsModel> modelOf(const Program& program) {
    // (by the uid, which names a Program and everything it points at; a Program made by hand has none and is read each time)
    struct Kept {
        uint64_t uid = 0;
        std::shared_ptr<const VsModel> model;
    };
    static thread_local Kept kept[64];
    if (program.uid == 0) return build(program);
    Kept& slot = kept[(program.uid * 0x9e3779b97f4a7c15ull) >> 58];
    if (slot.uid != program.uid || slot.model == nullptr) {
        slot.model = build(program);
        slot.uid = program.uid;
    }
    return slot.model;
}

Why capture(const Program& program, const Draw& draw, const Bindings& vsBindings, const uint32_t area[4], Geometry* out) {
    if (draw.indirect != nullptr || draw.instanceCount != 1 || draw.vertexCount == 0 || draw.vertexCount > kMaxVertices || draw.scissorEmpty ||
        (draw.primitive != Primitive::kTriangle && draw.primitive != Primitive::kTriangleStrip) || draw.vertexCount < 3) {
        return kShape;
    }
    Geometry& g = *out;
    g.vs = modelOf(program);
    if (!g.vs->usable) return kNoModel;
    for (uint32_t s = 0; s < g.vs->sources; s++) {
        const uint32_t index = g.vs->sourceIndex[s];
        if (index < vsBindings.buffers.size()) g.src[s] = Source{vsBindings.buffers[index].buffer, vsBindings.buffers[index].offset};
        else g.src[s] = Source{};
    }
    g.primitive = draw.primitive;
    g.vertexStart = draw.vertexStart;
    g.vertexCount = draw.vertexCount;
    g.indexBuffer = draw.indexBuffer;
    g.indexOffset = draw.indexOffset;
    g.indexType = draw.indexType;
    g.baseVertex = draw.baseVertex;
    // (a6xx::viewportWords' floats: what GRAS_CL_VPORT is given)
    const Viewport& vp = draw.viewport;
    const float xs = float(vp.width) / 2.0f, ys = float(vp.height) / 2.0f;
    g.xoffset = float(vp.x) + xs;
    g.xscale = xs;
    g.yoffset = float(vp.y) + ys;
    g.yscale = -ys;
    if (!std::isfinite(g.xoffset) || !std::isfinite(g.yoffset) || !(xs > 0) || !(ys > 0)) return kShape;
    // The guard band: GRAS_CL_GUARDBAND_CLIP_ADJ holds, rounded down to 3.6 floating point and capped, the NDC extent
    // whose window coordinates stay within -32768..32767 (a6xx::calcGuardband, fd_calc_guardband); a triangle inside
    // it is not clipped in x or y. Half of it, and of the cap's 254, leaves the rounding far behind.
    auto guard = [](double offset, double scale) {
        const double gb = std::min((32768.0 + offset) / scale, (32767.0 - offset) / scale);
        return gb >= 2.0 ? std::min(gb, 250.0) / 2.0 : 0.0;
    };
    g.guardX = guard(g.xoffset, xs);
    g.guardY = guard(g.yoffset, ys);
    g.area = Rect{int32_t(area[0]), int32_t(area[1]), int32_t(area[2]), int32_t(area[3])};
    return kOk;
}

bool Snapshots::take(const Buffer& b, uint64_t offset, uint32_t length, void* out) {
    const Bo& bo = b.bo();
    if (b.contents() == nullptr || bo.surface() || bo.gpuWritten() || offset > bo.size() || length > bo.size() - offset) return false;
    memcpy(out, static_cast<const uint8_t*>(b.contents()) + offset, length);
    const uint32_t at = uint32_t(bytes_.size());
    bytes_.insert(bytes_.end(), static_cast<const uint8_t*>(out), static_cast<const uint8_t*>(out) + length);
    records_.push_back(Record{&b, offset, length, at});
    return true;
}

bool Snapshots::same(uint32_t first, uint32_t count) const {
    for (uint32_t i = first; i < first + count && i < records_.size(); i++) {
        const Record& r = records_[i];
        if (r.buffer->contents() == nullptr || r.buffer->bo().gpuWritten()) return false;
        if (memcmp(static_cast<const uint8_t*>(r.buffer->contents()) + r.offset, bytes_.data() + r.at, r.length) != 0) return false;
    }
    return first + count <= records_.size();
}

uint32_t Snapshots::absorb(Snapshots& other) {
    const uint32_t firstRecord = uint32_t(records_.size()), firstByte = uint32_t(bytes_.size());
    bytes_.insert(bytes_.end(), other.bytes_.begin(), other.bytes_.end());
    for (Record r : other.records_) {
        r.at += firstByte;
        records_.push_back(r);
    }
    other.clear();
    return firstRecord;
}

Why evaluate(const Geometry& g, Snapshots* snapshots, Positions* out) {
    const VsModel& m = *g.vs;
    Positions& p = *out;
    p.count = 0;
    p.restart = false;
    // the vertices' numbers: the range, or the indices (each fetched vertex worked out once)
    uint32_t vertex[kMaxVertices];
    if (g.indexBuffer != nullptr) {
        const uint32_t size = g.indexType == 1 ? 4 : 2;
        uint8_t raw[4 * kMaxVertices];
        if (g.indexOffset > g.indexBuffer->length() || uint64_t(size) * g.vertexCount > g.indexBuffer->length() - g.indexOffset ||
            !snapshots->take(*g.indexBuffer, g.indexOffset, size * g.vertexCount, raw)) {
            return kNoData;
        }
        for (uint32_t i = 0; i < g.vertexCount; i++) {
            uint32_t x = 0;
            memcpy(&x, raw + size * i, size);
            if (x == (size == 4 ? 0xffffffffu : 0xffffu)) {
                p.restart = true;
                continue;
            }
            const int64_t v = int64_t(x) + g.baseVertex;
            if (v < 0 || v > 0x7fffffff) return kNoData;
            vertex[p.count++] = uint32_t(v);
        }
    } else {
        for (uint32_t i = 0; i < g.vertexCount; i++) vertex[p.count++] = g.vertexStart + i;
    }
    if (p.count == 0) return kShape;
    Consts consts;
    consts.g = &g;
    consts.snaps = snapshots;
    Val regs[kRegs];
    for (uint32_t i = 0; i < p.count; i++) {
        bool again = false;
        for (uint32_t j = 0; j < i && !again; j++) {
            if (vertex[j] == vertex[i]) {
                p.p[i] = p.p[j];
                again = true;
            }
        }
        if (again) continue;
        for (uint32_t r = 0; r < m.registers && r < kRegs; r++) regs[r] = kUnknownVal;
        for (const VsModel::Input& in : m.inputs) {
            if (in.comps == 0) continue;   // (left unknown)
            // (VFD_FETCH: the buffer from the binding's offset to its length; a vertex past it reads zeros there, and
            // is refused here. A component the format has not is 0, the fourth 1.)
            const Source& s = g.src[in.source];
            float f[4] = {0, 0, 0, 1};
            if (s.buffer == nullptr || s.offset > s.buffer->length()) return kNoData;
            const uint64_t at = uint64_t(vertex[i]) * in.stride + in.offset, bytes = s.buffer->length() - s.offset;
            if (at + 4ull * in.comps > bytes || !snapshots->take(*s.buffer, s.offset + at, 4u * in.comps, f)) return kNoData;
            for (uint32_t c = 0; c < 4; c++) {
                if (((in.compmask >> c) & 1) && in.regid + c < kRegs) regs[in.regid + c] = std::isfinite(f[c]) ? Val{double(f[c]), 0, true} : kUnknownVal;
            }
        }
        if (!run(m, regs, &consts)) return kNoModel;
        if (consts.failed) return kNoData;
        const Val &X = regs[m.positionRegid], &Y = regs[m.positionRegid + 1], &Z = regs[m.positionRegid + 2], &W = regs[m.positionRegid + 3];
        if (!X.known || !Y.known || !W.known) return kUnknown;
        const double w = W.v - W.e;   // (the least the GPU's w may be)
        if (!(w > 0)) return kW;
        const double nx = X.v / W.v, ny = Y.v / W.v;
        const double ex = (X.e + std::fabs(nx) * W.e) / w, ey = (Y.e + std::fabs(ny) * W.e) / w;
        Positions::P& q = p.p[i];
        q.x = double(g.xoffset) + double(g.xscale) * nx;
        q.y = double(g.yoffset) + double(g.yscale) * ny;
        const double e = std::max(std::fabs(double(g.xscale)) * ex, std::fabs(double(g.yscale)) * ey);
        if (!std::isfinite(q.x) || !std::isfinite(q.y) || !(e <= 0.25) || std::fabs(q.x) > 1e6 || std::fabs(q.y) > 1e6) return kRange;
        q.u = kMargin + e;
        // (Metal's clip volume, as GRAS_CL_CNTL has it: 0 <= z <= w; and x, y inside the guard band)
        q.inside = Z.known && Z.v - Z.e >= 0 && Z.v + Z.e <= w && g.guardX > 0 && g.guardY > 0 && std::fabs(nx) + ex <= g.guardX &&
                   std::fabs(ny) + ey <= g.guardY;
    }
    return kOk;
}

Rect coverBox(const Geometry& g, const Positions& p) {
    double x0 = INFINITY, y0 = INFINITY, x1 = -INFINITY, y1 = -INFINITY;
    for (uint32_t i = 0; i < p.count; i++) {
        x0 = std::min(x0, p.p[i].x - p.p[i].u);
        y0 = std::min(y0, p.p[i].y - p.p[i].u);
        x1 = std::max(x1, p.p[i].x + p.p[i].u);
        y1 = std::max(y1, p.p[i].y + p.p[i].u);
    }
    // the pixels whose centres the widened box holds, in the draw's area
    Rect r;
    r.x0 = std::max(g.area.x0, int32_t(std::max(-1e6, std::ceil(x0 - 0.5))));
    r.y0 = std::max(g.area.y0, int32_t(std::max(-1e6, std::ceil(y0 - 0.5))));
    r.x1 = std::min(g.area.x1, int32_t(std::min(1e6, std::floor(x1 - 0.5))));
    r.y1 = std::min(g.area.y1, int32_t(std::min(1e6, std::floor(y1 - 0.5))));
    return r;
}

Why occluderRect(const Geometry& g, const Positions& p, Rect* out) {
    // two triangles: a list of six vertices, or a strip of four
    uint32_t tri[2][3];
    if (p.restart) return kShape;
    if (g.primitive == Primitive::kTriangle && p.count == 6) {
        for (uint32_t i = 0; i < 6; i++) tri[i / 3][i % 3] = i;
    } else if (g.primitive == Primitive::kTriangleStrip && p.count == 4) {
        for (uint32_t i = 0; i < 3; i++) tri[0][i] = i, tri[1][i] = i + 1;
    } else {
        return kShape;
    }
    // exactly two x and two y among the corners, as the interpreter puts them (equal inputs through equal instructions)
    double xs[2] = {p.p[0].x, p.p[0].x}, ys[2] = {p.p[0].y, p.p[0].y};
    uint32_t nx = 1, ny = 1, corner[kMaxVertices];
    double u = 0;
    for (uint32_t i = 0; i < p.count; i++) {
        const Positions::P& q = p.p[i];
        if (!q.inside) return kClip;
        uint32_t ix = q.x == xs[0] ? 0 : nx == 2 && q.x == xs[1] ? 1 : 2, iy = q.y == ys[0] ? 0 : ny == 2 && q.y == ys[1] ? 1 : 2;
        if (ix == 2) {
            if (nx == 2) return kNotRect;
            xs[nx++] = q.x, ix = 1;
        }
        if (iy == 2) {
            if (ny == 2) return kNotRect;
            ys[ny++] = q.y, iy = 1;
        }
        corner[i] = ix + 2 * iy;
        u = std::max(u, q.u);
    }
    if (nx != 2 || ny != 2) return kNotRect;
    // each triangle three corners, the two leaving out opposite ones: the halves either side of one diagonal, which the
    // rasterizer gives to exactly one of them pixel by pixel (the same two vertices, the same edge)
    uint32_t missing[2];
    for (uint32_t t = 0; t < 2; t++) {
        const uint32_t mask = (1u << corner[tri[t][0]]) | (1u << corner[tri[t][1]]) | (1u << corner[tri[t][2]]);
        if (mask != 0xe && mask != 0xd && mask != 0xb && mask != 0x7) return kNotRect;
        missing[t] = mask == 0xe ? 0 : mask == 0xd ? 1 : mask == 0xb ? 2 : 3;
    }
    if ((missing[0] ^ missing[1]) != 3) return kNotRect;
    // the pixels whose centres lie inside by more than the bound, in the draw's area
    const double x0 = std::min(xs[0], xs[1]) + u, x1 = std::max(xs[0], xs[1]) - u, y0 = std::min(ys[0], ys[1]) + u, y1 = std::max(ys[0], ys[1]) - u;
    Rect r;
    r.x0 = std::max(g.area.x0, int32_t(std::max(-1e6, std::ceil(x0 - 0.5))));
    r.y0 = std::max(g.area.y0, int32_t(std::max(-1e6, std::ceil(y0 - 0.5))));
    r.x1 = std::min(g.area.x1, int32_t(std::min(1e6, std::floor(x1 - 0.5))));
    r.y1 = std::min(g.area.y1, int32_t(std::min(1e6, std::floor(y1 - 0.5))));
    *out = r;
    return kOk;
}

bool Region::subtract(const Rect& o) {
    Rect left[kMaxRegion];
    uint32_t k = 0;
    auto put = [&](int32_t x0, int32_t y0, int32_t x1, int32_t y1) {
        if (x0 > x1 || y0 > y1) return true;
        if (k == kMaxRegion) return false;
        left[k++] = Rect{x0, y0, x1, y1};
        return true;
    };
    for (uint32_t i = 0; i < n; i++) {
        const Rect& a = r[i];
        if (o.empty() || o.x0 > a.x1 || o.x1 < a.x0 || o.y0 > a.y1 || o.y1 < a.y0) {
            if (!put(a.x0, a.y0, a.x1, a.y1)) return false;
            continue;
        }
        // the rows above and below it whole, then what is left of its own rows either side
        const int32_t y0 = std::max(a.y0, o.y0), y1 = std::min(a.y1, o.y1);
        if (!put(a.x0, a.y0, a.x1, o.y0 - 1) || !put(a.x0, o.y1 + 1, a.x1, a.y1) || !put(a.x0, y0, o.x0 - 1, y1) || !put(o.x1 + 1, y0, a.x1, y1)) return false;
    }
    n = k;
    std::copy(left, left + k, r);
    return true;
}

}  // namespace hakios::metal::cull
