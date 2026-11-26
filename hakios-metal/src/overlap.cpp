// Whether the primitives of one draw can cover a pixel twice (overlap.hpp). SPDX-License-Identifier: MIT
#include "hakios-metal/overlap.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "hakios-metal/a6xx.hpp"
#include "hakios-metal/command.hpp"

namespace hakios::metal::render {

size_t OverlapWitness::take(const uint8_t* live, size_t length) {
    if (live == nullptr || length == 0 || bytes.size() - first + length > kMaxBytes) return SIZE_MAX;
    const size_t at = bytes.size();
    bytes.resize(at + length);
    memcpy(bytes.data() + at, live, length);
    spans.push_back(Span{live, at, length});
    return at;
}

bool OverlapWitness::holds(size_t from, size_t to) const {
    for (size_t i = from; i < to && i < spans.size(); i++) {
        if (memcmp(spans[i].live, bytes.data() + spans[i].at, spans[i].length) != 0) return false;
    }
    return true;
}

namespace {

constexpr uint32_t kVertAttribGeneric0 = 15;   // VERT_ATTRIB_GENERIC0 (render.cpp's)
constexpr uint32_t kSyms = 8;                  // attribute components followed: two float4
constexpr uint32_t kRegs = 192;                // r0.x to r47.w: the registers a stage owns (r48 and up are shared or special)
constexpr uint32_t kMaxInstructions = 512;
constexpr uint32_t kMaxLines = 1024;           // (CoreAnimation's coverage spans: up to 768 lines a draw in the panel's traces)

// One register as the walk knows it: C[i] times attribute component i, summed, plus C[kSyms] -- with M the same sum
// taken over absolute values at every step (what bounds the rounding: a float32 evaluation of a sum of products N
// operations deep differs from the exact one by at most gamma_N times that sum of magnitudes), and N the operations
// on its longest path.
struct Form {
    double c[kSyms + 1];
    double m[kSyms + 1];
    uint32_t n;
    bool constant() const {
        for (uint32_t i = 0; i < kSyms; i++) {
            if (m[i] != 0) return false;
        }
        return true;
    }
};

Form constantForm(float v) {
    Form f{};
    f.c[kSyms] = double(v);
    f.m[kSyms] = std::fabs(double(v));
    return f;
}

Form negated(Form f) {
    for (double& x : f.c) x = -x;
    return f;
}

Form sum(const Form& a, const Form& b) {
    Form f;
    for (uint32_t i = 0; i <= kSyms; i++) {
        f.c[i] = a.c[i] + b.c[i];
        f.m[i] = a.m[i] + b.m[i];
    }
    f.n = std::max(a.n, b.n) + 1;
    return f;
}

// a product is affine only when one side depends on no attribute
bool product(const Form& a, const Form& b, Form* out) {
    const Form* k = a.constant() ? &a : b.constant() ? &b : nullptr;
    if (k == nullptr) return false;
    const Form& x = k == &a ? b : a;
    for (uint32_t i = 0; i <= kSyms; i++) {
        out->c[i] = k->c[kSyms] * x.c[i];
        out->m[i] = k->m[kSyms] * x.m[i];
    }
    out->n = a.n + b.n + 1;
    return true;
}

// --- the vertex stage's machine code, walked ------------------------------------------------------------------------------
// ir3's encoding, from Mesa 25.0.7's src/freedreno/isa: an instruction is 64 bits (ir3.xml:50), here LO its first
// word and HI its second (bit 32 + i of the instruction is bit i of HI); its category bits 61-63.
//   cat0 (ir3-cat0.xml:25-76)   OPC bits 55-58 with OPC_HI's low bit 49 clear: 0 nop, 6 end
//   cat1 (ir3-cat1.xml)         mov: OPC bits 57-58 00, bit 42 0; DST 32-39, REPEAT 40-41, SRC_R 43, DST_TYPE 46-48,
//                               DST_REL 49, SRC_TYPE 50-52 (ir3-common.xml #type: 1 f32, 3 u32, 5 s32), bits 53-54
//                               the source's kind: 10 an immediate (bits 0-31), 01 a constant (bits 0-10), 00 a
//                               register (bits 0-7; bit 11 set: relative to a0.x)
//   cat2 (ir3-cat2.xml:28-66)   DST 32-39, REPEAT 40-41, SAT 42, SRC1_R 43, DST_CONV 46, SRC2_R 51, FULL 52, OPC
//                               53-58: 0 add.f, 3 mul.f, 6 absneg.f (:186-212); SRC1 bits 0-15, SRC2 16-31, each a
//                               #multisrc (ir3-common.xml:127-240): bits 11-13 000 a register (0-7), x10 a constant
//                               (0-10), others immediates and relatives; ABSNEG bits 14-15 (1 neg, 2 abs, 3 both)
//   cat3 (ir3-cat3.xml:28-134)  mad.f32 is OPC bits 55-58 0111 (:249-252) with bit 13 clear (the encoding with
//                               constants): SRC1 bits 0-12, SRC1_NEG 14, SRC2_R 15, SRC3 16-28, SRC3_R 29, SRC2_NEG
//                               30, SRC3_NEG 31, DST 32-39, REPEAT 40-41, SAT 42, SRC1_R 43, DST_CONV 46, SRC2 47-54
//                               (a register); a #cat3-src with bit 12 set is a constant (bits 0-10), with bits 11-12
//                               clear a register (bits 0-7)
//   cat6 (ir3-cat6.xml:396-444) stp, a store to private memory: OPC bits 54-58 00101, bits 0, 23 and 40 as the
//                               pattern has them. It writes no register.
// A register is its number times 4 plus its component (#reg-gpr), a constant likewise (#reg-const). An instruction
// with REPEAT n runs n + 1 times, its destination one component further each time and so each source marked (r)
// (with REPEAT 0 the (r) bits count nops instead: ir3-common.xml:27-34). On gen 6 a half register is the low half of
// the full one of the same number (Kernel::mergedRegs), so a write of either makes the other's value unknown.
class Walk {
public:
    Walk(const Program& program, const Bindings& bindings, OverlapWitness* w) : vs_(program.vs), k_(program.vs.k), b_(bindings), w_(w) {
        memset(known_, 0, sizeof known_);
        memset(rangeAt_, 0xff, sizeof rangeAt_);
    }

    // the attribute component that is unknown I, as (input, component) -- filled by run()
    struct Sym { uint8_t location, component; };
    Sym syms[kSyms];
    uint32_t numSyms = 0;
    Form position[4];

    // false: the position is not known to be an affine function of the attributes
    bool run() {
        for (uint32_t i = 0; i < vs_.numInputs; i++) {
            const StageInput& in = vs_.inputs[i];
            // (an attribute fetched into a half register is not followed: its value there is the fetch's rounding)
            if (in.sysval || in.half || in.regid == a6xx::kRegidNone) continue;
            if (in.slot < kVertAttribGeneric0 || in.slot - kVertAttribGeneric0 >= kMaxVertexAttributes) continue;
            for (uint32_t c = 0; c < 4; c++) {
                const uint32_t reg = uint32_t(in.regid) + c;
                if (!((in.compmask >> c) & 1) || reg >= kRegs || numSyms >= kSyms) continue;
                Form f{};
                f.c[numSyms] = f.m[numSyms] = 1;
                set(reg, f);
                syms[numSyms++] = Sym{uint8_t(in.slot - kVertAttribGeneric0), uint8_t(c)};
            }
        }
        const uint32_t words = k_.constantDataSize != 0 ? std::min(k_.numWords, k_.constantDataOffset / 4) : k_.numWords;
        bool ended = false;
        for (uint32_t pc = 0; pc + 1 < words && pc / 2 < kMaxInstructions && !ended; pc += 2) {
            if (!step(k_.words[pc], k_.words[pc + 1], &ended)) return false;
        }
        if (!ended || vs_.positionRegid >= kRegs - 3) return false;
        for (uint32_t c = 0; c < 4; c++) {
            if (!known_[vs_.positionRegid + c]) return false;
            position[c] = r_[vs_.positionRegid + c];
            for (uint32_t i = 0; i <= kSyms; i++) {
                if (!std::isfinite(position[c].c[i]) || !std::isfinite(position[c].m[i])) return false;
            }
        }
        return true;
    }

private:
    void set(uint32_t reg, const Form& f) {
        if (reg < kRegs) {
            r_[reg] = f;
            known_[reg] = true;
        }
    }
    void forget(uint32_t reg) {
        if (reg < kRegs) known_[reg] = false;
    }
    bool reg(uint32_t n, Form* out) const {
        if (n >= kRegs || !known_[n]) return false;
        *out = r_[n];
        return true;
    }
    // the constant file's dword N as the draw will have it: an immediate of the program's, or a pushed range's word --
    // the program's own constant data, or a bound buffer's bytes, which go into the witness whole
    bool constant(uint32_t n, Form* out) {
        uint32_t bits = 0;
        const uint32_t imm = k_.immediatesOffsetVec4 * 4;
        if (k_.immediates != nullptr && n >= imm && n - imm < k_.numImmediates) {
            bits = k_.immediates[n - imm];
        } else {
            bool found = false;
            for (uint32_t i = 0; i < k_.numUboRanges && i < kMaxUboRanges && !found; i++) {
                const KernelUboRange& ur = k_.uboRanges[i];
                if (n < ur.offsetVec4 * 4 || n - ur.offsetVec4 * 4 >= ur.sizeDwords) continue;
                const uint32_t d = n - ur.offsetVec4 * 4;
                if (k_.constantDataUbo != 0 && ur.block == k_.constantDataUbo - 1) {
                    // (render.cpp: pushedRanges -- the program's words, zeros past their end)
                    const uint64_t at = uint64_t(ur.startDword + d) * 4;
                    bits = at + 4 <= k_.constantDataSize ? k_.words[k_.constantDataOffset / 4 + ur.startDword + d] : 0;
                    found = true;
                    continue;
                }
                if (rangeAt_[i] == SIZE_MAX) {
                    if (rangeTried_[i]) return false;
                    rangeTried_[i] = true;
                    if (ur.block >= b_.buffers.size() || b_.buffers[ur.block].buffer == nullptr) return false;
                    const Binding& x = b_.buffers[ur.block];
                    const uint64_t start = x.offset + 4ull * ur.startDword, bytes = 4ull * ur.sizeDwords;
                    if (x.buffer->contents() == nullptr || x.offset % 4 != 0 || start + bytes > x.buffer->length()) return false;
                    rangeAt_[i] = w_->take(static_cast<const uint8_t*>(x.buffer->contents()) + start, size_t(bytes));
                    if (rangeAt_[i] == SIZE_MAX) return false;
                }
                memcpy(&bits, w_->bytes.data() + rangeAt_[i] + 4ull * d, 4);
                found = true;
            }
            if (!found) return false;
        }
        float v;
        memcpy(&v, &bits, 4);
        if (!std::isfinite(v)) return false;
        *out = constantForm(v);
        return true;
    }
    // a cat2 source (#multisrc), at repetition I
    bool multisrc(uint32_t s, bool r, uint32_t i, Form* out) {
        const uint32_t kind = (s >> 11) & 7, absneg = (s >> 14) & 3;
        if (kind == 0) {
            if ((s >> 8) & 3) return false;
            if (!reg((s & 0xff) + (r ? i : 0), out)) return false;
        } else if ((kind & 3) == 2) {
            if (!constant((s & 0x7ff) + (r ? i : 0), out)) return false;
        } else {
            return false;   // (an immediate, or relative to a0.x)
        }
        if (absneg & 2) {
            if (!out->constant()) return false;
            out->c[kSyms] = std::fabs(out->c[kSyms]);
        }
        if (absneg & 1) *out = negated(*out);
        return true;
    }
    // a cat3 source (#cat3-src)
    bool cat3src(uint32_t s, bool r, uint32_t i, bool neg, Form* out) {
        if (s & 0x1000) {
            if ((s & 0x800) || !constant((s & 0x7ff) + (r ? i : 0), out)) return false;
        } else if ((s & 0x800) == 0) {
            if (((s >> 8) & 3) || !reg((s & 0xff) + (r ? i : 0), out)) return false;
        } else {
            return false;   // (relative to a0.x)
        }
        if (neg) *out = negated(*out);
        return true;
    }
    // A repeated instruction whose later repetitions would read what its earlier ones wrote: not followed (whether
    // the hardware runs the repetitions one after another is not what this may assume).
    static bool feedsItself(uint32_t dst, uint32_t rpt, uint32_t src, bool r) {
        if (rpt == 0) return false;
        return r ? dst > src && dst - src <= rpt : src >= dst && src - dst < rpt;
    }

    // false: an instruction the walk cannot go past
    bool step(uint32_t lo, uint32_t hi, bool* ended) {
        const uint32_t cat = hi >> 29, dst = hi & 0xff, rpt = (hi >> 8) & 3;
        switch (cat) {
        case 0: {
            const uint32_t opc = (hi >> 23) & 0xf;
            if (((hi >> 17) & 1) || ((hi >> 5) & 7)) return false;   // (OPC_HI, or a branch's type)
            if (opc == 6) *ended = true;
            return opc == 0 || opc == 6;
        }
        case 1: {
            if (((hi >> 25) & 3) != 0 || ((hi >> 10) & 1) || ((hi >> 17) & 1)) return false;   // (not a mov, or to a register named through a0.x)
            const bool r = (hi >> 11) & 1;
            const uint32_t dstType = (hi >> 14) & 7, srcType = (hi >> 18) & 7, kind = (hi >> 21) & 3;
            const bool whole = dstType == srcType && (srcType == 1 || srcType == 3 || srcType == 5);   // (32 bits copied as they are)
            if (kind == 3 || (kind == 2 && r)) return false;
            if (kind == 0 && !(lo & 0x800) && ((lo & 0xfffff300u) != 0 || feedsItself(dst, rpt, lo & 0xff, r))) return false;
            if (kind == 1 && (lo >> 11) != 0) return false;
            for (uint32_t i = 0; i <= rpt; i++) {
                Form f;
                bool ok = whole;
                if (ok && kind == 2) {
                    float v;
                    memcpy(&v, &lo, 4);
                    ok = std::isfinite(v);
                    f = constantForm(v);
                } else if (ok && kind == 1) {
                    ok = constant((lo & 0x7ff) + (r ? i : 0), &f);
                } else if (ok) {
                    ok = !(lo & 0x800) && reg((lo & 0xff) + (r ? i : 0), &f);
                }
                if (ok) set(dst + i, f); else forget(dst + i);
            }
            return true;
        }
        case 2: {
            const uint32_t opc = (hi >> 21) & 0x3f;
            const bool sat = (hi >> 10) & 1, conv = (hi >> 14) & 1, full = (hi >> 20) & 1;
            const bool r1 = rpt != 0 && ((hi >> 11) & 1), r2 = rpt != 0 && ((hi >> 19) & 1);
            const uint32_t s1 = lo & 0xffff, s2 = lo >> 16;
            const bool followed = (opc == 0 || opc == 3 || opc == 6) && full && !conv && !sat;
            // (a register source read again after an earlier repetition wrote it)
            const bool loops = (((s1 >> 11) & 7) == 0 && feedsItself(dst, rpt, s1 & 0xff, r1)) ||
                               (opc != 6 && ((s2 >> 11) & 7) == 0 && feedsItself(dst, rpt, s2 & 0xff, r2));
            for (uint32_t i = 0; i <= rpt; i++) {
                Form a, b, f;
                bool ok = followed && !loops && multisrc(s1, r1, i, &a);
                if (ok && opc == 6) {
                    f = a;
                } else if (ok) {
                    ok = multisrc(s2, r2, i, &b);
                    if (ok && opc == 0) f = sum(a, b);
                    if (ok && opc == 3) ok = product(a, b, &f);
                }
                if (ok) set(dst + i, f); else forget(dst + i);
            }
            return true;
        }
        case 3: {
            const uint32_t opc = (hi >> 23) & 0xf, s1 = lo & 0x1fff, s3 = (lo >> 16) & 0x1fff, s2 = (hi >> 15) & 0xff;
            const bool sat = (hi >> 10) & 1, conv = (hi >> 14) & 1, alt = (lo >> 13) & 1;
            const bool r1 = rpt != 0 && ((hi >> 11) & 1), r2 = rpt != 0 && ((lo >> 15) & 1), r3 = rpt != 0 && ((lo >> 29) & 1);
            const bool followed = opc == 7 && !alt && !sat && !conv;
            const bool loops = ((s1 & 0x1800) == 0 && feedsItself(dst, rpt, s1 & 0xff, r1)) || feedsItself(dst, rpt, s2, r2) ||
                               ((s3 & 0x1800) == 0 && feedsItself(dst, rpt, s3 & 0xff, r3));
            for (uint32_t i = 0; i <= rpt; i++) {
                Form a, b, c, p, f;
                bool ok = followed && !loops && cat3src(s1, r1, i, (lo >> 14) & 1, &a) && reg(s2 + (r2 ? i : 0), &b) && cat3src(s3, r3, i, (lo >> 31) & 1, &c);
                if (ok && ((lo >> 30) & 1)) b = negated(b);
                ok = ok && product(a, b, &p);
                if (ok) {
                    f = sum(p, c);   // (two operations whether the hardware fuses them or not: the bound's count)
                    set(dst + i, f);
                } else {
                    forget(dst + i);
                }
            }
            return true;
        }
        case 6:
            // stp alone: a store to private memory, which nothing followed here reads back
            return ((hi >> 22) & 0x1f) == 5 && ((hi >> 8) & 1) && ((lo >> 23) & 1) && !(lo & 1);
        default:
            return false;
        }
    }

    const Stage& vs_;
    const Kernel& k_;
    const Bindings& b_;
    OverlapWitness* w_;
    bool known_[kRegs];
    Form r_[kRegs];
    size_t rangeAt_[kMaxUboRanges];
    bool rangeTried_[kMaxUboRanges] = {};
};

// --- the draw's vertices ----------------------------------------------------------------------------------------------------

// what an attribute's format gives its components: floats and halfs alone (their values reach the register exactly;
// a normalised or scaled integer's conversion is the hardware's, not assumed here)
bool floatFormat(VertexFormat f, uint32_t* components, bool* half) {
    switch (f) {
    case VertexFormat::kFloat: *components = 1; *half = false; return true;
    case VertexFormat::kFloat2: *components = 2; *half = false; return true;
    case VertexFormat::kFloat3: *components = 3; *half = false; return true;
    case VertexFormat::kFloat4: *components = 4; *half = false; return true;
    case VertexFormat::kHalf: *components = 1; *half = true; return true;
    case VertexFormat::kHalf2: *components = 2; *half = true; return true;
    case VertexFormat::kHalf3: *components = 3; *half = true; return true;
    case VertexFormat::kHalf4: *components = 4; *half = true; return true;
    default: return false;
    }
}

double halfValue(uint16_t h) {
    const int e = (h >> 10) & 0x1f;
    const double m = h & 0x3ff, s = (h & 0x8000) ? -1.0 : 1.0;
    if (e == 0) return s * std::ldexp(m, -24);
    if (e == 31) return m == 0 ? s * INFINITY : NAN;
    return s * std::ldexp(m + 1024.0, e - 25);
}

// one attribute component the position depends on, where the witness holds it
struct Source {
    uint32_t sym;
    size_t at;        // of vertex VMIN's attribute in the witness's bytes; SIZE_MAX: the format's default for the component
    uint32_t stride, offset;
    bool half;
    double fixed;
};

struct Geometry {
    double ox, sx, oy, sy;   // the viewport: window = o + s ndc
    double gx, gy;           // the guardband, in ndc
};

// The largest ndc extent the rasterizer leaves unclipped on an axis, from what the driver writes to
// GRAS_CL_GUARDBAND_CLIP_ADJ for it: the 3.6 floating point of a6xx.hpp's calcGuardband decoded (rounded down when it
// was made). Its MAX_GB also stands for "not computed" (a viewport outside the range, a scale of zero): then 1, the
// viewport itself, which is inside any guardband.
double guardband(double offset, double scale) {
    const uint32_t v = a6xx::calcGuardband(float(offset), float(scale));
    if (v >= a6xx::kMaxGuardband) return 1.0;
    return std::ldexp(double(64 + (v & 63)) / 128.0, int(v >> 6) + 1);
}

}  // namespace

Overlap primitivesOverlap(const Program& program, const Draw& d, const Bindings& vsBindings, OverlapWitness* w) {
    w->first = w->bytes.size();
    const size_t spans0 = w->spans.size();
    auto may = [&]() {
        w->spans.resize(spans0);
        w->bytes.resize(w->first);
        return Overlap::kMay;
    };
    if (d.indirect != nullptr || d.instanceCount != 1) return Overlap::kMay;
    const uint32_t n = d.vertexCount;
    uint32_t primitives = 0;
    switch (d.primitive) {
    case Primitive::kPoint: primitives = n; break;
    case Primitive::kLine: primitives = n / 2; break;
    case Primitive::kLineStrip: primitives = n > 1 ? n - 1 : 0; break;
    case Primitive::kTriangle: primitives = n / 3; break;
    case Primitive::kTriangleStrip: primitives = n > 2 ? n - 2 : 0; break;
    default: return Overlap::kMay;
    }
    // 1. one primitive at most (a restart index among an indexed draw's makes them fewer, never more)
    if (primitives <= 1) return Overlap::kNever;
    const bool lines = d.primitive == Primitive::kLine;
    const bool pair = primitives == 2 && (d.primitive == Primitive::kTriangle || d.primitive == Primitive::kTriangleStrip);
    if (!pair && !(lines && primitives <= kMaxLines)) return Overlap::kMay;
    if (!(d.viewport.width > 0) || !(d.viewport.height > 0)) return Overlap::kMay;

    // 2a. the position as the vertex stage computes it
    Walk walk(program, vsBindings, w);
    if (!walk.run()) return may();
    const Form* pos = walk.position;
    bool used[kSyms] = {};
    for (uint32_t s = 0; s < walk.numSyms; s++) used[s] = pos[0].m[s] != 0 || pos[1].m[s] != 0 || pos[2].m[s] != 0 || pos[3].m[s] != 0;

    // the vertices the draw names: the triangles' six at most, or the lines' as they come
    const uint32_t count = lines ? primitives * 2 : d.primitive == Primitive::kTriangle ? 6u : 4u;
    static thread_local std::vector<uint64_t> index;
    index.resize(count);
    uint64_t vmin = UINT64_MAX, vmax = 0;
    if (d.indexBuffer != nullptr) {
        const uint32_t size = d.indexType == 1 ? 4 : 2;
        const uint64_t bytes = uint64_t(count) * size;
        if (d.indexBuffer->contents() == nullptr || d.indexOffset > d.indexBuffer->length() || bytes > d.indexBuffer->length() - d.indexOffset) return may();
        const size_t at = w->take(static_cast<const uint8_t*>(d.indexBuffer->contents()) + d.indexOffset, size_t(bytes));
        if (at == SIZE_MAX) return may();
        for (uint32_t i = 0; i < count; i++) {
            uint32_t x = 0;
            memcpy(&x, w->bytes.data() + at + size_t(i) * size, size);
            if (x == (size == 4 ? 0xffffffffu : 0xffffu)) return may();   // (primitive restart: other primitives than counted)
            const int64_t v = int64_t(x) + d.baseVertex;
            if (v < 0) return may();
            index[i] = uint64_t(v);
        }
    } else {
        for (uint32_t i = 0; i < count; i++) index[i] = uint64_t(d.vertexStart) + i;
    }
    for (uint32_t i = 0; i < count; i++) {
        vmin = std::min(vmin, index[i]);
        vmax = std::max(vmax, index[i]);
    }

    // the attribute components the position depends on, each attribute's bytes for vertices VMIN to VMAX in the witness
    Source sources[kSyms];
    uint32_t numSources = 0;
    size_t attributeAt[kMaxVertexAttributes];
    bool attributeTaken[kMaxVertexAttributes] = {};
    for (uint32_t s = 0; s < walk.numSyms; s++) {
        if (!used[s]) continue;
        const uint32_t loc = walk.syms[s].location, comp = walk.syms[s].component;
        const VertexAttribute& a = program.vertexDescriptor.attributes[loc];
        uint32_t components = 0;
        bool half = false;
        if (!floatFormat(a.format, &components, &half) || a.bufferIndex >= kMaxVertexBuffers) return may();
        const VertexLayout& l = program.vertexDescriptor.layouts[a.bufferIndex];
        if (l.step != VertexStep::kPerVertex || l.stride == 0) return may();
        Source src{s, SIZE_MAX, l.stride, comp * (half ? 2u : 4u), half, comp == 3 ? 1.0 : 0.0};
        if (comp < components) {
            if (!attributeTaken[loc]) {
                if (a.bufferIndex >= vsBindings.buffers.size() || vsBindings.buffers[a.bufferIndex].buffer == nullptr) return may();
                const Binding& x = vsBindings.buffers[a.bufferIndex];
                const uint64_t size = uint64_t(components) * (half ? 2 : 4);
                const uint64_t start = x.offset + vmin * l.stride + a.offset, bytes = (vmax - vmin) * l.stride + size;
                if (x.buffer->contents() == nullptr || vmax > UINT32_MAX || start + bytes > x.buffer->length()) return may();
                attributeAt[loc] = w->take(static_cast<const uint8_t*>(x.buffer->contents()) + start, size_t(bytes));
                if (attributeAt[loc] == SIZE_MAX) return may();
                attributeTaken[loc] = true;
            }
            src.at = attributeAt[loc];
        }
        sources[numSources++] = src;
    }

    // each vertex's clip x, y, z, w from the witness's bytes, and the largest magnitude of each attribute component
    static thread_local std::vector<double> clip, values;
    clip.resize(size_t(count) * 4);
    values.resize(size_t(count) * std::max(numSources, 1u));
    double largest[kSyms] = {};
    const uint8_t* bytes = w->bytes.data();
    for (uint32_t i = 0; i < count; i++) {
        double p[4] = {pos[0].c[kSyms], pos[1].c[kSyms], pos[2].c[kSyms], pos[3].c[kSyms]};
        for (uint32_t j = 0; j < numSources; j++) {
            const Source& s = sources[j];
            double v = s.fixed;
            if (s.at != SIZE_MAX) {
                const uint8_t* at = bytes + s.at + size_t(index[i] - vmin) * s.stride + s.offset;
                if (s.half) {
                    uint16_t h;
                    memcpy(&h, at, 2);
                    v = halfValue(h);
                } else {
                    float f;
                    memcpy(&f, at, 4);
                    v = double(f);
                }
            }
            if (!std::isfinite(v)) return may();
            values[size_t(i) * numSources + j] = v;
            largest[s.sym] = std::max(largest[s.sym], std::fabs(v));
            for (uint32_t c = 0; c < 4; c++) p[c] += pos[c].c[s.sym] * v;
        }
        memcpy(&clip[size_t(i) * 4], p, sizeof p);
    }
    // The GPU's float32 against the exact value: gamma_n = n u / (1 - n u) times the terms' magnitudes (Higham,
    // Accuracy and Stability of Numerical Algorithms, 2nd ed., section 3.1), with u one ulp's relative size, 2^-23 --
    // twice a correctly rounded operation's 2^-24, for a GPU whose operations are merely faithful.
    double err[4];
    for (uint32_t c = 0; c < 4; c++) {
        double magnitude = pos[c].m[kSyms];
        for (uint32_t s = 0; s < walk.numSyms; s++) magnitude += pos[c].m[s] * largest[s];
        const double nu = double(pos[c].n) * std::ldexp(1.0, -23);
        if (!(nu < 0.01)) return may();
        err[c] = nu / (1 - nu) * magnitude + std::ldexp(1.0, -120);
        if (!std::isfinite(err[c])) return may();
    }
    // window coordinates, and how far the GPU's may lie from them: the position's error through the division, the
    // viewport transform's own rounding (four operations on numbers no larger than the offset plus the distance from
    // it), and the snap to the subpixel grid
    const Viewport& vp = d.viewport;
    Geometry g{vp.x + vp.width / 2, vp.width / 2, vp.y + vp.height / 2, -vp.height / 2, 0, 0};
    static thread_local std::vector<double> win;
    win.resize(size_t(count) * 2);
    double wLow = INFINITY, farX = 0, farY = 0;
    for (uint32_t i = 0; i < count; i++) {
        const double* p = &clip[size_t(i) * 4];
        const double wl = p[3] - err[3];
        if (!(wl > 0) || !std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[3])) return may();
        wLow = std::min(wLow, wl);
        const double dx = g.sx * p[0] / p[3], dy = g.sy * p[1] / p[3];
        win[size_t(i) * 2] = g.ox + dx;
        win[size_t(i) * 2 + 1] = g.oy + dy;
        farX = std::max(farX, std::fabs(dx));
        farY = std::max(farY, std::fabs(dy));
    }
    const double u4 = std::ldexp(4.0, -23);
    const double ex = (std::fabs(g.sx) * err[0] + farX * err[3]) / wLow + u4 * (std::fabs(g.ox) + farX);
    const double ey = (std::fabs(g.sy) * err[1] + farY * err[3]) / wLow + u4 * (std::fabs(g.oy) + farY);
    const double delta = kSubpixelSnap + std::hypot(ex, ey);
    if (!std::isfinite(delta)) return may();

    if (lines) {
        // 3. lines no two of which can reach the same pixel. A line of half width H covers nothing outside its
        // segment's bounding box grown by sqrt(2) H: turnip reports strictLines (tu_device.cc:1080), so a line is the
        // rectangle of that half width along the segment, its ends square across it (Vulkan 1.3, "Basic Line Segment
        // Rasterization"), and a diamond-exit line stays inside the same box -- which also takes in a cap of H past
        // each end, should the hardware draw one. H is what the driver programs: GRAS_SU_CNTL.LINEHALFWIDTH, half a
        // pixel (a6xx.hpp: grasSuCntl's two quarters). A clipped line is a part of its segment. So each line gets the
        // pixels whose centres lie in that box grown by DELTA; lines whose pixel boxes are apart cover no pixel twice.
        // Taken: boxes of one pixel row each (CoreAnimation's coverage spans, horizontal at pixel centres), no two
        // sharing a pixel -- runs on a row with a clear pixel between them. Anything else keeps the ordering.
        const double grow = std::sqrt(2.0) * kLineHalfWidth + delta;
        static thread_local std::vector<uint64_t> boxes;
        boxes.resize(primitives);
        bool sorted = true;
        for (uint32_t i = 0; i < primitives; i++) {
            const double* a = &win[size_t(i) * 4];
            // (a pixel centre C is in [lo, hi] when ceil(lo - 0.5) <= C - 0.5 <= floor(hi - 0.5))
            const double x0 = std::ceil(std::min(a[0], a[2]) - grow - 0.5), x1 = std::floor(std::max(a[0], a[2]) + grow - 0.5);
            const double y0 = std::ceil(std::min(a[1], a[3]) - grow - 0.5), y1 = std::floor(std::max(a[1], a[3]) + grow - 0.5);
            if (y0 != y1 || !(x0 > -32768.0) || !(x1 < 32768.0) || !(y0 > -32768.0) || !(y0 < 32768.0)) return may();
            if (x1 < x0) return may();   // (not reached: the box is more than a pixel wide)
            // row, then first column, then last: 16 bits each, offset to be positive
            boxes[i] = uint64_t(uint32_t(y0 + 32768.0)) << 32 | uint64_t(uint32_t(x0 + 32768.0)) << 16 | uint64_t(uint32_t(x1 + 32768.0));
            if (i != 0 && boxes[i] < boxes[i - 1]) sorted = false;
        }
        if (!sorted) std::sort(boxes.begin(), boxes.end());
        for (uint32_t i = 1; i < primitives; i++) {
            const uint64_t p = boxes[i - 1], q = boxes[i];
            if ((p >> 32) == (q >> 32) && ((q >> 16) & 0xffff) <= (p & 0xffff)) return may();
        }
        return Overlap::kNever;
    }

    // 2b. the two triangles' shared edge. Vertices are the same when they are one index or every attribute component
    // the position depends on has the same value in both.
    const uint32_t t[2][3] = {{0, 1, 2}, {d.primitive == Primitive::kTriangle ? 3u : 1u, d.primitive == Primitive::kTriangle ? 4u : 2u,
                                          d.primitive == Primitive::kTriangle ? 5u : 3u}};
    auto same = [&](uint32_t a, uint32_t b) {
        if (index[a] == index[b]) return true;
        for (uint32_t j = 0; j < numSources; j++) {
            if (values[size_t(a) * numSources + j] != values[size_t(b) * numSources + j]) return false;
        }
        return true;
    };
    int match[3] = {-1, -1, -1};
    bool taken[3] = {};
    uint32_t matched = 0;
    for (uint32_t i = 0; i < 3; i++) {
        for (uint32_t j = 0; j < 3 && match[i] < 0; j++) {
            if (!taken[j] && same(t[0][i], t[1][j])) {
                match[i] = int(j);
                taken[j] = true;
                matched++;
            }
        }
    }
    if (matched != 2) return may();
    uint32_t A = 0, C = 0, B = 0, D = 0;
    {
        uint32_t shared[2], k = 0;
        for (uint32_t i = 0; i < 3; i++) {
            if (match[i] >= 0) shared[k++] = t[0][i]; else B = t[0][i];
        }
        A = shared[0];
        C = shared[1];
        for (uint32_t j = 0; j < 3; j++) {
            if (!taken[j]) D = t[1][j];
        }
    }
    // 2c. no clipping of any of the four: inside the guardband, in front of the near plane and before the far one
    g.gx = guardband(g.ox, g.sx);
    g.gy = guardband(g.oy, g.sy);
    // (a component is the same for all four when every attribute component it depends on is: the GPU then computes
    // it from the same inputs four times)
    auto sameForAll = [&](const Form& f) {
        for (uint32_t j = 0; j < numSources; j++) {
            if (f.m[sources[j].sym] == 0) continue;
            const double v = values[size_t(A) * numSources + j];
            for (uint32_t x : {C, B, D}) {
                if (values[size_t(x) * numSources + j] != v) return false;
            }
        }
        return true;
    };
    const bool zSame = sameForAll(pos[2]), wSame = sameForAll(pos[3]);
    for (uint32_t v : {A, C, B, D}) {
        const double* p = &clip[size_t(v) * 4];
        const double wl = p[3] - err[3];
        if (!(std::fabs(p[0]) + err[0] <= g.gx * wl) || !(std::fabs(p[1]) + err[1] <= g.gy * wl)) return may();
        if (!std::isfinite(p[2])) return may();
        if (!zSame && !(p[2] - err[2] > 0)) return may();
        if (!(zSame && wSame) && !(p[3] - p[2] - err[3] - err[2] > 0)) return may();
    }
    // 2d. B and D strictly on opposite sides of the line AC, by more than every vertex moving DELTA can undo
    auto at = [&](uint32_t v, uint32_t c) { return win[size_t(v) * 2 + c]; };
    auto cross = [&](uint32_t p) { return (at(C, 0) - at(A, 0)) * (at(p, 1) - at(A, 1)) - (at(C, 1) - at(A, 1)) * (at(p, 0) - at(A, 0)); };
    auto length = [&](uint32_t p, uint32_t q) { return std::hypot(at(p, 0) - at(q, 0), at(p, 1) - at(q, 1)); };
    const double cb = cross(B), cd = cross(D);
    const double marginB = 2 * delta * (length(C, A) + length(B, A)) + 4 * delta * delta;
    const double marginD = 2 * delta * (length(C, A) + length(D, A)) + 4 * delta * delta;
    if (!(std::fabs(cb) > marginB) || !(std::fabs(cd) > marginD) || (cb > 0) == (cd > 0)) return may();
    return Overlap::kNever;
}

}  // namespace hakios::metal::render
