// stagecc: one function of a .metallib compiled to a stage as the driver compiles it (compileStage), on the host --
// what it comes to and how long it takes. For a build with the compiler (MESA=1: tools/compiler.mk).
//
//   stagecc LIBRARY.metallib FUNCTION [--zeros] [--constants "4=1 6=2 30=54"] [--known BINDING HEX] [--runs N] [--list]
//
// --zeros:     every constant (0 to 63) supplied as 0, as a caller that sets them all does; --constants after it.
// --constants: the function's constants as the plug-in's lines print them (index=value; the rest are not supplied).
// --known:     the stage compiled for these bytes of its constant buffer at [[buffer(BINDING)]] (compileStage's
//              KnownBuffer), as the plug-in specialises CoreAnimation's generic shader; the bytes read are printed.
// --runs N:    compiled N times, each timed (with HAKIOS_METAL_CACHE unset nothing is read from a cache).
// --list:      the library's functions.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "hakios-metal/compiler.hpp"
#include "hakios-metal/metallib.hpp"

using namespace hakios::metal;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("stagecc LIBRARY.metallib FUNCTION [--constants \"I=V ...\"] [--known BINDING HEX] [--runs N] [--list]\n");
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    MetalLibrary lib;
    if (bytes.empty() || !lib.parse(bytes.data(), bytes.size())) {
        std::printf("stagecc: %s: %s\n", argv[1], bytes.empty() ? "cannot be read" : lib.why().c_str());
        return 1;
    }
    FunctionConstants constants;
    KnownBuffer known;
    bool hasKnown = false, list = false;
    int runs = 1;
    for (int i = 3; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--list") list = true;
        else if (a == "--zeros") {
            // (every constant supplied, as CoreAnimation supplies its functions': 0 unless --constants, after this, says otherwise)
            for (uint32_t k = 0; k < 64; k++) {
                FunctionConstant c{};
                c.index = k;
                constants.push_back(c);
            }
        }
        else if (a == "--runs" && i + 1 < argc) runs = std::max(1, atoi(argv[++i]));
        else if (a == "--constants" && i + 1 < argc) {
            for (const char* p = argv[++i]; *p != 0;) {
                char* end = nullptr;
                const unsigned long index = strtoul(p, &end, 10);
                if (end == p || *end != '=') break;
                const unsigned long long value = strtoull(end + 1, &end, 10);
                bool had = false;
                for (FunctionConstant& c : constants) {
                    if (c.index == uint32_t(index)) c.bits = value, had = true;
                }
                if (!had) {
                    FunctionConstant c{};
                    c.index = uint32_t(index);
                    c.bits = value;
                    constants.push_back(c);
                }
                p = end;
                while (*p == ' ') p++;
            }
        } else if (a == "--known" && i + 2 < argc) {
            hasKnown = true;
            known.binding = uint32_t(atoi(argv[++i]));
            const char* hex = argv[++i];
            for (size_t k = 0; hex[k] != 0 && hex[k + 1] != 0; k += 2) {
                const char two[3] = {hex[k], hex[k + 1], 0};
                known.bytes.push_back(uint8_t(strtoul(two, nullptr, 16)));
            }
        }
    }
    if (list || std::string(argv[2]) == "--list") {
        for (const MetalFunction& f : lib.functions()) std::printf("%s\n", f.name.c_str());
        return 0;
    }
    const MetalFunction* f = lib.find(argv[2]);
    const uint8_t* bitcode = nullptr;
    uint64_t length = 0;
    if (f == nullptr || !lib.bitcode(*f, &bitcode, &length)) {
        std::printf("stagecc: no function %s in %s\n", argv[2], argv[1]);
        return 1;
    }
    for (int run = 0; run < runs; run++) {
        std::string why;
        const auto t0 = std::chrono::steady_clock::now();
        std::unique_ptr<CompiledStage> s = compileStage(bitcode, size_t(length), &why, constants, 0, hasKnown ? &known : nullptr);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const CompileTimes t = takeCompileTimes();
        if (s == nullptr) {
            std::printf("%s: refused in %.1f ms: %s\n", argv[2], ms, why.c_str());
            return 1;
        }
        std::printf("%s: %llu bytes of bitcode, %zu constants supplied -> %u instruction words, constlen %u, %zu resources, in %.1f ms (%s%.1f ms compiling: "
                    "%.1f the front end, %.1f folding the known bytes, %.1f the rest; %.1f ms reading the cache)",
                    argv[2], (unsigned long long)length, constants.size(), s->stage.k.numWords, s->stage.k.constlen, s->resources.size(), ms,
                    t.cached ? "from the cache; " : "", t.compiledUs / 1e3, t.frontEndUs / 1e3, t.foldedUs / 1e3,
                    (t.compiledUs - t.frontEndUs - t.foldedUs) / 1e3, t.loadedUs / 1e3);
        if (hasKnown) {
            std::printf("; of the %zu known bytes it read", known.bytes.size());
            for (size_t i = 0; i < s->knownReads.size(); i++) if (s->knownReads[i]) std::printf(" %zu", i);
        }
        std::printf("\n");
    }
    return 0;
}
