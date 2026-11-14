// warmup: the driver's warm-up (compiler.hpp: warmUp) by itself -- every recipe of the cache directory and its seeds
// compiled for this build where it is not filed yet, and what it came to. On the host for measuring it (with the
// compiler: tools/compiler.mk, or rebuild/cpu-tools' native Mesa build); on the phone it is what the plug-in runs as
// the device is made.
//
//   warmup [--cache DIR] [--seed DIR[:DIR...]] [--runs N]
//
// --cache: the writable cache (HAKIOS_METAL_CACHE), where recipes are read and what is compiled is filed; --seed: the
// read-only ones (HAKIOS_METAL_CACHE_SEED). --runs N: the walk N times (the second and later find everything filed).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "hakios-metal/compiler.hpp"

using namespace hakios::metal;

int main(int argc, char** argv) {
    int runs = 1;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--cache" && i + 1 < argc) setenv("HAKIOS_METAL_CACHE", argv[++i], 1);
        else if (a == "--seed" && i + 1 < argc) setenv("HAKIOS_METAL_CACHE_SEED", argv[++i], 1);
        else if (a == "--runs" && i + 1 < argc) runs = atoi(argv[++i]) > 0 ? atoi(argv[i]) : 1;
        else {
            std::printf("warmup [--cache DIR] [--seed DIR[:DIR...]] [--runs N]\n");
            return 2;
        }
    }
    if (!hasCompiler()) {
        std::printf("warmup: no compiler in this build\n");
        return 1;
    }
    int bad = 0;
    for (int run = 0; run < runs; run++) {
        const WarmUpReport r = warmUp();
        std::printf("warm-up %d: %u recipes, %u filed already, %u compiled in %.1f ms, %u failed; %.1f ms in all%s%s\n", run + 1, r.recipes, r.cached,
                    r.compiled, r.compileMilliseconds, r.failed, r.milliseconds, r.firstFailure.empty() ? "" : "; first failure: ",
                    r.firstFailure.c_str());
        bad += r.failed != 0;
    }
    return bad != 0 ? 1 : 0;
}
