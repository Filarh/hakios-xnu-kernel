// The compiled-shader cache's files, apart from the compiler: what a stage is as bytes, and how a file of the cache is
// written and read so that every process of the phone can share one directory.
//
// The cache (compiler.cpp: compileStage, compileAir, warmUp) files each stage under a hash of what made it. The
// switch HAKIOS_METAL_CACHE_V2=1 (the plug-in's metal_shared_cache=1) gives it the layout and the guarantees a
// directory shared by every GPU client needs, where until now one process (backboardd) wrote it:
//
//   <root>/v2/<generation>/<hh>/<hash>.stage     (hh: the hash's first two digits; .kernel and .known likewise)
//
//  - GENERATION: the compiler's identity (HAKIOS_METAL_COMPILER_ID, which the Makefile derives from the compiler's
//    sources and Mesa's archives; else the cache key the build was given). A new compiler is a new directory: nothing
//    old is ever loaded by it, and the old directory is the first thing trimCache removes.
//  - SEALED FILES: a header (magic HMC2, the payload's length and a 128-bit hash of it, the generation) before the
//    payload, checked on every read: a file cut short or damaged is not loaded (and is removed), it is compiled again.
//  - ATOMIC, PER WRITER: written under a temporary name of the writer's own (pid, thread, a counter) and renamed into
//    place -- two processes compiling the same stage at once each rename a whole file, never one interleaved with the
//    other's (the old ".tmp" name was the same for every process). A file already there is left alone: same key, same
//    bytes.
//  - BOUNDED: trimCache keeps the writable root under a size (HAKIOS_METAL_CACHE_MAX_MB), removing other generations
//    first and then the least recently used files of this one (a hit touches its file's time, once a process a day).
//
// Without the switch the files are as before (no header, one directory), except that the temporary names are the
// writer's own in both layouts.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hakios-metal/compiler.hpp"

namespace hakios::metal {

// a stage as the cache's .stage payload (the Kernel's pointers and inline samplers' ids written as zeros: the same
// compilation is the same bytes in any process), and back -- nullptr for bytes that do not read as one
std::string stageBytes(const CompiledStage& s);
std::unique_ptr<CompiledStage> stageFromBytes(const std::string& bytes);

// HAKIOS_METAL_CACHE_V2 set and not "0"
bool sharedCacheLayout();
// the compiler's identity (see above), at most 24 characters of [0-9a-z]
std::string cacheGeneration();
// where a file of hash HEX (32 hex digits) and extension EXT (".stage") lies under a root: "/<hash>.stage" in the old
// layout, "/v2/<generation>/<hh>/<hash>.stage" in the shared one
std::string cacheRelativePath(const std::string& hex, const char* ext);

// write PAYLOAD at PATH as a whole file (directories made as needed): sealed when SEALED. False when it could not be
// written; true, and nothing written, when the file is already there.
bool cacheWriteFile(const std::string& path, const std::string& payload, bool sealed);
// read PATH: a sealed file checked and unwrapped (a bad one removed, false), an unsealed one as it is
bool cacheReadFile(const std::string& path, std::string* payload);
// one record more at the end of PATH (directories made as needed), in one write(2) with O_APPEND: two processes'
// records do not mix
bool cacheAppendRecord(const std::string& path, const std::string& record);
// a hit: its file's time set to now, at most once a day per process and file (what trimCache's order is by)
void cacheTouch(const std::string& path);

struct CacheTrimReport {
    uint64_t bytesBefore = 0, bytesAfter = 0;
    uint32_t files = 0, removed = 0, generationsRemoved = 0;
};
// keep ROOT's v2 tree under MAX_BYTES: other generations than GENERATION go first (whole), then this generation's
// least recently used files until it is under 3/4 of MAX_BYTES; recipes/ is not touched (build-independent, small)
CacheTrimReport trimCache(const std::string& root, const std::string& generation, uint64_t maxBytes);

// The cache's directories as the environment gives them: HAKIOS_METAL_CACHE (written) and HAKIOS_METAL_CACHE_SEED
// (read only, ':' between them), each "" / empty when not given
std::string cacheWritableRoot();
std::vector<std::string> cacheSeedRoots();

}  // namespace hakios::metal
