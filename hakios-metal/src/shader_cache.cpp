#include "hakios-metal/shader_cache.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <set>
#include <thread>

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

namespace hakios::metal {

namespace {

struct Writer {
    std::string out;
    void raw(const void* p, size_t n) { out.append(static_cast<const char*>(p), n); }
    void u32(uint32_t v) { raw(&v, 4); }
    void str(const std::string& v) { u32(uint32_t(v.size())); raw(v.data(), v.size()); }
};
struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;
    bool raw(void* dst, size_t n) {
        if (!ok || size_t(end - p) < n) return ok = false;
        memcpy(dst, p, n);
        p += n;
        return true;
    }
    uint32_t u32() { uint32_t v = 0; raw(&v, 4); return v; }
    std::string str() {
        uint32_t n = u32();
        if (!ok || size_t(end - p) < n) { ok = false; return {}; }
        std::string v(reinterpret_cast<const char*>(p), n);
        p += n;
        return v;
    }
};

constexpr uint32_t kStageMagic = 0x53434d48;    // "HMCS": the payload, as the old layout's files are whole
constexpr uint32_t kSealMagic = 0x32434d48;     // "HMC2": a sealed file's header
constexpr uint32_t kSealVersion = 1;

struct Hash {
    uint64_t a = 0xcbf29ce484222325ull, b = 0x84222325cbf29ce4ull;
    void mix(const void* p, size_t n) {
        for (size_t i = 0; i < n; i++) {
            const uint8_t c = static_cast<const uint8_t*>(p)[i];
            a = (a ^ c) * 0x100000001b3ull;
            b = (b ^ c) * 0x1000193ull + (b >> 29);
        }
    }
};

#if defined(HAKIOS_METAL_COMPILER_ID)
const char* const kGeneration = HAKIOS_METAL_COMPILER_ID;
#elif defined(HAKIOS_METAL_CACHE_BUILD)
const char* const kGeneration = HAKIOS_METAL_CACHE_BUILD;
#else
const char* const kGeneration = "dev";
#endif

bool makeParents(const std::string& path) {
    for (size_t at = 1; (at = path.find('/', at)) != std::string::npos; at++) {
        const std::string d = path.substr(0, at);
        if (mkdir(d.c_str(), 0755) != 0 && errno != EEXIST) return false;
    }
    return true;
}

std::string uniqueTemporary(const std::string& path) {
    static std::atomic<uint32_t> counter{0};
    char tail[96];
    snprintf(tail, sizeof(tail), ".tmp.%d.%zx.%u", int(getpid()), std::hash<std::thread::id>()(std::this_thread::get_id()) & 0xffffff,
             counter.fetch_add(1));
    return path + tail;
}

}  // namespace

std::string stageBytes(const CompiledStage& s) {
    Writer w;
    w.u32(kStageMagic);
    Stage stage = s.stage;
    stage.k.words = nullptr;
    stage.k.immediates = nullptr;
    stage.k.name = nullptr;
    stage.k.source = nullptr;
    w.raw(&stage, sizeof(stage));
    w.u32(uint32_t(s.words.size()));
    w.raw(s.words.data(), s.words.size() * 4);
    w.u32(uint32_t(s.immediates.size()));
    w.raw(s.immediates.data(), s.immediates.size() * 4);
    w.u32(uint32_t(s.resources.size()));
    for (const Resource& r : s.resources) {
        w.u32(uint32_t(r.kind));
        w.u32(r.index);
        w.u32(r.sizeBytes);
        const uint8_t flags[8] = {r.read, r.write, r.irWrite, r.textureDim, r.textureArray, r.element, r.nullTexture, r.inlineSampler};
        w.raw(flags, sizeof(flags));
        w.str(r.typeName);
        w.u32(r.argbufSet);
        w.u32(r.argbufOffset);
    }
    w.u32(uint32_t(s.varyingNames.size()));
    for (const std::string& v : s.varyingNames) w.str(v);
    w.u32(uint32_t(s.inlineSamplers.size()));
    for (const auto& is : s.inlineSamplers) {
        // (the sampler's id is makeSampler's count in the process that compiled it: nothing binds an inline sampler by
        // it, and written as it was it made the same compilation two different files)
        Sampler sampler = is.second;
        sampler.id = 0;
        w.u32(is.first);
        w.raw(&sampler, sizeof(sampler));
    }
    w.u32(s.iboBase);
    w.u32(s.writesMemory ? 1 : 0);
    w.str(s.name);
    w.str(s.listing);
    return w.out;
}

std::unique_ptr<CompiledStage> stageFromBytes(const std::string& data) {
    Reader r{reinterpret_cast<const uint8_t*>(data.data()), reinterpret_cast<const uint8_t*>(data.data()) + data.size()};
    auto s = std::make_unique<CompiledStage>();
    if (r.u32() != kStageMagic || !r.raw(&s->stage, sizeof(s->stage))) return nullptr;
    const uint32_t words = r.u32();
    if (!r.ok || words > (1u << 24)) return nullptr;
    s->words.resize(words);
    r.raw(s->words.data(), s->words.size() * 4);
    const uint32_t immediates = r.u32();
    if (!r.ok || immediates > (1u << 20)) return nullptr;
    s->immediates.resize(immediates);
    r.raw(s->immediates.data(), s->immediates.size() * 4);
    const uint32_t resources = r.u32();
    if (!r.ok || resources > 4096) return nullptr;
    s->resources.resize(resources);
    for (Resource& res : s->resources) {
        res.kind = Resource::Kind(r.u32());
        res.index = r.u32();
        res.sizeBytes = r.u32();
        uint8_t flags[8] = {};
        r.raw(flags, sizeof(flags));
        res.read = flags[0], res.write = flags[1], res.irWrite = flags[2], res.textureDim = flags[3];
        res.textureArray = flags[4], res.element = flags[5], res.nullTexture = flags[6], res.inlineSampler = flags[7];
        res.typeName = r.str();
        res.argbufSet = r.u32();
        res.argbufOffset = r.u32();
    }
    const uint32_t varyings = r.u32();
    if (!r.ok || varyings > 4096) return nullptr;
    s->varyingNames.resize(varyings);
    for (std::string& v : s->varyingNames) v = r.str();
    const uint32_t samplers = r.u32();
    if (!r.ok || samplers > 4096) return nullptr;
    s->inlineSamplers.resize(samplers);
    for (auto& is : s->inlineSamplers) {
        is.first = r.u32();
        r.raw(&is.second, sizeof(is.second));
    }
    s->iboBase = r.u32();
    s->writesMemory = r.u32() != 0;
    s->name = r.str();
    s->listing = r.str();
    if (!r.ok || r.p != r.end) return nullptr;
    s->stage.k.words = s->words.data();
    s->stage.k.numWords = uint32_t(s->words.size());
    s->stage.k.immediates = s->immediates.empty() ? nullptr : s->immediates.data();
    s->stage.k.numImmediates = uint32_t(s->immediates.size());
    s->stage.k.name = s->name.c_str();
    s->stage.k.source = "";
    return s;
}

bool sharedCacheLayout() {
    const char* e = getenv("HAKIOS_METAL_CACHE_V2");
    return e != nullptr && e[0] != 0 && e[0] != '0';
}

std::string cacheGeneration() {
    std::string g;
    for (const char* p = kGeneration; *p != 0 && g.size() < 24; p++) {
        const char c = *p;
        if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z')) g.push_back(c);
        else if (c >= 'A' && c <= 'Z') g.push_back(char(c - 'A' + 'a'));
    }
    return g.empty() ? std::string("dev") : g;
}

std::string cacheRelativePath(const std::string& hex, const char* ext) {
    if (!sharedCacheLayout()) return "/" + hex + ext;
    return "/v2/" + cacheGeneration() + "/" + hex.substr(0, 2) + "/" + hex + ext;
}

bool cacheWriteFile(const std::string& path, const std::string& payload, bool sealed) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) return true;   // (same key, same bytes: whoever wrote it first wrote these)
    std::string bytes;
    if (sealed) {
        Writer w;
        Hash h;
        h.mix(payload.data(), payload.size());
        const uint64_t length = payload.size();
        char generation[24] = {};
        const std::string g = cacheGeneration();
        memcpy(generation, g.data(), std::min(g.size(), sizeof(generation)));
        w.u32(kSealMagic);
        w.u32(kSealVersion);
        w.raw(&length, 8);
        w.raw(&h.a, 8);
        w.raw(&h.b, 8);
        w.raw(generation, sizeof(generation));
        bytes = std::move(w.out);
        bytes += payload;
    }
    const std::string& out = sealed ? bytes : payload;
    if (!makeParents(path)) return false;
    const std::string tmp = uniqueTemporary(path);
    FILE* f = fopen(tmp.c_str(), "wb");
    if (f == nullptr) return false;
    const bool written = fwrite(out.data(), 1, out.size(), f) == out.size();
    const bool closed = fclose(f) == 0;
    if (!written || !closed || rename(tmp.c_str(), path.c_str()) != 0) {
        remove(tmp.c_str());
        return false;
    }
    return true;
}

bool cacheReadFile(const std::string& path, std::string* payload) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    std::string data;
    char chunk[65536];
    for (size_t n; (n = fread(chunk, 1, sizeof(chunk), f)) > 0;) data.append(chunk, n);
    fclose(f);
    uint32_t magic = 0;
    if (data.size() >= 4) memcpy(&magic, data.data(), 4);
    if (magic != kSealMagic) {
        *payload = std::move(data);
        return true;
    }
    constexpr size_t kHeader = 4 + 4 + 8 + 8 + 8 + 24;
    Reader r{reinterpret_cast<const uint8_t*>(data.data()), reinterpret_cast<const uint8_t*>(data.data()) + data.size()};
    r.u32();
    const uint32_t version = r.u32();
    uint64_t length = 0;
    Hash want;
    r.raw(&length, 8);
    r.raw(&want.a, 8);
    r.raw(&want.b, 8);
    Hash got;
    if (r.ok && version == kSealVersion && data.size() >= kHeader && data.size() - kHeader == length) got.mix(data.data() + kHeader, length);
    if (!r.ok || version != kSealVersion || data.size() < kHeader || data.size() - kHeader != length || got.a != want.a || got.b != want.b) {
        remove(path.c_str());   // (a damaged file of a content-addressed cache is only a miss to come; a seed's stays: not ours to write)
        return false;
    }
    payload->assign(data.data() + kHeader, length);
    return true;
}

bool cacheAppendRecord(const std::string& path, const std::string& record) {
    if (!makeParents(path)) return false;
    const int fd = open(path.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);
    if (fd < 0) return false;
    const bool written = write(fd, record.data(), record.size()) == ssize_t(record.size());
    close(fd);
    return written;
}

void cacheTouch(const std::string& path) {
    static std::mutex lock;
    static std::set<std::string> touched;
    {
        std::lock_guard<std::mutex> g(lock);
        if (!touched.insert(path).second) return;
    }
    struct stat st;
    if (stat(path.c_str(), &st) != 0 || time(nullptr) - st.st_mtime < 86400) return;
    utimes(path.c_str(), nullptr);
}

namespace {
struct FileAt {
    std::string path;
    uint64_t bytes;
    time_t mtime;
};
void walk(const std::string& dir, std::vector<FileAt>* out) {
    DIR* d = opendir(dir.c_str());
    if (d == nullptr) return;
    while (struct dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        if (n == "." || n == "..") continue;
        const std::string p = dir + "/" + n;
        struct stat st;
        if (lstat(p.c_str(), &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) walk(p, out);
        else if (S_ISREG(st.st_mode)) out->push_back(FileAt{p, uint64_t(st.st_size), st.st_mtime});
    }
    closedir(d);
}
void removeTree(const std::string& dir) {
    DIR* d = opendir(dir.c_str());
    if (d == nullptr) return;
    while (struct dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        if (n == "." || n == "..") continue;
        const std::string p = dir + "/" + n;
        struct stat st;
        if (lstat(p.c_str(), &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) removeTree(p);
        else remove(p.c_str());
    }
    closedir(d);
    rmdir(dir.c_str());
}
}  // namespace

CacheTrimReport trimCache(const std::string& root, const std::string& generation, uint64_t maxBytes) {
    CacheTrimReport report;
    if (root.empty()) return report;
    const std::string v2 = root + "/v2";
    std::vector<std::pair<std::string, std::vector<FileAt>>> generations;
    DIR* d = opendir(v2.c_str());
    if (d == nullptr) return report;
    while (struct dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        if (n == "." || n == "..") continue;
        std::vector<FileAt> files;
        walk(v2 + "/" + n, &files);
        generations.emplace_back(n, std::move(files));
    }
    closedir(d);
    for (const auto& g : generations) {
        for (const FileAt& f : g.second) report.bytesBefore += f.bytes, report.files++;
    }
    uint64_t total = report.bytesBefore;
    // other generations first, the least recently used of them first: a compiler that is not this one hits nothing
    std::sort(generations.begin(), generations.end(), [](const auto& a, const auto& b) {
        time_t ta = 0, tb = 0;
        for (const FileAt& f : a.second) ta = std::max(ta, f.mtime);
        for (const FileAt& f : b.second) tb = std::max(tb, f.mtime);
        return ta < tb;
    });
    for (const auto& g : generations) {
        if (total <= maxBytes) break;
        if (g.first == generation) continue;
        uint64_t bytes = 0;
        for (const FileAt& f : g.second) bytes += f.bytes;
        removeTree(v2 + "/" + g.first);
        total -= bytes;
        report.removed += uint32_t(g.second.size());
        report.generationsRemoved++;
    }
    if (total > maxBytes) {
        std::vector<FileAt> mine;
        for (const auto& g : generations) {
            if (g.first == generation) mine = g.second;
        }
        std::sort(mine.begin(), mine.end(), [](const FileAt& a, const FileAt& b) { return a.mtime != b.mtime ? a.mtime < b.mtime : a.path < b.path; });
        const uint64_t goal = maxBytes / 4 * 3;
        for (const FileAt& f : mine) {
            if (total <= goal) break;
            // (an index -- .known -- names stages: removing a stage under it leaves a record that loads nothing, a miss)
            if (remove(f.path.c_str()) == 0) {
                total -= f.bytes;
                report.removed++;
            }
        }
    }
    report.bytesAfter = total;
    return report;
}

std::string cacheWritableRoot() {
    const char* dir = getenv("HAKIOS_METAL_CACHE");
    return dir != nullptr ? std::string(dir) : std::string();
}

std::vector<std::string> cacheSeedRoots() {
    std::vector<std::string> dirs;
    const char* seed = getenv("HAKIOS_METAL_CACHE_SEED");
    for (const char* p = seed; p != nullptr && *p != 0;) {
        const char* colon = strchr(p, ':');
        std::string d(p, colon != nullptr ? size_t(colon - p) : strlen(p));
        p = colon != nullptr ? colon + 1 : nullptr;
        if (!d.empty()) dirs.push_back(std::move(d));
    }
    return dirs;
}

}  // namespace hakios::metal
