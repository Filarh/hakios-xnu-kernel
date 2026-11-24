// A pipeline's specialised programs, drawn with as they become ready: never waited for.
//
// An uber shader -- a fragment stage of thousands of instructions whose path a small constant buffer chooses, like
// CoreAnimation's fixed_frag_lph_cpf and its 52-byte FragmentShaderConfig -- is drawn at once with its own program (the
// generic one, which reads the buffer in every fragment) when a content of that buffer is new, and a program compiled
// for that content is asked of the compile queue (compile_queue.hpp) at a priority; the draws after it is published
// take it. This is the plug-in's -hakiosProgramFor: as it was (its slots, the masks of the bytes each compilation read,
// the aliases), in the core so that the host can test it and the queue can order it, with what it compiles left to
// the caller (HOOKS): the plug-in's compile is hakios_metal_program_specialize; the tests' a sleep and a fixture.
//
// The swap is a pointer published under the table's lock and read under it by every draw: a command buffer already
// encoded keeps the program it drew with (a CompiledProgram is shared with every command buffer that drew with it until
// they complete); the programs the table made live until the table goes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "hakios-metal/compile_queue.hpp"

namespace hakios::metal {

class SpecializationTable {
public:
    struct Hooks {
        // the program for these bytes (run on the queue's worker, or on a thread that waits for it); nullptr: it could
        // not be made -- the generic one goes on drawing that content
        std::function<void*(const uint8_t* bytes, size_t length)> compile;
        // the bytes its compilation read (0xff each, 0 for the rest), LENGTH long; false when not known
        std::function<bool(void* program, std::vector<uint8_t>* mask)> reads;
        // a program the table made, when the table goes
        std::function<void(void* program)> release;
    };
    struct Stats {
        uint32_t contents = 0;      // slots taken
        uint32_t ready = 0;         // ...of which with a program
        uint32_t failed = 0;        // ...of which could not be made
        uint32_t fallbackDraws = 0; // programFor answers that were the generic one while a program was coming
        uint32_t hits = 0;          // answers that were a specialised program
        uint32_t full = 0;          // contents turned away because every slot was taken
    };

    // BASE: the generic program (not owned). LENGTH: the deciding bytes (at most 256). MOST: slots. QUEUE and PRIORITY:
    // where the compilations go. WHAT: what a compilation is called while it runs (for the console's "behind").
    // EXACT: a content is another only when every byte is the same (the plug-in's metal_spec_mask=0).
    SpecializationTable(void* base, size_t length, size_t most, Hooks hooks, CompileQueue& queue, CompilePriority priority, std::string what,
                        bool exact = false);
    ~SpecializationTable();
    SpecializationTable(const SpecializationTable&) = delete;
    SpecializationTable& operator=(const SpecializationTable&) = delete;

    // The program to draw a content (LENGTH bytes at BYTES) with, now: its own when it is ready, the generic one
    // otherwise (and a compilation asked for when the content is new; *STARTED says so). Never waits for a compilation.
    void* programFor(const uint8_t* bytes, bool* started = nullptr);
    // Wait until the program for a content is ready (it is compiled here, now, if it is still queued): for a caller
    // that would rather wait than draw the generic one. Returns what programFor would then answer.
    void* programForWaiting(const uint8_t* bytes);
    Stats stats();
    size_t length() const;

private:
    struct State;
    std::shared_ptr<State> state_;
};

}  // namespace hakios::metal
