#include <cstring>

#include "golden.hpp"
#include "hakios-metal/command.hpp"
#include "test.hpp"

// tests [--golden] [--long] [NAME]: every test whose name holds NAME. With --golden, instead of the verdicts, what each
// test submitted to the fake GPU, summed (golden.hpp): a line a test, compared with tests/golden/tests-<chip>.txt by
// `make -f tools/dev.mk check`. With --long no draw is taken from a program's kept state (CommandBuffer::keepStates):
// the sums are the same, which check holds them to (the tests that count the draws so taken fail then, as they should).
// With --unchecked the draws taken from kept states are not each checked against the long way (CommandBuffer::
// verifyDraws): the driver as a device runs it, with the tests' own verdicts and the same sums. With --sameprog, a draw
// of the same program as the one before leaves out the program's loads (CommandBuffer::skipSameProgramLoads): the
// verdicts must hold, the sums differ.
int main(int argc, char** argv) {
    int ran = 0;
    bool sums = false, checked = true;
    const char* only = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--golden") == 0) sums = true;
        else if (strcmp(argv[i], "--long") == 0) hakios::metal::CommandBuffer::keepStates(false);
        else if (strcmp(argv[i], "--unchecked") == 0) checked = false;
        else if (strcmp(argv[i], "--sameprog") == 0) hakios::metal::CommandBuffer::skipSameProgramLoads(true);
        else only = argv[i];
    }
    golden::Sums s;
    if (sums) golden::sumInto(&s);
    // every draw the tests make from a program's kept state is checked against the same draw encoded in full
    hakios::metal::CommandBuffer::verifyDraws(checked);
    // ...and memory a command buffer takes over from another is not the other's tables (CommandQueue::poisonReused)
    hakios::metal::CommandQueue::poisonReused(true);
    // tests whose submissions depend on which thread got there first (a command buffer let go while the GPU has it:
    // whether its memory is back in the queue's pools when the next one asks is the race the test is about), so that
    // the addresses in their streams differ from run to run: counted, not summed
    static const char* const kTimed[] = {"a_command_buffer_released_in_flight_gives_its_buffers_back"};
    for (const test::Case& c : test::cases()) {
        if (only != nullptr && strstr(c.name, only) == nullptr) {
            continue;
        }
        int before = test::failures();
        s.reset();
        c.run();
        if (sums) {
            bool timed = false;
            for (const char* t : kTimed) timed |= strcmp(t, c.name) == 0;
            // (a test that submits nothing says so with zeros: its line still tells a test added or taken away)
            std::printf("%s submits %u words %llu draws %u loads %u stream %016llx state %016llx names %u %016llx%s%s\n", c.name, s.submits,
                        (unsigned long long)s.words, s.draws, s.loads, (unsigned long long)(s.submits && !timed ? s.stream : 0),
                        (unsigned long long)(s.submits && !timed ? s.state : 0), timed ? 0 : s.named, (unsigned long long)(timed ? 0 : s.names),
                        s.bad ? " (unread)" : "", timed ? " (by timing: not summed)" : "");
        } else {
            std::printf("%s %s\n", test::failures() == before ? "ok  " : "FAIL", c.name);
        }
        ran++;
    }
    if (!sums) std::printf("%d tests, %d failed checks\n", ran, test::failures());
    return test::failures() == 0 ? 0 : 1;
}
