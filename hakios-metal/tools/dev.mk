# tools/dev.mk: the host tests and drawbench built object by object and in parallel, for either chip's constants
# (the Makefile's own `make test` compiles every source in one compiler run, a minute each time).
#
#   make -f tools/dev.mk -j4 [CHIP=a635|a610] [OPT="-O2 -g"] tests       out/dev-$(CHIP)/tests
#   make -f tools/dev.mk -j4 [CHIP=...] drawbench                       out/dev-$(CHIP)/drawbench (tools/drawbench.cpp)
#   make -f tools/dev.mk -j4 [CHIP=...] passcost                        out/dev-$(CHIP)/passcost (tools/passcost.cpp)
#   make -f tools/dev.mk -j4 [CHIP=...] check                           the tests, then the golden streams (tests/golden/)
#   make -f tools/dev.mk -j4 [CHIP=...] golden                          write the golden streams again (a change of the stream, said in its commit)
#
# CHIP=a610 builds with -DHAKIOS_CHIP_A610, the constants the phone's bundle is built with (make ios CHIP=a610).
CHIP ?= a635
CXX  ?= c++
OPT  ?= -O2 -g
OUT   = out/dev-$(CHIP)
DEFS  = $(if $(filter a610,$(CHIP)),-DHAKIOS_CHIP_A610)
CXXFLAGS = $(OPT) -std=c++17 -Wall -Wextra -Werror -Iinclude $(DEFS) -MMD -MP

CORE  = pm4 gpu command compute texture render gmem link compiler metallib c_api fake_gpu
TESTS = $(basename $(notdir $(wildcard tests/*_test.cpp))) main

CORE_O = $(CORE:%=$(OUT)/obj/src/%.o)
TEST_O = $(TESTS:%=$(OUT)/obj/tests/%.o)

.PHONY: tests drawbench passcost check golden
tests: $(OUT)/tests
drawbench: $(OUT)/drawbench
passcost: $(OUT)/passcost

$(OUT)/obj/src/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $<
$(OUT)/obj/tests/%.o: tests/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -Itests -c -o $@ $<
$(OUT)/obj/tools/%.o: tools/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -Itests -c -o $@ $<
$(OUT)/tests: $(CORE_O) $(TEST_O)
	$(CXX) $(OPT) -pthread -o $@ $^
$(OUT)/drawbench: $(CORE_O) $(OUT)/obj/tools/drawbench.o
	$(CXX) $(OPT) -pthread -o $@ $^
$(OUT)/passcost: $(CORE_O) $(OUT)/obj/tools/passcost.o
	$(CXX) $(OPT) -pthread -o $@ $^

# the golden streams: every scene of drawbench and every test's submissions, word for word what they were when
# tests/golden/ was written (drawbench --golden, tests --golden), with the draws taken from their programs' kept states
# (each checked against the long way as it is taken, and again with no such check: --unchecked, the driver as a device
# runs it) and with every draw written whole (--long)
check: $(OUT)/tests $(OUT)/drawbench
	$(OUT)/tests > $(OUT)/tests.log 2>&1; tail -1 $(OUT)/tests.log; grep -q ' 0 failed checks' $(OUT)/tests.log
	$(OUT)/drawbench --golden | diff -u tests/golden/drawbench-$(CHIP).txt -
	$(OUT)/tests --golden | diff -u tests/golden/tests-$(CHIP).txt -
	$(OUT)/drawbench --golden --long | diff -u tests/golden/drawbench-$(CHIP).txt -
	$(OUT)/tests --golden --long 2>/dev/null | diff -u tests/golden/tests-$(CHIP).txt -
	$(OUT)/tests --unchecked > $(OUT)/tests-unchecked.log 2>&1; tail -1 $(OUT)/tests-unchecked.log; grep -q ' 0 failed checks' $(OUT)/tests-unchecked.log
	$(OUT)/tests --golden --unchecked | diff -u tests/golden/tests-$(CHIP).txt -
	@echo "golden streams: the same -- from kept states (each draw checked, and unchecked) and written whole ($(CHIP))"
golden: $(OUT)/tests $(OUT)/drawbench
	@mkdir -p tests/golden
	$(OUT)/drawbench --golden > tests/golden/drawbench-$(CHIP).txt
	$(OUT)/tests --golden > tests/golden/tests-$(CHIP).txt

-include $(CORE_O:.o=.d) $(TEST_O:.o=.d) $(OUT)/obj/tools/drawbench.d $(OUT)/obj/tools/passcost.d
