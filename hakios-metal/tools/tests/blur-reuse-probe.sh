#!/bin/sh
# the probe on a synthetic capture: 2 comparisons, the screen identical in both, the capture identical in the first and
# changed by its Tmpc draw in the second
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d)
python3 -I "$HERE/gen-blur-reuse-trace.py" "$T/trace.txt"
python3 -I "$HERE/../blur-reuse-probe.py" --verbose "$T/trace.txt" | tee "$T/out.txt"
grep -q 'compared with the previous frame: 2)' "$T/out.txt"
grep -q 'identical (and not depending on what they loaded): 2 ' "$T/out.txt"
grep -q 'capture reusable (upper bound): 1 ' "$T/out.txt"
grep -q 'Tmpc49_Idst *1$' "$T/out.txt"
rm -rf "$T"
echo "blur-reuse-probe: ok"
