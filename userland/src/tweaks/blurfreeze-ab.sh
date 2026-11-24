#!/bin/sh
# blurfreeze-ab.sh OUTDIR NAME="args" [NAME="args"...]  (untested: written on the Mac, without the phone)
# For each condition: writes /var/tmp/blurfreeze.args on the phone (read by the tweak when no session runs: no respring),
# then three times arms a frame trace of 45 command buffers and opens and closes the Control Center, and brings the
# traces back to OUTDIR/NAME/. Before: blurfreeze.dylib installed and SpringBoard restarted once, and metal_trace=frame
# in /var/tmp/metal.args with backboardd restarted once (each switch of the driver is read once a process).
#   blurfreeze-ab.sh ~/ab-freeze A="mode=0" B="mode=1" C="mode=2 scale=0.5"
set -eu
OUT=$1; shift
S() { ssh -o ConnectTimeout=8 sapphire "export PATH=/var/jb/usr/bin:/var/jb/usr/sbin:/var/jb/bin:\$PATH; $*"; }
for kv in "$@"; do
	c=${kv%%=*}; a=${kv#*=}
	S "printf '%s\n' '$a' > /var/tmp/blurfreeze.args; rm -f /var/tmp/metal-trace-*.txt; H=/var/root/hidtouch; sleep 1.5; \
	   for n in 1 2 3; do echo 45 > /var/tmp/metal.trace; sleep 0.3; \$H swipe 0.92 0.003 0.92 0.65 300; sleep 3; \$H tap 0.5 0.92; sleep 2.5; done"
	mkdir -p "$OUT/$c"
	for f in $(S 'ls /var/tmp/metal-trace-*.txt' 2>/dev/null); do scp -q "sapphire:$f" "$OUT/$c/"; done
	echo "$c ($a): $(ls "$OUT/$c" | wc -l | tr -d ' ') traces"
done
S "printf 'mode=0\n' > /var/tmp/blurfreeze.args; rm -f /var/tmp/metal-trace-*.txt /var/tmp/metal.trace"
