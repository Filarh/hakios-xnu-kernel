#!/bin/sh
# blurfreeze.sh [OUTDIR]: builds blurfreeze.dylib (SpringBoard tweak, src/tweaks/blurfreeze.m) with the iPhoneOS SDK
# xcrun finds (or SDK=), for iOS 15 and later (so that ARC calls no entry point newer than iOS 16.5 has), ad-hoc signed, with its filter list beside it. Default OUTDIR: obj/tweaks.
# Installed by copying both files to /var/jb/usr/lib/TweakInject/ and restarting SpringBoard; switched by
# /var/tmp/blurfreeze.args (see the comment at the top of blurfreeze.m).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HERE/../../obj/tweaks}
SDK=${SDK:-$(xcrun --sdk iphoneos --show-sdk-path)}
mkdir -p "$OUT"
xcrun --sdk iphoneos clang -target arm64-apple-ios15.0 -isysroot "$SDK" -O2 -g -fobjc-arc -Wall -Wextra -Werror \
	-Wno-unused-parameter -dynamiclib -install_name /var/jb/usr/lib/TweakInject/blurfreeze.dylib \
	-framework Foundation -framework QuartzCore -framework CoreGraphics -framework IOSurface \
	-o "$OUT/blurfreeze.dylib" "$HERE/blurfreeze.m"
cp "$HERE/blurfreeze.plist" "$OUT/blurfreeze.plist"
codesign -f -s - "$OUT/blurfreeze.dylib" 2>/dev/null || ${LDID:-ldid} -S "$OUT/blurfreeze.dylib"
echo "blurfreeze.sh: $OUT/blurfreeze.dylib"
