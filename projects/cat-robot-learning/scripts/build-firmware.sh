#!/bin/sh
set -eu
console_mode=cdc
if [ "${1:-}" = "--com" ]; then
  console_mode=default
  shift
fi
fqbn="esp32:esp32:esp32s3:FlashSize=16M,PartitionScheme=huge_app,CDCOnBoot=$console_mode"
learning_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
sketch="$learning_root/../cat-robot-wiring-visual/sketches/cat_robot"
if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then
  tool_root="${XDG_CACHE_HOME:-$HOME/.cache}/cat-robot-tools/ctags-5.8-arduino11-arm64"
  if [ ! -x "$tool_root/source/ctags" ]; then
    mkdir -p "$tool_root"
    if [ ! -d "$tool_root/source/.git" ]; then
      git clone --quiet --depth 1 --branch 5.8-arduino11 https://github.com/arduino/ctags.git "$tool_root/source"
    fi
    # This old ctags macro conflicts with the current Apple SDK's attribute token.
    for source in "$tool_root/source/"*.c "$tool_root/source/"*.h; do
      perl -pi -e 's/\b__unused__\b/CTAGS_UNUSED/g' "$source"
    done
    (cd "$tool_root/source" && ./configure > "$tool_root/configure.log" 2>&1 && make -j4 > "$tool_root/build.log" 2>&1)
  fi
  "$tool_root/source/ctags" --version
  arduino-cli compile --fqbn "$fqbn" \
    --build-property "runtime.tools.ctags.path=$tool_root/source" \
    --build-property "runtime.tools.ctags-5.8-arduino11.path=$tool_root/source" "$sketch" "$@"
else
  arduino-cli compile --fqbn "$fqbn" "$sketch" "$@"
fi
