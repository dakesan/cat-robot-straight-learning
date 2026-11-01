#!/bin/sh
set -eu
robot_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
robot_build=$(mktemp -d "${TMPDIR:-/tmp}/cat-robot-control.XXXXXX")
c++ -std=c++17 -Wall -Wextra -Werror -fsanitize=undefined "$robot_root/tests/control_test.cpp" -o "$robot_build/control-test"
node -e 'const{spawnSync}=require("node:child_process");const r=spawnSync(process.argv[1],[],{encoding:"utf8",timeout:5000,env:{...process.env,UBSAN_OPTIONS:"halt_on_error=1"}});process.stdout.write(r.stdout||"");process.stderr.write(r.stderr||"");if(r.error)process.stderr.write(r.error.message+"\n");process.exit(r.status===0?0:1)' "$robot_build/control-test"
c++ -std=c++17 -Wall -Wextra -Werror -fsanitize=undefined "$robot_root/tests/drive_test.cpp" -o "$robot_build/drive-test"
"$robot_build/drive-test"
