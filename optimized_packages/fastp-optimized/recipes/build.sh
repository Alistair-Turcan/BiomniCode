#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="$(realpath "${1:-$PWD}")"
ENV="$ROOT/env/toolchain"
NAME="$(basename "$SOURCE")"
BUILD="$ROOT/env/build/$NAME"
mkdir -p "$BUILD"
export PKG_CONFIG_PATH="$ENV/lib/pkgconfig"
export LD_LIBRARY_PATH="$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
make -C "$SOURCE" -j 4 DIR_OBJ="$BUILD/obj" BIN_TARGET="$BUILD/fastp" \
  INCLUDE_DIRS="$ENV/include" LIBRARY_DIRS="$ENV/lib" \
  PKG_CONFIG="$ENV/bin/pkg-config" \
  LD_FLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib -lisal -ldeflate -lhwy -lpthread"
printf '%s\n' "$BUILD/fastp"
