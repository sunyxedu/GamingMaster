#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runtime
archive=runtime/factorio-headless_linux_2.0.72.tar.xz
if [ ! -f "$archive" ]; then
  curl --fail --location --retry 3 https://factorio.com/get-download/2.0.72/headless/linux64 -o "$archive.part"
  mv "$archive.part" "$archive"
fi
sha256sum -c engine.sha256
if [ ! -x runtime/factorio/bin/x64/factorio ]; then
  tar -xJf "$archive" -C runtime
fi
runtime/factorio/bin/x64/factorio --version
