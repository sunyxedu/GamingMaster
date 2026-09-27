#!/usr/bin/env bash
# fetch_engines.sh — install the external engines used by `gmdata label` (engine
# teacher labels for gm-shards), OUTSIDE the repository.
#
# Usage:
#   scripts/engines/fetch_engines.sh stockfish     # official release binary (best build for this CPU)
#   scripts/engines/fetch_engines.sh egaroucid     # Othello: clone + build the console engine
#   scripts/engines/fetch_engines.sh yaneuraou     # Shogi: clone + build NNUE edition + download eval
#   scripts/engines/fetch_engines.sh katago        # Go: release binary (backend by host) + network
#   scripts/engines/fetch_engines.sh pikafish      # Xiangqi: clone + build + NNUE network
#   scripts/engines/fetch_engines.sh all           # all of the above
#   scripts/engines/fetch_engines.sh paths         # print what is installed
#
# Layout (root = ${GM_ENGINES:-$HOME/gm-engines}):
#   bin/stockfish  bin/egaroucid  bin/yaneuraou  bin/katago  bin/pikafish
#                                                             entry points (symlinks / small
#                                                             wrappers that cd into the engine dir)
#   <engine>/<version>/...                                    the installs themselves
#   nets/...                                                  NNUE evals / KataGo networks
#   downloads/                                                download cache
# Every subcommand is idempotent (an existing install of the same version is kept) and
# ends by printing the installed paths. Supported hosts: macOS arm64 (x86-64 best effort)
# and Linux x86-64 (aarch64 best effort). Needs curl (or wget) and tar; the source builds
# need git, make / cmake and a C++20 compiler (g++ >= 11 or clang++).
#
# Engines and licences — all run as separate processes by gmdata, never linked or vendored:
#   Stockfish  GPL-3.0. Official release binaries, github.com/official-stockfish/Stockfish.
#              Releases >= sf_19 ship one "universal" binary per OS that selects the best
#              instruction set at runtime (x86-64 up to avx512icl / vnni512); older tags have
#              per-arch builds, chosen here from the CPU flags (STOCKFISH_ARCH overrides).
#   Egaroucid  GPL-3.0. Built from source, github.com/Nyanyan/Egaroucid (console engine, one
#              translation unit; AVX-512 / AVX2 / generic build by CPU flags, generic on ARM;
#              patched for labelling: transposition table on the heap (TT_USE_STACK=false)
#              and a fixed RNG seed, so every process searches identically — see
#              egaroucid_patch; the smoke test checks both). Its evaluation file
#              (bin/resources/eval.egev2) ships in the same repository under the same licence.
#   YaneuraOu  GPL-3.0. Built from source, github.com/yaneurao/YaneuraOu (tag v9.40 with two
#              small source patches, see yaneuraou_patch: the upstream macOS isready-deadlock
#              fix f129339a, and a ~12 ms instead of ~150 ms isready), standard NNUE edition
#              (halfkp_256x2-32-32). Eval: "Háo" (tanuki- project, release
#              tanuki-.halfkp_256x2-32-32.2023-05-08 at github.com/nodchip/tanuki-, a free
#              download; the archive ships gpl-3.0.txt: GPL-3.0; about as strong as tanuki-'s
#              WCSC33 eval "Lí" per its authors; needs the USI option FV_SCALE=20, which
#              `gmdata label shogi` sets by default).
#   Pikafish   GPL-3.0. Built from source, github.com/official-pikafish/Pikafish (a release
#              tag, Stockfish's Makefile: ARCH by CPU flags). Its NNUE network (pikafish.nnue,
#              GPL-3.0) comes from the Networks repository's master-net release, which the
#              build downloads; the install records the network's SHA-256 (the release
#              asset is replaced as the network improves).
#   KataGo     Engine: github.com/lightvector/KataGo (see its LICENSE: MIT-style, with
#              separately licensed third-party parts). Networks from katagotraining.org
#              under the "KataGo Neural Network License" (MIT-style,
#              https://katagotraining.org/network_license/).
#
# Environment overrides:
#   GM_ENGINES             install root (default $HOME/gm-engines)
#   JOBS                   build parallelism (default: all cores)
#   STOCKFISH_TAG          release tag (default: latest, e.g. sf_19)
#   STOCKFISH_ARCH         per-arch build for pre-universal tags (x86-64-avx512icl, ...)
#   EGAROUCID_REF          git tag / branch (default: v7.8.3, the version the othello labeller's
#                          defaults were measured and fitted with)
#   EGAROUCID_CXX          compiler (default: g++ on Linux, else clang++)
#   YANEURAOU_REF          git tag / branch (default: v9.40, pinned: labels record the engine
#                          version, and later versions may drop the halfkp_256x2-32-32 edition)
#   YANEURAOU_TARGET_CPU   Makefile TARGET_CPU (default: from the CPU flags)
#   YANEURAOU_EVAL_URL     .7z / .zip archive containing nn.bin (default: Háo)
#   PIKAFISH_REF           git tag (default: Pikafish-2026-09-06)
#   PIKAFISH_ARCH          Makefile ARCH (default: from the CPU flags, e.g. x86-64-avx512icl)
#   KATAGO_TAG             release tag (default: v1.18.1 for every Linux backend — the newest
#                          tag with CPU / OpenCL / TensorRT / ROCm assets; pinned so that labels
#                          from GPU and CPU nodes share one engine version in meta.teacher)
#   KATAGO_BACKEND         eigenavx2 | eigen | opencl | cuda12.8-cudnn9.8.0 | trt10.9.0-cuda12.8 |
#                          onnx-openvino | ... (the part between the tag and "-linux-x64" in the
#                          release asset name). Default, all without root: CUDA when an NVIDIA
#                          GPU (nvidia-smi) plus cuDNN 9 and cuBLAS 12 are loadable (ldconfig or
#                          LD_LIBRARY_PATH — e.g. after `module load cuda cudnn`, or pip's
#                          nvidia-cudnn-cu12 / nvidia-cublas-cu12 lib dirs on LD_LIBRARY_PATH);
#                          OpenCL when a GPU OpenCL ICD is present; else eigenavx2 / eigen (the
#                          CPU builds are static-pie executables and need no system libraries).
#                          TensorRT (fastest on NVIDIA; needs libnvinfer 10) only on request.
#   KATAGO_NET_URL         network URL (default: kata1 b18c384nbt — strong and the cheapest
#                          of the current kata1 nets per evaluation)
#
# KataGo on macOS: Homebrew's bottle (Metal backend; `brew install katago`) plus the
# b18c384nbt network it bundles (reused, not downloaded again). `gmdata label go` runs
# it with KataGo's Metal "mux" layout (2 GPU + 2 Neural Engine threads) by default.
# Measured on an M4 Pro: ~550 network evaluations/s mux vs ~200 GPU-only.
set -euo pipefail

ROOT="${GM_ENGINES:-$HOME/gm-engines}"
BIN="$ROOT/bin"
DL="$ROOT/downloads"
OS="$(uname -s)"
ARCH="$(uname -m)"

log() { printf '[fetch_engines] %s\n' "$*" >&2; }
die() { printf '[fetch_engines] error: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

jobs() {
    if [ -n "${JOBS:-}" ]; then echo "$JOBS"; return; fi
    if have nproc; then nproc; elif [ "$OS" = Darwin ]; then sysctl -n hw.ncpu; else echo 4; fi
}

# CPU feature flags, lower case, space separated (empty when unknown, e.g. Apple silicon).
cpu_flags() {
    if [ -r /proc/cpuinfo ]; then
        grep -m1 -E '^(flags|Features)' /proc/cpuinfo | cut -d: -f2 | tr '[:upper:]' '[:lower:]' || true
    elif [ "$OS" = Darwin ]; then
        { sysctl -n machdep.cpu.features machdep.cpu.leaf7_features 2>/dev/null || true; } | tr 'A-Z.' 'a-z_' | tr '\n' ' '
    fi
}
FLAGS=" $(cpu_flags) "
has_flag() { case "$FLAGS" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
VENDOR="$({ grep -m1 -E '^vendor_id' /proc/cpuinfo 2>/dev/null || true; } | awk '{print $NF}')"
is_amd() { [ "$VENDOR" = AuthenticAMD ]; }

# Download $1 to $2 (atomic; kept if present).
fetch() {
    local url="$1" dest="$2"
    if [ -s "$dest" ]; then log "have $(basename "$dest")"; return 0; fi
    mkdir -p "$(dirname "$dest")"
    log "downloading $url"
    if have curl; then
        curl -fL --retry 3 --connect-timeout 30 -o "$dest.part" "$url" || { rm -f "$dest.part"; return 1; }
    elif have wget; then
        wget -q -O "$dest.part" "$url" || { rm -f "$dest.part"; return 1; }
    else
        die "need curl or wget"
    fi
    mv "$dest.part" "$dest"
}

url_exists() {
    if have curl; then curl -fsIL --connect-timeout 30 -o /dev/null "$1"; else wget -q --spider "$1"; fi
}

# Latest release tag of github repo $1 (owner/name), via the /releases/latest redirect
# (no API call, so no rate limit).
latest_tag() {
    local u
    if have curl; then
        u="$(curl -fsSIL --connect-timeout 30 -o /dev/null -w '%{url_effective}' "https://github.com/$1/releases/latest")" || return 1
    else
        u="$(wget -q --max-redirect=5 -S --spider "https://github.com/$1/releases/latest" 2>&1 | awk '/^  Location: /{l=$2} END{print l}')"
    fi
    u="${u%/}"
    case "$u" in */tag/*) echo "${u##*/}" ;; *) return 1 ;; esac
}

# Extract archive $1 into directory $2 (.tar*, .zip, .7z).
extract() {
    local a="$1" d="$2"
    mkdir -p "$d"
    case "$a" in
        *.tar.gz | *.tgz) tar -xzf "$a" -C "$d" ;;
        *.tar.xz) tar -xJf "$a" -C "$d" ;;
        *.tar) tar -xf "$a" -C "$d" ;;
        *.zip)
            if have unzip; then unzip -q -o "$a" -d "$d"
            elif have bsdtar; then bsdtar -xf "$a" -C "$d"
            elif have python3; then python3 -m zipfile -e "$a" "$d"
            else die "need unzip, bsdtar or python3 to extract $a"; fi ;;
        *.7z) extract_7z "$a" "$d" ;;
        *) die "unknown archive type: $a" ;;
    esac
}

extract_7z() {
    local a="$1" d="$2" tool=""
    for t in 7zz 7z 7za 7zr; do if have "$t"; then tool="$t"; break; fi; done
    if [ -n "$tool" ]; then "$tool" x -y -o"$d" "$a" >/dev/null; return; fi
    if have bsdtar; then bsdtar -xf "$a" -C "$d"; return; fi
    if tar --version 2>/dev/null | grep -q bsdtar; then tar -xf "$a" -C "$d"; return; fi
    # standalone 7-Zip (7zz) for Linux
    if [ "$OS" = Linux ]; then
        local a7 tag v url
        case "$ARCH" in x86_64) a7=x64 ;; aarch64 | arm64) a7=arm64 ;; *) die "no 7-Zip build for $ARCH; install p7zip" ;; esac
        tag="$(latest_tag ip7z/7zip || echo 26.03)"
        v="$(echo "$tag" | tr -d .)"
        url="https://github.com/ip7z/7zip/releases/download/$tag/7z$v-linux-$a7.tar.xz"
        fetch "$url" "$DL/7z$v-linux-$a7.tar.xz" || die "cannot download 7-Zip ($url); install p7zip"
        extract "$DL/7z$v-linux-$a7.tar.xz" "$ROOT/tools/7zip"
        "$ROOT/tools/7zip/7zz" x -y -o"$d" "$a" >/dev/null
        return
    fi
    die "need 7z / bsdtar to extract $a"
}

# Point $BIN/$1 at $2 (a symlink).
link_bin() {
    mkdir -p "$BIN"
    ln -sfn "$2" "$BIN/$1"
}

# A wrapper $BIN/$1 that runs $3 from directory $2 (engines that find their data files
# relative to the working directory).
wrap_bin() {
    mkdir -p "$BIN"
    rm -f "$BIN/$1"
    cat > "$BIN/$1" <<EOF
#!/bin/sh
# generated by fetch_engines.sh: run $1 from its install directory
cd "$2" && exec "$3" "\$@"
EOF
    chmod +x "$BIN/$1"
}

# ------------------------------------------------------------------ Stockfish

stockfish_candidates() {
    local tag="$1" a
    case "$OS-$ARCH" in
        Darwin-arm64)
            echo "stockfish-macos-universal.tar.gz stockfish-macos-m1-apple-silicon.tar" ;;
        Darwin-x86_64)
            echo -n "stockfish-macos-universal.tar.gz "
            for a in $(stockfish_x86_arches); do
                case "$a" in avx2 | bmi2 | sse41-popcnt | x86-64) ;; *) continue ;; esac
                [ "$a" = x86-64 ] && echo -n "stockfish-macos-x86-64.tar " || echo -n "stockfish-macos-x86-64-$a.tar "
            done; echo ;;
        Linux-x86_64)
            echo -n "stockfish-linux-x86-64-universal.tar.gz "
            for a in $(stockfish_x86_arches); do
                [ "$a" = x86-64 ] && echo -n "stockfish-ubuntu-x86-64.tar " || echo -n "stockfish-ubuntu-x86-64-$a.tar "
            done; echo ;;
        Linux-aarch64 | Linux-arm64)
            echo "stockfish-linux-arm64-universal.tar.gz" ;;
        *) echo "" ;;
    esac
}

# Per-arch x86 builds this CPU can run, best first (pre-universal releases).
stockfish_x86_arches() {
    if [ -n "${STOCKFISH_ARCH:-}" ]; then echo "${STOCKFISH_ARCH#x86-64-}"; return; fi
    local out=""
    if has_flag avx512_vnni && has_flag avx512_vbmi2 && has_flag avx512_bitalg && has_flag avx512_vpopcntdq && has_flag avx512ifma; then
        out="$out avx512icl"
    fi
    if has_flag avx512_vnni && has_flag avx512bw && has_flag avx512vl; then out="$out vnni512"; fi
    if has_flag avx512bw && has_flag avx512f; then out="$out avx512"; fi
    if has_flag avx_vnni; then out="$out avxvnni"; fi
    if has_flag bmi2 && has_flag avx2; then out="$out bmi2"; fi
    if has_flag avx2; then out="$out avx2"; fi
    if has_flag sse4_1 && has_flag popcnt; then out="$out sse41-popcnt"; fi
    echo "$out x86-64"
}

install_stockfish() {
    local tag cand url dir exe
    tag="${STOCKFISH_TAG:-$(latest_tag official-stockfish/Stockfish || true)}"
    [ -n "$tag" ] || die "cannot resolve the latest Stockfish release (set STOCKFISH_TAG, e.g. sf_19)"
    dir="$ROOT/stockfish/$tag"
    if [ -x "$dir/stockfish" ]; then
        log "Stockfish $tag already installed"
    else
        local picked=""
        for cand in $(stockfish_candidates "$tag"); do
            url="https://github.com/official-stockfish/Stockfish/releases/download/$tag/$cand"
            if url_exists "$url"; then picked="$cand"; break; fi
        done
        if [ -z "$picked" ]; then
            if [ "$OS" = Darwin ] && have brew; then
                log "no release binary for $OS-$ARCH in $tag; using Homebrew"
                brew install stockfish
                mkdir -p "$dir"
                ln -sfn "$(brew --prefix)/bin/stockfish" "$dir/stockfish"
            else
                die "no Stockfish $tag binary for $OS-$ARCH (flags:${FLAGS:0:80}...)"
            fi
        else
            fetch "$url" "$DL/stockfish-$tag-$picked"
            rm -rf "$dir.tmp"
            extract "$DL/stockfish-$tag-$picked" "$dir.tmp/pkg"
            exe="$(find "$dir.tmp/pkg" -type f -name 'stockfish*' ! -name '*.*' -perm -u+x | head -1)"
            [ -n "$exe" ] || exe="$(find "$dir.tmp/pkg" -type f -name 'stockfish-*' ! -name '*.md' ! -name '*.txt' ! -name '*.sh' | head -1)"
            [ -n "$exe" ] || die "no stockfish executable in $picked"
            chmod +x "$exe"
            rm -rf "$dir"
            mv "$dir.tmp" "$dir"
            ln -sfn "$dir/${exe#"$dir.tmp/"}" "$dir/stockfish"
        fi
    fi
    link_bin stockfish "$dir/stockfish"
    # smoke test: UCI handshake + the build the binary runs (universal builds pick at runtime)
    local id arch
    id="$(printf 'uci\nquit\n' | "$BIN/stockfish" 2>/dev/null | awk '/^id name/{sub(/^id name /,""); print; exit}')"
    [ -n "$id" ] || die "$BIN/stockfish does not answer the UCI handshake"
    arch="$(printf 'compiler\nquit\n' | "$BIN/stockfish" 2>/dev/null | awk -F: '/Compilation architecture/{gsub(/^ +| +$/,"",$2); print $2; exit}')"
    echo "stockfish: $BIN/stockfish -> $(readlink "$dir/stockfish") ($id${arch:+, $arch build}; GPL-3.0)"
}

# ------------------------------------------------------------------ Egaroucid

# Egaroucid build defines for this host: its SIMD code is x86-only (AVX2; AVX-512 paths
# need F/VL/CD/DQ), so ARM and pre-AVX2 CPUs get the generic build.
egaroucid_defs() {
    case "$ARCH" in
        arm64 | aarch64) echo "-DHAS_NO_AVX2 -DHAS_ARM_PROCESSOR"; return ;;
    esac
    local d=""
    if ! has_flag avx2; then d="-DHAS_NO_AVX2"
    elif has_flag avx512f && has_flag avx512vl && has_flag avx512cd && has_flag avx512dq; then d="-DHAS_AVX512"; fi
    if is_amd; then d="${d:+$d }-DHAS_AMD_PROCESSOR"; fi
    echo "$d"
}

egaroucid_cpu_flags() {
    case "$OS-$ARCH" in
        Darwin-arm64) echo "-mcpu=native" ;;
        *) echo "-march=native -mtune=native" ;;
    esac
}

# Install directory tag: arch + build flavour (the binary is built with -march=native, so
# build it on the kind of node that runs the labelling; a shared home keeps one directory
# per flavour, and bin/egaroucid points at the last one built).
egaroucid_flavor() {
    local f defs
    defs="$(egaroucid_defs)"
    case "$defs" in
        *HAS_NO_AVX2*) f=generic ;;
        *HAS_AVX512*) f=avx512 ;;
        *) f=avx2 ;;
    esac
    # the AMD tag mirrors the build define (x86 only: ARM builds get no AMD tuning)
    case "$defs" in *HAS_AMD_PROCESSOR*) f="$f-amd" ;; esac
    echo "$ARCH-$f"
}

# Revision of the source patches below: part of the install directory name, so a changed
# patch set rebuilds instead of reusing an older build of the same tag.
EGAROUCID_PATCH_REV=gm1
# Fixed seed of Egaroucid's RNG (see egaroucid_patch).
EGAROUCID_SEED=20260926

# Source patches for labelling, applied to a checkout at $1 (both keep the search itself):
#  * TT_USE_STACK=false: the transposition table lives on the heap, sized by -hash (the
#    default build also constructs a static 2^25-entry table at start-up: ~1 GB resident
#    per process whatever -hash says; heap tables are a supported configuration).
#  * a fixed RNG seed: the repository ships transposition-table hash codes only for -hash
#    25-27 (resources/hash/hashNN.eghs); at any other level Egaroucid draws them from an
#    RNG seeded with the start-up time in ms, so every process indexes its table
#    differently and searches differ slightly (measured at the labeller defaults, -hash 20
#    level 15: ~1.5% of positions got a 1-3 disc different score, node counts differ on
#    almost every search). With a fixed seed the codes are identical in every process
#    and labels are reproducible across workers, runs and nodes. The RNG is otherwise only
#    used by features the labeller does not call (book randomness, self-play, tuning).
egaroucid_patch() {
    local e="$1/src/engine"
    sed -i.orig 's/^#define TT_USE_STACK true/#define TT_USE_STACK false/' "$e/setting.hpp"
    grep -q '^#define TT_USE_STACK false' "$e/setting.hpp" \
        || log "warning: TT_USE_STACK patch did not apply (expect ~1 GB more memory per process)"
    sed -i.orig "s/^std::mt19937 raw_myrandom(tim());/std::mt19937 raw_myrandom($EGAROUCID_SEED);/" "$e/common.hpp"
    grep -q "^std::mt19937 raw_myrandom($EGAROUCID_SEED);" "$e/common.hpp" \
        || log "warning: RNG seed patch did not apply (searches would differ between processes)"
}

# The probe of the determinism smoke test: raw hint rows (depth, move, score, nodes) of a
# 44-empties WTHOR position; identical node counts <=> identical searches.
egaroucid_probe() {
    printf 'setboard ------------O-----XXOOX---XXXXO---OOXOX---O-OO------O-----------X\nhint 3\nquit\n' \
        | "$1" -quiet -nobook -thread 1 -hash 16 -level 11 2>/dev/null | awk -F'|' '/^[|]/ {print $3, $4, $5, $7}'
}

install_egaroucid() {
    have git || die "egaroucid needs git"
    local ref dir exe src cxx cpu defs out
    ref="${EGAROUCID_REF:-v7.8.3}"
    dir="$ROOT/egaroucid/$ref-$(egaroucid_flavor)-$EGAROUCID_PATCH_REV"
    exe="$dir/bin/Egaroucid_for_Console.out"
    if [ -x "$exe" ]; then
        log "Egaroucid $ref already built"
    else
        src="$dir.src"
        rm -rf "$src" "$dir.tmp"
        git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$ref" https://github.com/Nyanyan/Egaroucid.git "$src"
        egaroucid_patch "$src"
        cxx="${EGAROUCID_CXX:-}"
        if [ -z "$cxx" ]; then
            if [ "$OS" = Linux ] && have g++; then cxx=g++; elif have clang++; then cxx=clang++; else cxx=c++; fi
        fi
        cpu="$(egaroucid_cpu_flags)"
        defs="$(egaroucid_defs)"
        mkdir -p "$dir.tmp/bin"
        out="$(cd "$dir.tmp/bin" && pwd)/Egaroucid_for_Console.out"
        log "building Egaroucid $ref: $cxx -O2 $cpu -std=c++20 $defs"
        # the console engine is a single translation unit (what its CMakeLists builds)
        # shellcheck disable=SC2086  # $cpu / $defs are intentionally word-split flag lists
        (cd "$src/src" && $cxx -O2 $cpu -pthread -std=c++20 -w $defs Egaroucid_for_Console.cpp -o "$out") \
            || die "Egaroucid build failed (set EGAROUCID_CXX to try another compiler)"
        # keep the binary, its data files (evaluation, hash codes, book) and the licence
        cp -R "$src/bin/resources" "$dir.tmp/bin/resources"
        cp "$src/LICENSE" "$dir.tmp/LICENSE"
        rm -rf "$src" "$dir"
        mv "$dir.tmp" "$dir"
    fi
    [ -f "$dir/bin/resources/eval.egev2" ] || die "$dir/bin/resources/eval.egev2 missing"
    # the macOS build finds resources/ relative to the working directory
    wrap_bin egaroucid "$dir/bin" ./Egaroucid_for_Console.out
    # smoke test: version line + black's four opening moves
    local ver n
    ver="$(printf 'version\nquit\n' | "$BIN/egaroucid" -quiet -nobook -thread 1 -hash 16 2>/dev/null | awk '/^Egaroucid/{print; exit}')" || true
    n="$(printf 'setboard ---------------------------OX------XO---------------------------X\nhint 4\nquit\n' \
        | "$BIN/egaroucid" -quiet -nobook -thread 1 -hash 16 -level 5 2>/dev/null | grep -c -E '[|][[:space:]]+(c4|d3|e6|f5)[|]')" || true
    if [ -z "$ver" ] || [ "$n" != 4 ]; then
        die "$BIN/egaroucid failed its smoke test (version: ${ver:-none}; $n of 4 opening moves)"
    fi
    # determinism: two processes must search identically (see egaroucid_patch)
    local p1 p2
    p1="$(egaroucid_probe "$BIN/egaroucid")" || true
    p2="$(egaroucid_probe "$BIN/egaroucid")" || true
    if [ "$(printf '%s\n' "$p1" | grep -c .)" != 4 ] || [ "$p1" != "$p2" ]; then
        die "$BIN/egaroucid is not deterministic across processes (the RNG seed patch is missing?):
$p1
---
$p2"
    fi
    echo "egaroucid: $BIN/egaroucid -> $exe ($ver; eval $dir/bin/resources/eval.egev2; GPL-3.0)"
}

# ------------------------------------------------------------------ YaneuraOu

yaneuraou_target_cpu() {
    if [ -n "${YANEURAOU_TARGET_CPU:-}" ]; then echo "$YANEURAOU_TARGET_CPU"; return; fi
    case "$OS-$ARCH" in
        Darwin-arm64) echo APPLEM1 ;;
        Darwin-x86_64) if has_flag avx2; then echo APPLEAVX2; else echo APPLESSE42; fi ;;
        # GRAVITON2 = -march=armv8.2-a+fp16+rcpc+dotprod+crypto; OTHER = portable C++
        Linux-aarch64 | Linux-arm64)
            if has_flag asimddp && has_flag asimdhp && has_flag lrcpc && has_flag aes; then echo GRAVITON2; else echo OTHER; fi ;;
        *)
            if has_flag avx512_vnni && has_flag avx512bw; then echo AVX512VNNI
            elif has_flag avx512bw && has_flag avx512f; then echo AVX512
            elif has_flag avx_vnni; then echo AVXVNNI
            elif has_flag avx2; then echo AVX2
            elif has_flag sse4_2; then echo SSE42
            else echo SSE2; fi ;;
    esac
}

# Source fixes applied before building (each a no-op when its pattern is absent):
#  * v9.40 prints "shared memory: fallback to local memory (…)" with two nested sync_cout
#    (a non-recursive output mutex) and deadlocks in `isready` wherever POSIX shared memory
#    is unavailable — always on macOS, on Linux without /dev/shm. Upstream removed the
#    message in f129339a (2026-06-05, after v9.40); here the inner writes go to std::cout.
#  * every `isready` runs through Engine::run_heavy_job, whose keep-alive thread is polled
#    in 100 ms sleeps: ~150 ms per isready. `gmdata label shogi` sends isready before every
#    position (it clears the hash, so labels do not depend on search order); 1 ms polling
#    makes that ~12 ms. Keep-alive newlines still go out every 5 s.
yaneuraou_patch() {
    local shm="$1/shm.h" eng="$1/engine.cpp"
    if [ -f "$shm" ]; then
        LC_ALL=C sed -e 's/^\( *\)sync_cout << " (" << \*error_message << ")";/\1std::cout << " (" << *error_message << ")";/' \
            -e 's/^\( *\)sync_cout << sync_endl;/\1std::cout << sync_endl;/' "$shm" > "$shm.gm" && mv "$shm.gm" "$shm"
    fi
    if [ -f "$eng" ]; then
        LC_ALL=C sed -e 's|std::this_thread::sleep_for(std::chrono::milliseconds(100));|std::this_thread::sleep_for(std::chrono::milliseconds(1));|' \
            -e 's|if (++count >= 50 /\* 5秒 \*/)|if (++count >= 5000 /* 5秒 */)|' \
            -e 's|^\( *\)Tools::sleep(100);|\1Tools::sleep(1);|' "$eng" > "$eng.gm" && mv "$eng.gm" "$eng"
    fi
}

# USI handshake + isready (eval load) within ${YANEURAOU_SMOKE_TIMEOUT:-120} s; prints the
# engine's `id name`.
yaneuraou_smoke() {
    local exe="$1" tmp pid i limit="${YANEURAOU_SMOKE_TIMEOUT:-120}"
    tmp="$(mktemp -d)"
    printf 'usi\nsetoption name Threads value 1\nsetoption name USI_Hash value 16\nsetoption name USI_OwnBook value false\nsetoption name BookFile value no_book\nsetoption name FV_SCALE value 20\nisready\nquit\n' > "$tmp/in"
    "$exe" < "$tmp/in" > "$tmp/out" 2>&1 &
    pid=$!
    for i in $(seq 1 "$limit"); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        grep -v '^option' "$tmp/out" >&2 || true
        rm -rf "$tmp"
        die "$exe: no readyok within $limit s (hangs in isready)"
    fi
    wait "$pid" 2>/dev/null || true
    if ! grep -q '^readyok' "$tmp/out"; then
        grep -v '^option' "$tmp/out" >&2 || true
        rm -rf "$tmp"
        die "$exe: isready failed (eval file missing / unreadable?)"
    fi
    awk '/^id name/{sub(/^id name /,""); print; exit}' "$tmp/out"
    rm -rf "$tmp"
}

install_yaneuraou() {
    have git || die "yaneuraou needs git"
    have make || die "yaneuraou needs make"
    local ref dir cpu cxx exe url arch
    ref="${YANEURAOU_REF:-v9.40}"
    cpu="$(yaneuraou_target_cpu)"
    # -gm1: this script's build recipe (source patches); bump it when yaneuraou_patch changes
    dir="$ROOT/yaneuraou/$ref-$cpu-gm1"
    if [ -x "$dir/YaneuraOu" ]; then
        log "YaneuraOu $ref ($cpu) already built"
    else
        rm -rf "$dir.tmp"
        git -c advice.detachedHead=false clone -q --depth 1 --branch "$ref" https://github.com/yaneurao/YaneuraOu.git "$dir.tmp"
        yaneuraou_patch "$dir.tmp/source"
        if [ "$OS" = Darwin ] || ! have g++; then cxx=clang++; else cxx=g++; fi
        log "building YaneuraOu $ref NNUE edition, TARGET_CPU=$cpu, $cxx (log: $dir/build.log)"
        make -C "$dir.tmp/source" -j"$(jobs)" normal TARGET_CPU="$cpu" \
            YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE COMPILER="$cxx" > "$dir.tmp/build.log" 2>&1 \
            || { tail -30 "$dir.tmp/build.log" >&2; die "YaneuraOu build failed (see $dir.tmp/build.log)"; }
        exe="$(find "$dir.tmp/source" -maxdepth 1 -type f -name 'YaneuraOu*' -perm -u+x | head -1)"
        [ -n "$exe" ] || die "YaneuraOu build produced no executable"
        mkdir -p "$dir.tmp/eval"
        mv "$exe" "$dir.tmp/YaneuraOu"
        rm -rf "$dir.tmp/obj"
        rm -rf "$dir"
        mv "$dir.tmp" "$dir"
    fi
    if [ ! -s "$dir/eval/nn.bin" ]; then
        url="${YANEURAOU_EVAL_URL:-https://github.com/nodchip/tanuki-/releases/download/tanuki-.halfkp_256x2-32-32.2023-05-08/tanuki-.halfkp_256x2-32-32.2023-05-08.7z}"
        arch="$DL/$(basename "$url")"
        fetch "$url" "$arch" || die "cannot download the NNUE eval ($url)"
        rm -rf "$ROOT/nets/yaneuraou.tmp"
        extract "$arch" "$ROOT/nets/yaneuraou.tmp"
        local nn
        nn="$(find "$ROOT/nets/yaneuraou.tmp" -type f -name 'nn.bin' | head -1)"
        [ -n "$nn" ] || die "no nn.bin in $(basename "$url")"
        mkdir -p "$ROOT/nets/yaneuraou" "$dir/eval"
        mv "$nn" "$ROOT/nets/yaneuraou/$(basename "$url" | sed 's/\.[^.]*$//').nn.bin"
        rm -rf "$ROOT/nets/yaneuraou.tmp"
        ln -sfn "$ROOT/nets/yaneuraou/$(basename "$url" | sed 's/\.[^.]*$//').nn.bin" "$dir/eval/nn.bin"
    fi
    wrap_bin yaneuraou "$dir" ./YaneuraOu
    local id
    id="$(yaneuraou_smoke "$BIN/yaneuraou")"
    echo "yaneuraou: $BIN/yaneuraou -> $dir/YaneuraOu ($id; eval $dir/eval/nn.bin -> $(readlink "$dir/eval/nn.bin"), FV_SCALE=20 for Háo; GPL-3.0)"
}

# ------------------------------------------------------------------ Pikafish

# Pikafish keeps Stockfish's Makefile ARCH names.
pikafish_arch() {
    if [ -n "${PIKAFISH_ARCH:-}" ]; then echo "$PIKAFISH_ARCH"; return; fi
    case "$OS-$ARCH" in
        Darwin-arm64) echo apple-silicon ;;
        Linux-aarch64 | Linux-arm64) if has_flag asimddp; then echo armv8-dotprod; else echo armv8; fi ;;
        *)
            if has_flag avx512_vnni && has_flag avx512_vbmi2 && has_flag avx512_vpopcntdq; then echo x86-64-avx512icl
            elif has_flag avx512_vnni && has_flag avx512bw; then echo x86-64-vnni512
            elif has_flag avx512bw && has_flag avx512f; then echo x86-64-avx512
            elif has_flag avx_vnni; then echo x86-64-avxvnni
            elif has_flag bmi2 && has_flag avx2; then echo x86-64-bmi2
            elif has_flag avx2; then echo x86-64-avx2
            elif has_flag sse4_1 && has_flag popcnt; then echo x86-64-sse41-popcnt
            else echo x86-64; fi ;;
    esac
}

install_pikafish() {
    have git || die "pikafish needs git"
    have make || die "pikafish needs make"
    local ref arch dir comp id sha
    ref="${PIKAFISH_REF:-Pikafish-2026-09-06}"
    arch="$(pikafish_arch)"
    dir="$ROOT/pikafish/$ref-$arch"
    if [ -x "$dir/pikafish" ] && [ -s "$dir/pikafish.nnue" ]; then
        log "Pikafish $ref ($arch) already built"
    else
        rm -rf "$dir.tmp"
        git -c advice.detachedHead=false clone -q --depth 1 --branch "$ref" https://github.com/official-pikafish/Pikafish.git "$dir.tmp"
        if [ "$OS" = Darwin ] || ! have g++; then comp=clang; else comp=gcc; fi
        log "building Pikafish $ref, ARCH=$arch, COMP=$comp (log: $dir/build.log)"
        make -C "$dir.tmp/src" -j"$(jobs)" build ARCH="$arch" COMP="$comp" > "$dir.tmp/build.log" 2>&1 \
            || { tail -30 "$dir.tmp/build.log" >&2; die "Pikafish build failed (see $dir.tmp/build.log)"; }
        [ -x "$dir.tmp/src/pikafish" ] && [ -s "$dir.tmp/src/pikafish.nnue" ] \
            || die "the Pikafish build left no executable / network in $dir.tmp/src"
        mkdir -p "$dir.tmp/install"
        mv "$dir.tmp/src/pikafish" "$dir.tmp/src/pikafish.nnue" "$dir.tmp/build.log" "$dir.tmp/install/"
        cp "$dir.tmp/Copying.txt" "$dir.tmp/install/" 2>/dev/null || true
        rm -rf "$dir"
        mv "$dir.tmp/install" "$dir"
        rm -rf "$dir.tmp"
    fi
    # the binary finds pikafish.nnue in its working directory
    wrap_bin pikafish "$dir" ./pikafish
    id="$(printf 'uci\nisready\nquit\n' | "$BIN/pikafish" 2>/dev/null | awk '/^id name/{sub(/^id name /,""); print; exit}')"
    [ -n "$id" ] || die "$BIN/pikafish does not answer the UCI handshake"
    printf 'uci\nisready\nquit\n' | "$BIN/pikafish" 2>/dev/null | grep -q '^readyok' || die "$BIN/pikafish: isready failed"
    if have sha256sum; then sha="$(sha256sum "$dir/pikafish.nnue" | cut -c1-16)"; else sha="$(shasum -a 256 "$dir/pikafish.nnue" | cut -c1-16)"; fi
    echo "pikafish: $BIN/pikafish -> $dir/pikafish ($id, ARCH=$arch; network pikafish.nnue sha256 $sha...; GPL-3.0)"
}

# ------------------------------------------------------------------ KataGo

# The ldconfig cache listing (empty when unavailable). ldconfig lives in /sbin on
# Debian-like systems, which is not on a normal user's PATH ($LDCONFIG overrides).
ldconfig_cache() {
    if [ -n "${LDCONFIG:-}" ]; then "$LDCONFIG" -p 2>/dev/null || true; return 0; fi
    local c
    for c in "$(command -v ldconfig 2>/dev/null || true)" /sbin/ldconfig /usr/sbin/ldconfig; do
        if [ -n "$c" ] && [ -x "$c" ]; then "$c" -p 2>/dev/null || true; return 0; fi
    done
}

# Whether shared library $1 (e.g. libcudnn.so.9) is loadable: known to ldconfig or in a
# directory of LD_LIBRARY_PATH (module systems / pip wheels, no root needed). The cache
# is captured first: `ldconfig -p | grep -q` under `set -o pipefail` fails whenever grep
# exits before ldconfig has written everything (SIGPIPE), i.e. on an early match.
have_lib() {
    local cache
    cache="$(ldconfig_cache)"
    case "$cache" in *"$1"*) return 0 ;; esac
    local d
    local IFS=:
    for d in ${LD_LIBRARY_PATH:-}; do
        [ -n "$d" ] && ls "$d/$1"* >/dev/null 2>&1 && return 0
    done
    return 1
}

katago_backend() {
    if [ -n "${KATAGO_BACKEND:-}" ]; then echo "$KATAGO_BACKEND"; return; fi
    if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1 && have_lib libcudnn.so.9 && have_lib libcublas.so.12; then
        echo cuda12.8-cudnn9.8.0; return
    fi
    if ls /etc/OpenCL/vendors/*.icd >/dev/null 2>&1 && have_lib libOpenCL.so && \
        { have nvidia-smi || [ -e /dev/kfd ] || [ -e /dev/dri/renderD128 ]; }; then
        echo opencl; return
    fi
    if has_flag avx2; then echo eigenavx2; else echo eigen; fi
}

# Release tag for backend $1 (KATAGO_TAG overrides). One pinned tag for every Linux
# backend: the KataGo version is part of meta.teacher (`gmdata label go`), and the loader
# refuses to mix shards labelled by different versions — GPU- and CPU-node labels (or two
# installs made months apart) must come from the same release. v1.18.1 is the newest tag
# with CPU / OpenCL / TensorRT / ROCm assets (v1.18.2 ships CUDA builds only).
KATAGO_LINUX_TAG=v1.18.1
katago_tag() {
    if [ -n "${KATAGO_TAG:-}" ]; then echo "$KATAGO_TAG"; return; fi
    echo "$KATAGO_LINUX_TAG"
}

install_katago() {
    local dir exe tag backend asset net_url net configs
    if [ "$OS" = Darwin ]; then
        have brew || die "on macOS, KataGo is installed with Homebrew (brew install katago)"
        if ! brew list katago >/dev/null 2>&1; then brew install katago; fi
        exe="$(brew --prefix)/bin/katago"
        configs="$(brew --prefix)/share/katago/configs"
        dir="$ROOT/katago/brew"
        mkdir -p "$dir"
        link_bin katago "$exe"
    else
        [ "$ARCH" = x86_64 ] || die "KataGo release binaries are x86-64 only; build from source for $ARCH"
        backend="$(katago_backend)"
        tag="$(katago_tag "$backend")"
        asset="katago-$tag-$backend-linux-x64.zip"
        dir="$ROOT/katago/$tag-$backend"
        if [ ! -x "$dir/katago" ]; then
            fetch "https://github.com/lightvector/KataGo/releases/download/$tag/$asset" "$DL/$asset" \
                || die "no $asset (set KATAGO_TAG / KATAGO_BACKEND)"
            rm -rf "$dir.tmp"
            extract "$DL/$asset" "$dir.tmp"
            exe="$(find "$dir.tmp" -type f -name katago | head -1)"
            [ -n "$exe" ] || die "no katago executable in $asset"
            chmod +x "$exe"
            if [ "$exe" != "$dir.tmp/katago" ]; then mv "$(dirname "$exe")"/* "$dir.tmp/" 2>/dev/null || true; fi
            rm -rf "$dir"
            mv "$dir.tmp" "$dir"
        fi
        exe="$dir/katago"
        if have ldd && ldd "$exe" 2>/dev/null | grep -q 'not found'; then
            log "warning: $exe misses shared libraries:"
            ldd "$exe" | grep 'not found' >&2 || true
            log "install them or pick another backend (KATAGO_BACKEND=eigenavx2 / eigen need the fewest)"
        fi
        link_bin katago "$exe"
        configs="$dir"
    fi
    net_url="${KATAGO_NET_URL:-https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz}"
    net="$ROOT/nets/katago/$(basename "$net_url")"
    local bundled=""
    [ "$OS" = Darwin ] && bundled="$(brew --prefix)/share/katago/$(basename "$net_url")"
    if [ ! -s "$net" ] && [ -n "$bundled" ] && [ -s "$bundled" ]; then
        mkdir -p "$(dirname "$net")"
        ln -sfn "$bundled" "$net"
        log "using the network bundled with Homebrew's katago ($bundled)"
    fi
    fetch "$net_url" "$net" || die "cannot download the KataGo network ($net_url)"
    ln -sfn "$net" "$ROOT/nets/katago/default.bin.gz"
    local ver
    ver="$("$BIN/katago" version 2>/dev/null | head -1)" || true
    echo "katago: $BIN/katago -> $exe (${ver:-version unknown}); network $net (KataGo Neural Network License); example configs in $configs"
}

# ------------------------------------------------------------------ main

show_paths() {
    local e
    for e in stockfish egaroucid yaneuraou katago pikafish; do
        if [ -e "$BIN/$e" ]; then echo "$e: $BIN/$e"; else echo "$e: (not installed)"; fi
    done
    [ -e "$ROOT/nets/katago/default.bin.gz" ] && echo "katago network: $(readlink "$ROOT/nets/katago/default.bin.gz")"
    return 0
}

usage() { sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

main() {
    [ $# -gt 0 ] || { usage; exit 2; }
    mkdir -p "$ROOT" "$BIN" "$DL"
    local what
    for what in "$@"; do
        case "$what" in
            stockfish) install_stockfish ;;
            egaroucid) install_egaroucid ;;
            yaneuraou) install_yaneuraou ;;
            katago) install_katago ;;
            pikafish) install_pikafish ;;
            all) install_stockfish; install_egaroucid; install_yaneuraou; install_katago; install_pikafish ;;
            paths) show_paths ;;
            -h | --help | help) usage ;;
            *) usage; die "unknown engine: $what" ;;
        esac
    done
}

# executed (not sourced: tests source the file to check the host-detection functions)
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
