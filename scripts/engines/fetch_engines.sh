#!/usr/bin/env bash
# fetch_engines.sh — install the external engines used by `gmdata label` (engine
# teacher labels for gm-shards), OUTSIDE the repository.
#
# Usage:
#   scripts/engines/fetch_engines.sh stockfish     # official release binary (best build for this CPU)
#   scripts/engines/fetch_engines.sh egaroucid     # Othello: clone + build the console engine
#   scripts/engines/fetch_engines.sh yaneuraou     # Shogi: clone + build NNUE edition + download eval
#   scripts/engines/fetch_engines.sh katago        # Go: release binary (backend by host) + network
#   scripts/engines/fetch_engines.sh all           # all of the above
#   scripts/engines/fetch_engines.sh paths         # print what is installed
#
# Layout (root = ${GM_ENGINES:-$HOME/gm-engines}):
#   bin/stockfish  bin/egaroucid  bin/yaneuraou  bin/katago   entry points (symlinks / small
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
#   Egaroucid  GPL-3.0. Built from source, github.com/Nyanyan/Egaroucid; its evaluation file
#              (bin/resources/eval.egev2) ships in the same repository under the same licence.
#   YaneuraOu  GPL-3.0. Built from source, github.com/yaneurao/YaneuraOu (tag v9.40 with two
#              small source patches, see yaneuraou_patch: the upstream macOS isready-deadlock
#              fix f129339a, and a ~12 ms instead of ~150 ms isready), standard NNUE edition
#              (halfkp_256x2-32-32). Eval: "Háo" (tanuki- project, release
#              tanuki-.halfkp_256x2-32-32.2023-05-08 at github.com/nodchip/tanuki-, a free
#              download; the archive ships gpl-3.0.txt: GPL-3.0; about as strong as tanuki-'s
#              WCSC33 eval "Lí" per its authors; needs the USI option FV_SCALE=20, which
#              `gmdata label shogi` sets by default).
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
#   EGAROUCID_REF          git tag / branch (default: latest release tag)
#   YANEURAOU_REF          git tag / branch (default: v9.40, pinned: labels record the engine
#                          version, and later versions may drop the halfkp_256x2-32-32 edition)
#   YANEURAOU_TARGET_CPU   Makefile TARGET_CPU (default: from the CPU flags)
#   YANEURAOU_EVAL_URL     .7z / .zip archive containing nn.bin (default: Háo)
#   KATAGO_TAG             release tag (default: v1.18.1 for CPU/OpenCL builds — later
#                          tags so far ship CUDA builds only — and latest for CUDA)
#   KATAGO_BACKEND         eigenavx2 | eigen | opencl | onnx-openvino | cuda12.8-cudnn9.8.0 | ...
#                          (default: CUDA if an NVIDIA GPU + cuDNN 9 are present, OpenCL if a
#                          GPU OpenCL ICD is present, else eigenavx2 / eigen — the CPU builds
#                          are static-pie executables, so they need no extra system libraries)
#   KATAGO_NET_URL         network URL (default: kata1 b18c384nbt, the fast choice on CPUs)
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
is_amd() { [ -r /proc/cpuinfo ] && grep -q -m1 AuthenticAMD /proc/cpuinfo; }

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

install_egaroucid() {
    have git || die "egaroucid needs git"
    local ref dir exe cxx defs=""
    ref="${EGAROUCID_REF:-$(latest_tag Nyanyan/Egaroucid || echo main)}"
    dir="$ROOT/egaroucid/$ref"
    exe="$dir/bin/Egaroucid_for_Console.out"
    if [ -x "$exe" ]; then
        log "Egaroucid $ref already built"
    else
        rm -rf "$dir.tmp"
        git clone --depth 1 --branch "$ref" https://github.com/Nyanyan/Egaroucid.git "$dir.tmp"
        case "$ARCH" in
            arm64 | aarch64) defs="-DHAS_ARM_PROCESSOR" ;;
            *)
                if ! has_flag avx2; then defs="-DHAS_NO_AVX2"
                elif has_flag avx512bw && has_flag avx512vl && has_flag avx512dq; then defs="-DHAS_AVX512"; fi
                if is_amd; then defs="$defs -DHAS_AMD_PROCESSOR"; fi ;;
        esac
        if have g++ && [ "$OS" = Linux ]; then cxx=g++; elif have clang++; then cxx=clang++; else cxx=c++; fi
        local march="-march=native"
        [ "$OS" = Darwin ] && march=""
        log "building Egaroucid $ref with $cxx $defs"
        # the console engine is a single translation unit (what its CMakeLists builds)
        # shellcheck disable=SC2086  # $march / $defs are intentionally word-split flag lists
        (cd "$dir.tmp/src" && $cxx -O2 -mtune=native $march -pthread -std=c++20 $defs \
            Egaroucid_for_Console.cpp -o ../bin/Egaroucid_for_Console.out)
        rm -rf "$dir"
        mv "$dir.tmp" "$dir"
    fi
    [ -f "$dir/bin/resources/eval.egev2" ] || log "warning: $dir/bin/resources/eval.egev2 missing"
    wrap_bin egaroucid "$dir/bin" ./Egaroucid_for_Console.out
    echo "egaroucid: $BIN/egaroucid -> $exe (eval $dir/bin/resources/eval.egev2; GPL-3.0)"
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

# ------------------------------------------------------------------ KataGo

katago_backend() {
    if [ -n "${KATAGO_BACKEND:-}" ]; then echo "$KATAGO_BACKEND"; return; fi
    if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libcudnn.so.9'; then
        echo cuda12.8-cudnn9.8.0; return
    fi
    if ls /etc/OpenCL/vendors/*.icd >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q libOpenCL.so && \
        { have nvidia-smi || [ -e /dev/kfd ] || [ -e /dev/dri/renderD128 ]; }; then
        echo opencl; return
    fi
    if has_flag avx2; then echo eigenavx2; else echo eigen; fi
}

install_katago() {
    local dir exe tag backend asset net_url net
    if [ "$OS" = Darwin ]; then
        have brew || die "on macOS, KataGo is installed with Homebrew (brew install katago)"
        if ! brew list katago >/dev/null 2>&1; then brew install katago; fi
        exe="$(brew --prefix)/bin/katago"
        dir="$ROOT/katago/brew"
        mkdir -p "$dir"
        link_bin katago "$exe"
    else
        [ "$ARCH" = x86_64 ] || die "KataGo release binaries are x86-64 only; build from source for $ARCH"
        backend="$(katago_backend)"
        case "$backend" in
            eigen* | opencl | onnx* | rocm*) tag="${KATAGO_TAG:-v1.18.1}" ;;
            *) tag="${KATAGO_TAG:-$(latest_tag lightvector/KataGo || echo v1.18.2)}" ;;
        esac
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
    fi
    net_url="${KATAGO_NET_URL:-https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz}"
    net="$ROOT/nets/katago/$(basename "$net_url")"
    fetch "$net_url" "$net" || die "cannot download the KataGo network ($net_url)"
    ln -sfn "$net" "$ROOT/nets/katago/default.bin.gz"
    local ver
    ver="$("$BIN/katago" version 2>/dev/null | head -1)" || true
    echo "katago: $BIN/katago -> $exe (${ver:-version unknown}); network $net (KataGo Neural Network License); configs in $(dirname "$exe")"
}

# ------------------------------------------------------------------ main

show_paths() {
    local e
    for e in stockfish egaroucid yaneuraou katago; do
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
            all) install_stockfish; install_egaroucid; install_yaneuraou; install_katago ;;
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
