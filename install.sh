#!/bin/sh
# curl -fsSL https://snowllm.dev/install.sh | sh
#
# Options:
#   --uninstall        remove SnowLLM
#   --help
#
# Env:
#   SNOWLLM_HOME       where the environment lives        (default: $XDG_DATA_HOME/snowllm)
#   SNOWLLM_BIN_DIR    where the snowllm command goes     (default: $XDG_BIN_HOME)
#   SNOWLLM_PYTHON     interpreter to build the venv with (default: autodetected)
#   SNOWLLM_ALLOW_WSL  proceed on WSL2, which nobody has verified

# Wrapped in main() so a truncated download cannot execute half a script.
main() {

set -eu

TORCH="torch[device-gfx1151]==2.12.0+rocm7.14.0"
TORCHVISION="torchvision==0.27.0+rocm7.14.0"
ROCM_INDEX="https://repo.amd.com/rocm/whl-multi-arch/"
GFX_TARGET=110501
NEED_GIB=15

SNOWLLM_HOME="${SNOWLLM_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/snowllm}"
BIN_DIR="${SNOWLLM_BIN_DIR:-${XDG_BIN_HOME:-$HOME/.local/bin}}"
VENV="$SNOWLLM_HOME/venv"
PYTHON=

say() { printf '==> %s\n' "$*" >&2; }
warn() { printf 'install.sh: %s\n' "$*" >&2; }
die() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

usage() {
    cat >&2 <<EOF
Install SnowLLM into $VENV and link it into $BIN_DIR.

    --uninstall    remove both
    --help         this

Env: SNOWLLM_HOME SNOWLLM_BIN_DIR SNOWLLM_PYTHON
EOF
}

uninstall() {
    [ -e "$SNOWLLM_HOME" ] || [ -L "$BIN_DIR/snowllm" ] || die "SnowLLM is not installed under $SNOWLLM_HOME."
    case "$(readlink "$BIN_DIR/snowllm" 2>/dev/null)" in
        "$VENV"/*) rm -f "$BIN_DIR/snowllm" ;;
    esac
    rm -rf "$SNOWLLM_HOME"
    say "removed $SNOWLLM_HOME"
    exit 0
}

nearest_dir() {
    d="$1"
    while [ ! -d "$d" ]; do d=$(dirname "$d"); done
    printf '%s\n' "$d"
}

usable_python() {
    [ -n "$1" ] && [ -x "$1" ] || return 1
    "$1" - >/dev/null 2>&1 <<'EOF' || return 1
import sys, venv, ensurepip
raise SystemExit(0 if (3, 10) <= sys.version_info < (3, 15) else 1)
EOF
}

find_python() {
    for c in /usr/bin/python3.12 /usr/bin/python3.13 /usr/bin/python3.11 /usr/bin/python3.14 \
             /usr/bin/python3.10 python3.12 python3.13 python3.11 python3.14 python3.10 python3; do
        p=$(command -v "$c" 2>/dev/null) || continue
        usable_python "$p" || continue
        printf '%s\n' "$p"
        return 0
    done
    return 1
}

no_python() {
    printf 'install.sh: no usable CPython found. SnowLLM needs 3.10-3.14 with the venv module.\n' >&2
    if have apt-get; then
        printf '\nDebian and Ubuntu ship venv separately:\n\n    sudo apt install python3-venv\n\n' >&2
    fi
    printf 'Then re-run, or point the script at one: SNOWLLM_PYTHON=/path/to/python3\n' >&2
    exit 1
}

is_wsl() {
    case "$(uname -r)" in *icrosoft*|*WSL*|*wsl*) return 0 ;; esac
    return 1
}

wsl_note() {
    cat >&2 <<'EOF'
install.sh: this looks like WSL2, which SnowLLM has never been tested on.

WSL2 has no /dev/kfd -- ROCm reaches the GPU through /dev/dxg instead -- so the
GPU checks below cannot tell you whether yours will work. Set WSL up with AMD's
guide first:

  https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/wsl/howto_wsl.html

If ROCm then reports far less memory than the machine has, check that Resizable
BAR is enabled in the BIOS. To try anyway:

    SNOWLLM_ALLOW_WSL=1 curl -fsSL https://snowllm.dev/install.sh | sh

Native Linux is the supported path.
EOF
    exit 1
}

check_machine() {
    [ "$(uname -s)" = Linux ] || die "SnowLLM runs on Linux only (this is $(uname -s))."
    [ "$(uname -m)" = x86_64 ] || die "SnowLLM runs on x86_64 only (this is $(uname -m))."
    have curl || die "curl is required."

    if is_wsl; then
        [ -n "${SNOWLLM_ALLOW_WSL:-}" ] || wsl_note
        warn "WSL2 detected. This is untested; the GPU checks below are being skipped."
        [ -c /dev/dxg ] || die \
            "no /dev/dxg: this distro cannot reach the GPU. Set WSL up with AMD's guide first."
        return 0
    fi

    [ -c /dev/kfd ] || die "no /dev/kfd: the amdgpu driver is not loaded, or there is no AMD GPU here."
    { [ -r /dev/kfd ] && [ -w /dev/kfd ]; } || die \
        "/dev/kfd is not readable by $(id -un). Run: sudo usermod -aG render,video $(id -un), then log out and back in."

    ok=""
    seen=""
    for p in /sys/class/kfd/kfd/topology/nodes/*/properties; do
        [ -r "$p" ] || continue
        v=$(awk '$1 == "gfx_target_version" { print $2 }' "$p")
        case "${v:-0}" in 0|"") continue ;; esac
        seen="$seen $v"
        if [ "$v" = "$GFX_TARGET" ]; then ok=1; fi
    done
    [ -n "$ok" ] || die \
        "no gfx1151 GPU found (kfd reports:${seen:- nothing}). SnowLLM's kernels are compiled for gfx1151 (Ryzen AI Max 300 series) and run nowhere else."
}

check_python() {
    if [ -n "${SNOWLLM_PYTHON:-}" ]; then
        usable_python "$SNOWLLM_PYTHON" || die \
            "$SNOWLLM_PYTHON is not usable: SnowLLM needs CPython 3.10-3.14 with the venv module."
        PYTHON="$SNOWLLM_PYTHON"
    else
        PYTHON=$(find_python) || no_python
    fi
}

check_dirs() {
    for d in "$SNOWLLM_HOME" "$BIN_DIR"; do
        near=$(nearest_dir "$d")
        [ -w "$near" ] || die \
            "$near is not writable by $(id -un). Set SNOWLLM_HOME and SNOWLLM_BIN_DIR, or fix its permissions."
    done

    if [ -e "$BIN_DIR/snowllm" ]; then
        case "$(readlink "$BIN_DIR/snowllm" 2>/dev/null)" in
            "$VENV"/*) ;;
            *) warn "$BIN_DIR/snowllm already exists and is not ours; it will be replaced." ;;
        esac
    fi

    near=$(nearest_dir "$SNOWLLM_HOME")
    free=$(df -Pk "$near" | awk 'NR == 2 { print int($4 / 1048576) }')
    [ "$free" -ge "$NEED_GIB" ] || die \
        "only ${free}G free on $near; torch and the ROCm libraries need about ${NEED_GIB}G. Set SNOWLLM_HOME to a roomier disk."
}

http_code() {
    url=$1
    shift
    code=$(curl -sS --max-time 15 --retry 2 -o /dev/null -w '%{http_code}' "$@" "$url" 2>/dev/null) || code=000
    printf '%s\n' "$code"
}

check_network() {
    code=$(http_code "$ROCM_INDEX" -I)
    [ "$code" = 200 ] || die \
        "cannot reach $ROCM_INDEX (HTTP $code) -- that is where the ROCm build of torch comes from. Check your network or proxy."

    for pkg in snowllm snowllm-kernels; do
        code=$(http_code "https://pypi.org/simple/$pkg/")
        case "$code" in
            200) ;;
            404) die "$pkg is not on PyPI. See https://pypi.org/project/$pkg/" ;;
            *) die "cannot reach PyPI (HTTP $code). Check your network or proxy." ;;
        esac
    done
}

pip_install() {
    "$VENV/bin/python" -m pip install --quiet --disable-pip-version-check "$@"
}

case "${1:-}" in
    --help|-h) usage; exit 0 ;;
    --uninstall) uninstall ;;
    "") ;;
    *) usage; exit 2 ;;
esac

check_machine
check_python
check_dirs
check_network
say "gfx1151 ok, ${free}G free, python $("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])') at $PYTHON"

mkdir -p "$SNOWLLM_HOME"
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c '' 2>/dev/null; then
    say "reusing $VENV"
else
    say "building $VENV"
    rm -rf "$VENV"
    "$PYTHON" -m venv "$VENV"
fi
pip_install --upgrade pip

say "installing torch from repo.amd.com (a few gigabytes)"
pip_install --index-url "$ROCM_INDEX" "$TORCH" "$TORCHVISION"

say "installing snowllm"
pip_install --upgrade snowllm snowllm-kernels

mkdir -p "$BIN_DIR"
ln -sf "$VENV/bin/snowllm" "$BIN_DIR/snowllm"

printf '\n'
"$VENV/bin/python" -c 'from snowllm._version import __version__; print("snowllm", __version__)'
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) printf '\n%s is not on your PATH. Add it:\n\n    export PATH="%s:$PATH"\n' "$BIN_DIR" "$BIN_DIR" ;;
esac
cat <<EOF

Get a model, then serve it:

    hf download Qwen/Qwen3.6-35B-A3B-FP8 --local-dir ~/models/Qwen3.6-35B-A3B-FP8
    snowllm ~/models/Qwen3.6-35B-A3B-FP8

Uninstall:  curl -fsSL https://snowllm.dev/install.sh | sh -s -- --uninstall
EOF

}

main "$@"
