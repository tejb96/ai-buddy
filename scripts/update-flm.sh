#!/usr/bin/env bash
# Update FastFlowLM (the NPU engine behind Lemonade's *-FLM models) to the latest stable release.
#
#   scripts/update-flm.sh            check, show the release notes, install on confirmation, smoke test
#   scripts/update-flm.sh --check    only report whether a newer stable release exists
#   scripts/update-flm.sh --rollback switch back to the previous FLM (or to Lemonade's bundled one)
#
# Only standard (XRT) stable releases are considered: drafts, pre-releases and the HRX preview packages
# are skipped. Nothing runs automatically or on a schedule.
#
# Layout (needs sudo, asked for when installing):
#   /opt/fastflowlm/<version>/   the unpacked portable package (fastflowlm_<version>_linux.tar.gz)
#   /opt/fastflowlm/current      symlink to the version in use; /opt/fastflowlm/previous to the one before
#   /usr/local/bin/flm           tiny wrapper that runs /opt/fastflowlm/current/flm (lemond's PATH has it)
# Lemonade uses it through its `flm.prefer_system` setting. Model weights stay where Lemonade keeps them.
set -euo pipefail

REPO="FastFlowLM/FastFlowLM"
PREFIX="/opt/fastflowlm"
WRAPPER="/usr/local/bin/flm"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="$ROOT/.venv/bin/python"
LEMONADE_URL="${LEMONADE_URL:-http://localhost:13305/v1}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

say() { printf '\033[1m%s\033[0m\n' "$*"; }
die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

# Version Lemonade currently runs for the flm recipe, e.g. 1.0.6.
lemonade_flm_version() { lemonade backends 2>/dev/null | awk '$1 == "flm" {print $4}' | sed 's/^v//'; }
prefer_system() { lemonade config 2>/dev/null | awk '$1 == "flm.prefer_system" {print $2}'; }

releases_json() {
    if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
        gh api "repos/$REPO/releases?per_page=30"
    else
        curl -fsSL "https://api.github.com/repos/$REPO/releases?per_page=30"
    fi
}

# Latest stable release with a standard Linux portable package: prints "<tag> <asset url>".
latest_stable() {
    releases_json | python3 -I -c '
import json, sys
for r in json.load(sys.stdin):
    if r["draft"] or r["prerelease"]:
        continue
    tag = r["tag_name"]
    want = "fastflowlm_" + tag.lstrip("v") + "_linux.tar.gz"   # never the HRX_ preview packages
    for a in r["assets"]:
        if a["name"] == want:
            print(tag, a["browser_download_url"]); sys.exit(0)
sys.exit(1)'
}

# Release notes of every stable release newer than $1 up to and including $2.
notes_between() {
    releases_json | python3 -I -c '
import json, sys
def v(t): return tuple(int(x) for x in t.lstrip("v").split("."))
lo, hi = v(sys.argv[1]), v(sys.argv[2])
for r in sorted(json.load(sys.stdin), key=lambda r: v(r["tag_name"])):
    if not (r["draft"] or r["prerelease"]) and lo < v(r["tag_name"]) <= hi:
        print("=" * 78); print(r["tag_name"], "published", r["published_at"][:10]); print("=" * 78)
        print(r["body"]); print()' "$1" "$2"
}

version_gt() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" = "$1" ] && [ "$1" != "$2" ]; }

unload_all() { curl -fsS -X POST "$LEMONADE_URL/unload" -H 'Content-Type: application/json' -d '{}' >/dev/null || true; }

smoke() {
    say "Smoke test: load Kira's default models, one spoken turn, one tool call"
    [ -x "$PY" ] || die "No project environment at $PY (run 'uv sync' first)."
    "$PY" "$HERE/flm_smoke.py"
}

rollback() {
    say "Rolling back"
    unload_all
    if [ -L "$PREFIX/previous" ]; then
        local prev; prev="$(readlink "$PREFIX/previous")"
        sudo ln -sfn "$prev" "$PREFIX/current"
        sudo rm -f "$PREFIX/previous"
        echo "FLM is back to $(basename "$prev") (system install)."
    else
        lemonade config set flm.prefer_system=false >/dev/null
        echo "Lemonade is back to its bundled FLM (flm.prefer_system=false)."
    fi
    unload_all
    echo "Lemonade now reports FLM $(lemonade_flm_version)."
}

case "${1:-}" in
    --rollback) rollback; exit 0 ;;
    --check|"") ;;
    *) die "Usage: $0 [--check|--rollback]" ;;
esac

command -v lemonade >/dev/null || die "lemonade CLI not found."
lemonade status >/dev/null 2>&1 || die "Lemonade server is not running."
current="$(lemonade_flm_version)"
[ -n "$current" ] || die "Could not read the FLM version from 'lemonade backends'."
read -r tag url < <(latest_stable) || die "Could not find a stable FastFlowLM release on GitHub."
latest="${tag#v}"
echo "Lemonade uses FLM $current (flm.prefer_system=$(prefer_system)); latest stable release: $latest"
if ! version_gt "$latest" "$current"; then
    echo "Already up to date."
    exit 0
fi
[ "${1:-}" = "--check" ] && exit 0

notes_between "$current" "$latest" | ${PAGER:-less -R}
echo
say "Check the notes above for breaking changes: renamed model tags, weights to re-pull, new requirements."
read -r -p "Install FastFlowLM $latest for Lemonade now? [y/N] " answer
[[ "$answer" =~ ^[Yy]$ ]] || { echo "Nothing changed."; exit 0; }

say "Downloading $(basename "$url")"
curl -fL --progress-bar -o "$WORK/flm.tar.gz" "$url"
mkdir "$WORK/pkg"
tar -xzf "$WORK/flm.tar.gz" -C "$WORK/pkg"
[ -x "$WORK/pkg/flm" ] && [ -x "$WORK/pkg/flm-real" ] || die "Unexpected package layout; nothing changed."
"$WORK/pkg/flm" version || true

say "Installing to $PREFIX/$latest (sudo)"
sudo mkdir -p "$PREFIX"
sudo rm -rf "$PREFIX/$latest"
sudo cp -a "$WORK/pkg" "$PREFIX/$latest"
sudo chown -R root:root "$PREFIX/$latest"
sudo chmod -R a+rX "$PREFIX/$latest"
if [ -L "$PREFIX/current" ]; then
    sudo ln -sfn "$(readlink "$PREFIX/current")" "$PREFIX/previous"
else
    sudo rm -f "$PREFIX/previous"
fi
sudo ln -sfn "$PREFIX/$latest" "$PREFIX/current"
# A wrapper, not a symlink: FLM's own launcher finds its libraries next to the path it was started from.
printf '#!/bin/sh\nexec %s/current/flm "$@"\n' "$PREFIX" | sudo tee "$WRAPPER" >/dev/null
sudo chmod 755 "$WRAPPER"

say "Pointing Lemonade at the system FLM"
unload_all
lemonade config set flm.prefer_system=true >/dev/null
now="$(lemonade_flm_version)"
echo "Lemonade now reports FLM $now."
if [ "$now" != "$latest" ]; then
    echo "Lemonade did not pick up $latest."
    rollback
    exit 1
fi

if smoke; then
    say "FastFlowLM $latest is installed and working. Undo with: $0 --rollback"
else
    say "Smoke test failed."
    rollback
    exit 1
fi
