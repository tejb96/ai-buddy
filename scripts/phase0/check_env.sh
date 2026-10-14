#!/usr/bin/env bash
# Phase 0 environment check: Lemonade server, NPU, FLM, and required models.
set -u
URL="${LEMONADE_URL:-http://localhost:13305/v1}"
FLM=/var/cache/lemonade/bin/flm/npu/flm
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$*"; }

echo "Lemonade server"
if curl -sf "$URL/health" >/dev/null; then ok "server up at $URL ($(lemonade --version 2>/dev/null))"; else bad "server not reachable at $URL (sudo systemctl start lemond)"; fi

echo "NPU"
[ -e /dev/accel/accel0 ] && ok "/dev/accel/accel0 present" || bad "/dev/accel/accel0 missing (amdxdna driver not loaded)"
id -nG | grep -qw render && ok "user in 'render' group" || warn "user not in 'render' group"
npu=$(curl -s "$URL/system-info" | python3 -c "import json,sys;n=json.load(sys.stdin)['devices']['amd_npu'];print(n['available'],n['family'])" 2>/dev/null)
[ "${npu%% *}" = "True" ] && ok "Lemonade sees NPU: ${npu#* }" || bad "Lemonade does not report an available NPU"
if [ -x "$FLM" ]; then
  ok "FLM $("$FLM" version 2>/dev/null) at $FLM"
  FLM_DISABLE_UPDATE_CHECK=1 "$FLM" validate 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | sed 's/^/         /'
  echo "         (a memlock error above refers to YOUR shell; FLM under lemond raises its own limit)"
else
  bad "FLM backend not installed (lemonade backends install flm:npu)"
fi

echo "Models"
for m in whisper-v3-turbo-FLM qwen3-it-4b-FLM qwen3.6-moe-35b-a3b-FLM kokoro-v1; do
  lemonade list --downloaded 2>/dev/null | grep -q "^$m " && ok "$m downloaded" || bad "$m missing (lemonade pull $m)"
done
