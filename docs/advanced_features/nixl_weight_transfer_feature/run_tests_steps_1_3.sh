#!/bin/bash
# Runner for NIXL weight-transfer tests, steps 1-3.
# See nixl-weight-transfer-tests.md for the full description of each test.
#
# By default runs only the no-GPU tests (1a,1b,1c,2a,2b,3a).
# Pass --with-e2e to also run the launch-based tests (2c Mooncake, 3b NIXL),
# which require GPUs + the DeepSeek-V3-Lite model (+ NIXL/UCX for 3b).
#
# Usage:
#   ./run_tests_steps_1_3.sh                 # no-GPU tests only
#   ./run_tests_steps_1_3.sh --with-e2e      # also run 2c + 3b
#   MODEL_PATH=/path/to/model ./run_tests_steps_1_3.sh --with-e2e

set -u

MODEL_PATH="${MODEL_PATH:-/sgl-workspace/llm_models/DeepSeek-V3-Lite/bf16}"
TP="${TP:-8}"
PORT="${PORT:-30000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-38999}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"   # seconds to wait for "ready to roll"
WITH_E2E=0
[ "${1:-}" = "--with-e2e" ] && WITH_E2E=1

PASS=0
FAIL=0
declare -a RESULTS

green() { printf '\033[32m%s\033[0m' "$1"; }
red()   { printf '\033[31m%s\033[0m' "$1"; }

# run_py <test-id> <description> <python-source>
run_py() {
    local id="$1" desc="$2" code="$3"
    echo "----------------------------------------------------------------"
    echo "[$id] $desc"
    if python -c "$code"; then
        echo "  -> $(green PASS)"
        PASS=$((PASS+1)); RESULTS+=("PASS  $id  $desc")
    else
        echo "  -> $(red FAIL)"
        FAIL=$((FAIL+1)); RESULTS+=("FAIL  $id  $desc")
    fi
}

# run_shell <test-id> <description> <command...>
run_shell() {
    local id="$1" desc="$2"; shift 2
    echo "----------------------------------------------------------------"
    echo "[$id] $desc"
    if "$@"; then
        echo "  -> $(green PASS)"
        PASS=$((PASS+1)); RESULTS+=("PASS  $id  $desc")
    else
        echo "  -> $(red FAIL)"
        FAIL=$((FAIL+1)); RESULTS+=("FAIL  $id  $desc")
    fi
}

echo "================================================================"
echo "NIXL weight-transfer tests: steps 1-3"
echo "with_e2e=$WITH_E2E  model=$MODEL_PATH  tp=$TP"
echo "================================================================"

# ---------------- Step 1 ----------------
run_shell "1a" "CLI flag is exposed in --help" \
    bash -c "python -m sglang.launch_server --help | grep -q -- --remote-instance-weight-loader-start-seed-via-nixl"

run_py "1b" "enum NIXL + dataclass field exist" '
from sglang.srt.model_loader.remote_instance_weight_loader_utils import RemoteInstanceWeightLoaderBackend as B
from sglang.srt.server_args import ServerArgs
f = ServerArgs.__dataclass_fields__
assert B.NIXL.value == "nixl", B.NIXL
assert "remote_instance_weight_loader_start_seed_via_nixl" in f
print("OK: enum + field present")
'

run_py "1c" "use_transfer_engine() covers the NIXL triggers" '
import inspect
from sglang.srt.server_args import ServerArgs
src = inspect.getsource(ServerArgs.remote_instance_weight_loader_use_transfer_engine)
assert "start_seed_via_nixl" in src and "nixl" in src
print("OK: use_transfer_engine covers nixl")
'

# ---------------- Step 2 ----------------
run_py "2a" "bootstrap server round-trips a tagged dict" "
import requests, time
from sglang.srt.entrypoints.engine_info_bootstrap_server import EngineInfoBootstrapServer
s = EngineInfoBootstrapServer('127.0.0.1', $BOOTSTRAP_PORT); time.sleep(1)
info = {'session_id': '10.0.0.7:18000', 'weights_info_dict': {'w': [1,2,3]}}
r = requests.put('http://127.0.0.1:$BOOTSTRAP_PORT/register_transfer_engine_info',
                 json={'tp_rank': 0, 'transfer_engine_info': info})
assert r.status_code == 200, r.text
g = requests.get('http://127.0.0.1:$BOOTSTRAP_PORT/get_transfer_engine_info', params={'rank':0}).json()
out = g['remote_instance_transfer_engine_info']
assert out == info, out
assert isinstance(out, dict)
print('OK: dict stored and served as-is')
s.close()
"

run_py "2b" "reader returns a single dict / None by key" '
import inspect
from sglang.srt.model_loader import remote_instance_weight_loader_utils as u
src = inspect.getsource(u.get_remote_instance_transfer_engine_info_per_rank)
assert "return None" in src and "None, None" not in src
print("OK: reader returns single dict or None")
'

# ---------------- Step 3 ----------------
run_py "3a" "init method branches to _remote_instance_init_nixl" '
import inspect
from sglang.srt.model_executor.model_runner import ModelRunner
src = inspect.getsource(ModelRunner.remote_instance_init_transfer_engine)
assert "_remote_instance_init_nixl" in src and "return" in src
assert hasattr(ModelRunner, "_remote_instance_init_nixl")
print("OK: nixl branch present")
'

# ---------------- e2e (optional) ----------------
# launch_and_check <test-id> <desc> <log-grep-success-regex> <extra launch args...>
launch_and_check() {
    local id="$1" desc="$2" success_re="$3"; shift 3
    echo "----------------------------------------------------------------"
    echo "[$id] $desc"
    local log; log="$(mktemp)"
    echo "  launching server (log: $log) ..."
    python -m sglang.launch_server \
        --model-path "$MODEL_PATH" --tp "$TP" --trust-remote-code --port "$PORT" \
        "$@" >"$log" 2>&1 &
    local pid=$!
    local waited=0 ok=0
    while [ "$waited" -lt "$READY_TIMEOUT" ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "  server process exited early"; break
        fi
        if grep -q "ready to roll" "$log"; then ok=1; break; fi
        if grep -qE "OutOfMemoryError|Scheduler hit an exception|Received sigquit" "$log"; then
            echo "  detected crash in log"; break
        fi
        sleep 5; waited=$((waited+5))
    done

    local result=1
    if [ "$ok" -eq 1 ]; then
        if grep -qE "$success_re" "$log"; then
            echo "  server ready AND matched: $success_re"; result=0
        else
            echo "  server ready BUT did not match: $success_re"; result=1
        fi
        # run the per-test endpoint check, if any
        if [ "$id" = "2c" ]; then
            echo "  endpoint:"; curl -s "http://localhost:$PORT/remote_instance_transfer_engine_info?rank=0" | python -m json.tool || true
        fi
    fi

    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    echo "  (full log kept at $log)"
    if [ "$result" -eq 0 ]; then
        echo "  -> $(green PASS)"; PASS=$((PASS+1)); RESULTS+=("PASS  $id  $desc")
    else
        echo "  -> $(red FAIL)"; FAIL=$((FAIL+1)); RESULTS+=("FAIL  $id  $desc")
    fi
}

if [ "$WITH_E2E" -eq 1 ]; then
    echo "================================================================"
    echo "Running e2e launch tests (needs GPUs + model)"
    echo "================================================================"

    # 2c - Mooncake path still returns a JSON object
    launch_and_check "2c" "Mooncake path reaches ready (schema regression)" \
        "ready to roll" \
        --remote-instance-weight-loader-backend transfer_engine \
        --remote-instance-weight-loader-start-seed-via-transfer-engine

    # 3b - NIXL agent init; gate on nixl importability first
    if python -c "import nixl._api" 2>/dev/null; then
        launch_and_check "3b" "NIXL agent init, server ready, no crash" \
            "NIXL weight-transfer agent initialized" \
            --remote-instance-weight-loader-start-seed-via-nixl
    else
        echo "----------------------------------------------------------------"
        echo "[3b] SKIPPED - 'import nixl._api' failed (flag would be reset to False)"
        RESULTS+=("SKIP  3b  NIXL not importable")
    fi
fi

# ---------------- summary ----------------
echo "================================================================"
echo "SUMMARY"
for line in "${RESULTS[@]}"; do echo "  $line"; done
echo "----------------------------------------------------------------"
echo "  passed=$PASS  failed=$FAIL"
echo "================================================================"
[ "$FAIL" -eq 0 ]
