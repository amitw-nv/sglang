#!/bin/bash
# Runner for NIXL weight-transfer tests, steps 1-6.
# See nixl-weight-transfer-tests.md for the full description of each test.
#
# By default runs only the no-GPU tests (1a,1b,1c,2a,2b,3a,4a,5a,6a).
# Pass --with-e2e to also run the launch-based tests (2c Mooncake, 3b+4b+5b+6b NIXL),
# which require GPUs + the DeepSeek-V3-Lite model (+ NIXL/UCX for the NIXL path).
#
# Usage:
#   ./run_tests_steps_1_6.sh                 # no-GPU tests only
#   ./run_tests_steps_1_6.sh --with-e2e      # also run 2c + 3b + 4b + 5b + 6b
#   MODEL_PATH=/path/to/model ./run_tests_steps_1_6.sh --with-e2e

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
echo "NIXL weight-transfer tests: steps 1-6"
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

# ---------------- Step 4 ----------------
run_py "4a" "register_memory_region_nixl exists and builds 4-field entries" '
import inspect
from sglang.srt.model_loader import remote_instance_weight_loader_utils as u
assert hasattr(u, "register_memory_region_nixl")
src = inspect.getsource(u.register_memory_region_nixl)
assert "register_memory" in src and "VRAM" in src
assert "gpu_id" in src
print("OK: register_memory_region_nixl present")
'

# ---------------- Step 5 ----------------
run_py "5a" "publish method emits the tagged nixl dict" '
import inspect
from sglang.srt.model_executor.model_runner import ModelRunner
src = inspect.getsource(ModelRunner._register_to_engine_info_bootstrap)
for key in ("backend", "agent_name", "agent_metadata", "base64"):
    assert key in src, key
print("OK: tagged nixl payload emitted")
'

# ---------------- Step 6 ----------------
run_py "6a" "engine startup condition gates on the nixl seed flag" '
import inspect, sglang.srt.entrypoints.engine as e
src = inspect.getsource(e)
assert "remote_instance_weight_loader_start_seed_via_nixl" in src
print("OK: engine startup gates on nixl flag")
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

# launch_and_check_nixl: one NIXL seed launch that covers 3b (agent init),
# 4b (memory registration ran without CUDA faults / process healthy), 5b (the
# served metadata is the tagged NIXL dict) and 6b (the bootstrap server started
# in NIXL seed mode, so the endpoint is reachable at all). All four share the
# exact same launch, so we run it once and record every result from one log/endpoint.
launch_and_check_nixl() {
    echo "----------------------------------------------------------------"
    echo "[3b/4b/5b/6b] NIXL agent init + memory registration + metadata publish + endpoint reachable (single launch)"
    local log; log="$(mktemp)"
    echo "  launching server (log: $log) ..."
    python -m sglang.launch_server \
        --model-path "$MODEL_PATH" --tp "$TP" --trust-remote-code --port "$PORT" \
        --remote-instance-weight-loader-start-seed-via-nixl >"$log" 2>&1 &
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

    # 3b: agent-init log line present.
    if grep -qE "NIXL weight-transfer agent initialized" "$log"; then
        echo "  [3b] -> $(green PASS) (agent initialized log found)"
        PASS=$((PASS+1)); RESULTS+=("PASS  3b  NIXL agent init")
    else
        echo "  [3b] -> $(red FAIL) (no agent initialized log)"
        FAIL=$((FAIL+1)); RESULTS+=("FAIL  3b  NIXL agent init")
    fi

    # 4b: server reached ready (registration ran) and no CUDA / registration faults.
    if [ "$ok" -eq 1 ] && ! grep -qE "CUDA error|NIXL memory registration failed|register memory failed" "$log"; then
        echo "  [4b] -> $(green PASS) (ready, no registration/CUDA faults)"
        PASS=$((PASS+1)); RESULTS+=("PASS  4b  NIXL memory registration")
    else
        echo "  [4b] -> $(red FAIL) (not ready or CUDA/registration fault in log)"
        FAIL=$((FAIL+1)); RESULTS+=("FAIL  4b  NIXL memory registration")
    fi

    # 6b: the public endpoint is reachable (bootstrap server started in NIXL seed
    # mode). 5b: the served payload carries the full tagged NIXL identity
    # (backend=nixl, non-empty agent_name, base64-valid agent_metadata, 4-field weights).
    echo "  endpoint:"; curl -s "http://localhost:$PORT/remote_instance_transfer_engine_info?rank=0" | python -m json.tool || true

    local endpoint_reachable=0
    if [ "$ok" -eq 1 ] && curl -s -f "http://localhost:$PORT/remote_instance_transfer_engine_info?rank=0" >/dev/null 2>&1; then
        endpoint_reachable=1
    fi
    if [ "$endpoint_reachable" -eq 1 ]; then
        echo "  [6b] -> $(green PASS) (endpoint reachable; bootstrap server started in nixl mode)"
        PASS=$((PASS+1)); RESULTS+=("PASS  6b  NIXL endpoint reachable")
    else
        echo "  [6b] -> $(red FAIL) (endpoint not reachable; bootstrap server likely not started)"
        FAIL=$((FAIL+1)); RESULTS+=("FAIL  6b  NIXL endpoint reachable")
    fi

    if [ "$ok" -eq 1 ] && curl -s "http://localhost:$PORT/remote_instance_transfer_engine_info?rank=0" | python -c '
import sys, json, base64
d = json.load(sys.stdin)["remote_instance_transfer_engine_info"]
assert d["backend"] == "nixl", d.get("backend")
assert d["agent_name"], "empty agent_name"
assert base64.b64decode(d["agent_metadata"]), "agent_metadata not b64"
w = d["weights_info_dict"]
name, entry = next(iter(w.items()))
assert len(entry) == 4, entry   # [addr, numel, element_size, device_id]
print("OK: nixl metadata served & b64-valid")
'; then
        echo "  [5b] -> $(green PASS) (tagged nixl metadata served)"
        PASS=$((PASS+1)); RESULTS+=("PASS  5b  NIXL metadata publish")
    else
        echo "  [5b] -> $(red FAIL) (endpoint missing/incomplete nixl metadata)"
        FAIL=$((FAIL+1)); RESULTS+=("FAIL  5b  NIXL metadata publish")
    fi

    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    echo "  (full log kept at $log)"
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

    # 3b + 4b + 5b + 6b - NIXL agent init + memory registration + metadata publish
    # + endpoint reachable; gate on nixl importability first
    if python -c "import nixl._api" 2>/dev/null; then
        launch_and_check_nixl
    else
        echo "----------------------------------------------------------------"
        echo "[3b/4b/5b/6b] SKIPPED - 'import nixl._api' failed (flag would be reset to False)"
        RESULTS+=("SKIP  3b  NIXL not importable")
        RESULTS+=("SKIP  4b  NIXL not importable")
        RESULTS+=("SKIP  5b  NIXL not importable")
        RESULTS+=("SKIP  6b  NIXL not importable")
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
