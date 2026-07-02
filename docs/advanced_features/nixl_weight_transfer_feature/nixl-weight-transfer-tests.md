# Tests: NIXL backend for weight transfer

Full test details for the implementation roadmap in `nixl-weight-transfer-impl.md`.
The impl doc lists only the test name + what it checks; the runnable commands and
expected results live here.

## Conventions

- **Model path** (inside the dev container):
  `/sgl-workspace/llm_models/DeepSeek-V3-Lite/bf16` (use the `fp8` sibling if VRAM is tight).
  DeepSeek-V3-Lite is a MoE model, so launch tests use `--tp 8 --trust-remote-code`.
- **Test types**:
  - *static* — source inspection, no import side effects.
  - *unit* — imports a module / spins a local HTTP server; no GPU or model.
  - *e2e* — launches a real server; needs GPU (+ model, + NIXL/UCX for the NIXL path).
- A one-shot runner for the no-GPU tests of steps 1–6 lives at `run_tests_steps_1_6.sh`.

### Prerequisite gate (run before any NIXL e2e test)

```bash
python -c "import nixl._api; print('nixl importable')"
```

Not a feature test — it's a gate. Step 1's `validate_nixl()` (called from `__post_init__`)
checks exactly `nixl._api`. If it isn't importable, the
`--remote-instance-weight-loader-start-seed-via-nixl` flag is **silently reset to `False`**,
so the NIXL branch never runs and any NIXL e2e test would be vacuously "green". Confirm this
prints `nixl importable` before trusting tests 3b / 4 / 5 / 6.

---

## Step 1 — Backend enum + CLI arg

### Test 1a — CLI flag is exposed (unit, no GPU)

```bash
python -m sglang.launch_server --help | grep -- --remote-instance-weight-loader-start-seed-via-nixl
```

- **Checks:** the new argparse flag is wired into the launcher.
- **Expected:** one line printed showing the flag + its help text (`Start seed server via NIXL backend...`). Empty output = fail.

### Test 1b — enum + dataclass field exist (unit, no GPU)

```bash
python -c "
from sglang.srt.model_loader.remote_instance_weight_loader_utils import RemoteInstanceWeightLoaderBackend as B
from sglang.srt.server_args import ServerArgs
f = ServerArgs.__dataclass_fields__
assert B.NIXL.value == 'nixl', B.NIXL
assert 'remote_instance_weight_loader_start_seed_via_nixl' in f
print('OK: enum + field present')
"
```

- **Checks:** `NIXL = \"nixl\"` is in the enum and the new bool field exists on `ServerArgs`.
- **Expected:** prints `OK: enum + field present`, exit 0. `AssertionError`/`KeyError` = fail.

### Test 1c — `use_transfer_engine()` covers the NIXL triggers (static, no GPU)

```bash
python -c "
import inspect
from sglang.srt.server_args import ServerArgs
src = inspect.getsource(ServerArgs.remote_instance_weight_loader_use_transfer_engine)
assert 'start_seed_via_nixl' in src and 'nixl' in src
print('OK: use_transfer_engine covers nixl')
"
```

- **Checks:** the gating helper treats `start_seed_via_nixl` / `backend == \"nixl\"` as a transfer-engine path (so Step 3's init actually fires).
- **Expected:** prints `OK: use_transfer_engine covers nixl`.

> Note on the `validate_nixl()` gate: in `__post_init__`, if `nixl._api` is not importable the
> flag is silently reset to `False` (graceful fallback). So on a box without NIXL, a fully
> constructed `ServerArgs` will read the flag as `False` — that is correct behavior, not a bug.
> 1b/1c check the wiring without triggering that reset.

---

## Step 2 — Bootstrap server schema migration

### Test 2a — bootstrap server round-trips a tagged dict (unit, no GPU)

```bash
python -c "
import requests, time
from sglang.srt.entrypoints.engine_info_bootstrap_server import EngineInfoBootstrapServer
s = EngineInfoBootstrapServer('127.0.0.1', 38999); time.sleep(1)
info = {'session_id': '10.0.0.7:18000', 'weights_info_dict': {'w': [1,2,3]}}
r = requests.put('http://127.0.0.1:38999/register_transfer_engine_info',
                 json={'tp_rank': 0, 'transfer_engine_info': info})
assert r.status_code == 200, r.text
g = requests.get('http://127.0.0.1:38999/get_transfer_engine_info', params={'rank':0}).json()
out = g['remote_instance_transfer_engine_info']
assert out == info, out          # dict round-trips unchanged, no list() wrapping
assert isinstance(out, dict)     # NOT a bare array/tuple anymore
print('OK: dict stored and served as-is')
s.close()
"
```

- **Checks:** PUT stores the dict without positional unpacking, GET returns it directly (no `list()` wrapper), value type is `dict`.
- **Expected:** prints `OK: dict stored and served as-is`. Serializing to a list or dropping keys = fail.

### Test 2b — reader returns a single dict / None by key (static, no GPU)

```bash
python -c "
import inspect
from sglang.srt.model_loader import remote_instance_weight_loader_utils as u
src = inspect.getsource(u.get_remote_instance_transfer_engine_info_per_rank)
assert 'return None' in src and 'None, None' not in src   # single dict / None, not 2-tuple
print('OK: reader returns single dict or None')
"
```

- **Checks:** the consumer no longer returns a positional `(None, None)` 2-tuple; it returns one dict or `None`.
- **Expected:** prints `OK: reader returns single dict or None`.

### Test 2c — Mooncake path still works under the new schema (e2e, needs GPU)

```bash
python -m sglang.launch_server \
  --model-path /sgl-workspace/llm_models/DeepSeek-V3-Lite/bf16 \
  --tp 8 --trust-remote-code \
  --remote-instance-weight-loader-backend transfer_engine \
  --remote-instance-weight-loader-start-seed-via-transfer-engine &
# wait for "ready to roll", then:
curl -s "http://localhost:30000/remote_instance_transfer_engine_info?rank=0" | python -m json.tool
```

- **Checks:** the schema migration didn't break the existing Mooncake backend (backend-neutral regression).
- **Expected:** a JSON **object** — not a bare array. Before Step 5 it is the untagged
  `{"session_id": "...:...", "weights_info_dict": {...}}`; after Step 5 the Mooncake path is
  also tagged, so it becomes `{"backend": "mooncake", "session_id": "...:...", "weights_info_dict": {...}}`.
  Either shape passes this backend-neutral regression (the assertion is only that it stays a JSON object).
  Requires `mooncake` importable.

---

## Step 3 — NIXL agent init on the worker

### Test 3a — init method is branched and instance vars exist (static, no GPU)

```bash
python -c "
import inspect
from sglang.srt.model_executor.model_runner import ModelRunner
src = inspect.getsource(ModelRunner.remote_instance_init_transfer_engine)
assert '_remote_instance_init_nixl' in src and 'return' in src   # branch-and-return before Mooncake
assert hasattr(ModelRunner, '_remote_instance_init_nixl')
print('OK: nixl branch present')
"
```

- **Checks:** `remote_instance_init_transfer_engine()` branches to `_remote_instance_init_nixl()` and returns early; the NIXL init method exists.
- **Expected:** prints `OK: nixl branch present`.

### Test 3b — server reaches ready state, worker doesn't crash (e2e, needs GPU + NIXL)

```bash
python -c "import nixl._api; print('nixl importable')"   # gate first
python -m sglang.launch_server \
  --model-path /sgl-workspace/llm_models/DeepSeek-V3-Lite/bf16 \
  --tp 8 --trust-remote-code \
  --remote-instance-weight-loader-start-seed-via-nixl
```

- **Checks:** with NIXL installed, the worker constructs the `nixl_agent`, validates the backend plugin (default `UCX` from `SGLANG_REMOTE_INSTANCE_NIXL_BACKEND`), and captures `get_agent_metadata()` — all inside `ModelRunner.initialize()` without throwing. This is the agent-init contract only (memory reg = Step 4, publish = Step 5).
- **Expected:**
  - Log line per rank: `NIXL weight-transfer agent initialized (agent_name=<uuid>, backend=UCX) for tp_rank=0`
  - Server reaches `The server is fired up and ready to roll!`, worker process stays alive.
  - No exception/traceback from `_remote_instance_init_nixl()` or `initialize()`.
- **Failure signatures:**
  - `ValueError: NIXL backend 'UCX' not found. Available: [...]` → UCX plugin not installed (env/plugin issue, not a logic bug).
  - Warning `Please install NIXL...` then fall-through → NIXL not importable, agent stayed `None` (and the flag was reset in Step 1).
  - `torch.OutOfMemoryError` during `load_model` → not a Step-3 result; the crash happened before agent init. Free the GPUs (`nvidia-smi`, kill stale procs) and/or use `--tp 8` / the `fp8` variant, then rerun.

---

## Step 4 — Memory registration

Implemented. `register_memory_region_nixl(model, nixl_agent, gpu_id)` registers the merged
contiguous VRAM weight blocks with the NIXL agent (`agent.register_memory([...], "VRAM")`,
keeping the returned `descs` alive on `nixl_agent._weight_descs`) and builds 4-field
`weights_info_dict[name] = (data_ptr, numel, element_size, gpu_id)` entries. The
registration block in `ModelRunner.initialize()` now branches: it calls
`register_memory_region_nixl` when `remote_instance_nixl_agent` is set, and the existing
`register_memory_region` when `remote_instance_transfer_engine` is set.

### Test 4a — `register_memory_region_nixl` exists and builds 4-field entries (static, no GPU)

```bash
python -c "
import inspect
from sglang.srt.model_loader import remote_instance_weight_loader_utils as u
assert hasattr(u, 'register_memory_region_nixl')
src = inspect.getsource(u.register_memory_region_nixl)
assert 'register_memory' in src and 'VRAM' in src       # agent.register_memory([...], 'VRAM')
assert 'gpu_id' in src                                   # 4th field widens the descriptor
print('OK: register_memory_region_nixl present')
"
```

- **Checks:** the NIXL registration helper exists, registers VRAM blocks, and widens each
  `weights_info_dict` entry to the 4-field `(data_ptr, numel, element_size, gpu_id)` tuple.
- **Expected:** prints `OK: register_memory_region_nixl present`.

### Test 4b — no CUDA errors during/after registration (e2e, needs GPU + NIXL)

Same launch as Test 3b:

```bash
python -c "import nixl._api; print('nixl importable')"   # gate first
python -m sglang.launch_server \
  --model-path /sgl-workspace/llm_models/DeepSeek-V3-Lite/bf16 \
  --tp 8 --trust-remote-code \
  --remote-instance-weight-loader-start-seed-via-nixl
```

- **Checks:** with the Step 4 branch landed, the NIXL memory-registration block runs for the
  NIXL agent (`register_memory_region_nixl`) without CUDA faults and the process stays healthy
  after it completes. Unlike before Step 4, registration is no longer skipped on the NIXL path.
- **Expected:** server still reaches `ready to roll`; no `CUDA error` / `NIXL memory registration failed`
  lines in logs; process alive after the registration block. (The published metadata is still the
  untagged `{session_id, weights_info_dict}` shape until Step 5 retags it.)

---

## Step 5 — Metadata publish

Implemented. `_register_to_engine_info_bootstrap()` now emits a backend-tagged dict. On the
NIXL path (`remote_instance_nixl_agent` set) it publishes
`{backend: "nixl", agent_name, agent_metadata (base64), weights_info_dict}`; otherwise it
publishes the Mooncake dict `{backend: "mooncake", session_id, weights_info_dict}`. The
`agent_metadata` bytes are base64-encoded for JSON transport.

### Test 5a — emitted payload is the tagged NIXL dict (static, no GPU)

```bash
python -c "
import inspect
from sglang.srt.model_executor.model_runner import ModelRunner
src = inspect.getsource(ModelRunner._register_to_engine_info_bootstrap)
for key in ('backend', 'agent_name', 'agent_metadata', 'base64'):
    assert key in src, key
print('OK: tagged nixl payload emitted')
"
```

- **Checks:** the publish method emits `backend` / `agent_name` / `agent_metadata` and base64-encodes the metadata.
- **Expected:** prints `OK: tagged nixl payload emitted`.

### Test 5b — endpoint serves all NIXL fields (e2e, needs GPU + NIXL)

```bash
curl -s "http://localhost:30000/remote_instance_transfer_engine_info?rank=0" | python -m json.tool
```

- **Checks:** end-to-end the served dict carries the full NIXL identity.
- **Expected:** JSON object with `backend == "nixl"`, a non-empty `agent_name`, a non-empty base64 `agent_metadata` string, and a `weights_info_dict` whose entries are 4-element `[addr, numel, element_size, device_id]`.
- Optional stronger check — `agent_metadata` decodes as base64:

```bash
curl -s "http://localhost:30000/remote_instance_transfer_engine_info?rank=0" \
 | python -c "import sys,json,base64; d=json.load(sys.stdin)['remote_instance_transfer_engine_info']; assert d['backend']=='nixl'; assert base64.b64decode(d['agent_metadata']); print('OK: nixl metadata served & b64-valid')"
```

---

## Step 6 — Engine startup condition

Implemented. In `engine.py` the bootstrap-server startup gate now fires when **either**
`remote_instance_weight_loader_start_seed_via_transfer_engine` **or**
`remote_instance_weight_loader_start_seed_via_nixl` is set (on `node_rank == 0`), so the
registry/HTTP server also starts in NIXL seed mode. Without this, the NIXL seed launch never
starts the bootstrap server, Step 5's publish PUT is refused, and the endpoint returns an
error (this was the `KeyError: 'remote_instance_transfer_engine_info'` symptom).

### Test 6a — startup condition includes the NIXL flag (static, no GPU)

```bash
python -c "
import inspect, sglang.srt.entrypoints.engine as e
src = inspect.getsource(e)
assert 'remote_instance_weight_loader_start_seed_via_nixl' in src
print('OK: engine startup gates on nixl flag')
"
```

- **Checks:** the NIXL seed flag is referenced in the bootstrap-server startup gate.
- **Expected:** prints `OK: engine startup gates on nixl flag`.

### Test 6b — endpoint is reachable in NIXL seed mode (e2e, needs GPU + NIXL)

The Test 5b `curl` succeeding *is* the Step 6 check: it proves the bootstrap server actually
started under `--remote-instance-weight-loader-start-seed-via-nixl` and the public endpoint is
reachable.

- **Checks:** bootstrap server started in NIXL seed mode; endpoint serves (no connection refused / 404).
- **Expected:** HTTP 200 with the Step 5b payload.

---

## Summary

| Test | Step | Type | Needs GPU/model? | Pass signal |
|---|---|---|---|---|
| 1a | 1 | unit | no | flag printed in `--help` |
| 1b | 1 | unit | no | enum + field present |
| 1c | 1 | static | no | `use_transfer_engine` covers nixl |
| 2a | 2 | unit | no | dict round-trips unchanged |
| 2b | 2 | static | no | reader returns dict/None |
| 2c | 2 | e2e | yes | endpoint returns a JSON object (Mooncake) |
| 3a | 3 | static | no | nixl branch present |
| 3b | 3 | e2e | yes (+NIXL/UCX) | "agent initialized" log, no crash |
| 4a | 4 | static | no | `register_memory_region_nixl` present |
| 4b | 4 | e2e | yes (+NIXL) | no CUDA errors, process healthy |
| 5a | 5 | static | no | tagged nixl payload emitted |
| 5b | 5 | e2e | yes (+NIXL) | endpoint serves backend/agent_name/agent_metadata/4-field dict |
| 6a | 6 | static | no | engine startup gates on nixl flag |
| 6b | 6 | e2e | yes (+NIXL) | endpoint reachable (200) |

No-GPU tests (1a–1c, 2a–2b, 3a, 4a, 5a, 6a) are automated by `run_tests_steps_1_6.sh`. The `<model>`
e2e tests need the container + GPUs from `running_script.sh`; the NIXL ones additionally need NIXL +
the UCX plugin in the image. Steps 1–6 are all implemented; the NIXL e2e checks (3b/4b/5b/6b) share a
single NIXL seed launch, so the runner records all four from one server start.
