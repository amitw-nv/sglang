# Implementation: NIXL backend for weight transfer

Implements the design in `nixl-weight-transfer-design.md`.

## Implementation roadmap

Each step is independently testable without Miles. Complete them in order.

---

### Step 1 — Backend enum + CLI arg

**Files:** `remote_instance_weight_loader_utils.py`, `server_args.py`

- Add `NIXL = "nixl"` to `RemoteInstanceWeightLoaderBackend`.
- Add `"nixl"` to the `remote_instance_weight_loader_backend` `Literal` and argparse `choices`.
- Add `remote_instance_weight_loader_start_seed_via_nixl: bool = False` field and `--remote-instance-weight-loader-start-seed-via-nixl` CLI flag.
- Add `validate_nixl()`: check `nixl._api` is importable.
- In `__post_init__`, gate the new seed flag on `validate_nixl()` (mirrors the existing
  `remote_instance_weight_loader_start_seed_via_transfer_engine` validation block): if NIXL isn't
  importable, the flag is reset to `False` so the seed falls back gracefully.
- Extend `remote_instance_weight_loader_use_transfer_engine()` to return `True` for the new flag or `backend == "nixl"`.

**Test:**
```bash
python -m sglang.launch_server --help | grep nixl
# should show the new flag
python -c "from sglang.srt.server_args import ServerArgs; a = ServerArgs.__dataclass_fields__; print('nixl' in str(a))"
```

---

### Step 2 — Bootstrap server schema migration

**File:** `engine_info_bootstrap_server.py`

- Change storage from `Dict[int, Tuple]` → `Dict[int, dict]`.
- PUT handler stores the tagged dict as-is (no positional unpacking).
- GET handler returns the dict directly (remove `list()` wrapper).

This is backend-neutral — the Mooncake path still works, just with a tagged dict instead of a tuple.

**Test:** Start with the existing Mooncake backend and hit the info endpoint:
```bash
python -m sglang.launch_server --model-path <model> \
  --remote-instance-weight-loader-backend transfer_engine \
  --remote-instance-weight-loader-start-seed-via-transfer-engine &
curl "http://localhost:30000/remote_instance_transfer_engine_info?rank=0"
# response should now be a dict with a "backend" key instead of a bare array
```

---

### Step 3 — NIXL agent init on the worker

**File:** `model_runner.py`

- Add `remote_instance_nixl_agent` and `remote_instance_transfer_engine_agent_metadata` instance vars.
- In `remote_instance_init_transfer_engine()`, branch at entry: if NIXL seed or `backend == "nixl"`, call `_remote_instance_init_nixl()` and return; otherwise run the existing Mooncake path unchanged.
- `_remote_instance_init_nixl()`: construct `nixl_agent` (same pattern as `disaggregation/nixl/conn.py`), capture `get_agent_metadata()` bytes. SGLang is export-only — Miles handles `add_remote_agent`.

**Test:** Server reaches ready state with no worker crash:
```bash
python -m sglang.launch_server --model-path <model> \
  --remote-instance-weight-loader-start-seed-via-nixl
# check logs: no exception in ModelRunner.initialize(), worker process stays alive
```

---

### Step 4 — Memory registration

**File:** `remote_instance_weight_loader_utils.py`

- Add `register_memory_region_nixl(model, nixl_agent, gpu_id)`: build `weights_info_dict[name] = (data_ptr, numel, element_size, gpu_id)` and register merged contiguous VRAM blocks with `agent.register_memory([(addr, size, gpu_id, "")], "VRAM")`.
- In `model_runner.py` `initialize()`, branch the memory-registration block on whether `remote_instance_nixl_agent` or `remote_instance_transfer_engine` is set.

**Test:** Same launch as step 3; verify no CUDA errors appear in logs and the process stays healthy after the registration block completes.

---

### Step 5 — Metadata publish

**File:** `model_runner.py`

- In `_register_to_engine_info_bootstrap()`, emit the tagged dict:
  ```json
  {
    "backend": "nixl",
    "agent_name": "<uuid>",
    "agent_metadata": "<base64 of agent.get_agent_metadata()>",
    "weights_info_dict": { "<name>": [addr, numel, element_size, device_id] }
  }
  ```
- Base64-encode `agent_metadata` for JSON transport.

**Test:** Query the endpoint and verify all fields are present:
```bash
curl "http://localhost:30000/remote_instance_transfer_engine_info?rank=0" | python -m json.tool
# expect: backend=nixl, agent_name, agent_metadata (non-empty string), weights_info_dict
```

---

### Step 6 — Engine startup condition

**File:** `engine.py`

- Add `remote_instance_weight_loader_start_seed_via_nixl` to the bootstrap server startup condition (alongside the existing transfer-engine flag).

**Test:** The step 5 `curl` command succeeds — this confirms the bootstrap server started and the endpoint is reachable.

