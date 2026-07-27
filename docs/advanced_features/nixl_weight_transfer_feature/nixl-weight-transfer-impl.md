# Implementation: NIXL backend for weight transfer

Implements the design in `nixl-weight-transfer-design.md`.

## Implementation roadmap

Each step is independently testable without Miles. Complete them in order.

Each step below lists only its test names + what they check. The runnable commands and expected
results live in `nixl-weight-transfer-tests.md`.

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

**Tests** (details in `nixl-weight-transfer-tests.md`):
- **1a** — CLI flag is exposed in `--help`.
- **1b** — `NIXL` enum value + the new dataclass field exist.
- **1c** — `use_transfer_engine()` covers the NIXL triggers.

---

### Step 2 — Bootstrap server schema migration

**Files:** `engine_info_bootstrap_server.py`, `remote_instance_weight_loader_utils.py`, `loader.py`

- Change storage from `Dict[int, Tuple]` → `Dict[int, dict]`.
- PUT handler stores the tagged dict as-is (no positional unpacking).
- GET handler returns the dict directly (remove `list()` wrapper).
- Update the reader consumer so the existing Mooncake pull path survives the schema change:
  - `get_remote_instance_transfer_engine_info_per_rank()` returns a single info dict on success and
    `None` on failure (instead of a positional 2-tuple / `None, None`).
  - `load_model_from_remote_instance_by_transfer_engine()` reads `session_id` / `weights_info_dict`
    *by key* instead of positional unpacking.

This is backend-neutral — the Mooncake path still works, just with a tagged dict instead of a tuple.
(The reader update is a schema-compatibility tweak only; no NIXL read logic is added — see design §4.5.)

**Tests** (details in `nixl-weight-transfer-tests.md`):
- **2a** — bootstrap server stores/serves the tagged dict as-is (no list-wrapping).
- **2b** — reader returns a single dict / `None` by key (not a positional 2-tuple).
- **2c** — Mooncake path still returns a JSON object end-to-end (backend-neutral regression).

---

### Step 3 — NIXL agent init on the worker

**File:** `model_runner.py`

- Add `remote_instance_nixl_agent` and `remote_instance_transfer_engine_agent_metadata` instance vars.
- In `remote_instance_init_transfer_engine()`, branch at entry: if NIXL seed or `backend == "nixl"`, call `_remote_instance_init_nixl()` and return; otherwise run the existing Mooncake path unchanged.
- `_remote_instance_init_nixl()`: construct `nixl_agent` (same pattern as `disaggregation/nixl/conn.py`), store `agent_name`. Set `remote_instance_transfer_engine_agent_metadata = None` — **do not call `get_agent_metadata()` here**. The agent metadata must be captured after weight VRAM is registered (step 4) so the blob includes the RDMA rkeys for those buffers. SGLang is export-only — Miles handles `add_remote_agent`.

**Tests** (details in `nixl-weight-transfer-tests.md`):
- **3a** — `remote_instance_init_transfer_engine()` branches to `_remote_instance_init_nixl()`.
- **3b** — server reaches ready state with no worker crash; "agent initialized" log appears.

---

### Step 4 — Memory registration

**File:** `remote_instance_weight_loader_utils.py`

- Add `register_memory_region_nixl(model, nixl_agent, gpu_id)`: build `weights_info_dict[name] = (data_ptr, numel, element_size, gpu_id)` and register merged contiguous VRAM blocks with `agent.register_memory([(addr, size, gpu_id, "")], "VRAM")`.
- In `model_runner.py` `initialize()`, branch the memory-registration block on whether `remote_instance_nixl_agent` or `remote_instance_transfer_engine` is set.
- **After** `register_memory_region_nixl()` returns, call `agent.get_agent_metadata()` and store the result in `remote_instance_transfer_engine_agent_metadata`. This must happen here — not in step 3 — because `get_agent_metadata()` is a snapshot: it only includes rkeys for regions registered at call time. Calling it before registration produces a blob with no weight rkeys, so Miles' RDMA WRITEs silently write nowhere and the weight checker sees the random values from `reset_tensors` unchanged.

**Tests** (details in `nixl-weight-transfer-tests.md`):
- **4a** — `register_memory_region_nixl` exists and builds 4-field (`+gpu_id`) entries.
- **4b** — same launch as step 3; no CUDA errors, process healthy after registration.

---

### Step 5 — Metadata publish

**File:** `model_runner.py`

- In `_register_to_engine_info_bootstrap()`, emit a backend-tagged dict, branching on whether
  `remote_instance_nixl_agent` is set:
  ```json
  // NIXL agent set
  {
    "backend": "nixl",
    "agent_name": "<uuid>",
    "agent_metadata": "<base64 of agent.get_agent_metadata()>",
    "weights_info_dict": { "<name>": [addr, numel, element_size, device_id] }
  }
  // otherwise (Mooncake)
  {
    "backend": "mooncake",
    "session_id": "<host:port>",
    "weights_info_dict": { "<name>": [addr, numel, element_size] }
  }
  ```
- Base64-encode `agent_metadata` for JSON transport (`agent_name` reuses the existing
  `remote_instance_transfer_engine_session_id` slot, which `_remote_instance_init_nixl()` set to the agent uuid).
- Cleanup: `_register_to_engine_info_bootstrap()` was accidentally defined **twice** in `ModelRunner`
  (the second shadowed the first at runtime). The dead earlier copy was removed so a single, tagged
  definition remains.

**Tests** (details in `nixl-weight-transfer-tests.md`):
- **5a** — publish method emits the tagged dict (`backend`/`agent_name`/`agent_metadata` b64).
- **5b** — endpoint serves `backend=nixl`, `agent_name`, non-empty `agent_metadata`, 4-field `weights_info_dict`.

---

### Step 6 — Engine startup condition

**File:** `engine.py`

- Add `remote_instance_weight_loader_start_seed_via_nixl` to the bootstrap server startup condition
  (alongside the existing transfer-engine flag), so the gate fires on either flag when
  `node_rank == 0`. Required for Step 5's publish to land end-to-end: without it the NIXL seed launch
  never starts the bootstrap server, the publish PUT is refused, and the endpoint returns an error.

**Tests** (details in `nixl-weight-transfer-tests.md`):
- **6a** — engine startup condition references the NIXL seed flag.
- **6b** — the step 5 `curl` succeeds, confirming the bootstrap server started and the endpoint is reachable.

