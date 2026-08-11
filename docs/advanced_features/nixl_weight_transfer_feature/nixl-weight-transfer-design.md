# Design: NIXL backend for weight transfer (minimal delta over the `transfer_engine` path)

## 1. Goal & framing

SGLang should expose its model weight buffers to an **external transfer engine (Miles, using NIXL)** so that
Miles can RDMA-write updated weights directly into SGLang's GPU memory.

SGLang is the **passive target**. It does two things and nothing else:

1. **Register** its weight GPU buffers with a NIXL agent (so RDMA can land in them).
2. **Publish** the connection identity + per-tensor descriptors over the existing HTTP/registry so the
   external engine can connect and address the buffers.

The actual data movement happens in **Miles**, not in SGLang.

This design **builds NIXL on top of the existing `transfer_engine` (Mooncake) infrastructure** rather than
standing up a parallel subsystem: the Mooncake path already does buffer registration and metadata export,
so we add a `nixl` branch inside that same machinery.

### How it is launched

Miles owns the entry point. It installs SGLang in-place and starts the whole flow through its own launcher. That launcher boots the SGLang seed itself.

Under the hood `run.py` launches the SGLang seed with
`--remote-instance-weight-loader-start-seed-via-nixl` and then runs the Miles NIXL peer that performs the
RDMA writes. So the design below describes what SGLang must do when `run.py` starts it in `nixl` mode; the
flag is an internal contract between Miles and SGLang, not something the user sets by hand.

## 2. Assumptions

These are the assumptions this design is built on. If any is wrong, the corresponding section changes.

- **A1 — NIXL is the SGLang-side library; Miles is the peer/driver.** SGLang only needs a NIXL `nixl_agent`
  to (a) register memory and (b) export agent metadata. SGLang does **not** issue `transfer()` calls for
  weights; Miles does. (NIXL is already vendored for PD-disaggregation, so the import/install path exists.)
- **A2 — We reuse the existing weight-transfer machinery as-is**, including its backend selection, the
  registry that publishes connection info, the HTTP endpoint that serves it, and the buffer-registration
  step. We add a `nixl` branch rather than a new parallel subsystem.
- **A3 — SGLang never reads/pulls weights for this feature.** The R-Fork client reader path
  (`load_model_from_remote_instance_by_transfer_engine`, READ xfers) is out of scope. Miles performs WRITE.
- **A4 — Memory is GPU (VRAM) weights.** Buffers are model parameters in CUDA memory; we register them as
  `"VRAM"` with a valid `device_id` (gpu_id). CPU-shadow / DRAM staging, if any, lives in Miles.
- **A5 — Same-shape / same-layout assumption holds** between what SGLang registers and what Miles writes,
  exactly as the Mooncake path already assumes (`weights_info_dict` keyed by parameter name; matching
  `numel`/`element_size`). Sharding/all-gather/`convert_to_hf` is Miles' responsibility.
- **A6 — NIXL requires an explicit metadata handshake.** Unlike Mooncake's implicit `P2PHANDSHAKE` +
  `session_id` string, NIXL needs the peer to call `add_remote_agent(agent_metadata)` before any transfer.
  Therefore the published metadata MUST carry the opaque `agent_metadata` blob. This is the single
  unavoidable schema change.
- **A7 — One agent per worker (per tp_rank).** Each `ModelRunner` owns one NIXL agent, registers its local
  shard's weights, and publishes under its `tp_rank`, mirroring how Mooncake publishes per `tp_rank`.
- **A8 — Backend transport for NIXL** (UCX, etc.) is selected via a dedicated
  `SGLANG_REMOTE_INSTANCE_NIXL_BACKEND` env (default `UCX`), scoped to weight transfer and kept separate
  from PD-disaggregation's `SGLANG_DISAGGREGATION_NIXL_BACKEND`; not a new required flag.
- **A9 — JSON transport for metadata.** The registry/HTTP layer is JSON, so the binary `agent_metadata` is
  base64-encoded for transport and decoded by the peer.

## 3. Baseline: what the `transfer_engine` (Mooncake) path already does

We get all of this for free and reuse it:

- **Backend selection** — `--remote-instance-weight-loader-backend transfer_engine` and the
  `RemoteInstanceWeightLoaderBackend` enum
  (`python/sglang/srt/model_loader/remote_instance_weight_loader_utils.py`).
- **Engine init on the worker** — `RemoteInstanceWeightTransporter.init_engine()` creates the engine and
  computes a `session_id`
  (`model_executor/model_runner_components/remote_instance_weight_transporter.py`).
  `ModelRunner` owns one `RemoteInstanceWeightTransporter` instance (initialized via
  `init_remote_instance_weight_transporter()`) and delegates all engine/agent state to it.
- **Buffer registration** — `register_memory_region()` / `register_memory_region_v2()` walk
  `model.named_parameters()`, register each buffer, and build
  `weights_info_dict[name] = (data_ptr, numel, element_size)`.
- **Metadata publish** — `RemoteInstanceWeightTransporter._register_to_engine_info_bootstrap()` PUTs
  `{session_id, weights_info_dict}` per `tp_rank` to the `EngineInfoBootstrapServer`, called from
  `maybe_register_and_publish_weight_info()`.
- **Metadata serve** — `EngineInfoBootstrapServer` stores `{tp_rank: (session_id, weights_info_dict)}` and
  serves it via `/get_transfer_engine_info`, proxied to the public
  `/remote_instance_transfer_engine_info` endpoint (`entrypoints/http_server.py`).

The peer (Miles) reads that endpoint, learns `session_id` + buffer pointers, and writes.

### 3.1 Block-by-block: each Mooncake API and the NIXL parallel we must provide

The table below walks the weight-transfer pipeline stage by stage. The "Mooncake API" column is what the
existing `transfer_engine` backend does today; the "NIXL parallel" column is what we have to implement to
satisfy the same contract.

| # | Pipeline block | Mooncake API (exists) | NIXL parallel (to add) | Who owns it |
|---|---|---|---|---|
| 1 | Engine/agent object | `TransferEngine()` | `nixl_agent(uuid, nixl_agent_config(...))` | SGLang worker |
| 2 | Connection identity | `initialize(ip, "P2PHANDSHAKE", "rdma", dev)` → `session_id` host:port string; handshake is implicit | `agent_name` (uuid); `agent.get_agent_metadata()` is called **after** step 3 so rkeys are included; peer must explicitly `add_remote_agent` | SGLang worker |
| 3 | Buffer registration | `engine.register_memory(addr, size)` per merged block | `agent.register_memory([(addr, size, gpu_id, "")], "VRAM")`, keep returned `descs` alive | SGLang worker |
| 4 | Per-tensor descriptors | `weights_info_dict[name] = (addr, numel, element_size)` | same, plus `device_id`: `(addr, numel, element_size, device_id)` | SGLang worker |
| 5 | Metadata publish | PUT `{session_id, weights_info_dict}` | PUT `{backend, agent_name, agent_metadata(b64), weights_info_dict}` | SGLang worker + bootstrap |
| 6 | Metadata serve | `/remote_instance_transfer_engine_info` | **identical path**, value is now a tagged dict | Shared HTTP (no change) |
| 7 | Data movement | `batch_transfer_sync_write/read(session_id, local, remote, lens)` | `add_remote_agent` → `get_xfer_descs` → `initialize_xfer("WRITE", …)` → `transfer()` + poll | Miles peer (no SGLang code) |

**Reading the map.** Stages 1–4 run inside `RemoteInstanceWeightTransporter` (owned by `ModelRunner`);
stages 5–6 are the shared registry/HTTP plumbing; stage 7 happens entirely on the Miles peer. SGLang is the
passive target, so the only blocks we *write new code for* are the NIXL variants of stages 1–5. Stages 6
and 7 need no SGLang work — the endpoint path is reused verbatim, and the transfer itself is Miles' job.

The crux of the difference is stage 2: Mooncake folds connection identity into a single self-describing
`session_id` string and handshakes implicitly over `P2PHANDSHAKE`, whereas NIXL splits identity into an
`agent_name` plus an opaque `agent_metadata` blob that the *peer* must register via `add_remote_agent`
before any transfer. That is why the metadata schema in §5 must carry the base64 `agent_metadata` — it is
the one unavoidable change (assumption **A6**). Everything else is a 1:1 substitution of the API call
inside an already-existing block.

## 4. Minimal delta for NIXL

Per touch point, the smallest change that makes NIXL work. Nothing is duplicated that can be branched.

### 4.1 Backend enum + arg (tiny)
- Add `NIXL = "nixl"` to `RemoteInstanceWeightLoaderBackend`.
- Add `"nixl"` to the `remote_instance_weight_loader_backend` `Literal` in `server_args.py`.
- Add `--remote-instance-weight-loader-start-seed-via-nixl` bool flag (analogous to the existing
  `--remote-instance-weight-loader-start-seed-via-transfer-engine`). This is the trigger for NIXL seed mode.
- Extend `remote_instance_weight_loader_use_transfer_engine()` to return `True` when either
  `remote_instance_weight_loader_start_seed_via_nixl` is set or `backend == "nixl"`.

### 4.2 Agent init on the worker (branch in `RemoteInstanceWeightTransporter.init_engine()`)
- `RemoteInstanceWeightTransporter.init_engine()` branches on backend:
  - `transfer_engine` / `mooncake` → existing path: create `TransferEngine`, call
    `initialize(..., "P2PHANDSHAKE", ...)`, store `session_id` as host:port string.
  - `nixl` → calls `_init_nixl()`: construct a `nixl_agent`, store `agent_name` as `session_id`.
    **Do not call `get_agent_metadata()` here** — the weight buffers are not yet registered, so the
    blob would carry no VRAM rkeys. `get_agent_metadata()` is called in
    `maybe_register_and_publish_weight_info()` immediately after `register_memory_region_nixl()`
    (§4.3), so the snapshot includes the rkeys the Miles peer needs for RDMA WRITEs.
- NIXL-specific state lives on `RemoteInstanceWeightTransporter` as:
  - `session_id` — the agent name (uuid string), same field used by Mooncake for host:port.
  - `_nixl_agent` — the `nixl_agent` object (replaces the `_nixl_manager` stub).
  - `_nixl_agent_metadata` — the opaque metadata bytes, set to `None` until registration completes.
  - `weight_info` — the per-tensor descriptors dict, same field used by Mooncake.

### 4.3 Buffer registration (branch in one helper)
- `register_memory_region(model, engine, backend)`:
  - Reuse the `v2` contiguous-block merging logic unchanged.
  - For NIXL, call `agent.register_memory([(addr, size, gpu_id, "")], "VRAM")` instead of
    `engine.register_memory(addr, size)`, and keep the returned `descs` handle alive on the agent.
  - Extend each `weights_info_dict` entry to `(data_ptr, numel, element_size, device_id)`. (Mooncake can
    ignore the 4th field; keeping a uniform tuple avoids a second code path.)
- **Immediately after** `register_memory_region_nixl()` returns, call `agent.get_agent_metadata()` and
  store the result in `remote_instance_transfer_engine_agent_metadata`. This ordering is critical:
  `get_agent_metadata()` returns a snapshot of the agent's current state, which only includes rkeys for
  already-registered regions. Calling it before registration would produce a blob with no weight rkeys,
  causing Miles' RDMA WRITEs to land nowhere and the weight checker to see the randomized values from
  `reset_tensors` unchanged.

### 4.4 Metadata schema (the one real change) — see §5.

### 4.5 Receiver path
- **No NIXL reader added.** SGLang still does not read/pull weights for the NIXL feature — Miles performs
  the WRITE. No NIXL-specific read logic is introduced.
- **Schema-compatibility update only.** Because the registry value migrates from a positional
  `(session_id, weights_info_dict)` tuple to a backend-tagged dict (§5), the existing Mooncake R-Fork
  reader `load_model_from_remote_instance_by_transfer_engine` is updated to read `session_id` /
  `weights_info_dict` *by key* instead of positional unpacking (and
  `get_remote_instance_transfer_engine_info_per_rank()` now returns a single dict / `None`). This is the
  minimum needed to keep the Mooncake pull path working under the new schema.

## 5. The metadata schema change (the crux)

This is the only structurally new thing, and it exists solely to satisfy assumption **A6** (NIXL needs
`agent_metadata`) and **A4** (device id).

Today the registry value is the implicit Mooncake tuple `(session_id, weights_info_dict)`. Make it a tagged
dict so it is backend-aware and forward-compatible:

```json
// backend == "mooncake" (unchanged behavior, just tagged)
{
  "backend": "mooncake",
  "session_id": "10.0.0.7:18000",
  "weights_info_dict": { "<name>": [addr, numel, element_size] }
}

// backend == "nixl" (new)
{
  "backend": "nixl",
  "agent_name": "<uuid>",
  "agent_metadata": "<base64 of agent.get_agent_metadata()>",
  "weights_info_dict": { "<name>": [addr, numel, element_size, device_id] }
}
```

Touch points (all already exist; we widen the payload):
- `RemoteInstanceWeightTransporter._register_to_engine_info_bootstrap()` — emit the tagged dict
  (base64 the metadata), called from `maybe_register_and_publish_weight_info()`.
- `EngineInfoBootstrapServer` — store/serve the dict instead of a positional tuple
  (`register_transfer_engine_info` / `get_transfer_engine_info`).
- `/remote_instance_transfer_engine_info` HTTP endpoint — passthrough; path **unchanged** so Miles' existing
  `query_remote_weight_info` discovery URL keeps working.

Miles reads this, base64-decodes `agent_metadata`, calls `add_remote_agent(...)`, then issues NIXL WRITE
xfers against the `(addr, size, device_id)` descriptors.

