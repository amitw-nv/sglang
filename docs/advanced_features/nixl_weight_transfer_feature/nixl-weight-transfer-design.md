# Design: NIXL backend for weight transfer (minimal delta over the `transfer_engine` path)

## 1. Goal & framing

SGLang should expose its model weight buffers to an **external transfer engine (Miles, using NIXL)** so that
Miles can RDMA-write updated weights directly into SGLang's GPU memory.

SGLang is the **passive target**. It does two things and nothing else:

1. **Register** its weight GPU buffers with a NIXL agent (so RDMA can land in them).
2. **Publish** the connection identity + per-tensor descriptors over the existing HTTP/registry so the
   external engine can connect and address the buffers.

The actual data movement (descriptor build, `transfer`, completion poll) happens in **Miles/NIXL**, not in SGLang.

### How it is launched

Miles owns the entry point. It installs SGLang in-place from the `sglang-miles` branch
(`pip install -e "python[all]" --no-deps`) and starts the whole flow through its own launcher,
`examples/p2p_weight_transfer/run.py`. That launcher boots the SGLang seed itself — the operator never
calls `sglang.launch_server` directly. The NIXL path is selected with `--mode nixl`:

```bash
python examples/p2p_weight_transfer/run.py run <model> --mode nixl
```

Under the hood `run.py` launches the SGLang seed with
`--remote-instance-weight-loader-start-seed-via-nixl` and then runs the Miles NIXL peer that performs the
RDMA writes. So the design below describes what SGLang must do when `run.py` starts it in `nixl` mode; the
flag is an internal contract between Miles and SGLang, not something the user sets by hand.

This is intentionally **not** R-Fork (SGL-to-SGL pull). We are only reusing the *plumbing* that the existing
`transfer_engine` (Mooncake) backend already established, because that plumbing already does buffer
registration + metadata export. We add the smallest possible delta on top of it.

## 2. Assumptions

These are the assumptions this design is built on. If any is wrong, the corresponding section changes.

- **A1 — NIXL is the SGLang-side library; Miles is the peer/driver.** SGLang only needs a NIXL `nixl_agent`
  to (a) register memory and (b) export agent metadata. SGLang does **not** issue `transfer()` calls for
  weights; Miles does. (NIXL is already vendored for PD-disaggregation, so the import/install path exists.)
- **A2 — We reuse the `remote_instance` / `transfer_engine` machinery as-is**, including
  `--remote-instance-weight-loader-backend`, the `EngineInfoBootstrapServer` registry, the
  `/remote_instance_transfer_engine_info` HTTP endpoint, and `register_memory_region`. We add a `nixl`
  branch rather than a new parallel subsystem.
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
- **A8 — Backend transport for NIXL** (UCX, etc.) is selected via the existing
  `SGLANG_DISAGGREGATION_NIXL_BACKEND` env (reused) or a defaulted value; not a new required flag.
- **A9 — JSON transport for metadata.** The registry/HTTP layer is JSON, so the binary `agent_metadata` is
  base64-encoded for transport and decoded by the peer.

## 3. Baseline: what the `transfer_engine` (Mooncake) path already does

We get all of this for free and reuse it:

- **Backend selection** — `--remote-instance-weight-loader-backend transfer_engine` and the
  `RemoteInstanceWeightLoaderBackend` enum
  (`python/sglang/srt/model_loader/remote_instance_weight_loader_utils.py`).
- **Engine init on the worker** — `ModelRunner.remote_instance_init_transfer_engine()` creates the engine and
  computes a `session_id` (`model_executor/model_runner.py`).
- **Buffer registration** — `register_memory_region()` / `register_memory_region_v2()` walk
  `model.named_parameters()`, register each buffer, and build
  `weights_info_dict[name] = (data_ptr, numel, element_size)`.
- **Metadata publish** — `ModelRunner._register_to_engine_info_bootstrap()` PUTs
  `{session_id, weights_info_dict}` per `tp_rank` to the `EngineInfoBootstrapServer`.
- **Metadata serve** — `EngineInfoBootstrapServer` stores `{tp_rank: (session_id, weights_info_dict)}` and
  serves it via `/get_transfer_engine_info`, proxied to the public
  `/remote_instance_transfer_engine_info` endpoint (`entrypoints/http_server.py`).

The peer (Miles, today Mooncake) reads that endpoint, learns `session_id` + buffer pointers, and writes.

## 4. Minimal delta for NIXL

Per touch point, the smallest change that makes NIXL work. Nothing is duplicated that can be branched.

### 4.1 Backend enum + arg (tiny)
- Add `NIXL = "nixl"` to `RemoteInstanceWeightLoaderBackend`.
- Add `"nixl"` to the `remote_instance_weight_loader_backend` `Literal` in `server_args.py`.
- Add `--remote-instance-weight-loader-start-seed-via-nixl` bool flag (analogous to the existing
  `--remote-instance-weight-loader-start-seed-via-transfer-engine`). This is the trigger for NIXL seed mode.
- Extend `remote_instance_weight_loader_use_transfer_engine()` to return `True` when either
  `remote_instance_weight_loader_start_seed_via_nixl` is set or `backend == "nixl"`.

### 4.2 Agent init on the worker (branch in one method)
- In `ModelRunner.remote_instance_init_transfer_engine()`, branch on backend:
  - `transfer_engine` → existing Mooncake `TransferEngine.initialize(..., "P2PHANDSHAKE", "rdma", ...)`.
  - `nixl` → construct a `nixl_agent` (reuse the construction in `disaggregation/nixl/conn.py`), and store
    `agent_name` + `agent.get_agent_metadata()` instead of `session_id`.
- Store backend-neutral connection info on the runner, e.g. keep
  `remote_instance_transfer_engine_session_id` for Mooncake and add
  `remote_instance_transfer_engine_agent_metadata` (bytes) for NIXL. (Minimal: two optional fields, not a
  new abstraction layer.)

### 4.3 Buffer registration (branch in one helper)
- `register_memory_region(model, engine, backend)`:
  - Reuse the `v2` contiguous-block merging logic unchanged.
  - For NIXL, call `agent.register_memory([(addr, size, gpu_id, "")], "VRAM")` instead of
    `engine.register_memory(addr, size)`, and keep the returned `descs` handle alive on the agent.
  - Extend each `weights_info_dict` entry to `(data_ptr, numel, element_size, device_id)`. (Mooncake can
    ignore the 4th field; keeping a uniform tuple avoids a second code path.)

### 4.4 Metadata schema (the one real change) — see §5.

### 4.5 Receiver path
- **No change / not added.** SGLang does not read. The R-Fork
  `load_model_from_remote_instance_by_transfer_engine` reader stays Mooncake-only and untouched.

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
- `ModelRunner._register_to_engine_info_bootstrap()` — emit the tagged dict (base64 the metadata).
- `EngineInfoBootstrapServer` — store/serve the dict instead of a positional tuple
  (`register_transfer_engine_info` / `get_transfer_engine_info`).
- `/remote_instance_transfer_engine_info` HTTP endpoint — passthrough; path **unchanged** so Miles' existing
  `query_remote_weight_info` discovery URL keeps working.

Miles reads this, base64-decodes `agent_metadata`, calls `add_remote_agent(...)`, then issues NIXL WRITE
xfers against the `(addr, size, device_id)` descriptors.

## 6. Out of scope (explicitly not touched)

- The broadcast / tensor update paths (`update_weights_from_distributed`, `update_weights_from_tensor`,
  `init_weights_update_group`, `init_weights_send_group_for_remote_instance`) — NCCL/IPC, no transfer engine.
- R-Fork SGL-to-SGL pull/reader path.
- Sharding, all-gather, dtype/`convert_to_hf`, CPU staging — all on the Miles side.
- ModelExpress backend.

## 7. Open questions / risks — resolved

- **Q1 — Decommission / re-register on weight realloc.** If SGLang reallocates weight memory (e.g. quant
  swap, CUDA-graph recapture), registered descriptors go stale and metadata must be re-published. The
  Mooncake path has the same hazard; confirm Miles re-queries the endpoint each round.
- **Q2 — Agent lifetime & teardown.** **Decision: keep for process lifetime.** Descriptors are pinned on the
  agent object (`nixl_agent._weight_descs`) and released only on process exit.
- **Q3 — Bidirectional metadata.** **Decision: SGLang is export-only.** Miles reads `agent_metadata` from
  the HTTP endpoint, calls `add_remote_agent()` on its side, and issues WRITE transfers. SGLang does not call
  `add_remote_agent()` back. A comment in `_remote_instance_init_nixl()` documents this assumption.
- **Q4 — Backend transport default.** **Decision: reuse `SGLANG_DISAGGREGATION_NIXL_BACKEND`.** No new env
  var is added; the same UCX default applies to both disaggregation and weight transfer.
