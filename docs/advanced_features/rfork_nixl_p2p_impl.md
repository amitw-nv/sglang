# Implementation: NIXL backend for weight transfer

Implements the design in `rfork_nixl_p2p_design.md`.

## What was built

SGLang can now expose its model weight buffers to an external NIXL peer (Miles) for RDMA writes. SGLang is purely the passive target: it registers weight VRAM buffers with a NIXL agent and publishes connection metadata over the existing bootstrap HTTP endpoint. Miles discovers the metadata, calls `add_remote_agent()`, and issues WRITE transfers.

## How to use

Start SGLang as a NIXL seed (exposes weights; does not load from a remote):

```bash
python -m sglang.launch_server \
  --model-path <model> \
  --remote-instance-weight-loader-start-seed-via-nixl
```

The endpoint `GET /remote_instance_transfer_engine_info?rank=<tp_rank>` returns:

```json
{
  "rank": 0,
  "remote_instance_transfer_engine_info": {
    "backend": "nixl",
    "agent_name": "<uuid>",
    "agent_metadata": "<base64-encoded bytes>",
    "weights_info_dict": {
      "<param_name>": [addr, numel, element_size, device_id],
      ...
    }
  }
}
```

Miles decodes `agent_metadata`, calls `add_remote_agent(decoded_bytes)`, then issues NIXL WRITE transfers using `(addr, numel * element_size, device_id)` per tensor.

The NIXL transport backend is selected via `SGLANG_DISAGGREGATION_NIXL_BACKEND` (default `"UCX"`).

## Files changed

### `python/sglang/srt/model_loader/remote_instance_weight_loader_utils.py`
- Added `NIXL = "nixl"` to `RemoteInstanceWeightLoaderBackend` enum.
- Added `register_memory_region_nixl(model, nixl_agent, gpu_id)`: builds a 4-tuple `(data_ptr, numel, element_size, gpu_id)` weight dict and registers merged contiguous VRAM blocks with the NIXL agent in a single `register_memory` call (reuses the v2 merging logic).
- Updated `get_remote_instance_transfer_engine_info_per_rank`: parses the new tagged-dict response format, returning `(session_id, weights_info_dict)` so the existing Mooncake caller interface is unchanged.

### `python/sglang/srt/server_args.py`
- Added `"nixl"` to the `remote_instance_weight_loader_backend` `Literal` type and argparse `choices`.
- Added `remote_instance_weight_loader_start_seed_via_nixl: bool = False` field.
- Added `--remote-instance-weight-loader-start-seed-via-nixl` CLI flag.
- Added `validate_nixl()`: checks `nixl._api` is importable, same pattern as `validate_transfer_engine()`.
- Updated `__post_init__` validation to handle the `nixl` backend and the new seed flag.
- Extended `remote_instance_weight_loader_use_transfer_engine()` to return `True` when `remote_instance_weight_loader_start_seed_via_nixl` is set or `backend == "nixl"`.

### `python/sglang/srt/entrypoints/engine_info_bootstrap_server.py`
- Changed storage from `Dict[int, Tuple]` → `Dict[int, dict]`.
- PUT handler now stores the entire tagged info dict as-is (no positional unpacking).
- GET handler returns the dict directly (removed `list()` wrapper).
- Updated `get_transfer_engine_info()` return type annotation.

### `python/sglang/srt/model_executor/model_runner.py`
- Added `remote_instance_nixl_agent` and `remote_instance_transfer_engine_agent_metadata` instance vars.
- Added `register_memory_region_nixl` to the import.
- `remote_instance_init_transfer_engine()`: branches at entry — if NIXL seed or `backend == "nixl"`, delegates to `_remote_instance_init_nixl()` and returns; otherwise runs the existing Mooncake path unchanged.
- Added `_remote_instance_init_nixl()`: constructs `nixl_agent` (reusing the same pattern as `disaggregation/nixl/conn.py`), captures `get_agent_metadata()` bytes. Includes a comment that SGLang is export-only and Miles handles the `add_remote_agent` handshake.
- `initialize()`: memory registration block now branches on whether `remote_instance_nixl_agent` or `remote_instance_transfer_engine` is set.
- `_register_to_engine_info_bootstrap()` (the active definition): emits tagged dict `{"backend": "nixl", ...}` or `{"backend": "mooncake", ...}` depending on backend; base64-encodes `agent_metadata` for JSON transport.

### `python/sglang/srt/entrypoints/engine.py`
- Bootstrap server startup condition now also triggers for `remote_instance_weight_loader_start_seed_via_nixl`.

## Schema migration note

The metadata format changed from a 2-element JSON array `[session_id, weights_info_dict]` to a tagged dict. The Mooncake consumer (`get_remote_instance_transfer_engine_info_per_rank`) was updated to parse the new format while keeping its return signature `(session_id, weights_info_dict)` unchanged, so `load_model_from_remote_instance_by_transfer_engine` required no changes.

## Out of scope (per design)

- R-Fork SGL-to-SGL pull/reader path.
- Any change to broadcast/NCCL/IPC weight-update paths.
- Sharding, dtype conversion, CPU staging — Miles' responsibility.
- Bidirectional `add_remote_agent` handshake from SGLang side.
