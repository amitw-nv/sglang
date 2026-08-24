# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SGLang is a high-performance LLM/VLM serving framework. The repo has two major runtime subsystems:
- **`python/sglang/srt/`** — The main LLM/VLM serving runtime (SRT = SGLang Runtime)
- **`python/sglang/multimodal_gen/`** — Diffusion model serving (image/video generation); see its own `CLAUDE.md` at `python/sglang/multimodal_gen/.claude/CLAUDE.md`

Supporting packages:
- **`sgl-kernel/`** — AOT-compiled CUDA/C++ kernels (`import sgl_kernel`), built separately
- **`python/sglang/jit_kernel/`** — JIT-compiled Triton/CUDA kernels loaded at runtime
- **`python/sglang/lang/`** — The SGLang frontend DSL (program tracing, `@sgl.function`)

## Installation

```bash
# Install SGLang (from repo root)
pip install -e "python[all]"

# Install SGLang with diffusion support
pip install -e "python[all,diffusion]"

# Build sgl-kernel from source (requires CMake ≥3.31, CUDA)
cd sgl-kernel && make build
# Limit parallelism if needed:
make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
```

## Running the Server

```bash
# Start an OpenAI-compatible server
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000

# Multi-GPU (tensor parallel)
python -m sglang.launch_server --model-path ... --tp-size 4

# With key options
python -m sglang.launch_server --model-path ... --mem-fraction-static 0.9 --enable-flashinfer
```

## Linting & Formatting

Pre-commit hooks enforce code style. Run manually:

```bash
# Format Python (isort + black) and C++/CUDA (clang-format)
cd sgl-kernel && make format

# Or run pre-commit directly on changed files
pre-commit run --files <file1> <file2>
```

Tools: **isort** (import sorting), **ruff** (F401/F821 lint), **black** (formatting), **clang-format** (C/C++/CUDA), **codespell** (spelling). Generated protobuf files (`*_pb2.py`, `*_pb2_grpc.py`) are excluded.

## Running Tests

```bash
# Single test file
python3 test/registered/core/test_srt_endpoint.py

# Single test method
python3 test/registered/core/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# Run a CI suite
python3 test/run_suite.py --hw cuda --suite stage-a-test-cpu
python3 test/run_suite.py --hw cuda --suite stage-b-test-1-gpu-small

# JIT kernel tests
python3 python/sglang/jit_kernel/tests/test_add_constant.py
```

## CI Test Suite Selection

| GPU requirement | Suite |
|---|---|
| No GPU | `stage-a-test-cpu` |
| 1 GPU, ≤32GB VRAM | `stage-b-test-1-gpu-small` |
| 1 GPU, Hopper/large | `stage-b-test-1-gpu-large` |
| JIT/kernel unit tests | `stage-b-kernel-unit-1-gpu-large` |
| 2/4/8 GPUs | `stage-b-test-2/4/8-gpu-large` |
| Long-running / experimental | `nightly-*` |

For parallel local sharding: `python3 test/run_suite.py --hw cuda --suite <suite> --auto-partition-id 0 --auto-partition-size 4`

## SRT Architecture

The SRT runtime is a multi-process system communicating over ZMQ:

```
HTTP Client
    ↓ HTTP
TokenizerManager  (python/sglang/srt/managers/tokenizer_manager.py)
    ↓ ZMQ
Scheduler         (python/sglang/srt/managers/scheduler.py)
  ├── RadixCache / PrefixCache  (srt/mem_cache/)
  ├── ScheduleBatch             (srt/managers/schedule_batch.py)
  └── ModelRunner               (srt/model_executor/model_runner.py)
        └── TP Workers (one per GPU, via torch.distributed)
```

**Key process roles:**
- `TokenizerManager` — tokenizes requests, manages the HTTP interface, detokenizes outputs
- `Scheduler` — zero-overhead CPU scheduler; manages KV cache allocation, batching policy, prefix caching
- `ModelRunner` — executes forward passes on GPU; manages CUDA graphs, attention backends
- `DataParallelController` — optional; routes requests across multiple DP replicas

**Entry points:**
- `python/sglang/srt/entrypoints/engine.py` — `Engine` class (Python API + spawns all processes)
- `python/sglang/srt/entrypoints/http_server.py` — FastAPI HTTP server (OpenAI-compatible)
- `python/sglang/launch_server.py` — CLI entry point

## Key Subsystems

**Attention backends** (`srt/layers/attention/`): Multiple backends selectable at runtime — `flashinfer`, `flashattention`, `flashmla` (for MLA/DeepSeek), `cutlass_mla`, `nsa` (native sparse attention), `aiter` (AMD). Selected via `--attention-backend`.

**KV cache / RadixAttention** (`srt/mem_cache/`): RadixAttention enables prefix caching by storing KV cache in a radix tree. `allocator.py` manages GPU memory blocks.

**Speculative decoding** (`srt/speculative/`): EAGLE, EAGLE-2, N-gram, DFlash, and standalone draft workers.

**Disaggregation** (`srt/disaggregation/`): Prefill-Decode disaggregation — encode and decode servers run as separate processes communicating KV cache over RDMA/NVLink.

**Quantization** (`srt/layers/quantization/`): FP8, INT4, AWQ, GPTQ, FP4, compressed-tensors, bitsandbytes.

**Structured outputs** (`srt/constrained/`): Grammar-guided decoding using xgrammar and llguidance.

**Models** (`srt/models/`): Each model file implements `forward()` using SGLang's layer primitives. `registry.py` maps HuggingFace `architectures` strings to model classes.

**ServerArgs** (`srt/server_args.py`): Single dataclass containing all server configuration. Always pass `ServerArgs` when constructing an `Engine`.

**Env vars** (`srt/environ.py`): All `SGLANG_*` env vars are declared as typed `EnvField` descriptors in the `Envs` class. Do not use `os.environ` directly for SGLang vars.

**Overlap scheduling** (`srt/managers/overlap_utils.py`): Default-enabled CPU-GPU pipelining. The scheduler runs two event loops — `event_loop_normal()` (sequential) and `event_loop_overlap()` (pipelined). `FutureMap` is a circular buffer that defers GPU result processing to the next iteration, hiding tokenizer/grammar overhead. Disable with `--disable-overlap-schedule`.

**Batch transformation chain**: Requests flow through three representations:
```
ScheduleBatch  →  get_model_worker_batch()  →  ModelWorkerBatch
                                                      ↓ convert_to_forward_batch()
                                                  ForwardBatch  (GPU tensors)
```
`ScheduleBatch` and `ModelWorkerBatch` live in `srt/managers/schedule_batch.py`; `ForwardBatch` lives in `srt/model_executor/forward_batch_info.py`.

**Scheduler mixin pattern** (`srt/managers/`): `Scheduler` composes behavior via mixins — `scheduler_output_processor_mixin.py` (token output), `scheduler_pp_mixin.py` (pipeline parallelism), `scheduler_dp_attn_mixin.py` (DP attention), `scheduler_profiler_mixin.py`, `scheduler_runtime_checker_mixin.py` (watchdog), etc. Disaggregation logic is in `srt/disaggregation/prefill.py` and `decode.py` via their own mixins.

**IO structures** (`srt/managers/io_struct.py`): Defines all ZMQ message types between TokenizerManager and Scheduler. When adding new request fields or response fields, update this file.

## Adding a New Model

1. Create `python/sglang/srt/models/<model_name>.py` implementing `forward()` with SGLang layer primitives
2. Register in `python/sglang/srt/models/registry.py`
3. Add a test in `test/registered/models/`

## Adding a New CUDA Kernel (AOT)

See `sgl-kernel/README.md`. Steps: implement in `sgl-kernel/csrc/`, expose in `include/sgl_kernel_ops.h`, register in `csrc/common_extension.cc`, update `CMakeLists.txt`, expose Python in `sgl-kernel/python/sgl_kernel/`.

See skill: `/add-sgl-kernel`

## Adding a New JIT Kernel

JIT kernels live in `python/sglang/jit_kernel/` as individual `.py` files using Triton or `torch.compile`. Tests go in `python/sglang/jit_kernel/tests/`, benchmarks in `python/sglang/jit_kernel/benchmark/`.

See skill: `/add-jit-kernel`

## CI Test Registration

Every test file in `test/registered/` must register at module level:

```python
from sglang.test.ci.ci_register import register_cuda_ci
register_cuda_ci(est_time=80, suite="stage-b-test-1-gpu-small")
```

Keep `est_time` and `suite` as **literal values** — the CI runner collects them via AST parsing.

CI pipeline: **Stage A** (pre-flight, ~3 min, CPU) → **Stage B** (basic, ~30 min, 1 GPU) → **Stage C** (advanced, multi-GPU). Use the lightest suite that fits.

For full test/CI details see `test/README.md` and skills `/write-sglang-test`, `/ci-workflow-guide`.

## Useful Skills

- `/write-sglang-test` — templates and checklist for adding CI tests
- `/ci-workflow-guide` — CI stage flow, fast-fail, debugging failures
- `/add-sgl-kernel` — AOT CUDA kernel tutorial
- `/add-jit-kernel` — JIT kernel tutorial
- `/sglang-prod-incident-triage` — debug live serving issues
- `/debug-distributed-hang` — debug multi-GPU hangs
- `/sglang-sota-performance` — benchmark and optimize serving performance
