# SPDX-License-Identifier: Apache-2.0

import enum
import importlib
import importlib.util
import logging
import time
from typing import List

import requests

logger = logging.getLogger(__name__)


class RemoteInstanceWeightLoaderBackend(str, enum.Enum):
    NCCL = "nccl"
    TRANSFER_ENGINE = "transfer_engine"
    MODELEXPRESS = "modelexpress"
    NIXL = "nixl"


def trigger_init_weights_send_group_for_remote_instance_request(
    remote_instance_weight_loader_seed_instance_ip: str,
    remote_instance_weight_loader_seed_instance_service_port: int,
    remote_instance_weight_loader_send_weights_group_ports: List[int],
    remote_instance_weight_loader_client_id: str,
):
    seed_instance_service_url = f"http://{remote_instance_weight_loader_seed_instance_ip}:{remote_instance_weight_loader_seed_instance_service_port}"
    # Only support loading weights from instance with same parallelism strategy.
    # Per TP rank pair between seed and dst instances will build a communication group for sending weights.
    # i.e. seed TP 0 <-> dst TP 0, seed TP 1 <-> dst TP 1, etc.
    # Each communication group will have a world size 2.
    try:
        requests.post(
            f"{seed_instance_service_url}/init_weights_send_group_for_remote_instance",
            json={
                "master_address": remote_instance_weight_loader_seed_instance_ip,
                "ports": (
                    ",".join(
                        str(p)
                        for p in remote_instance_weight_loader_send_weights_group_ports
                    )
                ),
                "group_rank": 0,
                "world_size": 2,
                "group_name": f"send_weights_{remote_instance_weight_loader_client_id}",
                "backend": "nccl",
            },
        )
    except Exception as e:
        logger.error(
            f"Failed to trigger init_weights_send_group_for_remote_instance_request to seed instance {seed_instance_service_url}: {e}."
        )
        raise


def trigger_transferring_weights_request(
    remote_instance_weight_loader_seed_instance_ip: str,
    remote_instance_weight_loader_seed_instance_service_port: int,
    remote_instance_weight_loader_send_weights_group_ports: List[int],
    remote_instance_weight_loader_client_id: str,
):
    seed_instance_service_url = f"http://{remote_instance_weight_loader_seed_instance_ip}:{remote_instance_weight_loader_seed_instance_service_port}"
    try:
        requests.post(
            f"{seed_instance_service_url}/send_weights_to_remote_instance",
            json={
                "master_address": remote_instance_weight_loader_seed_instance_ip,
                "ports": (
                    ",".join(
                        str(p)
                        for p in remote_instance_weight_loader_send_weights_group_ports
                    )
                ),
                "group_name": f"send_weights_{remote_instance_weight_loader_client_id}",
            },
        )
    except Exception as e:
        logger.error(f"Failed to trigger send weights to remote instance request: {e}")
        raise


def get_remote_instance_transfer_engine_info_per_rank(seed_url: str, rank: int):
    """Fetch the per-rank transfer engine info from the seed.

    Returns the backend-tagged info dict on success (e.g. Mooncake:
    {"session_id", "weights_info_dict"}), or None on any failure.
    """
    try:
        response = requests.get(
            f"{seed_url}/get_remote_instance_transfer_engine_info",
            params={
                "rank": rank,
            },
        )

        if response.status_code == 200:
            data = response.json()

            if "remote_instance_transfer_engine_info" in data:
                return data["remote_instance_transfer_engine_info"]
            else:
                logger.error(
                    "Failed to get `remote_instance_transfer_engine_info` in response."
                )
                return None
        else:
            logger.error(f"request.get failed: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Exception: {e}")
        return None


def _debug_dump_weight_memory_layout(memory_snapshot, weight_addr_set, merged):
    """Print the CUDA allocator weight-block layout; call once BEFORE and once
    AFTER contiguous-block merging. Debug aid only (no effect on registration).

    Same function, called twice on the same snapshot:
    - ``merged=False`` -> shows every weight block exactly as the allocator laid
      it out (the "before merge" view).
    - ``merged=True``  -> shows the same weight blocks after physically-contiguous
      ones within a segment are coalesced -- i.e. the regions actually registered.

    For each segment it prints the segment size/type and the (raw or merged)
    weight blocks with their sizes, plus overall counts and total memory.
    """

    def _mb(n):
        return "n/a" if n is None or n < 0 else f"{n / (1024 * 1024):.3f} MiB"

    stage = "AFTER MERGE" if merged else "BEFORE MERGE"
    logger.info("=" * 88)
    logger.info(f"[REG-MR DEBUG] === {stage} ===")

    total_blocks = 0
    total_size = 0
    segments_with_weights = 0
    for seg_idx, segment in enumerate(memory_snapshot):
        seg_size = segment.get("total_size", segment.get("size", -1))
        seg_type = segment.get("segment_type", "?")

        # Weight blocks in this segment, in address order. When merged=True,
        # coalesce physically-contiguous ones (same rule the real path uses).
        seg_weight_blocks = []
        for block in segment.get("blocks", []):
            address = block.get("address", -1)
            size = block.get("size", -1)
            state = block.get("state", "")
            if address < 0 or size < 0 or state != "active_allocated":
                continue
            if address not in weight_addr_set:
                continue
            if (
                merged
                and seg_weight_blocks
                and seg_weight_blocks[-1][0] + seg_weight_blocks[-1][1] == address
            ):
                prev_addr, prev_size = seg_weight_blocks[-1]
                seg_weight_blocks[-1] = (prev_addr, prev_size + size)
            else:
                seg_weight_blocks.append((address, size))

        # Weight-focused: only print segments that actually hold weights.
        if not seg_weight_blocks:
            continue
        segments_with_weights += 1

        seg_weight_size = sum(size for _, size in seg_weight_blocks)
        logger.info(
            f"[REG-MR DEBUG] segment #{seg_idx}: seg_size={_mb(seg_size)} "
            f"type={seg_type} weight_blocks={len(seg_weight_blocks)} "
            f"weight_size={_mb(seg_weight_size)}"
        )
        for blk_idx, (address, size) in enumerate(seg_weight_blocks):
            logger.info(
                f"[REG-MR DEBUG]     weight block #{blk_idx}: "
                f"addr={hex(address)} size={_mb(size)}"
            )
            total_blocks += 1
            total_size += size

    logger.info("-" * 88)
    logger.info(
        f"[REG-MR DEBUG] SUMMARY ({stage}): "
        f"{total_blocks} weight block(s) across "
        f"{segments_with_weights} segment(s)"
    )
    logger.info(
        f"[REG-MR DEBUG] SUMMARY ({stage}): total size of all weights = "
        f"{_mb(total_size)} ({total_size} bytes)"
    )
    logger.info("=" * 88)


def register_memory_region(model, transfer_engine):
    if importlib.util.find_spec("torch") is None:
        return register_memory_region_v1(model, transfer_engine)
    else:
        return register_memory_region_v2(model, transfer_engine)


def register_memory_region_v1(model, transfer_engine):
    start_tic = time.time()

    weight_mr_dict = {}
    for name, weight in model.named_parameters():
        ret = transfer_engine.register_memory(
            weight.data_ptr(), weight.numel() * weight.element_size()
        )
        if ret != 0:
            raise RuntimeError(
                f"register memory failed for weight {name}, error: {ret}"
            )
        weight_mr_dict[name] = (
            weight.data_ptr(),
            weight.numel(),
            weight.element_size(),
        )

    end_tic = time.time()
    logger.debug(f"Register memory region time: {(end_tic - start_tic):.4f}s")
    return weight_mr_dict


def register_memory_region_v2(model, transfer_engine):
    start_tic = time.time()

    weight_mr_dict = {}
    weight_addr_set = set()
    for name, weight in model.named_parameters():
        weight_mr_dict[name] = (
            weight.data_ptr(),
            weight.numel(),
            weight.element_size(),
        )
        weight_addr_set.add(weight.data_ptr())

    import torch

    memory_snapshot = torch.cuda.memory.memory_snapshot()
    _debug_dump_weight_memory_layout(memory_snapshot, weight_addr_set, merged=False)
    weight_blocks_for_reg_mr = []
    # Blocks in each segment have continuous physical addresses,
    # so they can be merged for memory registration.
    for segment in memory_snapshot:
        current_weight_block = None
        blocks = segment.get("blocks", [])
        for block in blocks:
            address = block.get("address", -1)
            size = block.get("size", -1)
            state = block.get("state", "")
            if address < 0 or size < 0 or state == "":
                continue
            # Only register active allocated memory blocks that hold weights.
            if state == "active_allocated":
                if address in weight_addr_set:
                    if current_weight_block is None:
                        current_weight_block = (address, size)
                    elif current_weight_block[0] + current_weight_block[1] == address:
                        current_weight_block = (
                            current_weight_block[0],
                            current_weight_block[1] + size,
                        )
                    else:
                        weight_blocks_for_reg_mr.append(current_weight_block)
                        current_weight_block = (address, size)
        if current_weight_block is not None:
            weight_blocks_for_reg_mr.append(current_weight_block)

    _debug_dump_weight_memory_layout(memory_snapshot, weight_addr_set, merged=True)

    # Register merged memory blocks that hold weights.
    for weight_block in weight_blocks_for_reg_mr:
        address, size = weight_block
        ret = transfer_engine.register_memory(address, size)
        if ret != 0:
            raise RuntimeError(
                f"register memory failed for weight block at address {address} with size {size}, error: {ret}"
            )

    end_tic = time.time()
    logger.debug(f"Register memory region v2 time: {(end_tic - start_tic):.4f}s")
    return weight_mr_dict


def register_memory_region_nixl(model, nixl_agent, gpu_id):
    """NIXL parallel of ``register_memory_region_v2`` for the weight-transfer backend.

    Same structure as the Mooncake ``v2`` path (build the per-tensor dict, merge
    physically-contiguous weight blocks from the CUDA memory snapshot, register the
    merged blocks), with only the two NIXL-specific differences:

    - ``weights_info_dict`` entries widen to 4 fields
      ``(data_ptr, numel, element_size, gpu_id)`` -- the extra ``gpu_id`` /
      ``device_id`` the peer needs to address VRAM.
    - registration uses ``agent.register_memory([(addr, size, gpu_id, "")], "VRAM")``
      (instead of the per-block ``engine.register_memory(addr, size)``) and the
      returned ``descs`` handle is kept alive on ``nixl_agent._weight_descs`` for the
      process lifetime (design Q2).

    SGLang is the passive, export-only target: it registers buffers and publishes
    descriptors; Miles performs the RDMA WRITE. No ``transfer()`` is issued here.
    """
    start_tic = time.time()

    weight_mr_dict = {}
    weight_addr_set = set()
    for name, weight in model.named_parameters():
        weight_mr_dict[name] = (
            weight.data_ptr(),
            weight.numel(),
            weight.element_size(),
            gpu_id,
        )
        weight_addr_set.add(weight.data_ptr())

    import torch

    memory_snapshot = torch.cuda.memory.memory_snapshot()
    _debug_dump_weight_memory_layout(memory_snapshot, weight_addr_set, merged=False)
    weight_blocks_for_reg_mr = []
    # Blocks in each segment have continuous physical addresses,
    # so they can be merged for memory registration.
    for segment in memory_snapshot:
        current_weight_block = None
        blocks = segment.get("blocks", [])
        for block in blocks:
            address = block.get("address", -1)
            size = block.get("size", -1)
            state = block.get("state", "")
            if address < 0 or size < 0 or state == "":
                continue
            # Only register active allocated memory blocks that hold weights.
            if state == "active_allocated":
                if address in weight_addr_set:
                    if current_weight_block is None:
                        current_weight_block = (address, size)
                    elif current_weight_block[0] + current_weight_block[1] == address:
                        current_weight_block = (
                            current_weight_block[0],
                            current_weight_block[1] + size,
                        )
                    else:
                        weight_blocks_for_reg_mr.append(current_weight_block)
                        current_weight_block = (address, size)
        if current_weight_block is not None:
            weight_blocks_for_reg_mr.append(current_weight_block)

    _debug_dump_weight_memory_layout(memory_snapshot, weight_addr_set, merged=True)

    # Register merged memory blocks that hold weights. Unlike Mooncake (per-block
    # register returning an int), NIXL registers VRAM blocks in one call and returns
    # a descriptor handle that must be kept alive.
    reg_addrs = [
        (address, size, gpu_id, "") for address, size in weight_blocks_for_reg_mr
    ]
    descs = nixl_agent.register_memory(reg_addrs, "VRAM")
    if not descs:
        raise RuntimeError(
            f"NIXL memory registration failed for {len(reg_addrs)} weight blocks"
        )
    # Keep the descriptor handle alive on the agent for the process lifetime so the
    # RDMA registration is not torn down (design Q2: pinned until process exit).
    nixl_agent._weight_descs = descs

    end_tic = time.time()
    logger.debug(f"Register memory region nixl time: {(end_tic - start_tic):.4f}s")
    return weight_mr_dict
