"""Unit tests for ParallelismContext CPU-replica mock groups."""

import unittest
from unittest.mock import MagicMock

import torch.distributed as dist

from sglang.srt.distributed.parallel_state import (
    ParallelismContext,
    RankParallelismConfig,
    get_tp_group,
    get_world_group,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestParallelismContextCpuGroup(CustomTestCase):
    def test_cpu_group_is_a_real_process_group(self):
        """v0.5.16 passes coordinator.cpu_group to torch.distributed.

        A MagicMock is rejected with "not initialized in the world group map".
        ParallelismContext must attach a 1-rank gloo group so CPU replicas
        can still call get_model().
        """
        config = RankParallelismConfig(
            tp_size=4, tp_rank=1, pp_size=4, pp_rank=2, world_size=32, global_rank=7
        )
        with ParallelismContext(config):
            tp_group = get_tp_group()
            self.assertEqual(tp_group.world_size, 4)
            self.assertEqual(tp_group.rank_in_group, 1)
            self.assertIsInstance(tp_group.cpu_group, dist.ProcessGroup)
            self.assertIsInstance(tp_group.device_group, dist.ProcessGroup)
            self.assertNotIsInstance(tp_group.cpu_group, MagicMock)

            dist.barrier(group=tp_group.cpu_group)
            self.assertEqual(dist.get_world_size(group=tp_group.cpu_group), 1)
            self.assertNotEqual(dist.get_backend(group=tp_group.cpu_group), "nccl")

            world_group = get_world_group()
            self.assertIsInstance(world_group.cpu_group, dist.ProcessGroup)
            dist.barrier(group=world_group.cpu_group)


if __name__ == "__main__":
    unittest.main()
