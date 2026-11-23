from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.layer_boundary import ops
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "sizes,actual,rows,output_rows,cp,eligible,available,pull",
    [
        ([2] * 8, [2] * 8, 16, 2, 1, True, True, True),
        ([1, 3] * 4, [1, 3] * 4, 16, 2, 1, True, True, False),
        ([0, 4] * 4, [0, 4] * 4, 16, 2, 1, True, True, False),
        ([2] * 8, [0, 4] * 4, 16, 2, 1, True, True, False),
        ([2] * 8, [1, 3] * 4, 16, 2, 1, True, True, False),
        ([2] * 8, [2] * 8, 24, 2, 1, True, True, False),
        ([2] * 8, [2] * 8, 16, 1, 1, True, True, False),
        ([2] * 4, [2] * 4, 8, 2, 1, True, True, False),
        ([2] * 8, [2] * 8, 16, 2, 2, True, True, False),
        ([2] * 8, [2] * 8, 16, 2, 1, False, True, False),
        ([2] * 8, [2] * 8, 16, 2, 1, True, False, False),
        (None, None, 16, 2, 1, True, True, False),
    ],
)
def test_equal_rank_major_shards_only(
    sizes, actual, rows, output_rows, cp, eligible, available, pull
):
    group = Mock()
    comm = Mock()
    comm.can_reduce_scatter.return_value = eligible
    parallel = SimpleNamespace(tp_size=8, attn_dp_size=8, attn_cp_size=cp, tp_group=group)
    hidden = torch.randn(rows, 32)
    output = torch.empty(output_rows, 32)
    batch = SimpleNamespace(global_num_tokens_cpu=actual)
    with (
        patch.object(ops, "get_parallel", return_value=parallel),
        patch.object(ops, "get_dp_global_num_tokens", return_value=sizes),
        patch.object(ops, "_nvlink_dp_comm", return_value=comm if available else None),
    ):
        ops.dp_reduce_scatterv(output, hidden, batch)
    if pull:
        comm.can_reduce_scatter.assert_called_once_with(hidden, output)
        comm.reduce_scatter.assert_called_once_with(hidden, output)
        group.reduce_scatterv.assert_not_called()
    else:
        comm.reduce_scatter.assert_not_called()
        group.reduce_scatterv.assert_called_once_with(hidden, output=output, sizes=sizes)


@pytest.mark.parametrize("sizes,pull", [([2] * 8, True), ([1, 3] * 4, False)])
def test_deepseek_transport_uses_the_boundary_dispatch(sizes, pull):
    from sglang.srt.models.deepseek_common import moe_transport

    group, comm = Mock(), Mock()
    comm.can_reduce_scatter.return_value = True
    parallel = SimpleNamespace(tp_size=8, attn_dp_size=8, attn_cp_size=1, tp_group=group)
    hidden, output = torch.randn(16, 32), torch.empty(2, 32)
    transport = object.__new__(moe_transport.MoeRowTransport)
    transport.batch = SimpleNamespace(global_num_tokens_cpu=sizes)
    transport._use_cp = transport._use_tp_attn_a2a_scatter = False
    transport._use_tp_moe_gather = transport._use_reduce_scatterv = True
    transport._use_reduce_scatter = False
    transport._shared_local = transport._a2a_scatter_chunks = None
    with (
        patch.object(ops, "get_parallel", return_value=parallel),
        patch.object(ops, "get_dp_global_num_tokens", return_value=sizes),
        patch.object(ops, "_nvlink_dp_comm", return_value=comm),
        patch.object(moe_transport, "get_parallel", return_value=parallel),
        patch.object(moe_transport, "get_local_dp_buffer", return_value=output),
        patch.object(moe_transport, "should_use_dp_reduce_scatterv", return_value=True),
    ):
        result = transport.return_output(hidden)
    assert result is output
    assert comm.reduce_scatter.called == pull
    assert group.reduce_scatterv.called != pull


@pytest.mark.parametrize(
    "role,tbo,compiling,world_gather,disabled,expected",
    [
        ("decode", False, False, False, False, True),
        ("mix", False, False, False, False, False),
        ("prefill", False, False, False, False, False),
        ("decode", True, False, False, False, False),
        ("decode", False, True, False, False, False),
        ("decode", False, False, True, False, False),
        ("decode", False, False, False, True, False),
    ],
)
def test_existing_role_and_shared_semaphore_gates(
    role, tbo, compiling, world_gather, disabled, expected
):
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.srt.layers import dp_attention, dsv41_sm90
    from sglang.srt.layers.moe import utils

    comm = object.__new__(CustomAllReduceV2)
    comm.disabled = disabled
    parallel = SimpleNamespace(
        tp_size=8, attn_dp_size=8, tp_group=SimpleNamespace(ca_comm=comm)
    )
    with (
        patch.object(dsv41_sm90, "dsv41_sm90_on", return_value=True),
        patch.object(dsv41_sm90, "dsv41_sm90_role", return_value=role),
        patch.object(utils, "is_tbo_enabled", return_value=tbo),
        patch.object(torch.compiler, "is_compiling", return_value=compiling),
        patch.object(dp_attention, "world_dp_gather_enabled", return_value=world_gather),
        patch.object(dp_attention, "get_parallel", return_value=parallel),
    ):
        assert (dp_attention._nvlink_dp_comm() is comm) == expected
