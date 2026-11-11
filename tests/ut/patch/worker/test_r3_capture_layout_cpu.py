# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware-independent tests for the production capture functions.

Run with --noconftest on a workstation without vLLM/torch_npu. Functions and the
communication enum are read from their production AST; only the collective and
forward context are substituted. No Ascend plugin global patches are imported.
"""

import ast
import logging
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def capture_api():
    root = Path(__file__).parents[4]
    enum_tree = ast.parse((root / "vllm_ascend/ascend_forward_context.py").read_text(encoding="utf-8"))
    enum_tree.body = [n for n in enum_tree.body if isinstance(n, ast.ClassDef) and n.name == "MoECommType"]
    ns = {"Enum": Enum, "torch": torch, "logger": logging.getLogger(__name__)}
    exec(compile(enum_tree, "ascend_forward_context.py", "exec"), ns)
    context = SimpleNamespace(dp_metadata=None, moe_comm_type=ns["MoECommType"].ALLTOALL)
    ns["get_forward_context"] = lambda: context
    ns["get_tp_group"] = lambda: SimpleNamespace(device_group="mock-hccl")
    ns["_EXTRA_CTX"] = context
    ns["dist"] = SimpleNamespace()
    path = root / "vllm_ascend/patch/worker/patch_routed_experts_capture.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    exec(compile(tree, str(path), "exec"), ns)
    return SimpleNamespace(**ns), context


@pytest.mark.parametrize("comm_name", ["ALLTOALL", "MC2", "FUSED_MC2"])
@pytest.mark.parametrize("tokens", [1, 7, 8, 9, 63, 65])
@pytest.mark.parametrize("rank", [0, 7])
def test_tp8_single_dp_reconstructs_every_token(capture_api, comm_name, tokens, rank, caplog):
    api, ctx = capture_api
    ctx.moe_comm_type = getattr(api.MoECommType, comm_name)
    full = torch.arange(tokens * 2, dtype=torch.int32).view(tokens, 2)
    padded_count = max(tokens, 8) if comm_name == "ALLTOALL" else ((tokens + 7) // 8 + 1) * 8
    padded = F.pad(full, (0, 0, 0, padded_count - tokens), value=-100)
    shards = torch.tensor_split(padded, 8)
    ctx.moe_comm_method = SimpleNamespace(
        prepare_finalize=SimpleNamespace(num_tokens=tokens, replace_allreduce=False, enable_shared_expert_dp=False)
    )
    calls = []

    def all_gather(out, local, group):
        assert group == "mock-hccl"
        assert len({tuple(t.shape) for t in [*out, local]}) == 1
        width = max(len(s) for s in shards)
        assert local.shape[0] == width
        torch.testing.assert_close(local[: len(shards[rank])], shards[rank])
        for target, source in zip(out, shards, strict=True):
            target.copy_(F.pad(source, (0, 0, 0, width - len(source))))
        calls.append(True)

    api.dist.all_gather = all_gather
    capturer = SimpleNamespace(tp_size=8, dp_rank=0, device_buffer=torch.zeros(100, 3, 2, dtype=torch.int32))
    with caplog.at_level(logging.WARNING):
        api.capture(capturer, 1, shards[rank])
    assert len(calls) == 1
    torch.testing.assert_close(capturer.device_buffer[:tokens, 1], full)
    assert capturer.device_buffer[tokens:, 1].count_nonzero() == 0
    assert "equal_size_all_gather=True" in caplog.text


@pytest.mark.parametrize("layout", ["allgather", "replace_allreduce", "shared_expert_dp"])
def test_replicated_routes_never_gather(capture_api, layout):
    api, ctx = capture_api
    full = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    ctx.moe_comm_type = api.MoECommType.ALLGATHER if layout == "allgather" else api.MoECommType.ALLTOALL
    # No num_tokens: prepare deliberately does not set it when bypassing the split.
    ctx.moe_comm_method = SimpleNamespace(
        prepare_finalize=SimpleNamespace(
            replace_allreduce=layout == "replace_allreduce",
            enable_shared_expert_dp=layout == "shared_expert_dp",
        )
    )
    capturer = SimpleNamespace(tp_size=8, dp_rank=0, device_buffer=torch.zeros(100, 3, 2, dtype=torch.int32))
    api.capture(capturer, 1, full)
    torch.testing.assert_close(capturer.device_buffer[:3, 1], full)


@pytest.mark.parametrize(
    "tokens_by_dp,rank,rows,expected_start",
    [
        ([5, 7], 0, 12, 0),
        ([5, 7], 1, 12, 5),
        ([5, 7], 1, 7, 0),
        ([5, 7], 0, 14, 0),
        ([5, 7], 1, 14, 7),
    ],
)
def test_existing_dp_layouts(capture_api, tokens_by_dp, rank, rows, expected_start):
    api, ctx = capture_api
    ctx.dp_metadata = SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor(tokens_by_dp))
    full = torch.arange(rows * 2, dtype=torch.int32).view(rows, 2)
    capturer = SimpleNamespace(tp_size=1, dp_rank=rank, device_buffer=torch.zeros(100, 3, 2, dtype=torch.int32))
    api.capture(capturer, 1, full)
    count = tokens_by_dp[rank]
    torch.testing.assert_close(capturer.device_buffer[:count, 1], full[expected_start : expected_start + count])


def test_missing_prepare_state_and_invalid_layer_fail(capture_api):
    api, ctx = capture_api
    capturer = SimpleNamespace(tp_size=8, dp_rank=0, device_buffer=torch.zeros(100, 3, 2, dtype=torch.int32))
    routes = torch.ones(1, 2, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="missing prepare_finalize"):
        api.capture(capturer, 0, routes)
    ctx.moe_comm_type = api.MoECommType.ALLGATHER
    with pytest.raises(RuntimeError, match="layer_id"):
        api.capture(capturer, 3, routes)
