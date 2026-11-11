# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType

logger = logging.getLogger(__name__)


def _gather_tp_routes(topk_ids: torch.Tensor, token_count: int, tp_size: int, comm_type: MoECommType) -> torch.Tensor:
    """Reconstruct prepare()'s token layout using equal-sized HCCL collectives.

    ALLTOALL uses tensor_split (unequal shards for e.g. 65 tokens / TP=8).
    MC2 pads first. Padding the collective inputs avoids unsupported uneven
    all_gather sizes on HCCL; the ALLTOALL padding is removed per rank.
    """
    if comm_type == MoECommType.ALLTOALL:
        padded_count = max(token_count, tp_size)
        sizes = [padded_count // tp_size + (rank < padded_count % tp_size) for rank in range(tp_size)]
    else:
        sizes = [topk_ids.shape[0]] * tp_size
    width = max(sizes)
    if topk_ids.shape[0] > width:
        raise RuntimeError(f"[R3][TP_CAPTURE] unexpected local rows={topk_ids.shape[0]} shard_sizes={sizes}")
    local = torch.nn.functional.pad(topk_ids, (0, 0, 0, width - topk_ids.shape[0])).contiguous()
    gathered = [torch.empty_like(local) for _ in range(tp_size)]
    dist.all_gather(gathered, local, group=get_tp_group().device_group)
    return torch.cat([part[:size] for part, size in zip(gathered, sizes)], dim=0)[:token_count]


def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
    """Capture expert routing decisions for a specific layer.

    Under data parallelism, ``topk_ids`` may have four different batch
    layouts depending on where the DP combine happens and whether
    Sequence Parallelism (SP) is active for the MoE layer:
      - ``n == total`` (naive dispatch): all DP ranks' tokens are
        concatenated before routing; we slice out this rank's span
        using the cumulative per-rank counts.
      - ``n == token_num_per_dp`` (modular-kernel path): DP combine
        happens inside ``quant_method.apply``; ``select_experts`` only
        ever sees this rank's tokens, so we take the whole tensor.
      - ``n == total_with_padding`` (padded all-gather path): tokens are
        padded to max_tokens before all-gather across DP group; each
        DP rank occupies a contiguous block of size max_tokens, and we
        extract only the actual tokens for this rank (skip padding).
        When all DP ranks have equal token counts, ``total == total_with_padding``,
        so the naive dispatch branch fires instead (equivalent result).
      - ``n == ceil(token_num_per_dp / tp_size)`` (SP + modular-kernel
        path): tokens were split along dim=0 across the TP group by
        ``_sequence_parallel_context``
        (``moe_runner_base.py:_sequence_parallel_context``), so each
        TP rank only sees its shard. We all-gather along dim=0 to
        reconstruct this DP rank's full routing tensor. SP pads with
        ceil-div (see ``_compute_sp_num_tokens`` in
        ``forward_context.py``), so the gathered tensor may contain a
        few trailing padding rows which are trimmed by the downstream
        ``[:token_num_per_dp]`` slice.

    Args:
        layer_id: The layer index.
        topk_ids: Tensor of shape (batch_size, num_routed_experts).
    """

    ctx = get_forward_context()
    if ctx.dp_metadata is None:  # single dp
        start_loc = 0
        end_loc = topk_ids.shape[0]
        token_num_per_dp = topk_ids.shape[0]
        comm_type = getattr(ctx, "moe_comm_type", None)
        method = getattr(ctx, "moe_comm_method", None)
        prepare = getattr(method, "prepare_finalize", None)
        sharded = comm_type in (MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2)
        if self.tp_size > 1 and sharded:
            if prepare is None:
                raise RuntimeError("[R3][TP_CAPTURE] missing prepare_finalize for TP-sharded capture")
            split = not (prepare.replace_allreduce or prepare.enable_shared_expert_dp)
            full_count = getattr(prepare, "num_tokens", None) if split else topk_ids.shape[0]
            if not isinstance(full_count, int) or full_count <= 0:
                raise RuntimeError("[R3][TP_CAPTURE] missing prepare_finalize.num_tokens for TP-sharded capture")
            if split:
                local_rows = topk_ids.shape[0]
                topk_ids = _gather_tp_routes(topk_ids, full_count, self.tp_size, comm_type)
                # Python runs at graph capture, not graph replay. Request-level
                # validation in verl checks every returned routing tensor.
                key = (comm_type.name, self.tp_size)
                reported = self.__dict__.setdefault("_r3_capture_layouts", set())
                if key not in reported:
                    logger.warning(
                        "[R3][TP_CAPTURE] dp=%s tp=%s comm=%s local_rows=%s reconstructed_rows=%s "
                        "equal_size_all_gather=True; graph replay reuses this capture path",
                        self.dp_rank,
                        self.tp_size,
                        comm_type.name,
                        local_rows,
                        full_count,
                    )
                    reported.add(key)
            token_num_per_dp = full_count
            end_loc = full_count
    else:  # multi dp
        num_tokens_dp = ctx.dp_metadata.num_tokens_across_dp_cpu
        token_num_per_dp = int(num_tokens_dp[self.dp_rank].item())
        total = int(num_tokens_dp.sum().item())
        n = topk_ids.shape[0]

        # Calculate total with padding for all-gather scenario.
        # When tokens are padded to max_tokens before all-gather across DP group,
        # the total size becomes max_tokens * dp_size.
        # Example: DP0 has 5 tokens, DP1 has 7 tokens, max_tokens=7.
        # After padding: DP0 has 7 tokens, DP1 has 7 tokens.
        # After all-gather: total_with_padding = 7 * 2 = 14.
        max_tokens = int(num_tokens_dp.max().item())
        total_with_padding = max_tokens * len(num_tokens_dp)

        if n == total:
            # Naive dispatch: all DP ranks' tokens concatenated
            # before routing. This rank owns tokens
            # [end_loc - token_num_per_dp, end_loc).
            cumsum = torch.cumsum(num_tokens_dp, dim=0)
            end_loc = int(cumsum[self.dp_rank].item())
            start_loc = end_loc - token_num_per_dp
        elif n == token_num_per_dp:
            # Modular-kernel path: DP combine happens inside
            # quant_method.apply; select_experts only sees this
            # rank's tokens, take the whole tensor.
            start_loc = 0
            end_loc = token_num_per_dp
        elif n == total_with_padding:
            # NOTE(Ronald1995): When all DP ranks have equal token counts,
            # total == total_with_padding, so the first branch (n == total)
            # fires instead. This overlap is intentional since both branches
            # produce equivalent results in that case.

            # Padded all-gather path: tokens are padded to max_tokens before
            # all-gather across DP group. Each DP rank occupies a contiguous
            # block of size max_tokens. Extract only the actual tokens for
            # this rank (skip padding).
            # Example: dp_rank=0, max_tokens=7, token_num_per_dp=5.
            # start_loc = 0 * 7 = 0
            # end_loc = 0 + 5 = 5 (only first 5 tokens are valid)

            start_loc = self.dp_rank * max_tokens
            end_loc = start_loc + token_num_per_dp
        elif (
            self.tp_size > 1
            and n != token_num_per_dp
            and (
                # all2all scenario use tensor split, different tp rank have different
                # size of tokens.
                n == (token_num_per_dp + self.tp_size - 1) // self.tp_size
                or n == token_num_per_dp // self.tp_size
                # mc2 scenario will pad dp tokens to max_tokens and then ceil-div.
                or n == (max_tokens + self.tp_size - 1) // self.tp_size
            )
        ):
            # SP + modular-kernel path. All-gather across the TP
            # group along dim=0 to reconstruct the full per-DP-rank
            # tensor; keep only the first ``token_num_per_dp`` rows
            # (trailing rows are SP ceil-div padding). The TP group
            # is always initialized on real rollout workers, and
            # every rank in the group reaches this branch in
            # lockstep (bind is per-FusedMoE layer, SP is a global
            # condition), so a bare all_gather here will not
            # deadlock -- let it raise if the precondition is
            # violated rather than skip silently.
            #
            # ``topk_ids`` is already whatever the router produced
            # (typically int32/int64, both supported by NCCL); the
            # downstream ``device_buffer[...] = topk_ids[...]``
            # setitem narrows into int32 automatically.

            # NOTE(Ronald1995): if total_num_per_dp == max_tokens,
            # it will be both all2all and mc2 scenario.
            # but we fires all2all scenario first.
            # the result will be the same.
            # all2all scenario in vllm-ascend.
            topk_ids = _gather_tp_routes(topk_ids, token_num_per_dp, self.tp_size, _EXTRA_CTX.moe_comm_type)
            start_loc = 0
            end_loc = token_num_per_dp
        else:
            sp_expected = (token_num_per_dp + self.tp_size - 1) // self.tp_size if self.tp_size > 0 else -1
            raise AssertionError(
                "RoutedExpertsCapturer: unexpected topk_ids batch "
                f"dim {n} (expected {total}, {token_num_per_dp}, "
                f"{total_with_padding}, or {sp_expected} for "
                f"dp_rank={self.dp_rank}, tp_size={self.tp_size})"
            )

    if layer_id < 0 or layer_id >= self.device_buffer.shape[1]:
        raise RuntimeError(f"[R3][TP_CAPTURE] layer_id={layer_id} outside buffer {tuple(self.device_buffer.shape)}")
    if token_num_per_dp > self.device_buffer.shape[0] or end_loc - start_loc != token_num_per_dp:
        raise RuntimeError(f"[R3][TP_CAPTURE] token_count={token_num_per_dp} exceeds capture buffer/layout")

    self.device_buffer[:token_num_per_dp, layer_id, :] = topk_ids[start_loc:end_loc, :]


RoutedExpertsCapturer.capture = capture
