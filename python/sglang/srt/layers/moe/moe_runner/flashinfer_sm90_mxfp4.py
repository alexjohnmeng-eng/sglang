"""Opt-in FlashInfer 0.7.5 SM90 MXFP4 weight preparation.

The public FlashInfer MegaMoE API is kept separate from the existing SM100
``moe_ep`` integration. This first stage only prepares checkpoint weights so
the conversion can be validated without changing the online Humming runner.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.device import get_device_sm

logger = logging.getLogger(__name__)
_BUFFERS: dict[tuple[int, int, int], object] = {}


def enabled() -> bool:
    return (
        envs.SGLANG_OPT_DSV41_FLASHINFER_SM90_PREPARE.get()
        or envs.SGLANG_OPT_DSV41_FLASHINFER_SM90_RUN.get()
    )


def run_enabled() -> bool:
    return envs.SGLANG_OPT_DSV41_FLASHINFER_SM90_RUN.get()


def _use_deepgemm() -> bool:
    return envs.SGLANG_EXPERIMENT_DSV41_SM90_MEGAMOE_DEEPGEMM.get()


def available() -> bool:
    if not enabled() or get_device_sm() != 90:
        return False
    try:
        if _use_deepgemm():
            import deep_gemm_mega  # noqa: F401
        else:
            import flashinfer.megamoe  # noqa: F401
    except ImportError:
        return False
    return get_parallel().moe_ep_size == 8


def _raw_scale_bytes(scale: torch.Tensor) -> torch.Tensor:
    """Return the E8M0 byte representation expected by FlashInfer."""
    if scale.dtype == torch.uint8:
        return scale.contiguous()
    if scale.dtype == getattr(torch, "float8_e8m0fnu", None):
        return scale.contiguous().view(torch.uint8)
    if scale.is_floating_point():
        positive = scale.float()
        exponent = torch.where(
            positive > 0,
            torch.round(torch.log2(positive)).clamp(-127, 128) + 127,
            torch.zeros_like(positive),
        )
        return exponent.to(torch.uint8).contiguous()
    raise TypeError(f"unsupported MXFP4 scale dtype {scale.dtype}")


def _checkpoint_to_marlin_bytes(packed: torch.Tensor) -> torch.Tensor:
    """Transpose eight checkpoint nibbles into one Marlin four-byte word."""
    packed = packed.contiguous().view(torch.uint8)
    if packed.shape[-1] % 4:
        raise ValueError(
            f"packed MXFP4 K dimension is not word aligned: {packed.shape}"
        )
    words = packed.view(*packed.shape[:-1], -1, 4)
    low = words & 0x0F
    high = words >> 4
    codes = torch.stack(
        (
            low[..., 0],
            high[..., 0],
            low[..., 1],
            high[..., 1],
            low[..., 2],
            high[..., 2],
            low[..., 3],
            high[..., 3],
        ),
        dim=-1,
    )
    return ((codes[..., :4] << 4) | codes[..., 4:]).reshape_as(packed)


def prepare_layer(layer: torch.nn.Module) -> bool:
    """Prepare one EP8 layer, returning whether the conversion succeeded."""
    if not available():
        return False
    if getattr(layer, "_flashinfer_sm90_mxfp4_weights", None) is not None:
        return True
    if getattr(layer, "moe_tp_size", 1) != 1:
        return False
    try:
        w13 = layer.w13_weight.data
        w2 = layer.w2_weight.data
        w13_scale = _raw_scale_bytes(layer.w13_weight_scale_inv.data)
        w2_scale = _raw_scale_bytes(layer.w2_weight_scale_inv.data)
        if w13.shape[1:] != (4608, 2560) or w2.shape[1:] != (5120, 1152):
            logger.warning(
                "FlashInfer SM90 MXFP4 shape mismatch for layer %s: w13=%s w2=%s",
                getattr(layer, "layer_id", "?"),
                tuple(w13.shape),
                tuple(w2.shape),
            )
            return False
        if _use_deepgemm():
            from deep_gemm_mega import (
                transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90,
            )

            transformed = transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
                (w13, w13_scale), (w2, w2_scale), layout="tiled"
            )
        else:
            from flashinfer.megamoe import transform_mxfp4_weights_for_mega_moe

            transformed = transform_mxfp4_weights_for_mega_moe(
                (_checkpoint_to_marlin_bytes(w13), w13_scale),
                (_checkpoint_to_marlin_bytes(w2), w2_scale),
            )
        layer._flashinfer_sm90_mxfp4_weights = transformed
        layer._flashinfer_sm90_num_local_experts = w13.shape[0]
        logger.info(
            "%s SM90 MXFP4 prepared layer=%s local_experts=%d",
            "DeepGEMM" if _use_deepgemm() else "FlashInfer",
            getattr(layer, "layer_id", "?"),
            w13.shape[0],
        )
        return True
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning(
            "FlashInfer SM90 MXFP4 preparation failed for layer %s: %s",
            getattr(layer, "layer_id", "?"),
            exc,
        )
        return False


def get_prepared_weights(layer: torch.nn.Module):
    return getattr(layer, "_flashinfer_sm90_mxfp4_weights", None)


def _get_buffer(capacity: int, num_experts: int):
    parallel = get_parallel()
    group = parallel.moe_ep_group.device_group
    key = (id(group), capacity, num_experts)
    buffer = _BUFFERS.get(key)
    if buffer is None:
        if _use_deepgemm():
            from deep_gemm_mega import get_symm_buffer_for_sm90_mega_moe

            buffer = get_symm_buffer_for_sm90_mega_moe(
                group, num_experts, capacity, 6, 5120, 2304, fused=True
            )
        else:
            from flashinfer.megamoe import MegaMoESymmBuffer

            buffer = MegaMoESymmBuffer(group, num_experts, capacity, 6, 5120, 2304)
        _BUFFERS[key] = buffer
        logger.info(
            "%s SM90 MXFP4 buffer ready capacity=%d bytes=%d",
            "DeepGEMM" if _use_deepgemm() else "FlashInfer",
            capacity,
            buffer.buffer.nbytes,
        )
    return buffer


def run(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    routed_scaling_factor: Optional[float],
) -> torch.Tensor:
    """Run standard dispatch with IDs already localized by the dispatcher."""
    parallel = get_parallel()
    local_experts = layer._flashinfer_sm90_num_local_experts
    ids = topk_ids.to(torch.int64)
    ids = torch.where(ids >= 0, ids + parallel.moe_ep_rank * local_experts, ids)
    if (
        routed_scaling_factor is not None
        and not layer.should_fuse_routed_scaling_factor_in_topk
    ):
        topk_weights = topk_weights * routed_scaling_factor
    return run_global_ids(layer, hidden_states, topk_weights, ids)


def run_global_ids(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    tune_prefill: bool = False,
) -> torch.Tensor:
    """Dispatch CP-local rows whose expert IDs are physical global IDs."""
    if not run_enabled():
        raise RuntimeError("FlashInfer SM90 MXFP4 runner is not enabled")
    weights = get_prepared_weights(layer)
    if weights is None:
        raise RuntimeError("FlashInfer SM90 MXFP4 weights were not prepared")
    if (
        hidden_states.ndim != 2
        or hidden_states.dtype != torch.bfloat16
        or hidden_states.shape[1] != 5120
        or topk_ids.shape != (hidden_states.shape[0], 6)
    ):
        raise ValueError(
            "unsupported FlashInfer SM90 MXFP4 batch shape: "
            f"hidden={tuple(hidden_states.shape)} topk={tuple(topk_ids.shape)}"
        )
    num_tokens = hidden_states.shape[0]
    if num_tokens == 0:
        return hidden_states.new_empty(hidden_states.shape)
    parallel = get_parallel()
    total_experts = layer._flashinfer_sm90_num_local_experts * parallel.moe_ep_size
    capacity = envs.SGLANG_OPT_DSV41_FLASHINFER_SM90_MAX_TOKENS.get()
    if parallel.attn_cp_size != parallel.moe_ep_size:
        capacity = max(capacity, num_tokens)
    if capacity <= 0:
        raise ValueError("SM90 MegaMoE needs positive token capacity")
    if (
        tune_prefill
        and _use_deepgemm()
        and envs.SGLANG_OPT_DSV41_SM90_MEGAMOE_16K_PREFILL.get()
        and num_tokens in (16384, 32768)
        and parallel.attn_cp_size == parallel.moe_ep_size == 8
    ):
        # The default early-combine bitmap supports 16K local rows. The
        # experimental 32K setting is isolated because its bitmap/workspace
        # footprint is larger and may fail graph capture on older wheels.
        capacity = max(
            capacity,
            32768 if os.environ.get("SGLANG_EXPERIMENT_DSV41_MEGA_MOE_32K") == "1" else 16384,
        )
    buffer = _get_buffer(capacity, total_experts)
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    output = torch.empty_like(hidden_states)
    ids = topk_ids.to(torch.int64)
    route_weights = topk_weights.float().contiguous()
    if _use_deepgemm():
        from deep_gemm_mega import fp8_mxfp4_mega_moe

        for offset in range(0, num_tokens, capacity):
            part = slice(offset, min(offset + capacity, num_tokens))
            rows = part.stop - part.start
            sglang_per_token_group_quant_fp8(
                hidden_states[part],
                128,
                output_q=buffer.x[:rows],
                output_s=buffer.x_sf[:rows],
            )
            buffer.topk_idx[:rows].copy_(ids[part])
            buffer.topk_weights[:rows].copy_(route_weights[part])
            fp8_mxfp4_mega_moe(
                output[part],
                *weights,
                buffer,
                activation_clamp=10.0,
                num_tokens_bound=rows,
            )
        return output

    from flashinfer import megamoe

    if (
        tune_prefill
        and num_tokens == capacity == 8192
        and total_experts == 448
        and parallel.attn_cp_size == parallel.moe_ep_size == 8
        and buffer.num_sms == 78
    ):
        plan = megamoe.select_mxfp4_fused_plan(
            buffer.num_sms,
            parallel.moe_ep_size,
            total_experts,
            buffer.num_topk,
            buffer.hidden,
            buffer.intermediate_hidden,
            capacity,
            num_tokens,
        )
        sglang_per_token_group_quant_fp8(
            hidden_states, 128, output_q=buffer.x, output_s=buffer.x_sf
        )
        buffer.topk_idx.copy_(ids)
        buffer.topk_weights.copy_(route_weights)
        buffer.commit_staged(num_tokens)
        return megamoe._run_kernel(
            megamoe.gen_megamoe_sm90_mxfp4_module,
            "megamoe_sm90_mxfp4_fused",
            replace(plan, num_stages=4),
            output,
            *weights,
            buffer,
            num_tokens,
            None,
            None,
            None,
        )
    for offset in range(0, num_tokens, capacity):
        part = slice(offset, min(offset + capacity, num_tokens))
        x_fp8, x_sf = sglang_per_token_group_quant_fp8(hidden_states[part], 128)
        buffer.stage(x_fp8, x_sf, ids[part].contiguous(), route_weights[part])
        megamoe.mxfp4_mega_moe(
            output[part],
            weights[0],
            weights[1],
            buffer,
            num_tokens=x_fp8.shape[0],
        )
    return output
