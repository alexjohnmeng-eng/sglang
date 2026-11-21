"""Optional DeepSeek-V4.1 H20 gc-moe adapter.

gc-moe owns expert dispatch and combine through its symmetric workspace. This
adapter is intentionally narrow: it accepts the standard dispatcher output,
uses global expert ids, and only runs on decode or target-verify batches. Any
unsupported shape returns ``None`` so the caller can use Humming unchanged.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.device import get_device_sm

logger = logging.getLogger(__name__)

_GC_MOE_CAPACITY = 256


def enabled() -> bool:
    return envs.SGLANG_OPT_DSV41_GC_MOE.get()


def available() -> bool:
    if not enabled() or get_device_sm() != 90:
        return False
    try:
        import gc_moe  # noqa: F401
    except ImportError:
        return False
    return True


def should_use_for_current_batch() -> bool:
    if not available():
        return False
    try:
        mode = get_forward_batch().forward_mode
    except RuntimeError:
        return False
    return mode.is_decode() or mode.is_target_verify()


def prepare_layer(layer: torch.nn.Module) -> None:
    """Transform raw MXFP4 weights and retain a stable graph workspace."""
    if not available() or getattr(layer, "_gc_moe_weights", None) is not None:
        return
    if getattr(layer, "num_fused_shared_experts", 0):
        logger.warning_once(
            "gc-moe is disabled for fused shared experts; using Humming fallback."
        )
        return
    if get_parallel().moe_ep_size not in (4, 8):
        return
    try:
        import gc_moe

        layer._gc_moe_weights = gc_moe.transform_weights_for_gc_moe_sm90(
            (layer.w13_weight.data, layer.w13_weight_scale_inv.data),
            (layer.w2_weight.data, layer.w2_weight_scale_inv.data),
        )
    except (AttributeError, RuntimeError, ValueError) as exc:
        logger.warning(
            "gc-moe weight preparation failed for layer %s, using Humming: %s",
            getattr(layer, "layer_id", "?"),
            exc,
        )
        return


def run(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    routed_scaling_factor: Optional[float],
) -> Optional[torch.Tensor]:
    """Run gc-moe for a local source-token batch, or return ``None``."""
    if not should_use_for_current_batch():
        return None
    weights = getattr(layer, "_gc_moe_weights", None)
    if weights is None:
        return None
    parallel = get_parallel()
    if parallel.moe_ep_size not in (4, 8):
        return None
    if (
        hidden_states.ndim != 2
        or hidden_states.dtype != torch.bfloat16
        or not hidden_states.is_contiguous()
        or hidden_states.shape[1] != 5120
        or hidden_states.shape[0] > _GC_MOE_CAPACITY
        or topk_ids.ndim != 2
        or topk_ids.shape[1] != 6
    ):
        return None
    if hidden_states.shape[0] == 0:
        return hidden_states.new_empty(hidden_states.shape)

    try:
        import gc_moe
        from sglang.srt.layers.moe.moe_runner.humming import humming_quant_input_rows

        buffer = getattr(layer, "_gc_moe_buffer", None)
        if buffer is None:
            buffer = gc_moe.get_symm_buffer_for_gc_moe_sm90(
                parallel.moe_ep_group.device_group,
                384,
                _GC_MOE_CAPACITY,
                6,
                5120,
                2304,
            )
            layer._gc_moe_buffer = buffer
        quantized = humming_quant_input_rows(layer, hidden_states)
        if quantized is None:
            return None
        x_fp8, x_scale = quantized
        if x_fp8.shape != hidden_states.shape or x_scale.shape != (
            hidden_states.shape[0],
            hidden_states.shape[1] // 128,
        ):
            return None
        buffer.x[: hidden_states.shape[0]].copy_(x_fp8)
        buffer.x_sf[: hidden_states.shape[0]].copy_(x_scale)
        buffer.topk_idx[: hidden_states.shape[0]].copy_(topk_ids)
        weights_for_call = topk_weights.float()
        if routed_scaling_factor is not None:
            weights_for_call = weights_for_call * float(routed_scaling_factor)
        buffer.topk_weights[: hidden_states.shape[0]].copy_(weights_for_call)
        stats = getattr(layer, "_gc_moe_stats", None)
        if stats is None:
            local_experts = 384 // parallel.moe_ep_size
            stats = torch.zeros(
                local_experts, dtype=torch.int32, device=hidden_states.device
            )
            layer._gc_moe_stats = stats
        return gc_moe.sm90_fp8_fp4_gc_moe(
            hidden_states,
            weights[0],
            weights[1],
            buffer,
            stats,
            recipe=(1, 1, 32),
            activation="swiglu",
            activation_clamp=10.0,
            fast_math=True,
            num_tokens_bound=_GC_MOE_CAPACITY,
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        logger.warning_once("gc-moe call failed, falling back to Humming: %s", exc)
        return None
