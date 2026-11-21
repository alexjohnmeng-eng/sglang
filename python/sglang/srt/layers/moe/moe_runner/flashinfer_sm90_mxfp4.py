"""Opt-in FlashInfer 0.7.5 SM90 MXFP4 weight preparation.

The public FlashInfer MegaMoE API is kept separate from the existing SM100
``moe_ep`` integration. This first stage only prepares checkpoint weights so
the conversion can be validated without changing the online Humming runner.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.device import get_device_sm

logger = logging.getLogger(__name__)
_BUFFERS: dict[tuple[int, int], object] = {}


def enabled() -> bool:
    return (
        envs.SGLANG_OPT_DSV41_FLASHINFER_SM90_PREPARE.get()
        or envs.SGLANG_OPT_DSV41_FLASHINFER_SM90_RUN.get()
    )


def run_enabled() -> bool:
    return envs.SGLANG_OPT_DSV41_FLASHINFER_SM90_RUN.get()


def available() -> bool:
    if not enabled() or get_device_sm() != 90:
        return False
    try:
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


def prepare_layer(layer: torch.nn.Module) -> bool:
    """Prepare one EP8 layer, returning whether the conversion succeeded."""
    if not available():
        return False
    if getattr(layer, "_flashinfer_sm90_mxfp4_weights", None) is not None:
        return True
    if getattr(layer, "moe_tp_size", 1) != 1:
        return False
    try:
        from flashinfer.megamoe import transform_mxfp4_weights_for_mega_moe

        w13 = layer.w13_weight.data.contiguous().view(torch.uint8)
        w2 = layer.w2_weight.data.contiguous().view(torch.uint8)
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
        transformed = transform_mxfp4_weights_for_mega_moe(
            (w13, w13_scale), (w2, w2_scale)
        )
        layer._flashinfer_sm90_mxfp4_weights = transformed
        logger.info(
            "FlashInfer SM90 MXFP4 prepared layer=%s local_experts=%d bytes=(%d,%d)",
            getattr(layer, "layer_id", "?"),
            w13.shape[0],
            transformed[0].numel(),
            transformed[1].numel(),
        )
        return True
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning(
            "FlashInfer SM90 MXFP4 preparation failed for layer %s: %s",
            getattr(layer, "layer_id", "?"),
            exc,
        )
        return False


def get_prepared_weights(layer: torch.nn.Module) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    return getattr(layer, "_flashinfer_sm90_mxfp4_weights", None)


def _get_buffer(hidden: torch.Tensor, capacity: int):
    import flashinfer.megamoe as megamoe

    parallel = get_parallel()
    group = parallel.moe_ep_group.device_group
    key = (id(group), capacity)
    buffer = _BUFFERS.get(key)
    if buffer is None:
        buffer = megamoe.MegaMoESymmBuffer(
            group,
            384,
            capacity,
            6,
            5120,
            2304,
        )
        _BUFFERS[key] = buffer
        logger.info("FlashInfer SM90 MXFP4 buffer ready capacity=%d", capacity)
    return buffer


def run(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    routed_scaling_factor: Optional[float],
) -> torch.Tensor:
    """Run the SM90 fused EP kernel for a local CP/EP row block."""
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
    capacity = max(envs.SGLANG_OPT_DSV41_FLASHINFER_SM90_MAX_TOKENS.get(), num_tokens)
    buffer = _get_buffer(hidden_states, capacity)
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    x_fp8, x_sf = sglang_per_token_group_quant_fp8(hidden_states, 128)
    if x_sf.shape != (num_tokens, 40):
        raise ValueError(f"unexpected activation scale shape {tuple(x_sf.shape)}")
    parallel = get_parallel()
    global_ids = topk_ids.to(torch.int64)
    local_experts = 384 // parallel.moe_ep_size
    global_ids = torch.where(
        global_ids >= 0,
        global_ids + parallel.moe_ep_rank * local_experts,
        global_ids,
    )
    route_weights = topk_weights.float()
    if routed_scaling_factor is not None:
        route_weights = route_weights * float(routed_scaling_factor)
    buffer.stage(x_fp8, x_sf, global_ids, route_weights)
    output = torch.empty_like(hidden_states)
    import flashinfer.megamoe as megamoe

    megamoe.mxfp4_mega_moe(
        output,
        weights[0],
        weights[1],
        buffer,
        num_tokens=num_tokens,
    )
    return output
