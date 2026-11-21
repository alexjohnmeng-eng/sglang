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


def enabled() -> bool:
    return envs.SGLANG_OPT_DSV41_FLASHINFER_SM90_PREPARE.get()


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
