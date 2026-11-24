"""Experimental CP8 Humming W2 output shards for DeepSeek-V4.1 on SM90."""

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dsv41_sm90 import dsv41_sm90_on
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

_PAIR_TUNING = {
    "block_shape": [64, 128, 64],
    "warp_shape": [64, 32, 64],
    "num_ctas_per_sm": 4,
    "num_sms": 78,
    "num_stages": 6,
    "num_write_splits": 2,
    "use_f16_accum": False,
    "use_stream_k": False,
}


def init_pair_w2_weights(layer):
    parallel = get_parallel()
    if not (
        layer.layer_id < envs.SGLANG_EXPERIMENT_DSV41_W2_PAIR_LAYERS.get()
        and not layer.is_draft
        and dsv41_sm90_on("moe")
        and envs.SGLANG_SHARED_EXPERT_TP1.get()
        and not layer.with_bias
        and not layer.num_fused_shared_experts
        and layer.moe_tp_size == parallel.tp_size == parallel.attn_cp_size == 8
        and layer.moe_tp_rank == parallel.attn_cp_rank
        and layer.moe_ep_size == 1
        and not envs.SGLANG_HUMMING_USE_F16_ACCUM.get()
        and layer.w2_weight.shape == (384, 5120, 144)
        and layer.w2_weight_scale_inv.shape == (384, 5120, 9)
    ):
        return

    # Keep canonical K sharding for decode and short prefill. The experiment
    # stores a second W2 layout only for the selected target layers.
    pair = torch.nn.Module()
    for source, name, dtype, tile in (
        (layer.w2_weight, "w2_weight", torch.int8, 144),
        (layer.w2_weight_scale_inv, "w2_weight_scale", torch.float8_e8m0fnu, 9),
    ):
        dst = torch.empty((384, 2560, 2 * tile), dtype=dtype, device=source.device)
        pair.register_buffer(name, dst, persistent=False)
        original = source.weight_loader
        row = layer.moe_tp_rank % 2 * 2560
        col = layer.moe_tp_rank // 2 * (2 * tile)

        def load(
            param, loaded_weight, weight_name, shard_id, expert_id, *,
            dst=dst, original=original, row=row, col=col, tile=tile,
        ):
            assert shard_id == "w2" and loaded_weight.shape == (5120, 8 * tile)
            original(param, loaded_weight, weight_name, shard_id, expert_id)
            shard = loaded_weight[row : row + 2560, col : col + 2 * tile]
            dst[expert_id].copy_(shard)

        source.weight_loader = load

    # The auxiliary tensors stay nonpersistent buffers during canonical load.
    layer._pair_w2 = pair


def prepare_pair_w2_weights(layer):
    pair = getattr(layer, "_pair_w2", None)
    if pair is None:
        return

    from humming.layer import HummingMethod
    from humming.schema import HummingInputSchema, HummingWeightSchema

    pair.register_buffer(
        "locks", torch.zeros(1024, dtype=torch.int32, device=pair.w2_weight.device)
    )
    for name, dtype in (
        ("w2_weight", torch.int32),
        ("w2_weight_scale", torch.float8_e8m0fnu),
    ):
        data = getattr(pair, name).view(dtype)
        delattr(pair, name)
        pair.register_parameter(name, torch.nn.Parameter(data, requires_grad=False))
    HummingMethod.prepare_layer_meta(
        pair, 2560, 576,
        HummingWeightSchema(
            b_dtype="float4e2m1", bs_dtype="float8e8m0", weight_scale_group_size=32,
        ),
        HummingInputSchema(
            a_dtype="float8e4m3", input_scale_group_size=0, input_scale_dtype="float32",
        ),
        num_experts=384, pad_n_to_multiple=64, pad_k_to_multiple=64,
        torch_dtype=torch.bfloat16, sublayer_name="w2",
    )
    HummingMethod.transform_humming_layer(pair, sublayer_name="w2")
    if layer.moe_tp_rank == 0:
        logger.info("Prepared paired W2 shard for %s", layer.layer_name)


def run_pair_w2(layer, gate_up, swiglu_limit, outputs, gemm_kwargs, activation):
    from humming.layer import HummingMethod

    pair_input, pair_scale = activation(gate_up, swiglu_limit)
    HummingMethod.forward_layer(
        layer=layer._pair_w2,
        inputs=pair_input,
        input_scale=pair_scale,
        outputs=outputs,
        sublayer_name="w2",
        **{**gemm_kwargs, "tuning_config": _PAIR_TUNING},
    )
