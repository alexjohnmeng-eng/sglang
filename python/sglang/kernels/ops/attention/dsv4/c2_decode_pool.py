"""Fused ratio-2 decode / target-verify pair-pooling, bitwise identical to the
torch pool_pairs path; a rounding difference here can change the indexer's
top-k selection."""

from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _pool_pairs_fp32(p_kv, kv, p_score, score):
    # The exp and division match torch softmax. Keep the product roundings.
    m = tl.maximum(p_score, score)
    e0 = libdevice.exp(p_score - m)
    e1 = libdevice.exp(score - m)
    denom = e0 + e1
    t0 = p_kv * tl.div_rn(e0, denom) + 0.0
    t1 = kv * tl.div_rn(e1, denom) + 0.0
    return t0 + t1


@triton.jit
def _c2_decode_pool_kernel(
    kv_ptr,  # [n, D] fp32
    score_ptr,  # [n, D] fp32
    pos_ptr,  # [n] int64
    raw_out_loc_ptr,  # [n] int32/int64
    out_loc_ptr,  # [n] int32/int64
    req_ptr,  # [n] int64
    state_kv_ptr,  # [R, D] fp32, in/out
    state_score_ptr,  # [R, D] fp32, in/out
    pooled_ptr,  # [n, D] fp32, out
    group_pos_ptr,  # [n] int64, out
    slots_ptr,  # [n] int64, out
    pad_row,
    RING_SIZE: tl.constexpr,
    STATE_KV_STRIDE: tl.constexpr,
    STATE_SCORE_STRIDE: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IN_BATCH_PREDECESSOR: tl.constexpr = False,
):
    row = tl.program_id(0)

    pos = tl.load(pos_ptr + row)
    raw_loc = tl.load(raw_out_loc_ptr + row)
    out_loc = tl.load(out_loc_ptr + row)
    req = tl.load(req_ptr + row)

    # Raw location 0 is the padded-graph-row sentinel, and its req_pool_idx 0 may
    # be a live request, so such a row's pair state goes to the spare row.
    odd = (pos % 2) == 1
    if RING_SIZE:
        r = tl.where(
            (raw_loc == 0) | (pos == 0),
            pad_row,
            req * RING_SIZE + (pos - 1) % RING_SIZE,
        )
    else:
        r = tl.where(raw_loc == 0, pad_row, req)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    kv = tl.load(kv_ptr + row * D + offs, mask=mask, other=0.0)
    score = tl.load(score_ptr + row * D + offs, mask=mask, other=0.0)
    if IN_BATCH_PREDECESSOR:
        # A verify row whose predecessor is the previous row of the same request
        # reads its projection, not the ring slot another program is writing.
        prev_req = tl.load(req_ptr + row - 1, mask=row > 0, other=-1)
        prev_pos = tl.load(pos_ptr + row - 1, mask=row > 0, other=-2)
        prev_raw_loc = tl.load(raw_out_loc_ptr + row - 1, mask=row > 0, other=0)
        in_batch = (
            (row > 0)
            & (req == prev_req)
            & (pos == prev_pos + 1)
            & (raw_loc != 0)
            & (prev_raw_loc != 0)
        )
        p_kv = tl.load(
            state_kv_ptr + r * STATE_KV_STRIDE + offs,
            mask=mask & ~in_batch,
            other=0.0,
        )
        p_score = tl.load(
            state_score_ptr + r * STATE_SCORE_STRIDE + offs,
            mask=mask & ~in_batch,
            other=0.0,
        )
        batch_p_kv = tl.load(
            kv_ptr + (row - 1) * D + offs, mask=mask & in_batch, other=0.0
        )
        batch_p_score = tl.load(
            score_ptr + (row - 1) * D + offs, mask=mask & in_batch, other=0.0
        )
        p_kv = tl.where(in_batch, batch_p_kv, p_kv)
        p_score = tl.where(in_batch, batch_p_score, p_score)
    else:
        p_kv = tl.load(state_kv_ptr + r * STATE_KV_STRIDE + offs, mask=mask, other=0.0)
        p_score = tl.load(
            state_score_ptr + r * STATE_SCORE_STRIDE + offs, mask=mask, other=0.0
        )

    if RING_SIZE:
        # Decode has one row per request; verify rows write consecutive, distinct
        # ring positions. Pad rows never modify the ring.
        write_row = req * RING_SIZE + pos % RING_SIZE
        tl.store(
            state_kv_ptr + write_row * STATE_KV_STRIDE + offs,
            kv,
            mask=mask & (raw_loc != 0),
        )
        tl.store(
            state_score_ptr + write_row * STATE_SCORE_STRIDE + offs,
            score,
            mask=mask & (raw_loc != 0),
        )
    else:
        tl.store(
            state_kv_ptr + r * STATE_KV_STRIDE + offs,
            tl.where(odd, p_kv, kv),
            mask=mask,
        )
        tl.store(
            state_score_ptr + r * STATE_SCORE_STRIDE + offs,
            tl.where(odd, p_score, score),
            mask=mask,
        )

    pooled = _pool_pairs_fp32(p_kv, kv, p_score, score)
    tl.store(pooled_ptr + row * D + offs, pooled, mask=mask)

    # One program per row, so these are written exactly once each; no guard.
    tl.store(group_pos_ptr + row, tl.where(odd, pos - 1, pos))
    tl.store(slots_ptr + row, tl.where(out_loc >= 0, out_loc, 0))


def c2_decode_pool(
    kv: torch.Tensor,
    score: torch.Tensor,
    pos: torch.Tensor,
    raw_out_loc: torch.Tensor,
    out_loc: torch.Tensor,
    req: torch.Tensor,
    state_kv: torch.Tensor,
    state_score: torch.Tensor,
    pad_row: int,
    *,
    ring_size: int = 0,
    in_batch_predecessor: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ratio-2 decode pair pooling; updates `state_kv` / `state_score` in place.

    With ring_size > 0 the state halves may be views of an interleaved
    CompressStatePool ring, one row per request otherwise; pad_row is the
    padded-graph-row sentinel and is never written. in_batch_predecessor
    (target verify, ring only) takes several rows per request, consecutive in
    row order and position and fewer than ring_size, so the carried
    predecessor of a request's first row stays readable.
    """
    assert ring_size > 0 or not in_batch_predecessor
    assert kv.is_contiguous() and score.is_contiguous()
    assert state_kv.stride(1) == state_score.stride(1) == 1
    assert kv.dtype == torch.float32 and score.dtype == torch.float32
    n, D = kv.shape
    pooled = torch.empty_like(kv)
    group_pos = torch.empty_like(pos)
    slots = torch.empty(n, dtype=out_loc.dtype, device=out_loc.device)
    if n == 0:
        return pooled, group_pos, slots
    _c2_decode_pool_kernel[(n,)](
        kv,
        score,
        pos,
        raw_out_loc,
        out_loc,
        req,
        state_kv,
        state_score,
        pooled,
        group_pos,
        slots,
        pad_row,
        RING_SIZE=ring_size,
        STATE_KV_STRIDE=state_kv.stride(0),
        STATE_SCORE_STRIDE=state_score.stride(0),
        D=D,
        BLOCK_D=triton.next_power_of_2(D),
        IN_BATCH_PREDECESSOR=in_batch_predecessor,
        num_warps=4,
    )
    return pooled, group_pos, slots


@triton.jit
def _c2_prefill_pool_kernel(
    kv_ptr,
    score_ptr,
    pos_ptr,
    loc_ptr,
    req_ptr,
    state_ptr,
    pooled_ptr,
    ROWS: tl.constexpr,
    D: tl.constexpr,
    KV_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    RING_SIZE: tl.constexpr,
    PAD_ROW: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    CP_RANKS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, D)
    valid = rows < ROWS
    pos = tl.load(pos_ptr + rows, valid, other=0)
    req = tl.load(req_ptr + rows, valid, other=0)
    loc = tl.load(loc_ptr + rows, valid, other=0)
    prev_valid = valid & (rows > 0)
    prev_pos = tl.load(pos_ptr + rows - 1, prev_valid, other=-2)
    prev_req = tl.load(req_ptr + rows - 1, prev_valid, other=-1)
    prev_loc = tl.load(loc_ptr + rows - 1, prev_valid, other=0)
    in_batch = prev_valid & (loc != 0) & (prev_loc != 0)
    in_batch = in_batch & (pos == prev_pos + 1) & (req == prev_req)
    state_rows = tl.where(
        (pos > 0) & (loc != 0), req * RING_SIZE + (pos - 1) % RING_SIZE, PAD_ROW
    )
    prev_rows = rows - 1
    if CP_RANKS > 1:
        local_rows: tl.constexpr = ROWS // CP_RANKS
        projected_rows = rows % CP_RANKS * local_rows + rows // CP_RANKS
        prev_rows = prev_rows % CP_RANKS * local_rows + prev_rows // CP_RANKS
    else:
        projected_rows = rows
    kv = tl.load(
        kv_ptr + projected_rows[:, None] * KV_STRIDE + cols, valid[:, None], other=0.0
    )
    score = tl.load(
        score_ptr + projected_rows[:, None] * SCORE_STRIDE + cols,
        valid[:, None],
        other=0.0,
    )
    prev_kv = tl.load(
        kv_ptr + prev_rows[:, None] * KV_STRIDE + cols, in_batch[:, None], other=0.0
    )
    prev_score = tl.load(
        score_ptr + prev_rows[:, None] * SCORE_STRIDE + cols,
        in_batch[:, None],
        other=0.0,
    )
    ring_offsets = state_rows[:, None] * (2 * D) + cols
    carried = (valid & ~in_batch)[:, None]
    state_kv = tl.load(state_ptr + ring_offsets, carried, other=0.0)
    state_score = tl.load(state_ptr + ring_offsets + D, carried, other=0.0)
    pooled = _pool_pairs_fp32(
        tl.where(in_batch[:, None], prev_kv, state_kv),
        kv,
        tl.where(in_batch[:, None], prev_score, state_score),
        score,
    )
    tl.store(pooled_ptr + rows[:, None] * D + cols, pooled, valid[:, None])


def c2_prefill_pool(
    kv: torch.Tensor,
    score: torch.Tensor,
    pos: torch.Tensor,
    raw_out_loc: torch.Tensor,
    req: torch.Tensor,
    state: torch.Tensor,
    *,
    ring_size: int,
) -> torch.Tensor:
    """Pool in token order from [rows, D] or uniform [CP ranks, local rows, D]."""
    cp_ranks = kv.shape[0] if kv.ndim == 3 else 1
    kv = kv.flatten(0, -2)
    score = score.flatten(0, -2)
    rows, width = kv.shape
    pooled = torch.empty_like(kv, memory_format=torch.contiguous_format)
    _c2_prefill_pool_kernel[(triton.cdiv(rows, 4),)](
        kv,
        score,
        pos,
        raw_out_loc,
        req,
        state,
        pooled,
        rows,
        width,
        kv.stride(0),
        score.stride(0),
        ring_size,
        state.shape[0] - 1,
        4,
        cp_ranks,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return pooled
