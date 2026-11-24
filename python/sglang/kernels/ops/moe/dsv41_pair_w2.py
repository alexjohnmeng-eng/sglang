"""Pairwise SwiGLU quantization and output assembly for CP8 prefill."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.activation.silu_mul_fp8_row_quant import _swiglu


@triton.jit
def _store_gated_and_peak(
    gate_up, out, peaks, routes, WIDTH: tl.constexpr, LIMIT: tl.constexpr,
    ROWS: tl.constexpr, SPLIT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    a, cols_a, mask_a = _swiglu(
        gate_up, row, row < routes, 2 * WIDTH, 0, WIDTH, SPLIT, LIMIT,
    )
    tl.store(out + row[:, None] * WIDTH + cols_a[None, :], a.to(tl.bfloat16), mask_a)
    peak = tl.max(tl.abs(a), 1)
    if SPLIT < WIDTH:
        b, cols_b, mask_b = _swiglu(
            gate_up, row, row < routes, 2 * WIDTH, SPLIT, WIDTH, 32, LIMIT,
        )
        tl.store(out + row[:, None] * WIDTH + cols_b[None, :], b.to(tl.bfloat16), mask_b)
        peak = tl.maximum(peak, tl.max(tl.abs(b), 1))
    tl.store(peaks + row, peak, row < routes)


@triton.jit
def _quantize_and_push_pair(
    src, left_peak, right_peak, local_dst, remote_dst, scale_dst, routes,
    WIDTH: tl.constexpr, COLOR: tl.constexpr, ROWS: tl.constexpr, SPLIT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    peak_l = tl.load(left_peak + row, row < routes, 0)
    peak_r = tl.load(right_peak + row, row < routes, 0)
    scale = tl.maximum(tl.maximum(peak_l, peak_r), 1e-30) / 448
    inverse = 1 / scale
    tl.store(scale_dst + row, scale, row < routes)
    col = tl.arange(0, SPLIT)
    mask = (row[:, None] < routes) & (col[None, :] < WIDTH)
    src_row = src + row[:, None] * WIDTH
    dst = row[:, None] * (2 * WIDTH) + COLOR * WIDTH
    value = tl.load(src_row + col[None, :], mask, 0).to(tl.float32)
    code = (value * inverse[:, None]).to(tl.float8e4nv)
    tl.store(local_dst + dst + col[None, :], code, mask)
    tl.store(remote_dst + dst + col[None, :], code, mask)
    if SPLIT < WIDTH:
        tail = SPLIT + tl.arange(0, 32)
        value = tl.load(src_row + tail[None, :], row[:, None] < routes, 0).to(tl.float32)
        code = (value * inverse[:, None]).to(tl.float8e4nv)
        tl.store(local_dst + dst + tail[None, :], code, row[:, None] < routes)
        tl.store(remote_dst + dst + tail[None, :], code, row[:, None] < routes)


@triton.jit
def _sum_colored_slots(src, out, rows, HALF: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = offset // (HALF * 2)
    col = offset % (HALF * 2)
    color = col // HALF
    addr = src + (color * rows + row) * HALF + col % HALF
    accum = tl.load(addr, row < rows, 0).to(tl.float32)
    for pair in tl.static_range(1, 4):
        accum += tl.load(addr + pair * 2 * rows * HALF, row < rows, 0).to(tl.float32)
    tl.store(out + offset, accum.to(tl.bfloat16), row < rows)


def sum_colored_slots(slots, rows, width):
    out = torch.empty((rows, width), dtype=slots.dtype, device=slots.device)
    _sum_colored_slots[(triton.cdiv(rows * width, 2048),)](
        slots, out, rows, width // 2, 2048, num_warps=4
    )
    return out


def store_gated_and_peak(gate_up, gated, peaks, swiglu_limit):
    routes, channels = gate_up.shape
    assert gate_up.is_contiguous() and channels == 576
    _store_gated_and_peak[(triton.cdiv(routes, 8),)](
        gate_up, gated, peaks, routes, channels // 2, swiglu_limit, 8, 256, num_warps=4
    )


def quantize_and_push_pair(gated, left_peak, right_peak, local, remote, scale, color):
    routes, width = gated.shape
    _quantize_and_push_pair[(triton.cdiv(routes, 8),)](
        gated, left_peak, right_peak, local, remote, scale, routes,
        width, color, 8, 256, num_warps=4,
    )
