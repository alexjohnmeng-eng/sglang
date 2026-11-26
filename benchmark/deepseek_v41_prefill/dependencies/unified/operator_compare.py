"""Run isolated legacy/unified processes on identical deterministic tensors.

Examples: --module deep_gemm --suite dense --output FILE;
          --module deep_gemm_mhc --suite mhc --output FILE.
The comparator requires exact outputs, including CUDA graph replay.
"""

import argparse
import importlib
import json
from pathlib import Path

import torch

p = argparse.ArgumentParser(__doc__)
p.add_argument("--module", default="deep_gemm")
p.add_argument(
    "--suite", choices=["dense", "mhc", "metadata", "compare"], required=True
)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--reference", type=Path)
p.add_argument(
    "--chain-dense",
    action="store_true",
    help="Precede mHC with a dense GEMM with global PDL enabled",
)
p.add_argument(
    "--stock-compatible",
    "--skip-wqb",
    dest="stock_compatible",
    action="store_true",
    help="Use stock SM90 APIs: BMM gran128 and no fused WQB",
)
p.add_argument(
    "--vary-scales",
    action="store_true",
    help="Use nonuniform activation and weight scales",
)
a = p.parse_args()
if a.suite == "compare":
    expected = torch.load(a.reference, weights_only=True)
    actual = torch.load(a.output, weights_only=True)
    assert expected.keys() == actual.keys()
    summary = {}
    for name in actual:
        x, y = expected[name], actual[name]
        assert x.shape == y.shape and x.dtype == y.dtype
        same = torch.equal(
            x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)
        )
        summary[name] = dict(
            shape=list(x.shape), dtype=str(x.dtype), bitwise_equal=same
        )
        assert same, name
    a.output.with_suffix(".comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    raise SystemExit(0)
dg = importlib.import_module(a.module)
print(dg.__file__, dg.__version__, flush=True)
results = {}


def scales(shape, *, weight=False):
    if not a.vary_scales:
        return torch.full(shape, 0.125, device="cuda")
    if weight:
        return torch.exp2(torch.randint(-8, 1, shape, device="cuda").float())
    return torch.rand(shape, device="cuda") * 0.2 + 0.01


def run(name, fn, tensors):
    fn()
    torch.cuda.synchronize()
    eager = {k: v.clone() for k, v in tensors.items()}
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    for k, v in tensors.items():
        assert torch.isfinite(v.float()).all(), (name, k)
        assert torch.equal(
            v.contiguous().view(torch.uint8), eager[k].contiguous().view(torch.uint8)
        ), (name, k, "graph")
        results[f"{name}/{k}"] = v.cpu()
    print(name, "finite, graph bitwise OK", flush=True)


if a.suite == "dense":
    # Real DSV41 dense shapes, both original block128 and block32 scales.
    for m, n, k, gran in [
        (64, 5120, 1280, 128),
        (8192, 2560, 5120, 32),
        (8192, 5120, 2560, 32),
        (8192, 32768, 1280, 32),
        (8192, 4608, 5120, 32),
    ]:
        torch.manual_seed(42)
        x = (torch.randn(m, k, device="cuda") * 0.25).to(torch.float8_e4m3fn)
        w = (torch.randn(n, k, device="cuda") * 0.25).to(torch.float8_e4m3fn)
        xs = scales((m, k // gran))
        ws = scales((n // gran, k // gran), weight=True)
        y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        run(
            f"gemm.{m}.{n}.{k}.g{gran}",
            lambda: dg.fp8_gemm_nt((x, xs), (w, ws), y, recipe=(1, gran, gran)),
            {"y": y},
        )
    for m, groups, gran in ((65, 1, 128), (8192, 8, 32)):
        if a.stock_compatible:
            gran = 128
        torch.manual_seed(61)
        n, k = 1024, 4096
        x = (torch.randn(m, groups, k, device="cuda") * 0.25).to(torch.float8_e4m3fn)
        w = (torch.randn(groups, n, k, device="cuda") * 0.25).to(torch.float8_e4m3fn)
        xs = scales((m, groups, k // gran))
        ws = scales((groups, n // gran, k // gran), weight=True)
        y = torch.empty(m, groups, n, device="cuda", dtype=torch.bfloat16)
        run(
            f"einsum.{m}.{groups}.bf16",
            lambda: dg.fp8_einsum(
                "bhr,hdr->bhd", (x, xs), (w, ws), y, recipe=(1, gran, gran)
            ),
            {"y": y},
        )
    if not a.stock_compatible:
        m, n, k = 8192, 32768, 1280
        torch.manual_seed(84)
        x = (torch.randn(m, k, device="cuda") * 0.25).to(torch.float8_e4m3fn)
        w = (torch.randn(n, k, device="cuda") * 0.25).to(torch.float8_e4m3fn)
        xs = scales((m, k // 32))
        ws = scales((n // 32, k // 32), weight=True)
        y = torch.empty(m, n, device="cuda", dtype=torch.float8_e4m3fn)
        pos = torch.arange(m, device="cuda")
        freq = torch.randn(m, 64, device="cuda")
        # Runtime calls the Torch op to preserve graph/capture behavior.
        run(
            "wqb_q8_rope",
            lambda: torch.ops.deep_gemm.dsv41_wqb_q8_rope(x, xs, w, ws, y, pos, freq),
            {"y": y},
        )
elif a.suite == "mhc":
    for m in (32, 65, 8192):
        torch.manual_seed(42)
        h, hc = 5120, 4
        inputs = dict(
            x=torch.randn(m, h, device="cuda", dtype=torch.bfloat16),
            residual=torch.randn(m, hc, h, device="cuda", dtype=torch.bfloat16),
            shifted_prev_mix=torch.randn(m, hc, 1, device="cuda").sigmoid(),
            post_mix=torch.randn(m, hc, 1, device="cuda").sigmoid(),
            comb_res_mix=torch.rand(m, hc, hc, device="cuda") / hc,
            fn=torch.randn(24, hc * h, device="cuda") * 0.01,
            mix_scales=torch.randn(3, device="cuda") * 0.1,
            mix_bases=torch.randn(24, device="cuda") * 0.1,
            hc_mult=hc,
            hc_norm_eps=1e-6,
            hc_pre_eps=1e-6,
            hc_post_scale=2.0,
            sinkhorn_eps=1e-6,
            num_sinkhorn_iters=20,
            rmsnorm_weight=torch.randn(h, device="cuda", dtype=torch.bfloat16) * 0.1
            + 1,
            rmsnorm_eps=1e-6,
            rmsnorm_scale=1.0,
        )
        outputs = dict(
            new_residual=torch.empty_like(inputs["residual"]),
            new_prev_mix=torch.empty_like(inputs["shifted_prev_mix"]),
            new_post_mix=torch.empty_like(inputs["post_mix"]),
            new_comb_res_mix=torch.empty_like(inputs["comb_res_mix"]),
            y_bf16=torch.empty_like(inputs["x"]),
            y_fp8=torch.empty((m, h), device="cuda", dtype=torch.float8_e4m3fn),
            y_gemm_sf=torch.empty_strided(
                (m, h // 128), (1, (m + 3) // 4 * 4), device="cuda", dtype=torch.int32
            ),
        )
        if a.chain_dense:
            producer = importlib.import_module("deep_gemm")
            producer.set_pdl(True)
            k = 1280
            q = (torch.randn(m, k, device="cuda") * 0.25).to(torch.float8_e4m3fn)
            w = (torch.randn(h, k, device="cuda") * 0.25).to(torch.float8_e4m3fn)
            qs, ws = scales((m, k // 32)), scales((h // 32, k // 32), weight=True)

            def chained():
                producer.fp8_gemm_nt((q, qs), (w, ws), inputs["x"], recipe=(1, 32, 32))
                dg.mega_mhc(**inputs, **outputs)

            run(f"mhc.{m}", chained, {"producer": inputs["x"], **outputs})
            # Changing Q and poisoning X prevents a stale read of the preceding
            # replay's producer output from passing an otherwise constant test.
            chained()
            torch.cuda.synchronize()
            dependency_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(dependency_graph):
                chained()
            for _ in range(3):
                q.copy_((torch.randn(q.shape, device="cuda") * 0.25).to(q.dtype))
                chained()
                torch.cuda.synchronize()
                serial = {k: v.clone() for k, v in outputs.items()}
                serial["producer"] = inputs["x"].clone()
                inputs["x"].zero_()
                dependency_graph.replay()
                torch.cuda.synchronize()
                for k, value in {"producer": inputs["x"], **outputs}.items():
                    assert torch.equal(
                        value.contiguous().view(torch.uint8),
                        serial[k].contiguous().view(torch.uint8),
                    ), (m, k, "fresh producer graph dependency")
            print(f"mhc.{m}: fresh producer graph dependency bitwise OK", flush=True)

        else:
            run(f"mhc.{m}", lambda: dg.mega_mhc(**inputs, **outputs), outputs)
else:
    for next_n in (1, 2, 4, 5, 6):
        for block in (64,):
            torch.manual_seed(71)
            context = torch.randint(
                256, 65536, (16, next_n), device="cuda", dtype=torch.int32
            )
            sms = dg.get_num_sms()
            out = dg.get_paged_mqa_logits_metadata(context, block, sms)
            run(
                f"metadata.next{next_n}.block{block}",
                lambda: out.copy_(
                    dg.get_paged_mqa_logits_metadata(context, block, sms)
                ),
                {"schedule": out},
            )
torch.save(results, a.output)
