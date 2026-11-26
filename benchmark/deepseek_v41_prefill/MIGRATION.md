# Historical H20 prefill migration

For the current single-source DeepGEMM and complete CP8EP8/CP8EP1/DP8EP8 scripts,
use [H20_TESTING.md](H20_TESTING.md). The reviewed recipes explicitly disable
WQB/RoPE/FP8 fusion on all three topologies. The results and separate dependency
sites below describe earlier experiments; the script update was not rebenchmarked
or reevaluated, as requested by the user.

The integration retains EP1 paired-W2/mHC and enables the historical EP8 path
only with `SGLANG_OPT_DSV41_CP8_EP8_PREFILL=1`, attention-CP8, MoE-EP8,
attention-DP1 and attention-TP1. Normal attention-DP8 decode cannot enter it.

| Feature | Historical source | Integrated behavior |
| --- | --- | --- |
| Equal CP8 rank-major KV gather | `d03fb60653` | EP8 gate plus complete equal-shard lengths/offset/layout checks; fallback retains original order |
| Local C2 compressor projection/gather | `9c26237183` | Same EP8 gate, fused-gate support, single request and no late-layer tail |
| WQB BF16/RoPE/FP8 epilogue | `1d1c9bf68f` | EP8 gate plus `SGLANG_EXPERIMENT_DSV41_SM90_WQB_Q8_ROPE=1`; pinned dense extension |
| SM90 MXFP4 MegaMoE | `867031493e`, DeepGEMM #473 | CP-local routes, 448 physical expert slots, 16K pool for long calls; original 8K short-call pool |
| Dense projection multicast | `c95f620339` | `DG_DSV41_DENSE_MULTICAST=1`, pinned dense extension and measured projection shapes |
| Mega mHC | `d33c4f160b`, DeepGEMM #521 | Preserves the existing h20 mHC configuration; isolated extension symbols/cache |
| Contiguous SWA TMA | `f85da409df` | EP8 gate plus `SGLANG_EXPERIMENT_DSV41_Q8_TMA_SWA=1` and contiguous-layout eligibility |
| Paired W2/shared add | EP1 h20 baseline `82bebb1d34` | EP1 assembly and communication implementation restored to the validated baseline (formatting only) |

The historical `megamoe-32k` directory measured 99922.39/97555.53 input tok/s
and 1282/1319 GSM8K. Its archived source and GPU trace show a **16K** pool,
with two launches for 32K CP-local rows. The directory name is not a capacity
claim. Actual 32K pool probes failed the early-combine bitmap limit and are
excluded from this migration.

Dependency patches, exact source pins, source/binary hashes and rebuild commands
are in [dependencies/README.md](dependencies/README.md). CP8/EP8 uses DeepGEMM
for MoE and tests the image's FlashInfer 0.7.2 for attention; a separately mounted
FlashInfer 0.7.5 package is optional and is never required by the EP1 entrypoint.

Validation entrypoints:

- CP8/EP8: `PREFILL_DENSE_MULTICAST=1 bash benchmark/deepseek_v41_prefill/run_cp8_megamoe.sh 262144`.
- EP1: `run.sh` with chunk65536, Q8 attention, EP1, memory .84, ntail, paired-W2 layers40, shared TP1, mHC site and `PREFILL_RUN_GSM8K=1`.
- DP8/EP8: `decode_regression/run.py --cases accuracy control candidate --revision <tree-or-commit> --output <new-directory>`.

Both prefill cases send exact seed42 input IDs of 65536/262144 tokens, disable
radix cache, request one output token, warm up ten times and measure five times.
Both capture GPU-only traces from all eight ranks. Every GSM8K case uses all
1319 test questions, 4-shot, temperature0, max_tokens2048 and concurrency64.
DP8/EP8 GSM8K uses real prefill in unified mode without acceptance simulation;
its separate throughput/profile tests use fake prefill, 64x65536 inputs, 1024
outputs and simulated acceptance4.5.

Current final-isolation validation artifacts are under
`/data1/fjmeng2/dsv41-runtime/repro/h20-final-isolation-validation/`.
The final result table is maintained in the repository `OPTIMIZATION_PLAN.md`.
A runtime change is not adopted until its matched throughput and full GSM8K
results are available. Work remains local until the user requests a push.
