# DeepSeek-V4.1 Ant migration experiments

## 2026-11-21: startup compatibility pass

- Branch: `tmp/ant-full-migration` (base `ba988560ba`).
- Runtime: `lmsysorg/sglang:latest`, container `dsv41-h20`, H20 x8.
- Command: `/data1/fjmeng2/dsv41-runtime/run-migrated-exact.tmp.sh` with the
  requested SM90, FP8, Engram host per-rank, CP8, Humming, DSpark and decode
  CUDA-graph settings; local server port `30002`.
- Result: weight loading and decode graph capture completed, but the first CP
  prefill exposed four incomplete cross-version interfaces. Temporary fixes are
  kept uncommitted until the exact request path passes:
  `DSV4Metadata.real_num_tokens` guard, CP `LateLayerTail.global_token_indices`,
  `LogitsProcessorOutput.own_customized_info`, and the optional Humming
  `mma_type` field.
- Performance: not measured. No performance claim or performance commit is
  made from this pass.
- Next direction after startup: run matched Ant/local 64K and 256K prefill
  TTFT with radix cache disabled, then fake-prefill decode at batch 64; collect
  SGLang profiler traces and compare CP gather, Humming/DeepGEMM, attention,
  Engram and barrier kernels before changing performance code.

## 2026-11-21: first migrated benchmark

- The exact TP8/CP8 service reached `The server is fired up and ready to roll`
  on port 30002 after the compatibility fixes.
- Cold prefill, radix cache disabled, random IDs, one request, output 32:
  64K TTFT **1068.01 ms**; 256K TTFT **2722.04 ms**. These are comparable
  with the Ant reference logs (775.98 ms and 2057.87 ms respectively), but are
  slower by 37.6% and 32.3%; no optimization claim is made.
- A fake-prefill client against the ordinary (non-disaggregation) server still
  performed real prefill. Its 64x64K result (window 2856 tok/s) is invalid for
  the requested decode metric and is excluded.
- A DP8/EP4 decode role using the existing Ant preset and fake transfer mode
  failed during CUDA-graph capture with `CUDA_ERROR_ILLEGAL_ADDRESS`, both at
  graph max batch 128/mem 0.85 and at batch 64/mem 0.80. This blocks a valid
  migrated fake-prefill decode measurement; investigate the DP/graph path
  before claiming decode performance.

## 2026-11-23: fake-prefill decode isolation

- Added temporary support for decode-only `fake` transfer validation and CUDA
  device-only sequence lengths in DSpark draft metadata; these changes remain
  uncommitted.
- TP8/EP4/attention-DP1 fake-prefill results: 1 request completed at 132.6
  tok/s, and 8 requests at 457.7 tok/s. A 64-request run failed at DSpark
  draft `cuGraphLaunch`; `CUDA_LAUNCH_BLOCKING=1` confirms the draft graph
  replay as the first failing stage.
- A temporary `SGLANG_DSPARK_DRAFT_DECODE_GRAPH=0` switch allowed a 64-request,
  4-token smoke test to finish, but longer multi-step runs later failed in the
  NCCL watchdog. These are diagnostic results, not throughput claims.
- Next target: profile the first multi-step 64-batch verify/draft iteration and
  inspect graph buffer lifetime and NCCL event ordering. Core dumps are disabled
  for experiments to prevent exhausting the container root filesystem.

## 2026-11-23: exact 64K kernel profile and temporary SM90/communication fixes

- Compared fresh profiler traces for the exact TP8/CP8/Humming/Engram command,
  with radix cache disabled. The migrated code initially used the generic mHC
  path: `_hc_mix_stats_partial_kernel + _hc_mix_stats_reduce_kernel`; Ant used
  `_hc_mix_stats_partial_v2_kernel + _hc_mix_reduce_sinkhorn_tl_kernel`.
- Temporary mHC dispatch was added to `deepseek_v4_mhc.py` (SM90 split-K stats,
  fused reduce/Sinkhorn, combine norm and post-combine norm). It reduced local
  mHC aggregate kernel time from about **923 ms to 608 ms** in the trace; the
  corresponding Ant aggregate was **592 ms**. This is a diagnostic gain only;
  it is not committed yet.
- Temporary custom-all-reduce ABI fixes were required because the local C++ ABI
  expects `PushPlane.default_blocks`, a one-element int64 `EpochCounter`, and
  stateless barrier calls. `segment_a2a.cuh` also needed `PullPlane.capacity`.
  Startup now reports `symmetric_memory = 18.01 MB, multicast = True, pull = True`.
  The trace then showed local NCCL **168 ms** and barrier **4.11 s** aggregate,
  versus fresh Ant NCCL **199 ms** and barrier **3.20 s**. Barrier duration has
  large rank-dependent variance (one rank was 159 ms locally while another was
 596 ms), so this is not yet an accepted optimization result.
- The first local profile accidentally prepended `humming-pylibs` (0.1.18),
  disabling Ant's 0.1.12 overlay/tuned tables. Removing that path enabled
  `v0_1_12_bm32` and H20-3e tuned tables. The key Humming `5120x320` kernel then
  measured **865 ms**, matching Ant's **865 ms**; aggregate Humming was
  **2450 ms local vs 2540 ms Ant**.
- Serving checks after these temporary changes: local exact 64K TTFT was
  **1021--1033 ms** in two runs; fresh Ant exact was **950 ms**. The target is
  therefore still not aligned. Attention sparse kernel and DeepGEMM are within
  1%, while CP symmetric-memory barrier long-tail remains the leading suspect.
- No GSM8K accuracy run or performance commit is claimed for this temporary
  compatibility/profile pass. Next direction: repeat local/Ant profile after
  warmup with rank-wise barrier timelines, then isolate CP gather/barrier
  scheduling before changing the barrier implementation.


## 2026-11-23: strict prefill remeasurement (supersedes earlier mixed profiles)

**Correction:** the previous serving commands did not fix random-ID length: the
reported 64K runs actually sent 56,423 tokens and generated 20 output tokens.
The old CPU+GPU traces also generated 32 output tokens and enabled Python stack
collection. Their cross-rank barrier sums are diagnostic, not pure prefill or
communication bandwidth measurements. A short barrier on one rank does not imply
an incomplete trace. Withdraw the earlier 7--9% residual prefill-gap conclusion.

Protocol: `tools/strict_prefill.py`, seed 42 exact `input_ids`, output 1,
10 warmups then 5 serial repetitions at each length. It asserts prompt token
count, completion token count and `cached_tokens=0`, verifies radix cache is off,
and saves server info, input SHA256 and raw samples. Same container/image/H20x8,
TP8/CP8 interleave, Engram host per_rank, Humming 0.1.12 with bm32 overlay/tables,
DSPARK block 5, bounded replay, chunk 16384, mem 0.84, decode graph 64.
GPU-only profiles set `with_stack=False, record_shapes=False`; stack-enabled
mapping traces are only for source attribution.

| Exact input tokens | Ant TTFT median | Local candidate TTFT median | Local input/TTFT |
| ---: | ---: | ---: | ---: |
| 65,536 | 857.769 ms | 854.863 ms | 76,662 tok/s |
| 262,144 | 3533.834 ms | 3522.919 ms | 74,411 tok/s |

Ant/local GPU spans (TP0): 849.144/846.996 ms at 64K and
3506.041/3494.641 ms at 256K. All ranks have complete matching prefill kernel
counts for the main compute families. Mean per-rank kernel sums at 64K:
DeepGEMM 267.601/267.542 ms, sparse attention 99.617/99.589 ms,
Humming W13 184.765/184.730 ms, W2 109.949/109.798 ms,
mHC 28.772/28.752 ms, CP barrier 10.806/12.017 ms.
Thus current prefill compute paths match Ant; previous hundreds-of-ms barrier
per-rank values were strongly perturbed by stack profiling.

Temporary ABI review additionally found that `Barrier`'s third constructor
argument now means semaphore index, not number of arrivals. Removed legacy
`num_arrives=2` from pull gather/scatter and segment A2A so blocks use their own
`blockIdx.x` semaphore. This correctness fix is part of the candidate under test.

GSM8K sanity: EvalScope 1.15.0, identical 5-shot dataset prompts, temperature 0,
max_tokens 2048, 20 examples: Ant 20/20, local 19/20. The difference is the lemon
tree break-even question (local answers 12 instead of 13), not an HTTP failure.
Full GSM8K and a matched local pre-change baseline are pending before committing.

Artifacts under `/data1/fjmeng2/dsv41-runtime/`:
- `results/1123-{ant,local}-strict/measurements.json` and `server_info.json`
- `profiles/1123-{ant,local}-gpu/{65536,262144}/` (8 ranks each)
- `results/1123-compare/{65536,262144}/comparison.md` and `gpu_summary.json`
- `results/1123-{ant,local}-strict/triage64.md` (repository profiler skill tables)
- `profiles/1123-ant-mapping/65536/` (source mapping only)

Next: validate full accuracy and a matched pre-change baseline, then commit only
verified gains. For further improvements beyond Ant, prioritize Humming/DeepGEMM
(compute dominates); do not infer a benefit from changing CP barriers alone.


### Strict prefill candidate accuracy (2026-11-23)

The complete candidate EvalScope GSM8K run completed: **1279/1319 = 96.9674%**,
temperature 0, max_tokens 2048, batch/concurrency 64, same local test split and
5-shot harness as the earlier Ant 1280/1319 record. The 20-example sanity
mismatch was the lemon-tree break-even interpretation; the full run does not
show a large accuracy drop, but a paired local pre-change run is still needed
before attributing any difference to these patches. Raw reviewed outputs are at
`results/1123-local-gsm-full/reviews/local-prefill-aligned-1123/gsm8k_default.jsonl`.

- Existing SM90 mHC kernel/dispatch tests passed: 7 tests, 161 subtests, bitwise comparisons across row counts including 2048/4100; command `pytest -q test/registered/kernels/ops/layernorm/test_mhc_sm90_kernels.py`.


### Validation harness compatibility notes

The existing communication test file cannot import the migrated runner because
it imports the newer unused `choose_algo` helper. An external test-only adapter
(`tools/check_migrated_collectives.py`) supplies a fail-if-used placeholder and
maps the test harness's newer `forced_algo` spelling to `override_algo`. Product
code is not changed for this harness compatibility. An initial `-k` selection
also matched the test filename and included unrelated ROCm hierarchical tests,
which failed on the missing `_hier` interface after 276 passed/264 skipped.
The corrected run explicitly selects all-reduce, all-gather, reduce-scatter and
lagging-rank graph-chain nodes; results follow below.

- Corrected eight-rank collective suite completed successfully: **391 passed on
  every rank** (eager and CUDA Graph, all-reduce/all-gather/reduce-scatter and
  lagging-rank graph chain). Log: `logs/1123-collectives-selected.log`.
- Candidate source for four files saved, SHA256 manifest at
  `results/1123-candidate-source/manifest.json`; four `HEAD` versions temporarily
  restored solely for matched baseline measurement. All other worktree changes
  and the Humming 0.1.12 launch environment are identical. No commit yet.


### Matched pre-change vs candidate performance

All values use the strict protocol above, identical launch script and Humming
0.1.12. Only the four source files were restored to HEAD for the baseline.

| Input tokens | Before ms | Candidate ms | Throughput gain | Ant ms |
| ---: | ---: | ---: | ---: | ---: |
| 65,536 | 907.327 | 854.863 | 6.14% | 857.769 |
| 262,144 | 3729.804 | 3522.919 | 5.87% | 3533.834 |

64K GPU-only profile attribution (mean sum per rank): mHC **79.687 -> 28.752 ms**,
launches **1011 -> 619**. TP0 GPU span **899.443 -> 846.996 ms**. Humming,
DeepGEMM and attention are stable. CP barrier **10.741 -> 12.017 ms** did not
improve; the measured saving comes from restored SM90 mHC dispatch/fusions,
not from eliminating CP communication. Baseline/candidate source attribution
is stored under `results/1123-baseline-vs-candidate/`.


### Paired accuracy and submission decision

Matched full GSM8K, same EvalScope configuration and 1319 examples:
- Before: **1278/1319 = 96.8916%**.
- Candidate: **1279/1319 = 96.9674%**.
- Paired changes: 6 wrong-to-correct, 5 correct-to-wrong; 1273 correct in both,
  35 wrong in both. This is a one-run accuracy check, not proof of output or
  statistical equivalence. Both raw review sets are retained.

Candidate is restored from the verified snapshot; only formatting changed after
restoration. Repeat unprofiled strict measurements after restart to confirm the
before/candidate sequence. With repeated throughput gains, matching Ant kernel
paths, full accuracy checks and passing kernel/multirank tests, the four-file
SM90 dispatch/communicator repair is eligible for one performance commit.
Other compatibility/diagnostic changes remain uncommitted on the temporary branch.


### Reproduction commands and scope

Run in container `dsv41-h20` (`lmsysorg/sglang:latest`, image ID
`sha256:f3a303d4028dc4597e312c96ada28fa19509f46559601a94e89dc630ac1ef152`).
Launch candidate: `bash /data1/fjmeng2/dsv41-runtime/run-migrated-ant-humming.tmp.sh`.
This preserves the requested exact server flags and uses Humming 0.1.12 from the
image; do not prepend the 0.1.18 `humming-pylibs` path for this comparison.
The actual exact-command topology is TP8/CP8/**EP1**, matching Ant. EP8 and other
parallel layouts are separate experiments, not established by these numbers.

Strict measurement and GPU-only trace:
```bash
docker exec dsv41-h20 python3 -u /data1/fjmeng2/dsv41-runtime/tools/strict_prefill.py \
  --port 30002 --lengths 65536 262144 --warmup 10 --repeat 5 \
  --output /data1/fjmeng2/dsv41-runtime/results/<tag> \
  --profile-dir /data1/fjmeng2/dsv41-runtime/profiles/<tag>
```

Kernel comparison tool: `tools/compare_prefill_profiles.py --reference <8-rank-dir>
--candidate <8-rank-dir> --output <report-dir>`; it reports mean kernel sums per
rank separately from GPU timeline span/union/idle. Whole-request latency includes
serialization/HTTP; kernel sums can overlap and are not additive TTFT.

Further directions after parity: Humming W13/W2 and DeepGEMM dominate useful GPU
time. Evaluate prefill-role ntail overlay / tile choices and chunk sizes as
separate experiments; preserve strict lengths and cache-off protocol. CP barrier
sums alone no longer support a communication-fusion priority. The original
prefill +50%/decode +30% global goals are not inferred from this incremental gain.


### Candidate restart confirmation and final state

- 65,536 tokens: median **857.078 ms**, input/TTFT **76464.49 tok/s**, throughput gain **5.86%** vs matched baseline. Raw samples: 856.795, 857.372, 856.616, 857.451, 857.078 ms.
- 262,144 tokens: median **3528.397 ms**, input/TTFT **74295.49 tok/s**, throughput gain **5.71%** vs matched baseline. Raw samples: 3526.137, 3530.008, 3526.017, 3528.397, 3530.290 ms.

Second launch confirms the candidate throughput saving and Ant parity. Server
remains healthy on port 30002 with candidate source restored. Python formatting
preserved ASTs and C++ formatting preserved non-whitespace text relative to the
tested snapshots; selected Python F checks, py_compile and diff checks passed.
One commit records the four verified source repairs and this experiment log.
Other compatibility/diagnostic worktree changes remain uncommitted on
`tmp/ant-full-migration`; this benchmark does not validate their standalone
portability, fake-prefill decode throughput, or alternate EP layouts.
