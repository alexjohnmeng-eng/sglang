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


## 2026-11-23: Complete prefill reproduction scripts

Entry point: `benchmark/deepseek_v41_prefill/run.sh`; full instructions and
standalone client/profile comparison commands are in that directory's README.
This is benchmark packaging, not another model/kernel optimization.

The serving source is exported from performance commit
`7f9566cb967ad55fb24197a38a7fdcc36d45db4f` into an isolated directory. The 13
uncommitted compatibility/diagnostic changes used by the successful experiment
are preserved in `compatibility.patch`, not silently assumed or staged as new
production source changes. The zero-context patch is applied with
`--unidiff-zero`; each patched file's resulting Git blob hash is checked.
Patch SHA256: `75f851e9990cad127afb95d121815474d95fc463a9a1ff798edb0ffdbce79342`.
The four performance files and 13 compatibility files were checked byte-for-byte
against the original tested worktree.

The launcher checks the immutable image ID and all nine serving package
versions, saves launch command/environment/source/hardware manifests, and owns
only its dedicated container. Humming remains 0.1.12 with H20-3e bm32 tables;
the FA3/kernel dependency target is separate from the server installation.
Exact topology is TP8/CP8/EP1. Client protocol is unchanged: seed 42 IDs,
65536/262144 inputs, output 1, cache off, serial concurrency 1, warmup 10 and
repeat 5 per length. Timed requests and GPU-only profiles are separate. Eight
complete gzip rank traces per length are required for client success.

Validation and failed preparation attempts are retained, not overwritten:

- `repro/1123-client-smoke`: actual server client smoke test (warmup 0, repeat 1;
  1015.501 ms cold 64K request), only a protocol check, not a performance result.
- `repro/1123-script-full`: port preflight initially rejected a recently closed
  socket in TIME_WAIT. Preflight now sets SO_REUSEADDR, matching server binding.
- `repro/1123-script-full-v2`: stopped before inference after discovering that
  the output parent is itself a Git repo and git apply had skipped paths.
  Git discovery is now bounded with GIT_CEILING_DIRECTORIES. A negative check
  against this directory confirms the blob verifier rejects skipped patches.
- `repro/1123-script-full-v3`: fresh image/container and isolated source passed
  both lengths (10 warmups + 5 samples each) and all 16 complete GPU-only traces.
  Median TTFT was 857.039 / 3529.827 ms, 76467.89 / 74265.40 input tok/s.
  The client wrote completed.json successfully. The outer shell exited 2 because
  its script was edited during execution; the final wrapper verification uses
  an immutable byte-identical copy of run.sh to avoid changing a running script.
- `repro/1123-script-final-source`: final zero-context patch preparation passed,
  including automatic blob verification for all 13 compatibility files.
- `repro/1123-script-final-full`: final one-shot run exited **0**, including
  automatic container cleanup. All 30 warmup/sample requests and two profile
  requests passed exact-length/cache/output checks; all 16 traces completed.
  Median TTFT: **857.278 / 3527.451 ms**; input throughput:
  **76446.60 / 74315.42 tok/s**, for 65536 / 262144 tokens respectively.
  Raw measurements, summary, launch command, source/runtime/image/hardware
  manifests and server log are saved in this directory. The final compatibility
  patch was verified automatically against all 13 expected resulting blobs.

Paths above are relative to `/data1/fjmeng2/dsv41-runtime/`. The working tree's
original 13 source modifications are preserved. Standalone kernel comparison
also passed against the existing baseline/candidate traces and newly captured
Ant/candidate GPU-only traces at both lengths. Reports are under
`repro/1123-repro-vs-ant/{65536,262144}/`: mean mHC 28.772/28.779 ms (Ant/local)
at 64K and 113.227/113.436 ms at 256K; TP0 spans 849.144/848.342 ms and
3506.041/3503.580 ms. These are profile diagnostics, separate from unprofiled
TTFT; kernel sums can overlap. Full GSM8K results for the unchanged serving
source remain the paired evaluation recorded above.

Submission checks: shell syntax and Python compilation/AST checks, exact input
hashes, negative cache/patch guards, source equality and all expected runtime
artifacts passed. Ruff 0.15.1 lint/format ran with the existing pinned binary
instead of downloading another copy for pre-commit. Applicable pre-commit
checks include whitespace/EOF, AST, executable modes, private keys, isort and
codespell; C++/notebook hooks have no applicable files. The launcher explicitly
passes the test admin key to the client, preserving the verified value even if
an unrelated shell admin-key environment variable is set.


## 2026-11-23: Commit the runtime compatibility fixes directly

The user requested normal source commits instead of a runtime patch. The
required compatibility code is now being committed in its source files. Eleven
files retain the tested Humming metadata, CP tail/shared-read, DSPARK plan/API,
output ownership, row-argmax and fake-transfer adapters. The temporary
SGLANG_DSPARK_DRAFT_DECODE_GRAPH switch is removed; its previously enabled
default execution path is preserved. An unused DSV4Metadata import is removed
and two files are formatted; neither changes execution.
Required import lint cleanup also imports existing symbols from their defining
modules and removes duplicate model imports, without replacing implementations.

Validation before the source commit:

- The eleven adapters matched the previously benchmarked source byte for byte
  before formatting/import cleanup. Default graph behavior is unchanged.
- RunnerIds, customized output streaming, deferred-KV commit, and DSPARK draft
  path tests: **71 passed, 33 subtests passed**. Log:
  `repro/1123-compat-selected-tests.log`.
- A broader first run had 73 passing tests plus four failures in existing
  deferred-KV epilogue cases using an unsupported `deferred_draft_kv` constructor
  argument. Both that test file and the DsparkVerifyEpilogue source are identical
  to pre-change HEAD. This pre-existing suite/API mismatch is retained in
  `repro/1123-compat-tests.log`; it is not reported as a passing suite.
- CUDA row_argmax agrees with torch for [1,64] x [4096,129280] shapes, in eager
  mode and three CUDA graph replays with new inputs per shape. Owned typed
  customized-info D2H rows survive overwriting the original persistent GPU
  buffer. Log: `repro/1123-compat-cuda.log`.
- Ruff 0.15.1 lint and formatting checks passed for all eleven source files.
- Applicable pre-commit checks passed, including indirect-import and runtime
  import-boundary checks. The selected tests and CUDA probes passed again after
  the required import cleanup.

Paths above are relative to `/data1/fjmeng2/dsv41-runtime/`. This is a source
compatibility commit, not a claim of another throughput improvement. The
reproduction launcher will export a committed revision directly and drop
compatibility.patch and patch-specific manifest logic. Clean-source full
prefill/profile validation is recorded in the follow-up entry.


## 2026-11-23: Prefill directly from committed source, no applied patch

Runtime adapters are committed as
`378dd4b110be92a41ffce162c4afdb671e9a0140`. The reproduction launcher now exports
the selected commit directly (HEAD by default, optional PREFILL_REVISION),
records source_revision and per-file SHA256, and applies no patches.
compatibility.patch and all patch-application/verification code are removed.

Full fresh-container run: `repro/1123-no-patch-full/`; wrapper log:
`repro/1123-no-patch-wrapper.log`. The source manifest records the runtime commit
above, and dependency-version mismatches are empty. Preparation also checked
that all eleven runtime adapter files equal that committed source.

| Input tokens | Median TTFT ms | Input tok/s | Samples |
| ---: | ---: | ---: | ---: |
| 65,536 | 857.227 | 76451.20 | 5 |
| 262,144 | 3528.738 | 74288.33 | 5 |

Both lengths used the unchanged Ant configuration, radix cache disabled,
output=1, concurrency=1, ten warmups per length and five timed samples. All
thirty warmup/sample requests and two separate GPU-only profile requests passed
the exact input/cache/output checks. All sixteen TP rank traces completed;
completed.json is true, the wrapper exited **0**, and automatic container
cleanup succeeded. This confirms reproduction from committed runtime code
retains the earlier performance; no extra optimization gain is claimed.

Script lint/format/syntax and applicable pre-commit checks also passed. The
original four stale deferred-KV epilogue cases remain documented in the prior
entry; this run does not claim those tests or fake-prefill decode were fixed.
All paths above are relative to `/data1/fjmeng2/dsv41-runtime/`.


## 2026-11-23: Equal-shard DP8/EP8 expert output over NVLink pull

The DeepSeek-V4 MoE row transport bypassed the generic layer-boundary return and
called variable-size reduce-scatter directly. With Humming and DP8/EP8 this became
grouped NCCL Reduce, despite the input gather already using the pull plane.

Route that return through the boundary operation. Equal positive rank-major
shard sizes, exact full/local shapes and matching CPU token metadata may use the
existing CustomAllReduceV2.reduce_scatter capability. Unequal/idle metadata and
unsupported dtype/layout/capture/role/TBO states retain the existing fallback.
No new communicator or tuning flag is introduced. The multi-tokenizer warmup
label uses the process ID, matching the newer manager's IPC registration model.

Matched H20x8, TP8/attention-DP8/EP8, Engram host per_rank, Humming0.1.12 with FP8
input and FP8 target LM head, DSPARK block5, memory.85, graph32, workers8:

- Decode-only + fake bootstrap,64x65536 input IDs,1024 output, simulated acceptance4.5,
  warmup1+repeat3:7465.003/7412.191/7388.002 tok/s, median7412.191, versus the
  matched startup-fixed base median6880.962 (+7.720%). Inputs and actual topology
  match. No performance claim from the first unwired/no-gain attempt.
- All8GPU-only profiles:22target verify rounds/rank. Output pull43calls/rank/round;
  grouped NCCL Reduce absent. NCCL6.840->.103ms/rank/round; custom collectives
  1.697->5.746ms including the new output pull. Kernel sums include waiting/overlap.
- Eight-GPU component validation: BF16 width5120,shards1/8/16/32/48/64,eager and
  four changed-input CUDA Graph replays with a lagging rank; bitwise equality to
  rank-ordered FP32 accumulation then BF16, plus NCCL tolerance comparison and
  uneven/idle fallback. Maximum observed absolute rounding delta from NCCL.125.
- Boundary/FFN-exit regressions:38tests and11subtests passed.
- Full real-prefill unified GSM8K:1281/1319=97.1190%, unchanged correct count,
  zero request/scoring errors. EvalScope1.15.0,4shot,temp0,max2048,concurrency64,
  simulation unset. Unified role=mix does NOT exercise decode-only pull; numerical
  and replay coverage above is separate evidence for that path.

Artifacts: /data1/fjmeng2/dsv41-runtime/repro/1123-dp8ep8-opt/ and
1123-decode-reanalysis/matrix/local-ep8-{accuracy-}comm-pull-wired/.
Future E48 tile experiments remain separate from this communication change.

## 2026-11-23: E48 Humming tuning for DP8/EP8 decode

The original E48 expert GEMMs selected large M tiles, while a table targeting
only 2304 route slots missed padded graph buckets. Add a tied M32 W13/W2 pair
at exactly 2304, 2880 and 3456 global route slots (384/480/576 rows, top-k 6).
These cover 8/10/12 requests per attention-DP rank with DSPARK block size 5.
The estimator continues to count invalid EP entries. Other E48 sizes retain
Humming 0.1.12 heuristics; EP4 E96 and draft E16 table entries are unchanged.
FP8 input, MXFP4 group-32 weights, FP32 accumulation and disabled stream-K
remain the measured settings.

Matched H20x8 deployment: TP8, attention-DP8 (dp_size=1, attn_dp_size=8),
EP4 or EP8, Engram host per_rank, Humming 0.1.12/bm32, FP8 target LM head,
DSPARK block5, memory .85, graph max batch32, max requests256, tokenizer
workers8. Performance uses decode-only fake prefill, 64 requests with exactly
65536 input IDs and 1024 output tokens, zero cached tokens, simulated
acceptance4.5, one warmup and three repetitions per scheduling scenario.
Accuracy uses real prefill in unified mode, with simulation disabled.

| Scheduling | EP8 communication-only tok/s | EP8 tuned tok/s | Gain | EP4 same-source tok/s |
| --- | ---: | ---: | ---: | ---: |
| Balanced (8 per DP rank) | 7456.037 | 10047.817 | 34.76% | 10501.897 |
| Skewed | 7352.647 | 8707.808 | 18.43% | 7752.440 |
| Natural | 7415.114 | 9999.811 | 34.86% | 10693.802 |

GPU-only profiles and separate stack mapping captures completed for both EP
configurations, including skewed GPU captures. Real-checkpoint operator checks
covered representative, high-total-load and hot-expert routes, changed inputs
and graph replays. Supplemental padded 480/576-row checks passed. Shipped-table
and selection checks passed (10 tests plus 20 subtests). A legacy runner suite
has 12 fixture errors from missing HummingMethod.may_process_input in pinned
Humming 0.1.12; the same failure reproduces on the pre-change source.

Full GSM8K (EvalScope1.15.0, 4-shot, temperature0, max2048, concurrency64):

- EP8 tuned: 1277/1319 = 96.8158%, zero request/scoring errors; two max-token stops.
- EP4 same source: 1277/1319 = 96.8158%, zero request/scoring errors; one max-token stop.
- EP8 communication-only: 1281/1319 = 97.1190%. The tuned result has four fewer
  correct answers (-0.3033 percentage points); paired-answer attribution remains
  unresolved. This experiment branch is published at the user's request and
  does not establish accuracy parity or production readiness.

Unified role=mix does not exercise decode-only pull; its separate numerical
and graph coverage is described in the communication entry above. The
final_matrix_completed.json adopt_e48 field records a performance gate only.

Tested source: /data1/fjmeng2/dsv41-runtime/repro/1123-dp8ep8-opt/07-e48-padded-buckets/source.
Both changed runtime files match that tested snapshot byte-for-byte.
Artifacts under /data1/fjmeng2/dsv41-runtime/repro/:

- 1123-decode-reanalysis/matrix/local-ep8-comm-reference-scenarios/
- 1123-decode-reanalysis/matrix/local-ep8-e48-padded-final/
- 1123-decode-reanalysis/matrix/local-ep8-accuracy-e48-padded-final/
- 1123-decode-reanalysis/matrix/local-ep4-both-supported-final/
- 1123-decode-reanalysis/matrix/local-ep4-accuracy-both-supported-final/
- 1123-dp8ep8-opt/{final-matrix-run.log,final_scenario_gains.json,pull_arrival_analysis.json}

Next: compare paired GSM8K answers and truncations before claiming accuracy
alignment, then use matched all-rank kernel timelines to investigate residual
EP8/EP4 differences and EP4's skewed scheduling cost.
