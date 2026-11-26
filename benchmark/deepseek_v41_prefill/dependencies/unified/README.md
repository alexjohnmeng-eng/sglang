# H20 unified DeepGEMM

This is one source tree and **one** `deep_gemm._C` extension, not a package containing three renamed binaries. Version `2.8.0+h20.5`, distribution `deep-gemm-h20`.

The base is DeepGEMM #473 `724a799bd04382732df8ba73611b4a2702879ac6`, including native fused SM90 MXFP4 MegaMoE and its measured prefill heuristic. The self-contained patch ports the validated dense block32 GEMM/BMM, WQB→BF16-round→RoPE→FP8 epilogue and TMA multicast from the `b7f3816c7c0b` donor, and SM90 Mega mHC from #521 `410ce17fb3cc`. The provenance JSON records donor file hashes. `manifest.json` pins the resulting source files and submodules. All APIs share a single JIT runtime; hidden symbol visibility avoids collisions with unrelated extensions. The runtime is destroyed at Python exit before CUDA teardown.

The public module is `deep_gemm`; no `deep_gemm_mega` / `deep_gemm_mhc` directories are installed. The distribution has a unique name so the image's stock `sgl-deep-gemm` metadata cannot be mistaken for the active extension. No custom FlashInfer is needed: use image FlashInfer **0.7.2**, Humming **0.1.12**, Python **3.12.16**, Torch **2.14.1+cu130**.

Build using the image Python:

```bash
bash benchmark/deepseek_v41_prefill/dependencies/unified/build.sh /absolute/new-build-dir
```

Run the three deployment checks from a frozen SGLang revision/tree (these commands execute benchmarks and GSM8K; they were not rerun for the script-only update):

```bash
bash benchmark/deepseek_v41_prefill/dependencies/unified/validate.sh \
  /absolute/new-build-dir/site /absolute/new-validation-dir REVISION_OR_TREE
```

CP8/EP8 retains chunk262144, fixed 64-redundant-expert placement, the 16K MegaMoE buffer (32K CP-local rows use two launches), dense multicast, Mega mHC and Q8-TMA-SWA. CP8/EP1 retains chunk65536, paired-W2 and Mega mHC; WQB fusion, dense multicast and EP8 dispatch remain disabled. DP8/EP8 retains its Humming runner and communication path; only the dense library import changes. Decode performance uses fake prefill, 64×65536, output1024 and simulated acceptance4.5; GSM8K uses non-PD real prefill with no simulation. All three accuracy checks use full GSM8K1319, 4-shot, temperature0, max_tokens2048, concurrency64. Prefill measures exact65536/262144 token IDs, radix cache off, output1, warmup10/repeat5, and all8rank GPU-only profiles.

The current scripts explicitly **disable WQB/RoPE/FP8 fusion on all three topologies**, including CP8/EP8. Complete commands, switches, launch scripts, fake-prefill details and GSM8K protocol are in [H20_TESTING.md](../../H20_TESTING.md). `validate.sh` runs the same recipes serially. The script update was not rebenchmarked and GSM8K was not rerun, per the user's instruction. Earlier CP8/EP8 results had WQB fusion enabled and are historical evidence only.

The launcher accepts **only** `PREFILL_UNIFIED_DEEPGEMM_SITE` in unified mode and rejects simultaneous legacy DeepGEMM or FlashInfer site arguments. It sets `SGLANG_OPT_DSV41_UNIFIED_DEEPGEMM=1`. Per-topology optimization switches still determine which kernels run; unification does not independently enable EP8 on EP1 or decode. `SGLANG_DG_CACHE_DIR=/repro/cache/deepgemm-h20-unified` isolates JIT artifacts: setting `DG_JIT_CACHE_DIR` alone is insufficient because SGLang rewrites it. `runtime_manifest.json` records the active module, unique package version, extension hash, provenance and cache path.

`operator_compare.py` executes deterministic dense/WQB or mHC workloads, checks finite outputs and graph replay, and can compare snapshots from separate donor/unified processes bitwise. The separate-process contract avoids duplicate Torch operator registration. End-to-end test status and artifact paths belong in `OPTIMIZATION_PLAN.md`; source consolidation alone is not evidence of a throughput gain.

Standalone numerical and graph checks (execute inside the pinned image, with the unified `site` first on `PYTHONPATH`):

```bash
export PYTHONPATH=/absolute/new-build-dir/site
export DG_JIT_CACHE_DIR=/absolute/new-operator-cache
python3 benchmark/deepseek_v41_prefill/dependencies/unified/operator_compare.py \
  --suite dense --output /absolute/unified-dense.pt
python3 benchmark/deepseek_v41_prefill/dependencies/unified/operator_compare.py \
  --suite mhc --output /absolute/unified-mhc.pt
python3 benchmark/deepseek_v41_prefill/dependencies/unified/operator_megamoe.py \
  --np 8 --m 8 64 --fm 1 --hidden 5120 --ih 2304 --experts-per-rank 56 \
  --topk 6 --graph --bound --repeat 3
```

Dense BMM uses BF16 output on SM90; the FP8-output BMM API is restricted to SM100 in both the donor and unified builds. MegaMoE additionally checks an independent PyTorch reference and per-expert receive counts on all eight ranks. To compare donor outputs, run the dense suite in a separate process with only the donor dense site, and mHC with only its donor site plus `--module deep_gemm_mhc`, then use `--suite compare --reference LEGACY.pt --output UNIFIED.pt`.

The unified runtime keeps launch policy per kernel family. Dense GEMMs honor SGLang's global PDL setting. SM90 mHC explicitly disables PDL because its kernel has no griddepcontrol wait/trigger. SM90 MXFP4 MegaMoE adds a per-call `pdl: Optional[bool] = False`; `False` preserves the isolated donor's large-prefill policy, small calls retain their forced PDL, and explicit `None` inherits the global setting. No global switch is temporarily toggled, so another stream's dense launch is unaffected. Non-grouped GEMM `compiled_dims` defaults remain the dense donor's empty string. `DG_DSV41_DENSE_MULTICAST=1` also gates the CP8/EP8-specific forced tiles; EP1 and decode use generic tile selection.

The stronger operator checks use `--vary-scales` for nonuniform activation/weight scales, `--stock-compatible` for the stock image's generic SM90 interface (BMM gran128, without the custom WQB op), and `--chain-dense --suite mhc` for dense-with-PDL → mHC eager/graph dependency checks. The custom donor also supports BMM gran32; the image's stock SM90 BMM rejects it, so it is tested only against the custom donor. `operator_megamoe.py --dense-pdl --m 8192` checks a shared-runtime large-prefill call while dense global PDL is enabled. Metadata checks include next_n6 (DSPARK block5 target verification).

Historical measurements and source trees: [VALIDATION.md](VALIDATION.md). Current complete reproduction commands: [H20_TESTING.md](../../H20_TESTING.md). The chained graph test changes the producer input and clears its intermediate output before replay, so reusing the preceding replay's data cannot pass the check. For future uncommitted experiments, pass a frozen source tree explicitly rather than assuming that `HEAD` includes working-tree changes.

The generated patch uses zero context so inherited whitespace is not rewritten by repository formatting hooks. Apply it to the pinned base with `git apply --unidiff-zero`; the build script does this and verifies every final modified-file hash. Reconstructing the patch reproduces the exact tested native tree.
