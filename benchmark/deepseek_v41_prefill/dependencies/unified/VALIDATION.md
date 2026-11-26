# Unified DeepGEMM H20 validation

**Historical results, before the WQB-off script update.** CP8/EP8 used WQB/RoPE/FP8 fusion; CP8/EP1 and DP8/EP8 did not. The current three-recipe scripts disable it everywhere and were not rebenchmarked or reevaluated, per the user's instruction. Do not assign the scores or throughput below to the changed CP8/EP8 recipe. Full current commands: [H20_TESTING.md](../../H20_TESTING.md).

One `deep_gemm._C`, distribution `deep-gemm-h20==2.8.0+h20.5`. All tested unified deployments load extension SHA256 `b8bd8d79f3c9a989927359b84bc9e693e14a7f8cdba2491210b743e512e922fb`. Image FlashInfer 0.7.2 and Humming 0.1.12 remain active.

| Deployment | Workload | Repeated original tok/s | Unified tok/s | Delta |
| --- | --- | ---: | ---: | ---: |
| cp8ep8 | 65536 input, output1 | 99,767.43 | 99,673.50 | -0.094% |
| cp8ep8 | 262144 input, output1 | 97,579.60 | 97,664.47 | +0.087% |
| ep1 | 65536 input, output1 | 92,681.94 | 92,757.03 | +0.081% |
| ep1 | 262144 input, output1 | 89,408.52 | 89,401.75 | -0.008% |
| DP8/EP8 | fake prefill64×65536, output1024 | 10,000.11 | 10,027.33 | +0.272% |

| Deployment | Original GSM8K | Unified GSM8K | Request errors |
| --- | ---: | ---: | ---: |
| cp8ep8 | 1277/1319 | 1282/1319 | 0 |
| ep1 | 1277/1319 | 1281/1319 | 0 |
| DP8/EP8 non-PD | 1281/1319 (previous validated baseline) | 1282/1319 | 0 |

Prefill: exact seed42 token IDs, cache disabled, 10 warmups and 5 samples per length; GPU-only profiles on all 8 ranks. Decode: warmup1/repeat3, simulated acceptance4.5 only for performance. Accuracy: real prefill, non-PD, no acceptance simulation, full GSM8K1319, 4-shot, temperature0, max_tokens2048, concurrency64.

CP8/EP8 kernel names, template parameters and counts match the original for both 64K and 256K. EP1 keeps generic dense tiles and paired-W2; decode keeps its Humming and pull communication paths. Kernel sums include synchronization/overlap and are not wall time.

Operator checks cover nonuniform scales against stock/donor dense, dense-with-PDL→mHC graph dependencies with changing producer inputs and poisoned intermediate outputs, metadata next_n1/2/4/5/6, and 8-rank M8192 MXFP4 MegaMoE with dense global PDL enabled.

This was source/dependency consolidation; the measured throughput stayed near the original baseline. No performance-gain claim. At the time of these measurements the work was uncommitted; the later local integration is documented in `OPTIMIZATION_PLAN.md`. No remote push was requested.

SGLang tree: `ab48f9269ab74ae4795d53f9d2da472cbd599866`; native patched tree: `3aaa7dc59774f997e95b387c6f9b2a9697c90fe4`.

Historical validation used SGLang tree `ab48f9269ab74ae4795d53f9d2da472cbd599866` and the launch/client scripts archived in each deployment's `tools/` directory. The current reviewed configuration is reproduced with:

```bash
bash benchmark/deepseek_v41_prefill/dependencies/unified/validate.sh \
  /data1/fjmeng2/dsv41-runtime/repro/deepgemm-unified-20261126/site-v5 \
  /absolute/new-output-dir perf/dsv41-h20
```

The current `validate.sh` uses the reviewed WQB-off recipes instead of the earlier WQB-on CP8/EP8 configuration. See `dependencies/unified/README.md` for the pinned source build and launch policy; see `H20_TESTING.md` for current commands. The tree above identifies historical source and is not the current integration commit.

Complete artifacts: `/data1/fjmeng2/dsv41-runtime/repro/deepgemm-unified-20261126`. Raw measurements and score reports are listed in `results-v5.json`; all experiments, including rejected candidates, are recorded in `/data1/fjmeng2/optimization_log.md`.
