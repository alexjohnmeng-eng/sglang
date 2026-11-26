# H20 custom DeepGEMM dependencies

These packages extend pinned upstream sources. They are independent of the
SGLang runtime source and must be reproduced separately; a stock pip wheel
will not provide `torch.ops.deep_gemm.dsv41_wqb_q8_rope` or the isolated
`deep_gemm_mega` / `deep_gemm_mhc` modules.

| Package | Source pin | Patch | Purpose |
| --- | --- | --- | --- |
| dense | sgl-project/DeepGEMM b7f3816c7c0b | dense-wqb-rope-multicast.patch | SM90 block32 BMM support, WQB BF16-round/RoPE/FP8 output epilogue, shape-specific TMA multicast |
| mega | deepseek-ai/DeepGEMM 724a799bd043 (#473) | sm90-mxfp4-megamoe.patch | Isolated C++ symbols and DG473 cache namespace; prefill staging/shared-memory tuning |
| mhc | deepseek-ai/DeepGEMM 410ce17fb3cc (#521) | mega-mhc-isolation.patch | Isolated C++ symbols and DG521 cache namespace |

The patches retain FP32 accumulation and the checkpoint MXFP4 scales. SM90
uses FP8 WGMMA with in-kernel MXFP4 decoding. The dense epilogue preserves
the BF16 rounding before and after RoPE, followed by FP8 saturation. Multicast
selection is restricted to the measured projection shapes. The source and
binary fingerprints of the historical builds are in `manifest.json`.

Build from the host using the same `lmsysorg/sglang:latest` image (Python
3.12.16, torch 2.14.1+cu130). Choose fresh output directories:

```bash
bash benchmark/deepseek_v41_prefill/dependencies/build.sh dense /data1/fjmeng2/dsv41-runtime/build-dense-repro
bash benchmark/deepseek_v41_prefill/dependencies/build.sh mega /data1/fjmeng2/dsv41-runtime/build-mega-repro
bash benchmark/deepseek_v41_prefill/dependencies/build.sh mhc /data1/fjmeng2/dsv41-runtime/build-mhc-repro
```

The builder fetches each exact revision and submodules, checks/applies the
patch, verifies the patched source hashes, then builds in the image. It
installs `libdw-dev` in the disposable build container. `PIP_INDEX_URL` can
select another package index; it does not change the source revision pins.
The `.dist-info` remains alongside renamed Python packages for provenance.
New binary hashes are reported separately: compiler paths and build metadata
can make a valid rebuild differ from the historical binary fingerprint.

The serving entrypoint takes the resulting `site/` directories through
`PREFILL_DENSE_DEEPGEMM_SITE`, `PREFILL_DEEPGEMM473_SITE` and
`PREFILL_DEEPGEMM521_SITE`. It mounts them read-only. Dense uses the
`deep_gemm` name, whereas MegaMoE and mHC use isolated package names and C++
symbol visibility to avoid collision with the image's dense extension.

The EP8 protocol uses the included `placement-variable-copies.json` (384
logical experts, 448 physical expert slots). The JSON and its SHA256 are
checked in. No placement is applied to EP1 or DP8 decode by default.

The historical run directory `megamoe-32k` used a 16384-row pool for 32K
CP-local rows, split into two launches. Its archived source and GPU traces
confirm that the experimental 32K flag had no effect in that successful run.
Actual 32768-row pool attempts (`megamoe-32k-v2/v3`) failed the early-combine
bitmap limit. The tested #473 build has no `DG_SM90_PREFILL_EARLY_COMBINE`
override. This reproducer retains the validated 16K pool; it does not expose
the failed capacity experiment. Short calls keep the 8192-row pool.
